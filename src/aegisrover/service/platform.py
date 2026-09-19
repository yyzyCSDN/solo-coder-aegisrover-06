"""Application service: the operational contract the console and robots talk to.

The service layer is where three operational guarantees live. Commands carry an
idempotency key, so a console that retries after a timeout does not start a mission
twice. Writes carry the revision the caller read, so a stale edit is rejected instead
of overwriting somebody else's change. And every failure is reported as a
``ServiceError`` with a stable code and an HTTP status, so the console can show a
sensible message instead of a stack trace.
"""
from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass
from typing import Any

from aegisrover.mission.lifecycle import InvalidTransition, MissionError, MissionService
from aegisrover.mapping.revisions import MapFormatError, MapRepository, MapRevision
from aegisrover.runtime.handover import PENDING, HandoverError, HandoverManager
from aegisrover.runtime.session import SessionError, SessionRegistry
from aegisrover.storage.audit import AuditLog
from aegisrover.storage.event_store import EventStore
from aegisrover.storage.repository import NotFound, Repository, VersionConflict, canonical_json

__all__ = ('ServiceError', 'PlatformService')

IDEMPOTENCY_NAMESPACE = 'idempotency'


@dataclass(frozen=True)
class ServiceError(RuntimeError):
    code: str
    message: str
    status: int = 400
    details: dict | None = None

    def __str__(self) -> str:  # pragma: no cover - trivial
        return f'{self.code}: {self.message}'

    def to_dict(self) -> dict:
        return {'error': {'code': self.code, 'message': self.message,
                          'details': self.details or {}}}


class PlatformService:
    def __init__(self, repository: Repository, *, clock=time.time):
        self.repository = repository
        self.clock = clock
        self.audit = AuditLog(repository, clock)
        self.missions = MissionService(repository, self.audit, clock)
        self.sessions = SessionRegistry(repository, self.audit, clock)
        self.handovers = HandoverManager(repository, self.sessions, self.missions, self.audit, clock)
        self.maps = MapRepository(repository, self.audit, clock)
        self.events = EventStore(repository, clock)

    # -- missions --------------------------------------------------------------
    def create_mission(self, mission_id: str, waypoints, *, priority: int = 0,
                       required_capabilities=(), actor: str = 'operator',
                       idempotency_key: str | None = None) -> dict:
        def action() -> dict:
            mission = self.missions.create(mission_id, waypoints, priority=priority,
                                           required_capabilities=required_capabilities, actor=actor)
            return {'mission': mission.to_dict(), 'applied': True}

        return self._idempotent(f'create:{mission_id}', idempotency_key, action)

    def command_mission(self, mission_id: str, command: str, *, actor: str = 'operator',
                        expected_revision: int | None = None, assignee: str | None = None,
                        idempotency_key: str | None = None) -> dict:
        def action() -> dict:
            return self.missions.command(mission_id, command, actor=actor,
                                         expected_revision=expected_revision,
                                         assignee=assignee)

        return self._idempotent(f'{mission_id}:{command}', idempotency_key, action)

    def get_mission(self, mission_id: str) -> dict:
        try:
            return self.missions.get(mission_id).to_dict()
        except NotFound:
            raise ServiceError('mission_not_found', f'unknown mission {mission_id}', 404) from None

    def list_missions(self, *, state: str | None = None) -> list[dict]:
        return [m.to_dict() for m in self.missions.list_missions(state=state)]

    # -- maps ------------------------------------------------------------------
    def save_map(self, map_id: str, cells: dict, *, actor: str = 'operator',
                 if_match: int | None = None, note: str = '') -> dict:
        try:
            revision = self.maps.save(map_id, cells, expected_revision=if_match, actor=actor, note=note)
        except VersionConflict as exc:
            raise ServiceError('revision_conflict',
                               f'expected revision {exc.expected}, found {exc.actual}', 409) from None
        return {'revision': revision.revision, 'etag': revision.digest, 'cells': len(revision.cells)}

    def get_map(self, map_id: str, revision: int | None = None) -> dict:
        try:
            item = self.maps.get(map_id, revision)
        except KeyError:
            raise ServiceError('map_not_found', f'unknown map {map_id}', 404) from None
        return {'map_id': item.map_id, 'revision': item.revision, 'etag': item.digest,
                'cells': item.cells, 'parent': item.parent}

    # -- sessions --------------------------------------------------------------
    def open_session(self, robot: str, operator: str, **kwargs) -> dict:
        try:
            session = self.sessions.open(robot, operator, **kwargs)
        except SessionError as exc:
            raise ServiceError('session_conflict', str(exc), 409) from None
        return session.to_dict()

    def close_session(self, session_id: str, *, actor: str = 'operator', reason: str = '') -> dict:
        try:
            return self.sessions.close(session_id, actor=actor, reason=reason).to_dict()
        except NotFound:
            raise ServiceError('session_not_found', f'unknown session {session_id}', 404) from None

    # -- handover --------------------------------------------------------------
    def offer_handover(self, robot: str, to_operator: str, *, actor: str, note: str = '',
                       ttl_seconds: float = 900.0, idempotency_key: str | None = None) -> dict:
        def action() -> dict:
            handover = self.handovers.offer(robot, to_operator, actor=actor, note=note,
                                            ttl_seconds=ttl_seconds)
            return {'handover': handover.to_dict(), 'applied': True}

        return self._idempotent(f'handover:offer:{robot}:{to_operator}', idempotency_key, action)

    def accept_handover(self, handover_id: str, *, actor: str, lease_seconds: float = 30.0,
                        idempotency_key: str | None = None) -> dict:
        def action() -> dict:
            try:
                handover, session = self.handovers.accept(handover_id, actor=actor,
                                                          lease_seconds=lease_seconds)
            except NotFound:
                raise ServiceError('handover_not_found',
                                   f'unknown handover {handover_id}', 404) from None
            # A fresh snapshot rides along so the new operator sees where every
            # mission stands at the moment control actually changes hands.
            return {'handover': handover.to_dict(), 'session': session.to_dict(),
                    'snapshot': self.handovers.snapshot(handover.robot), 'applied': True}

        return self._idempotent(f'handover:accept:{handover_id}', idempotency_key, action)

    def cancel_handover(self, handover_id: str, *, actor: str) -> dict:
        try:
            handover = self.handovers.cancel(handover_id, actor=actor)
        except NotFound:
            raise ServiceError('handover_not_found', f'unknown handover {handover_id}', 404) from None
        return {'handover': handover.to_dict(), 'applied': True}

    def get_handover(self, handover_id: str) -> dict:
        try:
            return self.handovers.get(handover_id).to_dict()
        except NotFound:
            raise ServiceError('handover_not_found', f'unknown handover {handover_id}', 404) from None

    def list_handovers(self, *, robot: str | None = None, state: str | None = None,
                       to_operator: str | None = None) -> list[dict]:
        return [h.to_dict() for h in self.handovers.list_handovers(
            robot=robot, state=state, to_operator=to_operator)]

    def report_mission_progress(self, mission_id: str, waypoint_index: int, *,
                                position=None, note: str = '', actor: str = 'robot') -> dict:
        def action() -> dict:
            try:
                mission = self.missions.report_progress(mission_id, waypoint_index=waypoint_index,
                                                        position=position, note=note, actor=actor)
            except NotFound:
                raise ServiceError('mission_not_found', f'unknown mission {mission_id}', 404) from None
            return {'mission': mission.to_dict(), 'applied': True}

        return self._idempotent(f'{mission_id}:progress', None, action)

    # -- operations ------------------------------------------------------------
    def health(self) -> dict:
        breaks = self.audit.verify()
        expired = self.sessions.expire()
        expired_offers = self.handovers.expire()
        states: dict[str, int] = {}
        for mission in self.missions.list_missions():
            states[mission.state] = states.get(mission.state, 0) + 1
        tail = self.events.tail()
        checks = {
            'repository': {'ok': True, 'namespaces': list(self.repository.namespaces())},
            'audit_chain': {'ok': not breaks, 'breaks': [b.__dict__ for b in breaks]},
            'sessions': {'ok': True, 'active': len(self.sessions.list_sessions(state='active')),
                         'expired_now': len(expired)},
            'handovers': {'ok': True, 'pending': len(self.handovers.list_handovers(state=PENDING)),
                          'expired_now': len(expired_offers)},
            'events': {'ok': True, 'tail_seq': None if tail is None else tail.seq},
        }
        return {'status': 'ok' if all(c['ok'] for c in checks.values()) else 'degraded',
                'checks': checks, 'missions': states}

    def audit_export(self, *, after_seq: int = 0) -> dict:
        entries = [e.__dict__ for e in self.audit.entries() if e.seq > after_seq]
        digest = hashlib.sha256(canonical_json(entries).encode()).hexdigest()
        return {'entries': entries, 'etag': digest, 'breaks': [b.__dict__ for b in self.audit.verify()]}

    # -- internals -------------------------------------------------------------
    def _idempotent(self, scope: str, key: str | None, action) -> dict:
        record_key = None if key is None else f'{scope}:{key}'
        if record_key is not None:
            existing = self.repository.maybe_get(IDEMPOTENCY_NAMESPACE, record_key)
            if existing is not None:
                stored = dict(existing.payload['response'])
                stored['idempotent'] = True
                return stored
        try:
            response = action()
        except VersionConflict as exc:
            raise ServiceError('revision_conflict',
                               f'expected revision {exc.expected}, found {exc.actual}', 409) from None
        except InvalidTransition as exc:
            raise ServiceError('invalid_transition', str(exc), 409) from None
        except HandoverError as exc:
            raise ServiceError('handover_conflict', str(exc), 409) from None
        except MissionError as exc:
            raise ServiceError('mission_error', str(exc), 422) from None
        except MapFormatError as exc:
            raise ServiceError('map_format_error', str(exc), 422) from None
        if record_key is not None:
            self.repository.put(IDEMPOTENCY_NAMESPACE, record_key,
                                {'response': response, 'recorded_at': self.clock()})
        return response
