"""Shift handover: move a robot and its in-flight missions to the next operator.

The old routine was "stop everything, close the session, next shift starts
from scratch". A handover instead transfers the lease together with a frozen
snapshot of mission progress, so the incoming operator sees exactly which
waypoint each mission had reached before they accept anything.

Mutual exclusion is the invariant that matters here: at every instant at most
one operator may command the robot. While a handover is ``pending`` the
command channel is frozen for *both* sides — the robot keeps executing its
current mission autonomously, but neither console can issue new operator
commands, and nobody can open a fresh session on the robot from the side.
The outgoing operator can always :meth:`HandoverManager.cancel` to take
control back; the handover also expires on its own deadline so a robot is
never locked forever by an operator who walked away.
"""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, replace
from typing import Iterable

from aegisrover.mission.progress import ProgressStore
from aegisrover.runtime.session import ACTIVE, Session, SessionRegistry
from aegisrover.storage.audit import AuditLog
from aegisrover.storage.repository import NotFound, Repository, VersionConflict

__all__ = (
    'HANDOVER_NAMESPACE', 'PENDING', 'ACCEPTED', 'REJECTED', 'CANCELLED', 'EXPIRED',
    'Handover', 'HandoverError', 'HandoverManager',
)

HANDOVER_NAMESPACE = 'handovers'
PENDING = 'pending'
ACCEPTED = 'accepted'
REJECTED = 'rejected'
CANCELLED = 'cancelled'
EXPIRED = 'expired'


class HandoverError(RuntimeError):
    pass


@dataclass(frozen=True)
class Handover:
    handover_id: str
    robot: str
    from_operator: str
    from_session: str
    to_operator: str
    note: str
    state: str
    created_at: float
    deadline: float
    resolved_at: float | None = None
    accepted_session: str | None = None
    progress: tuple[dict, ...] = ()
    revision: int = 1

    def to_dict(self) -> dict:
        return {
            'handover_id': self.handover_id,
            'robot': self.robot,
            'from_operator': self.from_operator,
            'from_session': self.from_session,
            'to_operator': self.to_operator,
            'note': self.note,
            'state': self.state,
            'created_at': self.created_at,
            'deadline': self.deadline,
            'resolved_at': self.resolved_at,
            'accepted_session': self.accepted_session,
            'progress': [dict(p) for p in self.progress],
            'revision': self.revision,
        }

    @staticmethod
    def from_dict(payload: dict) -> 'Handover':
        return Handover(
            handover_id=payload['handover_id'],
            robot=payload['robot'],
            from_operator=payload['from_operator'],
            from_session=payload['from_session'],
            to_operator=payload['to_operator'],
            note=payload.get('note', ''),
            state=payload['state'],
            created_at=float(payload['created_at']),
            deadline=float(payload['deadline']),
            resolved_at=payload.get('resolved_at'),
            accepted_session=payload.get('accepted_session'),
            progress=tuple(payload.get('progress') or ()),
            revision=int(payload.get('revision', 1)),
        )


class HandoverManager:
    """Drives the pending → accepted/rejected/cancelled/expired state machine."""

    def __init__(self, repository: Repository, sessions: SessionRegistry,
                 progress: ProgressStore | None = None, audit: AuditLog | None = None,
                 clock=time.time, id_factory=None):
        self._repo = repository
        self._sessions = sessions
        self._progress = progress
        self._audit = audit if audit is not None else AuditLog(repository, clock)
        self._clock = clock
        self._new_id = id_factory or (lambda: uuid.uuid4().hex[:12])

    # -- commands --------------------------------------------------------------
    def initiate(self, robot: str, to_operator: str, *, note: str = '',
                 ttl_seconds: float = 300.0, actor: str | None = None) -> Handover:
        """Offer the robot (and a progress snapshot) to the next shift."""
        if ttl_seconds <= 0:
            raise HandoverError('handover ttl must be positive')
        holder = self._sessions.lease_holder(robot)
        if holder is None:
            raise HandoverError(f'{robot} has no active session to hand over')
        if holder.operator == to_operator:
            raise HandoverError(f'{to_operator} already holds {robot}')
        if self.pending_for(robot) is not None:
            raise HandoverError(f'{robot} already has a pending handover')
        now = self._clock()
        snapshot = tuple(self._progress.briefing(robot)) if self._progress is not None else ()
        handover = Handover(
            handover_id=self._new_id(),
            robot=robot,
            from_operator=holder.operator,
            from_session=holder.session_id,
            to_operator=to_operator,
            note=note,
            state=PENDING,
            created_at=now,
            deadline=now + ttl_seconds,
            progress=snapshot,
        )
        self._repo.put(HANDOVER_NAMESPACE, handover.handover_id, handover.to_dict())
        self._audit.append(actor or holder.operator, 'handover.initiate', handover.handover_id,
                           {'robot': robot, 'from': holder.operator, 'to': to_operator,
                            'missions': len(snapshot)})
        return handover

    def accept(self, handover_id: str, *, actor: str, lease_seconds: float = 30.0,
               cpu_limit: float = 600.0, wall_limit: float = 1800.0,
               capabilities: Iterable[str] = ()) -> tuple[Handover, Session]:
        """Take over: close the old session, open the new one, mark accepted.

        The lease moves exactly once — the old session is closed before the new
        one opens, so the robot never has two commanders. Retrying an accept
        that crashed mid-way is safe: an already-closed old session is skipped.
        """
        handover = self.get(handover_id)
        self._require_pending(handover)
        if actor != handover.to_operator:
            raise HandoverError(f'only {handover.to_operator} can accept handover {handover_id}')
        if self._clock() >= handover.deadline:
            self._mark(handover, EXPIRED, 'handover expired before accept')
            raise HandoverError(f'handover {handover_id} expired')
        try:
            old = self._sessions.get(handover.from_session)
        except NotFound:
            old = None
        if old is not None and old.state == ACTIVE:
            self._sessions.close(old.session_id, actor=actor,
                                 reason=f'handed over to {handover.to_operator}')
        session = self._sessions.open(
            handover.robot, handover.to_operator, lease_seconds=lease_seconds,
            cpu_limit=cpu_limit, wall_limit=wall_limit, capabilities=capabilities, actor=actor)
        updated = replace(handover, state=ACCEPTED, resolved_at=self._clock(),
                          accepted_session=session.session_id, revision=handover.revision + 1)
        self._commit(handover, updated)
        self._audit.append(actor, 'handover.accept', handover_id,
                           {'robot': handover.robot, 'session': session.session_id})
        return updated, session

    def reject(self, handover_id: str, *, actor: str, reason: str = '') -> Handover:
        handover = self.get(handover_id)
        self._require_pending(handover)
        if actor != handover.to_operator:
            raise HandoverError(f'only {handover.to_operator} can reject handover {handover_id}')
        return self._mark(handover, REJECTED, reason or 'rejected by incoming operator',
                          actor=actor)

    def cancel(self, handover_id: str, *, actor: str) -> Handover:
        handover = self.get(handover_id)
        self._require_pending(handover)
        if actor != handover.from_operator:
            raise HandoverError(f'only {handover.from_operator} can cancel handover {handover_id}')
        return self._mark(handover, CANCELLED, 'cancelled by outgoing operator', actor=actor)

    def expire(self, now: float | None = None) -> list[Handover]:
        moment = self._clock() if now is None else now
        expired = []
        for handover in self.list_handovers(state=PENDING):
            if moment >= handover.deadline:
                expired.append(self._mark(handover, EXPIRED, 'handover deadline passed'))
        return expired

    # -- reads -----------------------------------------------------------------
    def get(self, handover_id: str) -> Handover:
        try:
            record = self._repo.get(HANDOVER_NAMESPACE, handover_id)
        except NotFound:
            raise HandoverError(f'unknown handover {handover_id!r}') from None
        return Handover.from_dict(record.payload)

    def list_handovers(self, *, robot: str | None = None, state: str | None = None) -> list[Handover]:
        items = [Handover.from_dict(r.payload) for r in self._repo.scan(HANDOVER_NAMESPACE)]
        if robot is not None:
            items = [h for h in items if h.robot == robot]
        if state is not None:
            items = [h for h in items if h.state == state]
        return sorted(items, key=lambda h: (h.created_at, h.handover_id))

    def pending_for(self, robot: str) -> Handover | None:
        """The live pending handover for ``robot``; lapses are expired on read."""
        for handover in self.list_handovers(robot=robot, state=PENDING):
            if self._clock() >= handover.deadline:
                self._mark(handover, EXPIRED, 'handover deadline passed')
                continue
            return handover
        return None

    def command_authority(self, robot: str) -> Session | None:
        """Who may command the robot right now — nobody while a handover pends."""
        if self.pending_for(robot) is not None:
            return None
        return self._sessions.lease_holder(robot)

    def briefing(self, robot: str) -> dict:
        """What the incoming operator looks at before accepting."""
        holder = self._sessions.lease_holder(robot)
        pending = self.pending_for(robot)
        missions = self._progress.briefing(robot) if self._progress is not None else []
        return {
            'robot': robot,
            'holder': None if holder is None else holder.to_dict(),
            'pending_handover': None if pending is None else pending.to_dict(),
            'missions': missions,
        }

    # -- internals -------------------------------------------------------------
    def _require_pending(self, handover: Handover) -> None:
        if handover.state != PENDING:
            raise HandoverError(f'handover {handover.handover_id} is {handover.state}')

    def _mark(self, handover: Handover, state: str, reason: str,
              actor: str = 'registry') -> Handover:
        updated = replace(handover, state=state, resolved_at=self._clock(),
                          revision=handover.revision + 1)
        self._commit(handover, updated)
        self._audit.append(actor, f'handover.{state}', handover.handover_id,
                           {'robot': handover.robot, 'reason': reason})
        return updated

    def _commit(self, previous: Handover, updated: Handover) -> None:
        record = self._repo.get(HANDOVER_NAMESPACE, previous.handover_id)
        try:
            self._repo.put(HANDOVER_NAMESPACE, previous.handover_id, updated.to_dict(),
                           expected=record.version)
        except VersionConflict:
            raise HandoverError(
                f'handover {previous.handover_id} changed while resolving') from None
