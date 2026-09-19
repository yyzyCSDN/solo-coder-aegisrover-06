"""Shift handover: pass a robot and its in-flight missions to the next operator.

Without a handover protocol a shift change means stopping the robot and
re-creating whatever it was doing — unfinished missions start over. Here the
outgoing operator offers a handover instead. The offer captures a snapshot of
every live mission on the robot (state, waypoint progress, last event) plus a
free-text note, so the incoming operator can see exactly where the previous
shift got to before deciding to accept.

Control is never shared. While an offer is pending the outgoing operator keeps
the lease and the incoming operator has none; accepting closes the old session
before the new one is opened, so a crash mid-transfer fails closed (no holder,
and the accept can simply be retried) rather than open (two holders). Missions
themselves are never touched by the transfer — state, progress and history
carry over untouched.
"""
from __future__ import annotations

import copy
import time
import uuid
from dataclasses import dataclass, replace

from aegisrover.mission.lifecycle import MissionService
from aegisrover.runtime.session import ACTIVE, Session, SessionError, SessionRegistry
from aegisrover.storage.audit import AuditLog
from aegisrover.storage.repository import Repository

__all__ = ('Handover', 'HandoverError', 'HandoverManager',
           'PENDING', 'ACCEPTED', 'CANCELLED', 'EXPIRED')

HANDOVER_NAMESPACE = 'handovers'
PENDING = 'pending'
ACCEPTED = 'accepted'
CANCELLED = 'cancelled'
EXPIRED = 'expired'

#: mission states that represent work still on the robot's plate
LIVE_MISSION_STATES = ('assigned', 'running', 'paused')

DEFAULT_TTL_SECONDS = 900.0


class HandoverError(RuntimeError):
    pass


@dataclass(frozen=True)
class Handover:
    handover_id: str
    robot: str
    from_operator: str
    to_operator: str
    note: str
    state: str
    from_session_id: str
    snapshot: dict
    offered_at: float
    expires_at: float
    to_session_id: str | None = None
    closed_at: float | None = None
    revision: int = 1
    history: tuple[dict, ...] = ()

    def to_dict(self) -> dict:
        return {
            'handover_id': self.handover_id,
            'robot': self.robot,
            'from_operator': self.from_operator,
            'to_operator': self.to_operator,
            'note': self.note,
            'state': self.state,
            'from_session_id': self.from_session_id,
            'to_session_id': self.to_session_id,
            'snapshot': copy.deepcopy(self.snapshot),
            'offered_at': self.offered_at,
            'expires_at': self.expires_at,
            'closed_at': self.closed_at,
            'revision': self.revision,
            'history': [dict(h) for h in self.history],
        }

    @staticmethod
    def from_dict(payload: dict) -> 'Handover':
        return Handover(
            handover_id=payload['handover_id'],
            robot=payload['robot'],
            from_operator=payload['from_operator'],
            to_operator=payload['to_operator'],
            note=payload.get('note', ''),
            state=payload['state'],
            from_session_id=payload['from_session_id'],
            snapshot=copy.deepcopy(payload.get('snapshot') or {}),
            offered_at=float(payload['offered_at']),
            expires_at=float(payload['expires_at']),
            to_session_id=payload.get('to_session_id'),
            closed_at=payload.get('closed_at'),
            revision=int(payload.get('revision', 1)),
            history=tuple(payload.get('history') or ()),
        )


class HandoverManager:
    """Coordinate the two-phase offer/accept of a robot between shifts.

    The manager never mutates mission state: missions keep running on their own
    records and are only read for snapshots. What changes hands is the session
    lease, and the change is ordered close-then-open so the robot can never be
    driven by both shifts at once.
    """

    def __init__(self, repository: Repository, sessions: SessionRegistry,
                 missions: MissionService, audit: AuditLog | None = None,
                 clock=time.time, id_factory=None):
        self._repo = repository
        self._sessions = sessions
        self._missions = missions
        self._audit = audit if audit is not None else AuditLog(repository, clock)
        self._clock = clock
        self._new_id = id_factory or (lambda: uuid.uuid4().hex[:12])

    # -- lifecycle -------------------------------------------------------------
    def offer(self, robot: str, to_operator: str, *, actor: str, note: str = '',
              ttl_seconds: float = DEFAULT_TTL_SECONDS) -> Handover:
        """Offer ``robot`` to the next shift, capturing a progress snapshot.

        The robot stays under the caller's control until the offer is accepted;
        only the current lease holder may offer a handover.
        """
        if not to_operator:
            raise HandoverError('to_operator is required')
        if ttl_seconds <= 0:
            raise HandoverError('ttl must be positive')
        holder = self._sessions.lease_holder(robot)
        if holder is None:
            raise HandoverError(f'no active session for {robot}')
        if holder.operator != actor:
            raise HandoverError(f'{robot} is held by {holder.operator}, not {actor}')
        if to_operator == actor:
            raise HandoverError('cannot hand over to yourself')
        if self.list_handovers(robot=robot, state=PENDING):
            raise HandoverError(f'{robot} already has a pending handover')
        now = self._clock()
        handover = Handover(
            handover_id=self._new_id(),
            robot=robot,
            from_operator=actor,
            to_operator=to_operator,
            note=note,
            state=PENDING,
            from_session_id=holder.session_id,
            snapshot=self.snapshot(robot, now),
            offered_at=now,
            expires_at=now + ttl_seconds,
            history=({'at': now, 'event': 'offer', 'actor': actor},),
        )
        self._repo.put(HANDOVER_NAMESPACE, handover.handover_id, handover.to_dict())
        self._audit.append(actor, 'handover.offer', handover.handover_id,
                           {'robot': robot, 'to_operator': to_operator})
        return handover

    def accept(self, handover_id: str, *, actor: str, lease_seconds: float = 30.0,
               cpu_limit: float | None = None,
               wall_limit: float | None = None) -> tuple[Handover, Session]:
        """Accept a pending offer and take over the robot in one ordered transfer.

        The outgoing session is closed first and the incoming one opened after,
        so the robot never has two commanders. Budget limits are inherited from
        the outgoing session unless overridden. Re-accepting an already
        completed handover is a replay, not an error — the console retries
        after timeouts.
        """
        record = self._repo.get(HANDOVER_NAMESPACE, handover_id)
        handover = Handover.from_dict(record.payload)
        if handover.state == ACCEPTED and actor == handover.to_operator:
            return handover, self._sessions.get(handover.to_session_id)
        now = self._clock()
        if handover.state != PENDING:
            raise HandoverError(f'handover {handover_id} is {handover.state}')
        if now >= handover.expires_at:
            self._transition(handover, EXPIRED, actor, 'offer lapsed', expected=record.version)
            raise HandoverError(f'handover {handover_id} expired before it was accepted')
        if actor != handover.to_operator:
            raise HandoverError(
                f'handover {handover_id} is offered to {handover.to_operator}, not {actor}')
        outgoing = self._sessions.get(handover.from_session_id)
        if outgoing.state == ACTIVE:
            self._sessions.close(outgoing.session_id, actor=actor,
                                 reason=f'handover to {handover.to_operator}')
        holder = self._sessions.lease_holder(handover.robot)
        if holder is not None and holder.operator == actor:
            session = holder  # an earlier accept crashed after opening the session
        else:
            try:
                session = self._sessions.open(
                    handover.robot, actor,
                    lease_seconds=lease_seconds,
                    cpu_limit=outgoing.budget.cpu_limit if cpu_limit is None else cpu_limit,
                    wall_limit=outgoing.budget.wall_limit if wall_limit is None else wall_limit,
                    actor=actor)
            except SessionError as exc:
                raise HandoverError(str(exc)) from None
        updated = self._transition(handover, ACCEPTED, actor, 'lease transferred',
                                   expected=record.version, to_session_id=session.session_id)
        return updated, session

    def cancel(self, handover_id: str, *, actor: str) -> Handover:
        """Withdraw a pending offer; the robot stays with the outgoing operator."""
        record = self._repo.get(HANDOVER_NAMESPACE, handover_id)
        handover = Handover.from_dict(record.payload)
        if handover.state != PENDING:
            raise HandoverError(f'handover {handover_id} is {handover.state}')
        if actor not in (handover.from_operator, handover.to_operator):
            raise HandoverError('only the outgoing or incoming operator can cancel')
        return self._transition(handover, CANCELLED, actor, f'cancelled by {actor}',
                                expected=record.version)

    def expire(self, now: float | None = None, *, actor: str = 'registry') -> list[Handover]:
        """Lapse pending offers past their TTL. Control stays with the outgoing shift."""
        moment = self._clock() if now is None else now
        expired = []
        for handover in self.list_handovers(state=PENDING):
            if moment >= handover.expires_at:
                record = self._repo.get(HANDOVER_NAMESPACE, handover.handover_id)
                expired.append(self._transition(handover, EXPIRED, actor, 'offer lapsed',
                                                expected=record.version))
        return expired

    # -- reads -----------------------------------------------------------------
    def get(self, handover_id: str) -> Handover:
        return Handover.from_dict(self._repo.get(HANDOVER_NAMESPACE, handover_id).payload)

    def list_handovers(self, *, robot: str | None = None, state: str | None = None,
                       to_operator: str | None = None) -> list[Handover]:
        items = [Handover.from_dict(r.payload) for r in self._repo.scan(HANDOVER_NAMESPACE)]
        if robot is not None:
            items = [h for h in items if h.robot == robot]
        if state is not None:
            items = [h for h in items if h.state == state]
        if to_operator is not None:
            items = [h for h in items if h.to_operator == to_operator]
        return sorted(items, key=lambda h: (h.offered_at, h.handover_id))

    def snapshot(self, robot: str, now: float | None = None) -> dict:
        """Live view of everything the next shift would inherit on ``robot``."""
        moment = self._clock() if now is None else now
        missions = []
        for mission in self._missions.list_missions():
            if mission.assigned_to != robot or mission.state not in LIVE_MISSION_STATES:
                continue
            missions.append({
                'mission_id': mission.mission_id,
                'state': mission.state,
                'priority': mission.priority,
                'progress': None if mission.progress is None else dict(mission.progress),
                'waypoints_total': len(mission.waypoints),
                'last_event': dict(mission.history[-1]) if mission.history else None,
            })
        missions.sort(key=lambda m: (-m['priority'], m['mission_id']))
        return {'robot': robot, 'captured_at': moment, 'missions': missions}

    # -- internals -------------------------------------------------------------
    def _transition(self, handover: Handover, state: str, actor: str, reason: str, *,
                    expected: int, **fields) -> Handover:
        now = self._clock()
        entry = {'at': now, 'event': state, 'actor': actor, 'reason': reason}
        updated = replace(handover, state=state, closed_at=now,
                          history=handover.history + (entry,),
                          revision=handover.revision + 1, **fields)
        self._repo.put(HANDOVER_NAMESPACE, handover.handover_id, updated.to_dict(),
                       expected=expected)
        self._audit.append(actor, f'handover.{state}', handover.handover_id,
                           {'robot': handover.robot, 'from_operator': handover.from_operator,
                            'to_operator': handover.to_operator, 'reason': reason})
        return updated
