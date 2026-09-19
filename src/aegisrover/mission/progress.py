"""Durable execution checkpoints so a mission survives a shift change.

Today a shift change means stopping the robot and starting the mission over,
because :class:`~aegisrover.mission.execution.MissionExecution` lives only in
memory. This module snapshots the runner (which waypoint it is on, what it
skipped, the last known position) into the repository on every tick, so the
incoming shift can restore the execution and continue from the waypoint the
outgoing shift had reached instead of returning to waypoint zero.

A checkpoint for a finished mission is useless — there is nothing left to
resume — so saving a terminal state clears the record instead of leaving
stale progress in the handover briefing.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Iterable

from aegisrover.mission.execution import Geofence, MissionExecution, WaypointRunner
from aegisrover.storage.repository import Repository

__all__ = ('ExecutionCheckpoint', 'ProgressError', 'ProgressStore', 'PROGRESS_NAMESPACE')

PROGRESS_NAMESPACE = 'execution'
TERMINAL_STATES = ('completed', 'aborted')

Point = tuple[float, float]


class ProgressError(RuntimeError):
    pass


@dataclass(frozen=True)
class ExecutionCheckpoint:
    """Point-in-time snapshot of a running mission on a robot."""

    mission_id: str
    robot: str
    state: str
    waypoint_index: int
    total_waypoints: int
    waypoints: tuple[Point, ...]
    tolerance: float
    skipped: tuple[Point, ...]
    position: Point | None
    events: tuple[dict, ...]
    abort_reason: str | None
    saved_at: float
    revision: int = 1

    @property
    def progress(self) -> float:
        if self.total_waypoints <= 0:
            return 1.0
        return self.waypoint_index / self.total_waypoints

    def to_dict(self) -> dict:
        return {
            'mission_id': self.mission_id,
            'robot': self.robot,
            'state': self.state,
            'waypoint_index': self.waypoint_index,
            'total_waypoints': self.total_waypoints,
            'waypoints': [list(p) for p in self.waypoints],
            'tolerance': self.tolerance,
            'skipped': [list(p) for p in self.skipped],
            'position': None if self.position is None else list(self.position),
            'events': [dict(e) for e in self.events],
            'abort_reason': self.abort_reason,
            'saved_at': self.saved_at,
            'revision': self.revision,
        }

    @staticmethod
    def from_dict(payload: dict) -> 'ExecutionCheckpoint':
        position = payload.get('position')
        return ExecutionCheckpoint(
            mission_id=payload['mission_id'],
            robot=payload['robot'],
            state=payload['state'],
            waypoint_index=int(payload['waypoint_index']),
            total_waypoints=int(payload['total_waypoints']),
            waypoints=tuple((float(x), float(y)) for x, y in payload['waypoints']),
            tolerance=float(payload['tolerance']),
            skipped=tuple((float(x), float(y)) for x, y in payload.get('skipped') or ()),
            position=None if position is None else (float(position[0]), float(position[1])),
            events=tuple(payload.get('events') or ()),
            abort_reason=payload.get('abort_reason'),
            saved_at=float(payload['saved_at']),
            revision=int(payload.get('revision', 1)),
        )

    def summary(self) -> dict:
        """The "which step is it on" line shown in a handover briefing."""
        return {
            'mission_id': self.mission_id,
            'state': self.state,
            'waypoint_index': self.waypoint_index,
            'total_waypoints': self.total_waypoints,
            'progress': round(self.progress, 6),
            'skipped': len(self.skipped),
            'position': self.position,
            'saved_at': self.saved_at,
        }


class ProgressStore:
    """Saves and restores mission execution state through the repository."""

    def __init__(self, repository: Repository, clock=time.time):
        self._repo = repository
        self._clock = clock

    def save(self, execution: MissionExecution, robot: str,
             position: Point | None = None) -> ExecutionCheckpoint:
        """Snapshot ``execution``; a terminal state clears the record instead."""
        existing = self._repo.maybe_get(PROGRESS_NAMESPACE, execution.mission_id)
        revision = 1 if existing is None else int(existing.payload.get('revision', 1)) + 1
        checkpoint = ExecutionCheckpoint(
            mission_id=execution.mission_id,
            robot=robot,
            state=execution.state,
            waypoint_index=execution.runner.index,
            total_waypoints=len(execution.runner.waypoints),
            waypoints=tuple(execution.runner.waypoints),
            tolerance=execution.runner.tolerance,
            skipped=tuple(execution.runner.skipped),
            position=position,
            events=tuple(dict(e) for e in execution.events),
            abort_reason=execution.abort_reason,
            saved_at=self._clock(),
            revision=revision,
        )
        if execution.state in TERMINAL_STATES:
            self.clear(execution.mission_id)
            return checkpoint
        self._repo.put(PROGRESS_NAMESPACE, execution.mission_id, checkpoint.to_dict())
        return checkpoint

    def load(self, mission_id: str) -> ExecutionCheckpoint | None:
        record = self._repo.maybe_get(PROGRESS_NAMESPACE, mission_id)
        return None if record is None else ExecutionCheckpoint.from_dict(record.payload)

    def restore(self, mission_id: str, *, fence: Geofence | None = None) -> MissionExecution | None:
        """Rebuild an execution at the saved waypoint, or ``None`` if unknown."""
        checkpoint = self.load(mission_id)
        if checkpoint is None:
            return None
        if checkpoint.state in TERMINAL_STATES:
            raise ProgressError(f'{mission_id} finished as {checkpoint.state}; nothing to resume')
        runner = WaypointRunner(checkpoint.waypoints, tolerance=checkpoint.tolerance)
        runner.index = checkpoint.waypoint_index
        runner.skipped = list(checkpoint.skipped)
        execution = MissionExecution(checkpoint.mission_id, runner, fence)
        execution.state = checkpoint.state
        execution.events = [dict(e) for e in checkpoint.events]
        execution.abort_reason = checkpoint.abort_reason
        return execution

    def clear(self, mission_id: str) -> None:
        if self._repo.maybe_get(PROGRESS_NAMESPACE, mission_id) is not None:
            self._repo.delete(PROGRESS_NAMESPACE, mission_id)

    def briefing(self, robot: str) -> list[dict]:
        """Progress lines for every mission with a live checkpoint on ``robot``."""
        items = []
        for record in self._repo.scan(PROGRESS_NAMESPACE):
            checkpoint = ExecutionCheckpoint.from_dict(record.payload)
            if checkpoint.robot == robot:
                items.append(checkpoint.summary())
        return sorted(items, key=lambda item: item['mission_id'])
