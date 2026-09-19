"""Acceptance tests for shift handover: the robot and its in-flight missions
move to the next operator as one unit, commands are mutually exclusive while
the handover pends, and the incoming operator can see exactly how far the
outgoing shift got."""
import pytest

from aegisrover.mission.execution import MissionExecution, WaypointRunner
from aegisrover.mission.progress import ProgressStore
from aegisrover.runtime.handover import (
    ACCEPTED, EXPIRED, PENDING, REJECTED, HandoverError, HandoverManager,
)
from aegisrover.runtime.session import ACTIVE, CLOSED, SessionRegistry
from aegisrover.service.platform import PlatformService, ServiceError
from aegisrover.storage.audit import AuditLog
from aegisrover.storage.repository import Repository


class Clock:
    def __init__(self, now=0.0):
        self.now = now

    def __call__(self):
        return self.now

    def tick(self, delta):
        self.now += delta
        return self.now


@pytest.fixture()
def repo():
    clock = Clock(100.0)
    repository = Repository(':memory:', clock=clock)
    yield repository
    repository.close()


@pytest.fixture()
def service(repo):
    return PlatformService(repo, clock=Clock(100.0))


WAYPOINTS = [(1.0, 0.0), (2.0, 0.0), (3.0, 0.0), (4.0, 0.0)]


def _run_to_index(service, robot, mission_id, positions):
    """Simulate the robot runtime: execute and checkpoint after each tick."""
    execution = MissionExecution(mission_id, WaypointRunner(WAYPOINTS, tolerance=0.5))
    execution.start((0.0, 0.0))
    for position in positions:
        execution.tick(position)
        service.progress.save(execution, robot, position)
    return execution


# ----------------------------------------------------------------- happy path
def test_handover_carries_running_mission_to_next_shift(service):
    service.open_session('robot-1', 'alice')
    service.create_mission('m1', WAYPOINTS)
    service.command_mission('m1', 'queue')
    service.command_mission('m1', 'assign', assignee='robot-1')
    service.command_mission('m1', 'start')
    _run_to_index(service, 'robot-1', 'm1', [(1.0, 0.0), (2.0, 0.0)])

    handover = service.initiate_handover('robot-1', 'bob', note='night shift')['handover']
    assert handover['state'] == PENDING
    # The snapshot tells bob exactly where alice got to before he accepts.
    assert handover['progress'][0]['waypoint_index'] == 2
    assert handover['progress'][0]['total_waypoints'] == 4

    briefing = service.handover_briefing('robot-1')
    assert briefing['pending_handover']['handover_id'] == handover['handover_id']
    assert briefing['missions'][0]['mission_id'] == 'm1'
    assert briefing['missions'][0]['progress'] == 0.5
    assert briefing['missions'][0]['mission_state'] == 'running'

    result = service.accept_handover(handover['handover_id'], actor='bob')
    assert result['handover']['state'] == ACCEPTED
    assert result['session']['operator'] == 'bob'
    assert service.sessions.lease_holder('robot-1').operator == 'bob'
    assert service.sessions.get(handover['from_session']).state == CLOSED

    # Bob resumes the mission at waypoint 2 — not from scratch.
    resumed = service.progress.restore('m1')
    assert resumed.runner.index == 2
    assert resumed.state == 'running'
    resumed.tick((3.0, 0.0))
    resumed.tick((4.0, 0.0))
    assert resumed.state == 'completed'
    assert resumed.complete() == 'completed'


# ------------------------------------------------------------- mutual exclusion
def test_pending_handover_freezes_commands_for_both_sides(service):
    service.open_session('robot-1', 'alice')
    service.create_mission('m1', WAYPOINTS)
    service.command_mission('m1', 'queue')
    service.command_mission('m1', 'assign', assignee='robot-1')

    handover = service.initiate_handover('robot-1', 'bob')['handover']

    # Neither side can drive the robot while the handover pends.
    assert service.handovers.command_authority('robot-1') is None
    with pytest.raises(ServiceError) as excinfo:
        service.command_mission('m1', 'start', actor='alice')
    assert excinfo.value.code == 'handover_pending'
    with pytest.raises(ServiceError) as excinfo:
        service.command_mission('m1', 'start', actor='bob')
    assert excinfo.value.code == 'handover_pending'
    # And nobody can sneak a fresh session in from the side.
    with pytest.raises(ServiceError) as excinfo:
        service.open_session('robot-1', 'mallory')
    assert excinfo.value.code == 'handover_pending'

    service.accept_handover(handover['handover_id'], actor='bob')
    assert service.handovers.command_authority('robot-1').operator == 'bob'
    assert service.command_mission('m1', 'start', actor='bob')['applied'] is True


def test_reject_returns_control_to_outgoing_operator(service):
    service.open_session('robot-1', 'alice')
    service.create_mission('m1', WAYPOINTS)
    service.command_mission('m1', 'queue')
    service.command_mission('m1', 'assign', assignee='robot-1')
    handover = service.initiate_handover('robot-1', 'bob')['handover']

    rejected = service.reject_handover(handover['handover_id'], actor='bob',
                                       reason='short staffed')['handover']
    assert rejected['state'] == REJECTED
    assert service.sessions.lease_holder('robot-1').operator == 'alice'
    assert service.handovers.command_authority('robot-1').operator == 'alice'
    assert service.command_mission('m1', 'start', actor='alice')['applied'] is True


def test_cancel_returns_control_to_outgoing_operator(service):
    service.open_session('robot-1', 'alice')
    handover = service.initiate_handover('robot-1', 'bob')['handover']
    cancelled = service.cancel_handover(handover['handover_id'], actor='alice')['handover']
    assert cancelled['state'] == 'cancelled'
    assert service.handovers.command_authority('robot-1').operator == 'alice'


def test_expired_handover_releases_the_lock(repo):
    clock = Clock(100.0)
    service = PlatformService(repo, clock=clock)
    service.open_session('robot-1', 'alice')
    handover = service.initiate_handover('robot-1', 'bob', ttl_seconds=10)['handover']
    assert service.handovers.command_authority('robot-1') is None

    clock.tick(11.0)
    assert service.handovers.pending_for('robot-1') is None
    assert service.handovers.get(handover['handover_id']).state == EXPIRED
    assert service.handovers.command_authority('robot-1').operator == 'alice'
    with pytest.raises(HandoverError):
        service.handovers.accept(handover['handover_id'], actor='bob')


def test_handover_guards(repo):
    clock = Clock(100.0)
    sessions = SessionRegistry(repo, audit=AuditLog(repo, clock), clock=clock)
    manager = HandoverManager(repo, sessions, clock=clock)
    sessions.open('robot-1', 'alice')

    with pytest.raises(HandoverError):
        manager.initiate('robot-1', 'alice')  # already holds it
    with pytest.raises(HandoverError):
        manager.initiate('robot-2', 'bob')  # no session to hand over

    handover = manager.initiate('robot-1', 'bob')
    with pytest.raises(HandoverError):
        manager.initiate('robot-1', 'carol')  # already pending
    with pytest.raises(HandoverError):
        manager.accept(handover.handover_id, actor='mallory')  # wrong operator
    with pytest.raises(HandoverError):
        manager.cancel(handover.handover_id, actor='bob')  # not the initiator

    manager.accept(handover.handover_id, actor='bob')
    with pytest.raises(HandoverError):
        manager.accept(handover.handover_id, actor='bob')  # already resolved


# ------------------------------------------------------------- progress store
def test_checkpoint_restore_preserves_runner_state(repo):
    clock = Clock(100.0)
    store = ProgressStore(repo, clock)
    execution = MissionExecution('m1', WaypointRunner(WAYPOINTS, tolerance=0.5))
    execution.start((0.0, 0.0))
    execution.tick((1.0, 0.0))
    execution.tick((2.0, 0.0))
    execution.pause()
    store.save(execution, 'robot-1', (2.0, 0.0))

    clock.tick(3600.0)  # next shift, possibly another process
    resumed = store.restore('m1')
    assert resumed.runner.index == 2
    assert resumed.state == 'paused'
    # Events survive the JSON round-trip (positions come back as lists).
    assert [e['event'] for e in resumed.events] == [e['event'] for e in execution.events]
    resumed.resume()
    resumed.tick((3.0, 0.0))
    assert resumed.runner.index == 3


def test_finished_mission_clears_its_checkpoint(repo):
    store = ProgressStore(repo, Clock(100.0))
    execution = MissionExecution('m1', WaypointRunner([(1.0, 0.0)], tolerance=0.5))
    execution.start((0.0, 0.0))
    execution.tick((1.0, 0.0))
    assert execution.state == 'completed'
    store.save(execution, 'robot-1', (1.0, 0.0))
    assert store.load('m1') is None
    assert store.restore('m1') is None
    assert store.briefing('robot-1') == []


def test_briefing_only_lists_missions_on_that_robot(repo):
    store = ProgressStore(repo, Clock(100.0))
    for robot, mission_id in (('robot-1', 'm1'), ('robot-2', 'm2')):
        execution = MissionExecution(mission_id, WaypointRunner(WAYPOINTS, tolerance=0.5))
        execution.start((0.0, 0.0))
        store.save(execution, robot, (0.0, 0.0))
    assert [m['mission_id'] for m in store.briefing('robot-1')] == ['m1']
