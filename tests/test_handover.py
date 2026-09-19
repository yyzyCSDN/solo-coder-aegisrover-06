"""Acceptance tests for shift handover: lease transfer with mission continuity.

The scenarios mirror the shift-change requirements: the robot and its running
missions move to the next operator as one unit, the robot is never controllable
by both shifts at once, and the incoming operator can see exactly which step
the previous shift reached before (and after) taking over.
"""
from types import SimpleNamespace

import pytest

from aegisrover.mission.lifecycle import InvalidTransition, MissionError, MissionService
from aegisrover.runtime.handover import (
    ACCEPTED, CANCELLED, EXPIRED, PENDING, HandoverError, HandoverManager,
)
from aegisrover.runtime.session import ACTIVE, CLOSED, SessionError, SessionRegistry
from aegisrover.service.platform import PlatformService, ServiceError
from aegisrover.storage.audit import AuditLog
from aegisrover.storage.repository import Repository


class Clock:
    def __init__(self, now=1000.0):
        self.now = now

    def __call__(self):
        return self.now

    def tick(self, delta):
        self.now += delta
        return self.now


@pytest.fixture()
def env():
    clock = Clock(1000.0)
    repo = Repository(':memory:', clock=clock)
    audit = AuditLog(repo, clock)
    missions = MissionService(repo, audit, clock)
    sessions = SessionRegistry(repo, audit, clock)
    handovers = HandoverManager(repo, sessions, missions, audit, clock)
    yield SimpleNamespace(clock=clock, repo=repo, audit=audit, missions=missions,
                          sessions=sessions, handovers=handovers)
    repo.close()


def _running_mission(env, mission_id='m1', robot='robot-1', waypoints=5):
    env.missions.create(mission_id, [(float(i), 0.0) for i in range(waypoints)], priority=3)
    env.missions.command(mission_id, 'queue')
    env.missions.command(mission_id, 'assign', assignee=robot)
    env.missions.command(mission_id, 'start')
    return mission_id


def test_handover_transfers_robot_and_running_missions(env):
    alice = env.sessions.open('robot-1', 'alice', lease_seconds=60)
    _running_mission(env)
    env.missions.report_progress('m1', waypoint_index=3, position=(3.2, 0.1),
                                 note='aisle 4 clear', actor='robot-1')
    offer = env.handovers.offer('robot-1', 'bob', actor='alice',
                                note='battery at 40%, dock after m1')
    assert offer.state == PENDING and offer.note.startswith('battery')
    # the incoming shift sees exactly where the previous shift got to
    snap = offer.snapshot['missions'][0]
    assert snap['mission_id'] == 'm1' and snap['state'] == 'running'
    assert snap['progress']['waypoint_index'] == 3
    assert snap['progress']['waypoints_total'] == 5
    assert snap['progress']['note'] == 'aisle 4 clear'
    assert snap['last_event']['command'] == 'start'

    accepted, session = env.handovers.accept(offer.handover_id, actor='bob')
    assert accepted.state == ACCEPTED and accepted.to_session_id == session.session_id
    assert env.sessions.get(alice.session_id).state == CLOSED
    assert env.sessions.lease_holder('robot-1').operator == 'bob'
    # the operating envelope carries over to the next shift
    assert session.budget.cpu_limit == alice.budget.cpu_limit
    # the mission was never stopped: state, progress and history carry over
    mission = env.missions.get('m1')
    assert mission.state == 'running'
    assert mission.progress['waypoint_index'] == 3
    assert [h['command'] for h in mission.history] == ['create', 'queue', 'assign', 'start']


def test_handover_never_gives_two_operators_control(env):
    alice = env.sessions.open('robot-1', 'alice', lease_seconds=60)
    offer = env.handovers.offer('robot-1', 'bob', actor='alice')
    # while the offer is pending the incoming shift has no control at all
    with pytest.raises(SessionError):
        env.sessions.open('robot-1', 'bob', lease_seconds=60)
    assert env.sessions.lease_holder('robot-1').operator == 'alice'
    env.handovers.accept(offer.handover_id, actor='bob')
    # after the transfer the old session is dead: alice cannot even heartbeat
    with pytest.raises(SessionError):
        env.sessions.heartbeat(alice.session_id)
    assert env.sessions.lease_holder('robot-1').operator == 'bob'
    with pytest.raises(SessionError):
        env.sessions.open('robot-1', 'alice', lease_seconds=60)


def test_handover_accept_only_by_receiver_and_replayable(env):
    env.sessions.open('robot-1', 'alice', lease_seconds=60)
    offer = env.handovers.offer('robot-1', 'bob', actor='alice')
    with pytest.raises(HandoverError):
        env.handovers.accept(offer.handover_id, actor='carol')
    accepted, session = env.handovers.accept(offer.handover_id, actor='bob')
    # a console retry after a timeout replays instead of failing
    again, again_session = env.handovers.accept(offer.handover_id, actor='bob')
    assert again.revision == accepted.revision
    assert again_session.session_id == session.session_id


def test_handover_cancel_keeps_control_with_outgoing(env):
    env.sessions.open('robot-1', 'alice', lease_seconds=60)
    offer = env.handovers.offer('robot-1', 'bob', actor='alice')
    cancelled = env.handovers.cancel(offer.handover_id, actor='alice')
    assert cancelled.state == CANCELLED
    with pytest.raises(HandoverError):
        env.handovers.accept(offer.handover_id, actor='bob')
    assert env.sessions.lease_holder('robot-1').operator == 'alice'


def test_handover_offer_lapses_and_robot_stays_with_outgoing(env):
    env.sessions.open('robot-1', 'alice', lease_seconds=600)
    env.handovers.offer('robot-1', 'bob', actor='alice', ttl_seconds=10)
    env.clock.tick(11)
    expired = env.handovers.expire()
    assert [h.state for h in expired] == [EXPIRED]
    assert env.sessions.lease_holder('robot-1').operator == 'alice'


def test_handover_accept_after_deadline_marks_offer_expired(env):
    env.sessions.open('robot-1', 'alice', lease_seconds=600)
    offer = env.handovers.offer('robot-1', 'bob', actor='alice', ttl_seconds=10)
    env.clock.tick(11)
    with pytest.raises(HandoverError):
        env.handovers.accept(offer.handover_id, actor='bob')
    assert env.handovers.get(offer.handover_id).state == EXPIRED


def test_handover_offer_requires_holding_the_lease(env):
    with pytest.raises(HandoverError):
        env.handovers.offer('robot-1', 'bob', actor='alice')  # nobody holds the robot
    env.sessions.open('robot-1', 'alice', lease_seconds=60)
    with pytest.raises(HandoverError):
        env.handovers.offer('robot-1', 'bob', actor='carol')  # carol does not hold it
    with pytest.raises(HandoverError):
        env.handovers.offer('robot-1', 'alice', actor='alice')  # cannot hand over to yourself
    env.handovers.offer('robot-1', 'bob', actor='alice')
    with pytest.raises(HandoverError):
        env.handovers.offer('robot-1', 'carol', actor='alice')  # one pending offer at a time


def test_interrupted_accept_adopts_existing_session_on_retry(env):
    alice = env.sessions.open('robot-1', 'alice', lease_seconds=60)
    offer = env.handovers.offer('robot-1', 'bob', actor='alice')
    # simulate a crash after the lease moved but before the handover was marked
    env.sessions.close(alice.session_id, reason='handover to bob')
    crashed = env.sessions.open('robot-1', 'bob', lease_seconds=60)
    accepted, session = env.handovers.accept(offer.handover_id, actor='bob')
    assert session.session_id == crashed.session_id  # adopted, not duplicated
    assert accepted.state == ACCEPTED


def test_progress_reporting_validates_state_and_bounds(env):
    env.missions.create('m1', [(0, 0), (1, 0), (2, 0)])
    with pytest.raises(InvalidTransition):
        env.missions.report_progress('m1', waypoint_index=1)  # not running yet
    env.missions.command('m1', 'queue')
    env.missions.command('m1', 'assign', assignee='robot-1')
    env.missions.command('m1', 'start')
    with pytest.raises(MissionError):
        env.missions.report_progress('m1', waypoint_index=4)  # beyond the last waypoint
    with pytest.raises(MissionError):
        env.missions.report_progress('m1', waypoint_index=1, position=(1.0,))
    updated = env.missions.report_progress('m1', waypoint_index=2, position=(1.5, 0.0))
    assert updated.progress['waypoint_index'] == 2
    assert env.missions.get('m1').progress['reported_by'] == 'robot'


def test_handover_service_contract():
    clock = Clock()
    repo = Repository(':memory:', clock=clock)
    service = PlatformService(repo, clock=clock)
    service.open_session('robot-1', 'alice', lease_seconds=60)
    service.create_mission('m1', [(0, 0), (1, 0), (2, 0)])
    service.command_mission('m1', 'queue')
    service.command_mission('m1', 'assign', assignee='robot-1')
    service.command_mission('m1', 'start')
    service.report_mission_progress('m1', 2, actor='robot-1')

    offered = service.offer_handover('robot-1', 'bob', actor='alice',
                                     note='shift ends', idempotency_key='of1')
    replay = service.offer_handover('robot-1', 'bob', actor='alice', idempotency_key='of1')
    assert replay.get('idempotent') is True
    handover_id = offered['handover']['handover_id']

    with pytest.raises(ServiceError) as excinfo:
        service.accept_handover(handover_id, actor='carol')
    assert excinfo.value.status == 409 and excinfo.value.code == 'handover_conflict'
    with pytest.raises(ServiceError) as excinfo:
        service.get_handover('nope')
    assert excinfo.value.status == 404 and excinfo.value.code == 'handover_not_found'

    accepted = service.accept_handover(handover_id, actor='bob', idempotency_key='ac1')
    assert accepted['session']['operator'] == 'bob'
    assert accepted['snapshot']['missions'][0]['progress']['waypoint_index'] == 2
    again = service.accept_handover(handover_id, actor='bob', idempotency_key='ac1')
    assert again.get('idempotent') is True
    assert service.health()['checks']['handovers']['pending'] == 0
    repo.close()


def test_handover_api_endpoints():
    from fastapi.testclient import TestClient

    from aegisrover.service.api import app

    client = TestClient(app)
    assert client.post('/v1/sessions', json={
        'robot': 'api-r1', 'operator': 'alice', 'lease_seconds': 60}).status_code == 200
    assert client.post('/v1/missions', json={
        'mission_id': 'api-h1', 'waypoints': [[0, 0], [1, 0], [2, 0]]}).status_code == 200
    for payload in ({'command': 'queue'},
                    {'command': 'assign', 'assignee': 'api-r1'},
                    {'command': 'start'}):
        assert client.post('/v1/missions/api-h1/commands', json=payload).status_code == 200
    assert client.post('/v1/missions/api-h1/progress', json={'waypoint_index': 1}).status_code == 200

    offered = client.post('/v1/handovers', json={
        'robot': 'api-r1', 'to_operator': 'bob', 'actor': 'alice', 'note': 'night shift'})
    assert offered.status_code == 200
    handover = offered.json()['handover']
    assert handover['snapshot']['missions'][0]['progress']['waypoint_index'] == 1

    listed = client.get('/v1/handovers', params={'robot': 'api-r1', 'state': 'pending'})
    assert [h['handover_id'] for h in listed.json()] == [handover['handover_id']]

    accepted = client.post(f"/v1/handovers/{handover['handover_id']}/accept",
                           json={'actor': 'bob'})
    assert accepted.status_code == 200
    assert accepted.json()['session']['operator'] == 'bob'
    # the old shift cannot grab the robot back: exactly one commander remains
    conflict = client.post('/v1/sessions', json={'robot': 'api-r1', 'operator': 'alice'})
    assert conflict.status_code == 409
