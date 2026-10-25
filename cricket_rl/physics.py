"""SI-unit trajectories and explicit fielder pursuit. Parameters are not calibrated."""
from dataclasses import dataclass, field
import numpy as np
from cricket_rl.batting import BattingPlan
from cricket_rl.shots import DIRECTIONS, POWERS

DT = 0.02
PITCH_LENGTH = 20.12


@dataclass(frozen=True)
class Pitch:
    restitution: float = 0.55
    friction: float = 0.18
    boundary: float = 65.0
    gravity: float = 9.81
    swing: float = 0.5
    pace: float = 34.0
    accuracy: float = 0.14
    no_ball_rate: float = 0.015

    def __post_init__(self):
        values = tuple(vars(self).values())
        if not all(np.isfinite(v) for v in values):
            raise ValueError("Pitch parameters must be finite")
        if not 0 < self.restitution < 1 or not 0 <= self.friction < 1:
            raise ValueError("Invalid bounce coefficients")
        if not 40 <= self.boundary <= 90 or not 15 <= self.pace <= 45:
            raise ValueError("Boundary must be 40–90 m and pace 15–45 m/s")
        if self.gravity <= 0 or self.swing < 0 or self.accuracy < 0:
            raise ValueError("Invalid physical parameter")
        if not 0 <= self.no_ball_rate <= 0.2:
            raise ValueError("No-ball rate must be in [0, 0.2]")


PITCHES = {
    "standard": Pitch(),
    "green": Pitch(restitution=.62, friction=.14, swing=1.1),
    "dry": Pitch(restitution=.43, friction=.27, swing=.25),
}


@dataclass
class Delivery:
    runs: int = 0  # bat runs (or completed running runs on a run-out)
    wicket: bool = False
    dismissal: str | None = None
    boundary: bool = False
    trajectory: list = field(default_factory=list)
    landing: tuple = (0.0, 0.0)
    extras: int = 0
    extra_type: str | None = None
    legal: bool = True
    fielder_trajectory: list = field(default_factory=list)
    contact_frame: int | None = None
    bounced: bool = False
    event: str = "dot"
    anticipated_action: int | None = None
    anticipation: float = 0.0
    readiness: float = 0.0
    surprise: float = 0.0
    batting_position: float = 0.0


def pursue(fielders, target, dt=DT, speed=6.0):
    """Advance fielders toward the ball without exceeding running speed."""
    delta = target - fielders
    distance = np.linalg.norm(delta, axis=1)
    return fielders + delta * np.minimum(1, speed * dt / np.maximum(distance, 1e-9))[:, None]


def running_outcome(time, distance):
    """Conservative running: first run 3.5 s, subsequent runs 3.0 s.

    A marginal attempt can produce a run-out; completed runs still count.
    """
    available = time + distance / 28 + .5  # throw speed and gather time
    completed = max(0, min(3, int((available - .5) / 3)))
    next_time = 3.5 + completed * 3
    return completed, completed < 3 and 0 < next_time - available < .22


def simulate(rng, line, length, aggression, skill, preference, fielders,
             pitch=Pitch(), *, record=True, handedness=1, free_hit=False, batting_plan=None, shot=None):
    """Three lines × yorker/good/short. Each result includes cricket events.

    Bat contact, LBW adjudication and catching are approximations; pursuit and
    trajectories are explicit. The bowler releases 18 m from the striker.
    """
    fielders = np.asarray(fielders, dtype=float).copy()
    if fielders.shape != (9, 2) or not np.isfinite(fielders).all():
        raise ValueError("Nine finite 2D fielder positions required")
    plan = batting_plan or BattingPlan()
    readiness, surprise = plan.response(line, length)
    if shot is not None:
        aggression = POWERS[shot.power]
        preference = shot.stance
        # An explicit action replaces the automatic batter's premeditation.
        plan = BattingPlan()
        readiness, surprise = 0., 0.
    preferred_x = handedness * (-.55, 0, .55)[preference]
    expected_line = preference if plan.expected_action is None else plan.expected_action // 3
    prepared_x = handedness * (-.55, 0, .55)[expected_line]
    batting_position = (1 - plan.confidence) * preferred_x + plan.confidence * prepared_x
    result = Delivery(anticipated_action=plan.expected_action,
                      anticipation=plan.confidence, readiness=readiness,
                      surprise=surprise, batting_position=batting_position)
    speed = float(np.clip(rng.normal(pitch.pace, 1.2), 10, 50))
    target_x = handedness * (-.55, 0., .55)[line] + rng.normal(0, pitch.accuracy)
    bounce_y = (1.2, 5., 9.)[length]
    bounce_time = (18 - bounce_y) / speed
    pos = np.array([target_x - rng.normal(0, .05), 18., 2.0])
    vel = np.array([0., -speed, (.5 * pitch.gravity * bounce_time**2 - 2) / bounce_time])
    swing = rng.normal(0, pitch.swing)
    no_ball = bool(rng.random() < pitch.no_ball_rate)
    bounced = False

    def capture():
        if record:
            result.trajectory.append(pos.copy().tolist())
            result.fielder_trajectory.append(fielders.copy().tolist())

    # Velocity Verlet keeps free-flight acceleration independent of timestep.
    while pos[1] > 0:
        capture()
        acceleration = np.array([swing, 0, -pitch.gravity])
        pos += vel * DT + .5 * acceleration * DT**2
        vel += acceleration * DT
        if pos[2] <= 0:
            pos[2] = -pos[2] * pitch.restitution
            vel[2] = abs(vel[2]) * pitch.restitution
            vel[:2] *= 1 - pitch.friction
            bounced = True
    pos[1] = 0
    pos[2] = max(0, pos[2])
    capture()
    # Simplified limited-overs extras: no-ball takes precedence over wide.
    if no_ball:
        result.extras, result.extra_type, result.legal = 1, "no_ball", False
    elif abs(pos[0]) > .9 or pos[2] > 2.2:
        result.extras, result.extra_type, result.legal = 1, "wide", False
        result.event = "wide"
        result.landing = tuple(pos[:2])
        return result
    protected = free_hit or no_ball
    mismatch = abs(target_x - batting_position)
    contact_prob = float(np.clip(.80 + .15 * skill - .22 * mismatch
                                - .12 * aggression - .12 * max(pos[2] - 1, 0)
                                + .22 * readiness - .10 * surprise, .1, .985))
    if plan.confidence == 0:
        contact_prob = min(contact_prob, .98)
    if rng.random() > contact_prob:
        # Straight missed deliveries may hit stumps; pad interception approximates
        # LBW only for in-line impacts below the bails (no DRS or pitching test).
        on_stumps = abs(pos[0]) <= .114 and pos[2] < .711
        if on_stumps and not protected:
            result.wicket = True
            result.dismissal = "lbw" if rng.random() < .3 else "bowled"
        result.event = result.dismissal or result.extra_type or "dot"
        result.landing = tuple(pos[:2])
        return result
    result.contact_frame = len(result.trajectory) - 1 if record else None
    timing_mean = .70 + .22 * skill + .20 * readiness - .12 * surprise
    timing_sd = .16 * (1 - .45 * readiness) + .05 * surprise
    timing = np.clip(rng.normal(timing_mean, timing_sd), .1, 1.1)
    shot_speed = (12 + 25 * aggression) * timing
    preferred_angle = handedness * (-.9, 0., .9)[preference]
    if readiness > 0:
        # A prepared batter can premeditate a gap, rather than merely hit harder.
        candidates = np.array([-1.2, -.6, 0., .6, 1.2])
        angles = np.arctan2(fielders[:, 0], fielders[:, 1])
        coverage = np.exp(-((angles[:, None] - candidates[None, :]) / .3)**2).sum(axis=0)
        gap = float(candidates[np.argmin(coverage)])
        preferred_angle = (1 - .6 * readiness) * preferred_angle + .6 * readiness * gap
    if shot is not None:
        preferred_angle = handedness * DIRECTIONS[shot.direction]
    angle = preferred_angle + rng.normal(0, .35 + .3 * (1 - skill))
    loft = rng.uniform(.08, .22) + aggression * rng.uniform(0, .9)
    # Poor timing can edge the ball behind the wicket to the slip cordon.
    edge = bool(timing < .55)
    if edge:
        angle = rng.choice([-1, 1]) * rng.uniform(1.9, 2.8)
        shot_speed *= .55
        loft = rng.uniform(.1, .5)
    vel = np.array([np.sin(angle), np.cos(angle), 0.]) * shot_speed * np.cos(loft)
    vel[2] = shot_speed * np.sin(loft)
    pos[2] = max(.2, pos[2])
    time, grounded, failed_catches = 0., False, set()
    result.event = "edge" if edge else "shot"
    for _ in range(1500):
        time += DT
        pos += vel * DT + np.array([0, 0, -.5 * pitch.gravity * DT**2])
        vel[2] -= pitch.gravity * DT
        vel[:2] *= np.exp(-.025 * DT)
        if pos[2] <= 0:
            pos[2] = 0
            vel[2] = -vel[2] * .35 if abs(vel[2]) > 1 else 0
            # Constant rolling deceleration, rather than exponential stopping.
            horizontal = np.linalg.norm(vel[:2])
            vel[:2] *= max(0, 1 - 1.5 * DT / max(horizontal, 1e-9))
            grounded = True
        if time > .3:
            fielders = pursue(fielders, pos[:2])
        capture()
        distance = float(np.linalg.norm(pos[:2]))
        if distance >= pitch.boundary:
            result.runs = 4 if grounded else 6
            result.boundary = True
            result.event = "four" if grounded else "six"
            break
        distances = np.linalg.norm(fielders - pos[:2], axis=1)
        nearest = int(np.argmin(distances))
        if distances[nearest] <= 1.2 and pos[2] < 2.3:
            if not grounded:
                if nearest in failed_catches:
                    continue
                failed_catches.add(nearest)
                if rng.random() < .85:
                    if not protected:
                        result.wicket, result.dismissal, result.event = True, "caught", "caught"
                        break
                    # A protected catch remains live: drop to ground and field.
                    pos[2] = 0
                    vel *= .1
                    grounded = True
                    result.event = "protected_catch"
            else:
                result.runs, runout = running_outcome(time, distance)
                if runout:
                    result.wicket, result.dismissal = True, "run_out"
                result.event = result.dismissal or ("runs" if result.runs else "dot")
                break
    else:
        raise RuntimeError("Delivery exceeded integration horizon")
    result.bounced = grounded
    result.landing = tuple(float(x) for x in pos[:2])
    return result
