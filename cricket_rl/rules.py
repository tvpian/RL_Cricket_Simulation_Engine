"""Limited-overs scorekeeping, separated from delivery physics."""

from dataclasses import dataclass
from cricket_rl.physics import Delivery
from cricket_rl.validation import integer


@dataclass
class Innings:
    ball_limit: int = 24
    wicket_limit: int = 3
    target: int | None = None
    runs: int = 0
    wickets: int = 0
    balls: int = 0
    attempts: int = 0
    extras: int = 0
    striker: int = 0
    non_striker: int = 1
    next_batter: int = 2
    free_hit: bool = False

    def __post_init__(self):
        integer(self.ball_limit, "Ball limit", 1, 300)
        integer(self.wicket_limit, "Wicket limit", 1, 10)
        if self.target is not None:
            integer(self.target, "Target", 1, 500)

    @property
    def complete(self):
        return (
            self.balls >= self.ball_limit
            or self.wickets >= self.wicket_limit
            or (self.target is not None and self.runs >= self.target)
        )

    def apply(self, delivery: Delivery):
        if self.complete:
            raise RuntimeError("Innings already complete")
        integer(delivery.runs, "Runs", 0)
        integer(delivery.extras, "Extras", 0)
        if not isinstance(delivery.legal, bool) or not isinstance(
            delivery.wicket, bool
        ):
            raise ValueError("Legal-ball and wicket flags must be booleans")
        if delivery.extra_type not in (None, "wide", "no_ball"):
            raise ValueError("Unknown extras type")
        if delivery.extra_type in ("wide", "no_ball") and (
            delivery.legal or delivery.extras < 1
        ):
            raise ValueError(
                "Illegal deliveries require extras and cannot count as legal balls"
            )
        if delivery.extra_type is None and not delivery.legal:
            raise ValueError("Illegal delivery must declare its extras type")
        if delivery.wicket and delivery.dismissal not in (
            "bowled",
            "caught",
            "lbw",
            "run_out",
        ):
            raise ValueError("Wicket must declare a supported dismissal")
        if delivery.wicket and delivery.runs and delivery.dismissal != "run_out":
            raise ValueError("Only a run-out can score completed runs with a wicket")
        if min(delivery.runs, delivery.extras) < 0:
            raise ValueError("Runs cannot be negative")
        protected = self.free_hit or delivery.extra_type == "no_ball"
        if delivery.wicket and protected and delivery.dismissal != "run_out":
            raise ValueError("Illegal dismissal on no-ball or free hit")
        self.attempts += 1
        self.runs += delivery.runs + delivery.extras
        self.extras += delivery.extras
        self.balls += int(delivery.legal)
        # Preserve a free hit after a wide; a new no-ball always awards one.
        if delivery.extra_type == "no_ball":
            self.free_hit = True
        elif delivery.legal:
            self.free_hit = False
        if delivery.runs % 2:
            self.striker, self.non_striker = self.non_striker, self.striker
        if delivery.wicket:
            self.wickets += 1
            # Run-outs assume the runner approaching the striker's end is out.
            if self.wickets < self.wicket_limit:
                self.striker = self.next_batter
                self.next_batter += 1
        if delivery.legal and self.balls % 6 == 0:
            self.striker, self.non_striker = self.non_striker, self.striker

    def as_dict(self):
        return dict(
            runs=self.runs,
            wickets=self.wickets,
            balls=self.balls,
            attempts=self.attempts,
            extras=self.extras,
            striker=self.striker,
            non_striker=self.non_striker,
            free_hit=self.free_hit,
            overs=f"{self.balls // 6}.{self.balls % 6}",
            complete=self.complete,
        )
