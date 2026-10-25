# Public-release audit (v0.2)

This review fixes verified defects and records the remaining limits. It does not
certify that the software is bug-free, formally secure, scientifically calibrated,
or state of the art.

## Verification summary

The combined Python/browser suite completed **65 passing tests**, including six
axe-core WCAG cases (two roles plus guide, at two viewport widths), three bundled
Pages-runtime scenarios, algorithm/physics regressions and checkpoint tests.
Python lint/format and frontend formatting checks also pass. The corrected study
completed **18 jobs and 147,456 transitions**; model and report metadata are saved.

## Algorithm and experiment integrity

| Finding | Correction / evidence |
|---|---|
| Tabular Q-learning treated truncation as natural termination | Bootstraps on truncation; batting supplies a valid final delivery cue. Regression tests check both cases. |
| Adjacent episode seeds were reused by Q-learning runs | Seed once, advance the environment stream; use a separate exploration stream. Metadata records the protocol. |
| Existing checkpoints could be silently overwritten | Training and studies refuse existing artifacts unless `--overwrite` is explicit. |
| Interrupted JSON writes could leave partial reports | Core course metadata and aggregate study files use atomic replacement, rejecting nonfinite JSON numbers. |
| Q tables accepted nonfinite data or unknown encodings | Role, shape, finiteness and encoding checks run before inference. |
| Training configuration was incompletely recorded | Artifact version 2 records package versions, hyperparameters, requested/actual transitions and source hashes. |

All 18 course jobs were re-run at the same 8,192-transition budget. Current tables
are in [the course results](../course/RESULTS.md); the earlier values are archived
in [the v1 report](../research/COURSE_V1_RESULTS.md). This repeats published test
seeds as a regression evaluation; it is not a new blind test. Neural training
scores remain unchanged under the normal physics presets; tabular results change
with the corrected reset/exploration protocol. The bowling heuristic still beats
the learners on average. That negative result is retained.

The browser now includes Q-learning tables for both roles, selected by a declared
seed 0 rather than test performance. [The model card](../../web/models/model-card.json)
contains training metadata, checksums and limitations. They are frozen teaching
examples, not live training or a claim of superiority.

## Simulator and API reliability

- Bowling flight is bounded even for extreme custom friction; invalid physical
  configurations cannot silently hang the worker indefinitely.
- Action IDs reject booleans, fractional values, nonfinite values and out-of-range
  choices before advancing an episode. Shot components and score events are
  validated before changing the score.
- Illegal deliveries cannot be counted as legal balls. Only supported dismissals
  are accepted, and completed runs with a wicket require a run-out.
- Invalid HTTP Gym resets use a candidate environment and leave the old episode
  and its RNG untouched if validation fails.
- Previous playground environments are closed only after a new reset succeeds.
- The matched frozen batting/bowling pair still generates identical scores and
  opposite rewards through the two role interfaces over 50 evaluation seeds.

Default physics behavior and historical seeded reference rollouts are regression
checked. This audit did not tune physics coefficients to improve a reported RL
score.

## Browser, deployment and recovery

- Packaged application wheels use content-addressed URLs. Rebuilding in an empty
  wheel directory prevents selecting an older cached wheel after a version bump.
- Pyodide verifies the application wheel hash. Bundled demo models are checked
  before loading and use non-pickle NumPy loading. Corrupt demo weights disable
  that option while manual/random/rule-based play remains usable.
- Worker startup errors and timeouts produce a real retry path, creating a fresh
  worker instead of reusing a permanently rejected initialization promise.
- Mid-game runtime failure preserves client journal notes, resets stale replay
  state after recovery, and permits exporting the old notes without another game.
- The paused scene avoids repainting unchanged frames, reducing idle canvas work.
- Static-site tests run under a repository-style subpath, reject all third-party
  runtime requests, and exercise both roles and the bundled learned policy.
- Pages deployment still runs real browser checks before publishing the artifact.

The public Pages domain could not be retrieved from the audit workspace because
its outbound proxy rejected that destination. Browser checks used the generated
Pages artifact locally. A live deployment status or production uptime is not
claimed by those tests.

## Accessibility, presentation and maintainability

The review found and fixed missing mobile select names, an invalid ARIA label on
the outcome list, a low-contrast button transition, and keyboard-inaccessible
scrollable code examples. Pinned axe-core checks run for batting, bowling and the
field guide at mobile and desktop widths, alongside keyboard/replay/lesson and
connection-recovery browser tests. Automated checks are not a full human WCAG
audit, particularly for the visual canvas.

The site includes description/Open Graph metadata, a purpose-made share preview,
a favicon, plain-language learning guidance and explicit inference-vs-training
explanations. Python and frontend source are formatted for contributors; lint,
formatting and accessibility checks are included in CI.

## Limits that remain material

1. **Scientific validity:** pitch/contact/anticipation parameters are illustrative,
   scoring rates can be unrealistically high, and real-world calibration has not
   been performed. More training cannot validate those assumptions.
2. **Cricket scope:** approximate LBW and run-out logic, no full powerplay/field
   restrictions or full match tactics, and tactical delivery cues rather than
   pixel perception or frame-level bat control. Numerical flight/contact uses a
   fixed timestep rather than an exact continuous collision solver.
3. **Learning evidence:** only three training seeds at a pilot budget, wide
   uncertainty, aliased Q-table state and some collapsed neural policies. No
   simultaneous self-play, league, or demonstrated convergence.
4. **Hosting scope:** Pages runs privately in each visitor's worker. The separate
   Python HTTP playground is a local single-user development server, not an
   authenticated public backend. Browser training and PyTorch inference are not
   implemented; tabular inference is supported.
5. **Distribution:** no new license was selected on the owner's behalf. Choose
   and document appropriate licensing before describing the repository as
   permissively licensed open source.

## Reproduce the checks

```bash
pip install -e '.[dev,train,browser,analysis]'
npm ci --ignore-scripts
python -m playwright install chromium
ruff check cricket_rl tools tests
ruff format --check cricket_rl tools tests
npm run check:ui
python tools/build_site.py
CRICKET_BROWSER_TEST=1 CRICKET_STATIC_TEST=1 pytest -q
```

The corrected measurement command is:

```bash
python -m cricket_rl.course_study --steps 8192 --episodes 50 --seeds 0 1 2 \
  --output runs/audit-course-study
```

Use a new output path or explicit `--overwrite` for a repeat. Generated model and
measurement files stay under ignored `runs/`; compact public results and declared
small demo tables are committed.

## Wording the evidence supports

“Built a two-role cricket reinforcement-learning lab with Gymnasium environments,
a Python-in-browser teaching playground, baseline-controlled experiments, and
regression/accessibility checks. Investigated policy collapse and documented
where simple heuristics outperform trained agents.”

Do not turn that into a claim of real-world cricket accuracy, universal algorithm
superiority, formal security certification, or SOTA performance.
