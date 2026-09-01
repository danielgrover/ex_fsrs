# CLAUDE.md

Guidance for Claude Code (claude.ai/code) when working in this repository.

## Project overview

ExFsrs is an Elixir implementation of FSRS-6 (Free Spaced Repetition
Scheduler). Scheduling is dependency-free. An optional optimizer fits the 21
model weights to a user's review history and needs Nx, which is an optional
dependency: the `ExFsrs.Optimizer.*` modules are wrapped in
`if Code.ensure_loaded?(Nx)` and are not compiled without it.

`mix.exs` depends on stock hex Nx. `model: :loop` (used by the bench scripts,
not by the default optimizer path) needs an Nx carrying a fix for f64
gradients through `while` that is not yet upstream; a local checkout with it
is at `../nx/nx`, branch `fix/while-grad-f64`. Set `EX_FSRS_NX_PATH=../nx/nx`
before `mix deps.get` to use it. See `bench/NX_WHILE_GRAD_F64.md`.

## Commands

```bash
mix test                          # full suite, ~50s (one py-fsrs parity run dominates)
mix test --cover                  # coverage, 90% threshold; ExFsrs.AnkiDataset is ignored
mix test test/scheduler_test.exs  # one file
mix test --include dataset        # also run tests needing the gated Anki dataset
mix ci                            # compile --warnings-as-errors, format, test, credo --strict,
                                  # dialyzer, ex_dna --max-clones 0, reach.check
mix docs                          # ex_doc
EX_FSRS_NX_PATH=../nx/nx MIX_ENV=test mix run bench/temporal_eval.exs  # needs dataset + patched Nx
```

`mix ci` must pass before a change is done. `ex_dna --max-clones 0` means any
duplicated block across files fails the build; the FSRS math on the optimizer
side lives once, in `ExFsrs.Optimizer.Model`, for that reason.

## Architecture

Scheduling (`lib/fsrs.ex`, `lib/fsrs/`):

- `ExFsrs` — the card struct (`state`, `step`, `stability`, `difficulty`,
  `due`, `last_review`) and a convenience API over a default scheduler.
  `to_map/1` / `from_map/1` for storage; `from_map/1` raises on bad input.
- `ExFsrs.Scheduler` — the algorithm. `review_card/5` computes the new memory
  state (`compute_stability_difficulty`) and the new schedule
  (`next_schedule` → `step_through`/`graduate`) independently. Learning and
  relearning share `step_through/7`; they differ only in state atom and step
  list. `reschedule_card/3` replays logs.
- `ExFsrs.ReviewLog` — the record `review_card` returns: card *before* the
  review, rating, datetime, optional duration.

Optimizer (`lib/fsrs/optimizer.ex`, `lib/fsrs/optimizer/`):

- `ExFsrs.Optimizer` — `compute_optimal_parameters/2` (py-fsrs's training
  loop: 5 epochs, minibatches of 512 scored reviews, Adam with cosine
  annealing, per-parameter clamping, best-epoch selection by full-collection
  loss), `batch_loss/2`, `evaluate/2`, `predictions/2`. Options `:initialize`,
  `:regularization`, `:recency` are the fsrs-optimizer/fsrs-rs upgrades, all
  off by default so the parity test still pins py-fsrs.
- `Data` — review logs → `[{card_id, [%Review{}]}]`. A review is scored
  (`counts_for_loss?`) only when it follows an earlier review by ≥ 1 day.
  Card order (ascending id) and review order (datetime) fix the minibatch
  boundaries and therefore the whole optimization trajectory.
  `apply_recency_weights/2` ranks by time; `remove_outliers/1` is implemented
  but measured harmful.
- `Model` — the differentiable FSRS-6 math, the only copy on this side.
  Functions accept scalar or tensor ratings/elapsed days. Every float constant
  goes through `c/1` to stay f64.
- `Model.Batched` — padding, length-bucketing and masking so a minibatch is
  advanced as tensors; `step_columns/3` is one timestep for a column of cards.
- `Model.Loop` — the same as a `defn` `while` (fixed graph size, compilable
  with EXLA); calls `Batched.step_columns/3` from inside the loop.
- `Initialization` — fits `w[0..3]` from first-review outcomes (no Nx).
- `Adam`, `Loss` (BCE with torch's log clamp, L2 penalty), `Metrics`
  (RMSE(bins) and log loss as srs-benchmark defines them).

Test support: `test/support/fixtures.ex` loads py-fsrs's review-log CSV and the
recorded optimizer trace (`test/fixtures/`); `test/support/anki_dataset.ex`
loads fetched Anki collections for `:dataset` tests and the bench scripts.

## Key domain facts

- Ratings `:again | :hard | :good | :easy` map to 1..4. Card states
  `:learning → :review → :relearning`.
- Steps are **minutes**; `next_interval/2` and `maximum_interval` are
  **days**.
- Retrievability is `(1 + factor × elapsed_days / stability) ^ decay` with
  `decay = -w[20]` and `factor = 0.9^(1/decay) - 1`. Elapsed days are integer
  (`DateTime.diff/3` in `:day`), floored at 0.
- Reviews < 1 day after the last use the short-term stability formula;
  otherwise the long-term one. A first review takes initial values from
  `w[0..3]` (stability) and `w[4], w[5]` (difficulty).
- Difficulty is clamped to 1.0..10.0, stability floored at 0.001.

## Conventions

- Tests pin behaviour to reference implementations with golden values
  (`test/reference_test.exs`, `test/algorithm_validation_test.exs`, and the
  py-fsrs trace for the optimizer). When changing algorithm code, expect
  1e-9-level tolerances to catch it; do not loosen them.
- Bare float literals in Nx code become f32. Use `c/1` (`Nx.tensor(x, type: :f64)`).
- Outside `defn`, tensor arithmetic must use `Nx.*` functions, never `Kernel`
  operators.
- `Enum.sort/1` on `DateTime` structs is not chronological; sort with
  `DateTime` as the comparator or convert to unix time first.
- Deserialization raises `ArgumentError` on bad input; it never invents a
  default.
