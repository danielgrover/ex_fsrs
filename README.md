> **⚠️ Fork Notice:** Fork of [open-spaced-repetition/ex_fsrs](https://github.com/open-spaced-repetition/ex_fsrs) updated from FSRS v5 (19-parameter) to **FSRS v6 (21-parameter)**.
> 
> * **Validation:** Verified for parity by cross-checking against the `ts-fsrs`, `py-fsrs`, and `fsrs-rs` test suites. 
> * **Disclaimer:** Maintained for personal use; support and stability are best-effort—use at your own risk.

# ExFsrs

**An Elixir implementation of FSRS-6 (Free Spaced Repetition Scheduler)**

[![Elixir](https://img.shields.io/badge/Lang-Elixir-purple.svg)](https://elixir-lang.org/)

A spaced repetition scheduling library in Elixir implementing the FSRS-6 algorithm. Computes optimal review intervals for flashcards based on card difficulty, stability, and user feedback. Supports optional interval fuzzing to avoid predictable review dates. Scheduling has zero required dependencies; the optional parameter optimizer needs [Nx](https://hex.pm/packages/nx).

## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Usage](#usage)
- [Modules](#modules)
  - [ExFsrs](#exfsrs)
  - [Scheduler](#scheduler)
  - [ReviewLog](#reviewlog)
  - [Optimizer](#optimizer)
- [Testing](#testing)
- [Examples](#examples)
- [Contributing](#contributing)
- [License](#license)

---

## Overview

This project implements the [FSRS-6](https://github.com/open-spaced-repetition/fsrs4anki/wiki/The-Algorithm) algorithm in Elixir. FSRS-6 uses 21 trained model weights to schedule spaced repetition reviews with high accuracy.

Key features:
- **FSRS-6 algorithm** with 21 parameters (trained model weights).
- **Adaptive scheduling** based on a card's difficulty, stability, and prior performance.
- **Fuzzing (optional)** to randomize intervals, preventing overly predictable schedules.
- **Learning, Review, Relearning states** with dedicated logic for each phase.
- **Parameter optimization (optional)** — train the 21 weights on your own review history.
- **Rescheduling** — replay review logs through a new scheduler with different parameters.
- **Serialization** — cards and review logs convert to/from maps for storage.

---

## Installation

Add ExFsrs as a dependency in your `mix.exs`:

```elixir
def deps do
  [
    {:ex_fsrs, "~> 0.2.0", git: "https://github.com/danielgrover/ex_fsrs"}
  ]
end
```

Scheduling needs nothing else. To use `ExFsrs.Optimizer`, also add Nx — it is an
optional dependency, so it is not installed unless you ask for it:

```elixir
def deps do
  [
    {:ex_fsrs, "~> 0.2.0", git: "https://github.com/danielgrover/ex_fsrs"},
    {:nx, "~> 0.13"}
  ]
end
```

Without Nx, `ExFsrs.Optimizer` and its submodules are simply not compiled.

Then fetch and compile:

```bash
mix deps.get
mix compile
```

---

## Usage

Below is a quick example demonstrating how you might use the core ExFsrs module to process a review for a given card:

```elixir
# Create a new card
card = ExFsrs.new(state: :learning, step: 0)

# Review the card with a rating
{updated_card, review_log} = ExFsrs.review_card(card, :good)

# Or use the scheduler directly with custom parameters
scheduler = ExFsrs.Scheduler.new(
  parameters: [
    0.212, 1.2931, 2.3065, 8.2956, 6.4133, 0.8334, 3.0194,
    0.001, 1.8722, 0.1666, 0.796, 1.4835, 0.0614, 0.2629,
    1.6483, 0.6014, 1.8729, 0.5425, 0.0912, 0.0658, 0.1542
  ],
  desired_retention: 0.9,
  learning_steps: [1.0, 10.0],
  relearning_steps: [10.0],
  maximum_interval: 36500,
  enable_fuzzing: true
)

{updated_card, review_log} = ExFsrs.Scheduler.review_card(scheduler, card, :good)
```

What happens under the hood?

1. **Card State Update**
   The card's state is updated based on the rating and current state (learning, review, or relearning).

2. **Difficulty & Stability Calculation**
   The scheduler computes new difficulty and stability values based on the rating and time since last review.

3. **Interval Computation**
   Based on the new difficulty, stability, and rating, the next review interval is calculated. If fuzzing is enabled, the interval may be slightly randomized.

4. **Logging**
   A ReviewLog is created to track the review outcome, including the rating, review datetime, and updated card state.

---

## Modules

### ExFsrs
The main module that provides the card struct and public API entry point.

```elixir
defmodule ExFsrs do
  @type t :: %__MODULE__{
    card_id: integer(),
    state: :learning | :review | :relearning,
    step: integer() | nil,
    stability: float() | nil,
    difficulty: float() | nil,
    due: DateTime.t(),
    last_review: DateTime.t() | nil
  }

  defstruct [
    :card_id,
    :state,
    :step,
    :stability,
    :difficulty,
    :due,
    :last_review
  ]
end
```

### Scheduler
Handles the core FSRS-6 algorithm, including interval calculation, state transitions, and stability/difficulty updates. Configurable with 21 model weights.

```elixir
defmodule ExFsrs.Scheduler do
  @type t :: %__MODULE__{
    parameters: [float()],
    desired_retention: float(),
    learning_steps: [float()],
    relearning_steps: [float()],
    maximum_interval: integer(),
    enable_fuzzing: boolean(),
    decay: float(),
    factor: float()
  }

  defstruct [
    :parameters,
    :desired_retention,
    :learning_steps,
    :relearning_steps,
    :maximum_interval,
    :enable_fuzzing,
    :decay,
    :factor
  ]
end
```

### ReviewLog
Tracks the outcome of a review, including the rating and updated card state.

```elixir
defmodule ExFsrs.ReviewLog do
  @type t :: %__MODULE__{
    card: ExFsrs.t(),
    rating: ExFsrs.rating(),
    review_datetime: DateTime.t(),
    review_duration: integer() | nil
  }

  defstruct [
    :card,
    :rating,
    :review_datetime,
    :review_duration
  ]
end
```

### Optimizer

Trains the 21 model weights on your own review history. Requires the optional
`:nx` dependency.

```elixir
# Accepts %ExFsrs.ReviewLog{} structs or {card_id, rating, review_datetime}
# tuples -- the tuple form avoids building a card struct per row when loading
# a large history out of storage.
parameters = ExFsrs.Optimizer.compute_optimal_parameters(review_logs)

scheduler = ExFsrs.Scheduler.new(parameters: parameters)
```

Returns the default parameters unchanged if there is too little data (fewer
than 512 reviews that follow an earlier review of the same card by at least a
day). `ExFsrs.Optimizer.batch_loss/2` scores a parameter set against a
collection, which is the cheapest way to confirm the optimized weights actually
beat the defaults on your data.

Expect roughly 50 seconds for a 12,500-review collection (measured on an Apple
M2, Nx's default `BinaryBackend`); py-fsrs does the same run in 19s. Cost grows
a little faster than linearly with review count.

Cards in a minibatch are advanced together as tensors rather than one review at
a time, which is ~2.9x faster than the scalar formulation. Both are kept: pass
`model: :scalar` for the reference implementation, which the test suite diffs
the batched one against.

A third model, `model: :loop`, expresses the timestep loop as a `defn` `while`
so the expression graph stays a fixed size, which makes it compilable:

```elixir
ExFsrs.Optimizer.compute_optimal_parameters(logs, model: :loop, compiler: EXLA)
```

That runs the reference collection in ~25s. It requires `:exla` and an Nx that
computes f64 gradients through `while` correctly — released versions do not, and
fail silently rather than raising, so `model: :loop` checks at startup and
refuses to run otherwise. See `bench/NX_WHILE_GRAD_F64.md`.

### The upgrades, and what they are worth

Three of `fsrs-optimizer`/`fsrs-rs`'s refinements are implemented, each off by
default because each departs from the py-fsrs behaviour the parity tests pin:

```elixir
ExFsrs.Optimizer.compute_optimal_parameters(logs,
  initialize: true,      # fit w[0..3] from data instead of starting at defaults
  regularization: 2.0    # L2 pull toward the starting weights
)

# recency weighting is applied to the data rather than passed as an option
weighted = ExFsrs.Optimizer.Data.apply_recency_weights(sequences, days_by_card)
```

Measured across **83 real collections** from the Anki Revlogs 10K dataset, each
trained on its own past and scored on its own future
(`bench/temporal_eval.exs`), with the per-epoch card shuffle pinned so every
variant sees identical conditions:

| variant | total loss | per-collection mean | median | improved | Wilcoxon p |
|---|---|---|---|---|---|
| `initialize` + `regularization: 2.0` | +1.76% | +4.33% | +0.63% | 49/83 | **0.004** |
| the same, plus recency weighting | +2.03% | +4.85% | +1.03% | 56/83 | **0.0008** |

Recency weighting is worth adding on top: it beats the same configuration
without it on 54 of 83 collections (p = 0.012). A Friedman test across the three
configurations gives p = 0.0006.

Three aggregations are quoted because they disagree and each answers a different
question. **Total loss** is dominated by the collections with the largest losses.
**Per-collection mean** weights every collection equally and is pulled upward by
a few large winners — the best collection improves by 73%, the worst regresses
by 25%. **Median** is what a typical collection gets, and it is about 1%.

The effect depends strongly on collection size:

| training reviews | collections | effect |
|---|---|---|
| under 2,000 | 7 | +0.4% |
| 2,000-4,000 | 26 | **+7.4%** |
| 4,000-8,000 | 37 | +4.8% |
| over 8,000 | 13 | +2.3% |

Both ends have little to gain, for opposite reasons: below a couple of thousand
reviews there is not enough signal to fit anything better, and above eight
thousand plain py-fsrs optimization already has enough data to find good
parameters on its own. The upgrades earn their keep in the middle.

Two cautions worth carrying, both learned the hard way here:

* **Score held-out future, not held-out cards.** A card holdout rated
  initialization at +2.2% (p = 0.002) and regularization at noise. Splitting the
  same collections temporally reversed both. Held-out cards come from the period
  the parameters were fitted on, so the fit flatters itself.
* **Pin the shuffle.** The per-epoch card shuffle alone moves held-out loss by a
  median of 2% run to run — as much as the largest difference between any two
  variants. `bench/noise_floor.exs` measures it; any comparison that does not
  control for it is reading noise. An earlier 23-collection run found nothing
  significant, and it took both fixes plus 60 more collections to resolve
  effects this size.

### What the loss numbers do not say

Log loss understates what these parameters do to a schedule. Days until the next
review, after a card's first review:

| collection | again | hard | good | easy |
|---|---|---|---|---|
| defaults | 1d | 1d | 2d | 8d |
| 3929 | 1d | 1d | 4d | 8d |
| 9861 | 4d | 29d | **70d** | 70d |
| 9881 | 1d | 1d | 6d | 19d |

Collection 9861's fitted weights schedule a "Good" first review 70 days out
where the defaults say 2 — a 35x difference — on a collection whose loss barely
moved. Loss averages over thousands of reviews, most of them easy to predict
either way; the interval is a direct function of stability.

That is the real shape of the result. The defaults are themselves the product of
optimizing across ~20,000 collections, so they already are the average user's
personalized weights, and a typical collection has nothing to gain. The value is
concentrated in the atypical ones, where it is large.

`ExFsrs.Optimizer.evaluate/2` reports RMSE(bins) alongside log loss, using
`srs-benchmark`'s bin edges, so results here can be compared with the numbers
the FSRS project publishes.

This is a port of py-fsrs's optimizer and is verified against a recorded run of
it (see `test/fixtures/`). `fsrs-optimizer` and `fsrs-rs` still do more: recency
weighting of samples, and outlier removal. Those are not implemented here.

---

## Testing

The library comes with a test suite to ensure functionality works as expected.

To run the entire test suite:

```bash
mix test
```

Run a specific test file:

```bash
mix test test/scheduler_test.exs
```

Run a specific test by line number:

```bash
mix test test/scheduler_test.exs:42
```

Run tests with a specific tag:

```bash
mix test --only performance
```

---

## Examples

The `examples/` directory contains interactive scripts for exploring the FSRS-6 algorithm:

```bash
iex -S mix
c("examples/demo.exs")
ExFsrs.Demo.run()
```

This prints detailed output showing how cards move through learning, review, and relearning states with different ratings.

---

## Contributing

**Contributions are welcome!** If you would like to fix bugs or add new features:

1. Fork the repository
2. Create a new branch
3. Make your changes and commit them
4. Push to your fork
5. Create a pull request

Please ensure you include tests where appropriate.

---

## License
This project is available as open source under the terms of the **MIT License**. Feel free to use it, distribute it, and contribute.
