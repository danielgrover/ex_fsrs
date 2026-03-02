> **⚠️ Fork Notice:** Fork of [open-spaced-repetition/ex_fsrs](https://github.com/open-spaced-repetition/ex_fsrs) updated from FSRS v5 (19-parameter) to **FSRS v6 (21-parameter)**.
> 
> * **Validation:** Verified for parity by cross-checking against the `ts-fsrs`, `py-fsrs`, and `fsrs-rs` test suites. 
> * **Disclaimer:** Maintained for personal use; support and stability are best-effort—use at your own risk.

# ExFsrs

**An Elixir implementation of FSRS-6 (Free Spaced Repetition Scheduler)**

[![Elixir](https://img.shields.io/badge/Lang-Elixir-purple.svg)](https://elixir-lang.org/)

A spaced repetition scheduling library in Elixir implementing the FSRS-6 algorithm. Computes optimal review intervals for flashcards based on card difficulty, stability, and user feedback. Supports optional interval fuzzing to avoid predictable review dates. Zero external dependencies.

## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Usage](#usage)
- [Modules](#modules)
  - [ExFsrs](#exfsrs)
  - [Scheduler](#scheduler)
  - [ReviewLog](#reviewlog)
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
- **Rescheduling** — replay review logs through a new scheduler with different parameters.
- **Serialization** — cards and review logs convert to/from maps for storage.

---

## Installation

Add ExFsrs as a dependency in your `mix.exs`:

```elixir
def deps do
  [
    {:ex_fsrs, "~> 0.1.0", git: "https://github.com/danielgrover/ex_fsrs"}
  ]
end
```

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
