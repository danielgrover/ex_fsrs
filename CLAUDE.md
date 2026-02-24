# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ExFsrs is a pure Elixir implementation of the FSRS (Free Spaced Repetition Scheduler) algorithm. It schedules flashcard reviews based on card difficulty, stability, and user feedback. Zero external dependencies.

## Commands

```bash
mix compile          # Build
mix test             # Run all tests
mix test test/scheduler_test.exs          # Run a single test file
mix test test/scheduler_test.exs:42       # Run a specific test by line
mix test --only performance               # Run tagged tests
mix format            # Format code
```

## Architecture

Three modules with clear separation:

- **`ExFsrs`** (`lib/fsrs.ex`) — Card struct and public API entry point. Holds card state (`:learning`, `:review`, `:relearning`), step position, stability, difficulty, due date. Provides `review_card/4` which delegates to the Scheduler, plus serialization (`to_map`/`from_map`) and `get_retrievability/2`.

- **`ExFsrs.Scheduler`** (`lib/fsrs/scheduler.ex`) — Core FSRS-6 algorithm. Configurable with 21 float parameters (trained model weights), `desired_retention`, `learning_steps`/`relearning_steps` (in minutes), `maximum_interval` (in days), and `enable_fuzzing`. Includes `decay` and `factor` fields computed from `w[20]`. Implements state transitions, stability/difficulty calculations, interval computation, and optional interval fuzzing.

- **`ExFsrs.ReviewLog`** (`lib/fsrs/review_log.ex`) — Immutable record of a review event (card snapshot, rating, datetime, duration). Also serializable via `to_map`/`from_map`.

## Key Domain Concepts

- **Ratings**: `:again`, `:hard`, `:good`, `:easy` (mapped to 1-4)
- **Card states**: `:learning` → `:review` → `:relearning` (state machine with transitions driven by ratings)
- **Stability**: Memory strength (higher = longer retention). New cards get initial stability from parameters[0..3] indexed by rating.
- **Difficulty**: 1.0–10.0 range, adjusted via mean reversion after each review.
- **Retrievability**: Forgetting curve calculation `(1 + factor × elapsed/stability)^decay` where `decay = -w[20]` (default -0.1542) and `factor = 0.9^(1/decay) - 1`.
- **Time units**: Learning/relearning steps are in **minutes**. `next_interval()` returns **days**. `maximum_interval` is in **days**.
- **Short-term vs long-term**: Reviews within 1 day use short-term stability formulas; reviews after 1+ day use long-term formulas.
