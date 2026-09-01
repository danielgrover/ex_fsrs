# ExFsrs

**FSRS-6 spaced repetition scheduling in Elixir, with an optional parameter optimizer.**

[![Elixir](https://img.shields.io/badge/Lang-Elixir-purple.svg)](https://elixir-lang.org/)

ExFsrs schedules flashcard reviews with the [FSRS-6](https://github.com/open-spaced-repetition/fsrs4anki/wiki/The-Algorithm)
algorithm: each card carries a memory state (stability and difficulty), each
review updates it, and the next review lands when the card is predicted to be
on the edge of being forgotten. Scheduling has no dependencies. The optimizer,
which fits the 21 model weights to a user's own review history, needs
[Nx](https://hex.pm/packages/nx).

> This is a fork of [open-spaced-repetition/ex_fsrs](https://github.com/open-spaced-repetition/ex_fsrs)
> moved from FSRS v5 (19 parameters) to FSRS-6 (21 parameters). It is checked
> against the `py-fsrs`, `ts-fsrs` and `fsrs-rs` test suites and against a
> recorded run of py-fsrs's optimizer. Maintained for personal use; support is
> best-effort.

- [Installation](#installation)
- [Scheduling](#scheduling)
  - [Reviewing a card](#reviewing-a-card)
  - [Configuring the scheduler](#configuring-the-scheduler)
  - [Storing cards and review logs](#storing-cards-and-review-logs)
  - [Retrievability and rescheduling](#retrievability-and-rescheduling)
  - [Burying, suspending and late reviews](#burying-suspending-and-late-reviews)
- [Optimizing parameters](#optimizing-parameters)
  - [Fitting it into an application](#fitting-it-into-an-application)
  - [The upgrades, and what they are worth](#the-upgrades-and-what-they-are-worth)
  - [Outlier removal, which does not help](#outlier-removal-which-does-not-help)
  - [What the loss numbers do not say](#what-the-loss-numbers-do-not-say)
  - [Models and performance](#models-and-performance)
- [API overview](#api-overview)
- [Development](#development)
- [License](#license)

## Installation

```elixir
def deps do
  [
    {:ex_fsrs, git: "https://github.com/danielgrover/ex_fsrs"}
  ]
end
```

Scheduling needs nothing else. To use `ExFsrs.Optimizer`, also add Nx. It is
an optional dependency, so without it the optimizer modules are simply not
compiled:

```elixir
{:ex_fsrs, git: "https://github.com/danielgrover/ex_fsrs"},
{:nx, "~> 0.13"}
```

Stock Nx is all the optimizer needs. One opt-in path, `model: :loop`,
additionally needs `:exla` and an Nx carrying a fix for f64 gradients through
`while` that is not yet upstream (see
[`bench/NX_WHILE_GRAD_F64.md`](bench/NX_WHILE_GRAD_F64.md)). The default model
does not use `while`, and `:loop` checks at startup and refuses to run on an
unpatched Nx rather than train on wrong gradients. To use it, set
`EX_FSRS_NX_PATH` to a patched checkout before `mix deps.get`.

## Scheduling

### Reviewing a card

```elixir
card = ExFsrs.new()
#=> %ExFsrs{state: :learning, step: 0, stability: nil, difficulty: nil, ...}

{card, log} = ExFsrs.review_card(card, :good)

card.state      #=> :learning      (second learning step, due in 10 minutes)
card.stability  #=> 2.3065
card.difficulty #=> 2.118...

{card, log} = ExFsrs.review_card(card, :good, card.due)
card.state      #=> :review        (graduated; due in a couple of days)
```

Ratings are `:again`, `:hard`, `:good` and `:easy`. Every review returns the
updated card and an `ExFsrs.ReviewLog` snapshotting the card *before* the
review, the rating and the time.

A card moves through three states:

| state         | meaning                                            | scheduled by                |
|---------------|----------------------------------------------------|-----------------------------|
| `:learning`   | new card working through the learning steps        | `learning_steps` (minutes)  |
| `:review`     | graduated; intervals grow with stability           | stability and retention     |
| `:relearning` | lapsed (`:again` in review); relearning steps      | `relearning_steps` (minutes)|

Stability and difficulty are updated on every review regardless of state.
Reviews within a day of the last one use FSRS-6's short-term formula; reviews a
day or more later use the long-term formula, which depends on how likely the
card was to still be remembered at that moment.

### Configuring the scheduler

`ExFsrs.review_card/4` uses a default `ExFsrs.Scheduler`. Build your own to
change anything:

```elixir
scheduler =
  ExFsrs.Scheduler.new(
    desired_retention: 0.85,        # default 0.9
    learning_steps: [1.0, 10.0],    # minutes; [] graduates on the first review
    relearning_steps: [10.0],       # minutes
    maximum_interval: 365,          # days; default 36_500
    enable_fuzzing: false,          # default true
    parameters: personal_weights    # 21 floats; default FSRS-6 weights
  )

{card, log} = ExFsrs.Scheduler.review_card(scheduler, card, :good, DateTime.utc_now())
```

A scheduler is a plain struct. Build one per user (or per deck preset) and
reuse it; there is no process behind it.

Fuzzing nudges review-state intervals a few days either way so that cards
learned together do not stay due together forever. Disable it for
deterministic results, as the test suite does.

### Storing cards and review logs

`ExFsrs.to_map/1` and `ExFsrs.ReviewLog.to_map/1` produce string-keyed maps
with ISO 8601 datetimes, ready for JSON or a database. `from_map/1` on each
accepts string or atom keys, ISO strings or `DateTime`s, and raises
`ArgumentError` on anything it cannot interpret rather than substituting a
default.

```elixir
ExFsrs.to_map(card)
#=> %{"card_id" => 1, "state" => "review", "step" => nil, "stability" => 5.1,
#     "difficulty" => 4.9, "due" => "2024-06-12T09:00:00Z", "last_review" => "..."}

card == card |> ExFsrs.to_map() |> ExFsrs.from_map()
#=> true
```

Keep the review logs. They are what rescheduling and the optimizer work from,
and a table of `{card_id, rating, reviewed_at}` is all either needs.

### Retrievability and rescheduling

```elixir
ExFsrs.get_retrievability(card, DateTime.utc_now())
#=> 0.87   probability the card is still remembered right now
```

Changing a scheduler's parameters, retention or steps does not move cards that
were already scheduled. `reschedule_card/3` replays a card's logs through a
scheduler as if it had always been in use:

```elixir
new_scheduler = ExFsrs.Scheduler.new(parameters: personal_weights, enable_fuzzing: false)
card = ExFsrs.Scheduler.reschedule_card(new_scheduler, card, logs_for_that_card)
```

Log order does not matter. It raises if any log belongs to a different card.

### Burying, suspending and late reviews

None of that needs modelling here. `%ExFsrs{}` holds `state`, `step`,
`stability`, `difficulty`, `due` and `last_review`; burying and suspending are
queue decisions your application makes, and the scheduler never asks.

They do not distort the algorithm either. Both the scheduler and the optimizer
measure elapsed time from `last_review` to the review that actually happened,
never from `due`, so a card reviewed late — buried repeatedly, a holiday, a
month away — is simply a review with a longer gap, which is exactly what the
forgetting curve is for.

The one trap is at the boundary: **do not write a review log for a bury or a
suspend.** They are not reviews. Log what `review_card` returns and nothing
else.

## Optimizing parameters

The default weights are the result of optimizing across ~20,000 collections, so
they already are the average user's personalized weights. `ExFsrs.Optimizer`
fits weights to one user's history instead:

```elixir
# Accepts %ExFsrs.ReviewLog{} structs or {card_id, rating, review_datetime}
# tuples. The tuple form avoids building a card struct per row when loading a
# large history out of storage.
parameters = ExFsrs.Optimizer.compute_optimal_parameters(logs)
scheduler = ExFsrs.Scheduler.new(parameters: parameters)
```

Called with no options this reproduces py-fsrs's optimizer exactly: start from
the defaults and run Adam on binary cross-entropy for five epochs. The test
suite pins it to a recorded py-fsrs run to 1e-9 per parameter.

It returns the defaults unchanged when there are fewer than 512 *scoreable*
reviews — reviews that follow an earlier review of the same card by at least a
day. Same-day repeats update a card's state but are not predictions the model
can be scored on.

Three refinements from `fsrs-optimizer` and `fsrs-rs` are available, each off
by default because each departs from the py-fsrs behaviour the parity tests
pin:

```elixir
ExFsrs.Optimizer.compute_optimal_parameters(logs,
  initialize: true,      # fit w[0..3] from data instead of starting at the defaults
  regularization: 1.0,   # L2 pull toward the starting weights
  recency: true          # weight recent reviews more heavily
)
```

That is the configuration measured best below.

To see what a parameter set does on a collection:

```elixir
ExFsrs.Optimizer.batch_loss(logs, parameters)
#=> 0.3655   mean log loss; cheapest way to compare two parameter sets

ExFsrs.Optimizer.evaluate(logs, parameters)
#=> %{rmse_bins: 0.0557, log_loss: 0.3655, reviews: 1425}
```

`evaluate/2` reports RMSE(bins) and log loss using `srs-benchmark`'s
definitions and bin edges, so numbers here are comparable with the ones the
FSRS project publishes. RMSE(bins) is the more interpretable of the two: it is
roughly the gap, in probability, between what the model predicted and what the
user actually recalled.

### Fitting it into an application

The scheduler is stateless and the optimizer is a batch job, so integration is
about deciding *when* to run it and *whether to keep* what it produces.

**1. Keep the review logs.** `review_card` returns one per review; persist it. A
table of `{card_id, rating, reviewed_at}` per user is enough, and can be the
same rows you keep for a review history UI.

**2. Run it in the background, per user.** A 12,500-review collection takes
about 50 seconds on Nx's default backend (~25s with `model: :loop` and EXLA).
That is a job, not a request. Parameters are personal, so one run per user — or
per deck preset, if your app groups material the way Anki does.

**3. Only run it when there is enough history.** Below 512 scoreable reviews it
returns the defaults, so there is no point scheduling a job until a user is past
that. Re-run periodically afterwards: habits, material and settings drift, and
the parameters go stale with them.

**4. Never adopt blindly.** Score the candidate and the current parameters on
the user's own history and keep the better one. A collection can be unusual
enough that training makes it worse.

```elixir
def optimize(user) do
  logs =
    Repo.all(from r in Review, where: r.user_id == ^user.id, order_by: r.reviewed_at)

  tuples = Enum.map(logs, &{&1.card_id, &1.rating, &1.reviewed_at})
  current = user.fsrs_parameters || ExFsrs.Scheduler.new().parameters

  candidate =
    ExFsrs.Optimizer.compute_optimal_parameters(tuples,
      initialize: true,
      regularization: 1.0,
      recency: true
    )

  if ExFsrs.Optimizer.batch_loss(tuples, candidate) <
       ExFsrs.Optimizer.batch_loss(tuples, current) do
    {:ok, candidate}
  else
    :keep_current
  end
end
```

**5. Decide what happens to cards already scheduled.** New parameters do not
change existing due dates. Either let cards pick up the new parameters at their
next review, which is undisruptive and converges over a few weeks, or
reschedule them with `ExFsrs.Scheduler.reschedule_card/3`. Rescheduling can
move a due date by a lot (see the interval table below), so do it deliberately.

**6. Watch what it did.** Log `ExFsrs.Optimizer.evaluate/2` for both parameter
sets if you want to know whether optimization is earning its keep across your
users. Most will see very little change, and that is expected. The value is
concentrated in users whose habits are unusual.

### The upgrades, and what they are worth

Measured across **83 real collections** from the Anki Revlogs 10K dataset, each
trained on its own past and scored on its own future
(`bench/temporal_eval.exs`), with the per-epoch card shuffle pinned so every
variant sees identical conditions:

| variant                                              | total loss | per-collection mean | median | improved | Wilcoxon p |
|------------------------------------------------------|-----------:|--------------------:|-------:|---------:|-----------:|
| `initialize` + `regularization: 2.0`                 | +1.76%     | +4.33%              | +0.63% | 49/83    | **0.004**  |
| `initialize` + `regularization: 1.0` + `recency`     | +2.03%     | +4.85%              | +1.03% | 56/83    | **0.0008** |

Positive numbers are improvements over plain py-fsrs optimization. Recency
weighting is worth adding: it beats the same configuration without it on 54 of
83 collections (p = 0.012). A Friedman test across the three configurations
gives p = 0.0006.

Three aggregations are quoted because they disagree and each answers a different
question. **Total loss** is dominated by the collections with the largest losses.
**Per-collection mean** weights every collection equally and is pulled upward by
a few large winners — the best collection improves by 73%, the worst regresses
by 25%. **Median** is what a typical collection gets, and it is about 1%.

The effect depends strongly on collection size:

| training reviews | collections | effect    |
|------------------|------------:|----------:|
| under 2,000      | 7           | +0.4%     |
| 2,000-4,000      | 26          | **+7.4%** |
| 4,000-8,000      | 37          | +4.8%     |
| over 8,000       | 13          | +2.3%     |

Both ends have little to gain, for opposite reasons: below a couple of thousand
reviews there is not enough signal to fit anything better, and above eight
thousand plain optimization already has enough data to find good parameters on
its own. The upgrades earn their keep in the middle.

Two cautions, both learned the hard way here:

* **Score held-out future, not held-out cards.** A card holdout rated
  initialization at +2.2% (p = 0.002) and regularization at noise. Splitting the
  same collections temporally reversed both. Held-out cards come from the period
  the parameters were fitted on, so the fit flatters itself.
* **Pin the shuffle.** The per-epoch card shuffle alone moves held-out loss by a
  median of 2% run to run — as much as the largest difference between any two
  variants. `bench/noise_floor.exs` measures it; any comparison that does not
  control for it (the `:card_orders` option) is reading noise. An earlier
  23-collection run found nothing significant, and it took both fixes plus 60
  more collections to resolve effects this size.

### Outlier removal, which does not help

`fsrs-optimizer`'s fourth refinement drops reviews sitting in sparse or
implausibly long interval buckets. It is implemented as
`ExFsrs.Optimizer.Data.remove_outliers/1`, measured, and on this evidence it
should not be used:

| variant                          | vs py-fsrs | improved | Wilcoxon p |
|----------------------------------|-----------:|---------:|-----------:|
| initialize + L2 + recency        | +2.03%     | 56/83    | **0.0008** |
| the same, plus outlier removal   | +0.79%     | 44/83    | 0.39       |

Adding it turns a significant 2% gain into a non-significant 0.8% one. Compared
head to head it is 1.26% *worse* and wins on only 35 of 83 collections
(p = 0.054), with a worst case of -114%.

| training reviews | collections | effect of removing outliers |
|------------------|------------:|----------------------------:|
| under 2,000      | 7           | +0.1%                       |
| 2,000-4,000      | 26          | **-4.5%**                   |
| 4,000-8,000      | 37          | **-4.7%**                   |
| over 8,000       | 13          | +1.7%                       |

It hurts most in exactly the band where the other upgrades help most. On these
collections it discards a median of 18% of scored reviews — far more than its
nominal 5% budget, because past that budget every bucket with fewer than 6
reviews is dropped. `fsrs-optimizer` expands each card into one row per history
prefix, so its buckets hold many more rows than the one observation per card
this representation gives them; the same threshold therefore removes more here.
The port is faithful to the algorithm; the data it is applied to is shaped
differently.

Long absences interact with it too: a user returning after a summer produces
exactly the over-100-day intervals it drops, and those reviews are real evidence
about long-term retention.

### What the loss numbers do not say

Log loss understates what these parameters do to a schedule. Days until the next
review, after a card's first review:

| collection | again | hard | good   | easy |
|------------|------:|-----:|-------:|-----:|
| defaults   | 1d    | 1d   | 2d     | 8d   |
| 3929       | 1d    | 1d   | 4d     | 8d   |
| 9861       | 4d    | 29d  | **70d**| 70d  |
| 9881       | 1d    | 1d   | 6d     | 19d  |

Collection 9861's fitted weights schedule a "Good" first review 70 days out
where the defaults say 2 — a 35x difference — on a collection whose loss barely
moved. Loss averages over thousands of reviews, most of them easy to predict
either way; the interval is a direct function of stability.

### Models and performance

Three implementations of the forward model share one copy of the FSRS-6 math
(`ExFsrs.Optimizer.Model`) and differ in how they walk a minibatch:

| `model:`   | how                                                     | 12,500 reviews |
|------------|---------------------------------------------------------|---------------:|
| `:batched` | every card in a minibatch advanced together (default)   | ~50s           |
| `:scalar`  | one review at a time; the reference the others are diffed against | ~3x slower |
| `:loop`    | `:batched` as a `defn` `while`, so it compiles with EXLA | ~25s with EXLA |

```elixir
ExFsrs.Optimizer.compute_optimal_parameters(logs, model: :loop, compiler: EXLA)
```

Timings are on an Apple M2; py-fsrs does the same run in 19s. Cost grows a
little faster than linearly with review count. See
[Installation](#installation) for what `:loop` requires.

## API overview

| module                  | what it is                                                          |
|-------------------------|---------------------------------------------------------------------|
| `ExFsrs`                | the card struct; `new/1`, `review_card/4`, `get_retrievability/2`, `reschedule_card/2`, `to_map/1`, `from_map/1` |
| `ExFsrs.Scheduler`      | the algorithm and its configuration; `new/1`, `review_card/5`, `next_interval/2`, `get_fuzzed_interval/2`, `get_retrievability/3`, `reschedule_card/3` |
| `ExFsrs.ReviewLog`      | one review; `new/4`, `to_map/1`, `from_map/1`                        |
| `ExFsrs.Optimizer`      | `compute_optimal_parameters/2`, `batch_loss/2`, `evaluate/2`, `predictions/2` |
| `ExFsrs.Optimizer.Data` | review logs → training sequences; `build_sequences/2`, `build_sequences_from_elapsed/1`, `apply_recency_weights/2`, `remove_outliers/1` |
| `ExFsrs.Optimizer.Metrics` | `rmse_bins/1`, `log_loss/1`                                       |
| `ExFsrs.Optimizer.Initialization` | the `initialize: true` fit of `w[0..3]`                     |

Full documentation is generated with `mix docs`.

## Development

```bash
mix test                       # the suite; ~50s, dominated by one py-fsrs parity run
mix test --cover               # with coverage (threshold 90%)
mix test test/scheduler_test.exs:42
mix ci                         # compile (warnings as errors), format, test, credo,
                               # dialyzer, ex_dna (no clones), reach
mix docs                       # HTML docs into doc/
```

Tests tagged `:dataset` run the optimizer against real collections from the
Anki Revlogs 10K dataset, which is gated and not committed. Fetch it with
`bench/fetch_anki_revlogs.py` (needs a Hugging Face token in `.env`), then:

```bash
mix test --include dataset
```

The `bench/` directory holds the evaluation scripts behind the numbers above
(`temporal_eval.exs`, `holdout_eval.exs`, `gamma_sweep.exs`, `noise_floor.exs`)
and the write-up of the Nx gradient bug. They train with `model: :loop` for
speed, so they need the patched Nx described there:

```bash
EX_FSRS_NX_PATH=../nx/nx mix deps.get
EX_FSRS_NX_PATH=../nx/nx MIX_ENV=test mix run bench/temporal_eval.exs
mix deps.get   # afterwards: back to stock Nx (mix.lock is untouched either way)
```

`examples/demo.exs` walks cards through every state transition and prints what
happens:

```bash
iex -S mix
c("examples/demo.exs")
ExFsrs.Demo.run()
```

Contributions are welcome; please include tests.

## License

MIT. See the `LICENSE` file.
