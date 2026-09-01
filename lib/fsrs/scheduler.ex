defmodule ExFsrs.Scheduler do
  @moduledoc """
  The FSRS-6 scheduling algorithm.

  A scheduler holds the 21 model weights and the scheduling policy built around
  them: the target retention, the learning and relearning steps, the interval
  cap, and whether intervals are fuzzed. It is a plain struct with no process
  behind it, so one scheduler can serve any number of cards concurrently.

      scheduler = ExFsrs.Scheduler.new(desired_retention: 0.85, enable_fuzzing: false)
      {card, log} = ExFsrs.Scheduler.review_card(scheduler, card, :good)

  Build schedulers with `new/1` rather than the struct literal: `decay` and
  `factor` are derived from `w[20]` there.

  ## How a review is scheduled

  Every review does two independent things:

    * **Memory state.** Stability and difficulty are recomputed from the rating
      and the time since the last review. A first review takes the initial
      values from the weights; a review within a day uses the short-term
      formula; a review a day or more later uses the long-term formula, which
      depends on the card's retrievability at the moment of review.

    * **Scheduling.** The card moves through the `:learning` → `:review` →
      `:relearning` state machine. Learning and relearning walk a list of steps
      measured in minutes; a card that graduates to `:review` gets an interval
      in days computed from its stability and the desired retention.

  Time units: steps are **minutes**, `next_interval/2` returns **days**, and
  `maximum_interval` is in **days**.
  """

  @ratings [:again, :hard, :good, :easy]
  @rating_numbers %{again: 1, hard: 2, good: 3, easy: 4}

  # 1 minute, then 10 minutes.
  @learning_steps [1.0, 10.0]
  @relearning_steps [10.0]
  @maximum_interval 36_500
  @stability_min 0.001
  @parameter_count 21

  @default_parameters [
    # w[0..3]: initial stability after a first rating of Again, Hard, Good, Easy
    0.212,
    1.2931,
    2.3065,
    8.2956,
    # w[4]: initial difficulty base
    6.4133,
    # w[5]: initial difficulty scaling per rating
    0.8334,
    # w[6]: difficulty change per rating step away from Good
    3.0194,
    # w[7]: difficulty mean-reversion strength
    0.001,
    # w[8]: recall stability increase factor
    1.8722,
    # w[9]: recall stability decay exponent
    0.1666,
    # w[10]: recall retrievability sensitivity
    0.796,
    # w[11]: post-lapse stability base multiplier
    1.4835,
    # w[12]: post-lapse difficulty power (negative)
    0.0614,
    # w[13]: post-lapse previous-stability power
    0.2629,
    # w[14]: post-lapse retrievability sensitivity
    1.6483,
    # w[15]: hard penalty multiplier
    0.6014,
    # w[16]: easy bonus multiplier
    1.8729,
    # w[17]: short-term stability change rate
    0.5425,
    # w[18]: short-term rating offset
    0.0912,
    # w[19]: short-term last-stability decay exponent
    0.0658,
    # w[20]: forgetting-curve decay constant
    0.1542
  ]

  # Fuzz widens with the interval: each range adds `factor` per day of interval
  # that falls inside it. The last range is open-ended.
  @fuzz_ranges [
    %{start: 2.5, end: 7.0, factor: 0.15},
    %{start: 7.0, end: 20.0, factor: 0.1},
    %{start: 20.0, end: nil, factor: 0.05}
  ]

  @typedoc "A scheduler configuration. Build one with `new/1`."
  @type t :: %__MODULE__{
          parameters: [float()],
          desired_retention: float(),
          learning_steps: [number()],
          relearning_steps: [number()],
          maximum_interval: pos_integer(),
          enable_fuzzing: boolean(),
          decay: float(),
          factor: float()
        }

  defstruct parameters: @default_parameters,
            desired_retention: 0.9,
            learning_steps: @learning_steps,
            relearning_steps: @relearning_steps,
            maximum_interval: @maximum_interval,
            enable_fuzzing: true,
            decay: nil,
            factor: nil

  @doc """
  Builds a scheduler.

  ## Options

    * `:parameters` — the 21 FSRS-6 weights. Defaults to the published FSRS-6
      defaults, which are the result of optimizing across ~20,000 collections.
      `ExFsrs.Optimizer` fits personal ones.
    * `:desired_retention` — the probability of recall to schedule for.
      Default `0.9`.
    * `:learning_steps` — minutes between reviews of a new card until it
      graduates. Default `[1.0, 10.0]`. An empty list graduates on the first
      review.
    * `:relearning_steps` — the same for a lapsed card. Default `[10.0]`.
    * `:maximum_interval` — cap on the interval in days. Default `36_500`.
    * `:enable_fuzzing` — randomize review-state intervals slightly so cards
      learned together do not stay due together. Default `true`.

  Raises `ArgumentError` unless exactly 21 parameters are given.
  """
  @spec new(keyword()) :: t()
  def new(opts \\ []) do
    parameters = Keyword.get(opts, :parameters, @default_parameters)

    unless is_list(parameters) and length(parameters) == @parameter_count do
      raise ArgumentError,
            "expected #{@parameter_count} parameters, got #{length(List.wrap(parameters))}"
    end

    decay = -Enum.at(parameters, 20)

    %__MODULE__{
      parameters: parameters,
      desired_retention: Keyword.get(opts, :desired_retention, 0.9),
      learning_steps: Keyword.get(opts, :learning_steps, @learning_steps),
      relearning_steps: Keyword.get(opts, :relearning_steps, @relearning_steps),
      maximum_interval: Keyword.get(opts, :maximum_interval, @maximum_interval),
      enable_fuzzing: Keyword.get(opts, :enable_fuzzing, true),
      decay: decay,
      factor: :math.pow(0.9, 1 / decay) - 1
    }
  end

  @doc """
  Reviews a card, returning the updated card and a log of the review.

  `review_datetime` defaults to now; `review_duration` is an optional number of
  milliseconds recorded on the log and never used by the algorithm.

  The returned card has new stability, difficulty, state, step and due date.
  The log snapshots the card *as it was before* the review, so replaying logs
  through `reschedule_card/3` reconstructs the same history.
  """
  @spec review_card(t(), ExFsrs.t(), ExFsrs.rating(), DateTime.t(), non_neg_integer() | nil) ::
          {ExFsrs.t(), ExFsrs.ReviewLog.t()}
  def review_card(
        %__MODULE__{} = scheduler,
        %ExFsrs{} = card,
        rating,
        review_datetime \\ DateTime.utc_now(),
        review_duration \\ nil
      )
      when rating in @ratings do
    {stability, difficulty} =
      compute_stability_difficulty(card, rating, review_datetime, scheduler)

    {state, step, due} = next_schedule(card, rating, stability, review_datetime, scheduler)

    updated_card = %{
      card
      | state: state,
        step: step,
        stability: stability,
        difficulty: difficulty,
        due: due,
        last_review: review_datetime
    }

    {updated_card, ExFsrs.ReviewLog.new(card, rating, review_datetime, review_duration)}
  end

  # --- State machine ---------------------------------------------------------

  # Learning and relearning walk their step lists by identical rules; only the
  # state they stay in and the list they read differ. Review-state cards either
  # keep reviewing or lapse into relearning.
  defp next_schedule(%{state: :learning} = card, rating, stability, datetime, scheduler) do
    step_through(
      :learning,
      scheduler.learning_steps,
      card,
      rating,
      stability,
      datetime,
      scheduler
    )
  end

  defp next_schedule(%{state: :relearning} = card, rating, stability, datetime, scheduler) do
    step_through(
      :relearning,
      scheduler.relearning_steps,
      card,
      rating,
      stability,
      datetime,
      scheduler
    )
  end

  defp next_schedule(%{state: :review}, :again, _stability, datetime, %{
         relearning_steps: [step | _]
       }) do
    {:relearning, 0, add_minutes(datetime, step)}
  end

  defp next_schedule(%{state: :review}, _rating, stability, datetime, scheduler) do
    graduate(stability, datetime, scheduler)
  end

  # No steps configured: every rating graduates straight to review.
  defp step_through(_state, [], _card, _rating, stability, datetime, scheduler) do
    graduate(stability, datetime, scheduler)
  end

  defp step_through(state, [first | _], _card, :again, _stability, datetime, _scheduler) do
    {state, 0, add_minutes(datetime, first)}
  end

  defp step_through(state, steps, card, rating, stability, datetime, scheduler) do
    # A card can sit at a step index the current scheduler no longer has —
    # restored from storage, or after the steps were shortened. py-fsrs
    # graduates it; a learning card with no step at all starts at 0.
    step = card.step || 0

    cond do
      step >= length(steps) or rating == :easy ->
        graduate(stability, datetime, scheduler)

      rating == :hard ->
        {state, step, add_minutes(datetime, hard_interval(step, steps))}

      # :good
      step + 1 == length(steps) ->
        graduate(stability, datetime, scheduler)

      true ->
        {state, step + 1, add_minutes(datetime, Enum.at(steps, step + 1))}
    end
  end

  # Hard at the first step repeats it with a longer wait: 1.5x a lone step, or
  # the midpoint of the first two. Later steps simply repeat.
  defp hard_interval(0, [only]), do: only * 1.5
  defp hard_interval(0, [first, second | _]), do: (first + second) / 2.0
  defp hard_interval(step, steps), do: Enum.at(steps, step)

  defp graduate(stability, datetime, scheduler) do
    days = stability |> next_interval(scheduler) |> maybe_fuzz(scheduler)
    {:review, nil, DateTime.add(datetime, days, :day)}
  end

  # Steps may be fractional minutes (e.g. 5.5); schedule to the second to keep
  # that precision, as py-fsrs does.
  defp add_minutes(datetime, minutes) do
    DateTime.add(datetime, round(minutes * 60), :second)
  end

  defp maybe_fuzz(days, %{enable_fuzzing: true, maximum_interval: max_ivl}) do
    get_fuzzed_interval(days, max_ivl)
  end

  defp maybe_fuzz(days, _scheduler), do: days

  # --- Memory state ----------------------------------------------------------

  defp compute_stability_difficulty(%{stability: s, difficulty: d}, rating, _datetime, scheduler)
       when is_nil(s) or is_nil(d) or s <= 0 do
    {initial_stability(rating, scheduler), initial_difficulty(rating, scheduler)}
  end

  defp compute_stability_difficulty(card, rating, datetime, scheduler) do
    difficulty = next_difficulty(card.difficulty, rating, scheduler)

    if days_since_last_review(card, datetime) < 1 do
      {short_term_stability(card.stability, rating, scheduler), difficulty}
    else
      retrievability = get_retrievability(card, datetime, scheduler)

      {next_stability(card.difficulty, card.stability, retrievability, rating, scheduler),
       difficulty}
    end
  end

  defp days_since_last_review(%{last_review: nil}, _datetime), do: -1

  defp days_since_last_review(%{last_review: last_review}, datetime) do
    DateTime.diff(datetime, last_review, :day)
  end

  defp initial_stability(rating, scheduler) do
    max(w(scheduler, rating_number(rating) - 1), @stability_min)
  end

  defp initial_difficulty(rating, scheduler) do
    clamp_difficulty(initial_difficulty_unclamped(rating, scheduler))
  end

  defp initial_difficulty_unclamped(rating, scheduler) do
    w(scheduler, 4) - :math.exp(w(scheduler, 5) * (rating_number(rating) - 1)) + 1
  end

  # Difficulty moves against the rating (Again raises it, Easy lowers it), is
  # damped as it nears 10, and reverts slightly toward the Easy baseline.
  defp next_difficulty(difficulty, rating, scheduler) do
    delta = -(w(scheduler, 6) * (rating_number(rating) - 3))
    damped = difficulty + (10.0 - difficulty) * delta / 9.0
    baseline = initial_difficulty_unclamped(:easy, scheduler)
    w7 = w(scheduler, 7)

    clamp_difficulty(w7 * baseline + (1 - w7) * damped)
  end

  defp clamp_difficulty(difficulty), do: difficulty |> max(1.0) |> min(10.0)

  defp next_stability(difficulty, stability, retrievability, :again, scheduler) do
    max(next_forget_stability(difficulty, stability, retrievability, scheduler), @stability_min)
  end

  defp next_stability(difficulty, stability, retrievability, rating, scheduler) do
    max(
      next_recall_stability(difficulty, stability, retrievability, rating, scheduler),
      @stability_min
    )
  end

  # After a lapse, stability is the smaller of the long-term post-lapse formula
  # and the short-term Again result, so a lapse can never raise stability.
  defp next_forget_stability(difficulty, stability, retrievability, scheduler) do
    long_term =
      w(scheduler, 11) *
        :math.pow(difficulty, -w(scheduler, 12)) *
        (:math.pow(stability + 1, w(scheduler, 13)) - 1) *
        :math.exp((1 - retrievability) * w(scheduler, 14))

    short_term = stability / :math.exp(w(scheduler, 17) * w(scheduler, 18))

    min(long_term, short_term)
  end

  defp next_recall_stability(difficulty, stability, retrievability, rating, scheduler) do
    hard_penalty = if rating == :hard, do: w(scheduler, 15), else: 1.0
    easy_bonus = if rating == :easy, do: w(scheduler, 16), else: 1.0

    increase =
      :math.exp(w(scheduler, 8)) *
        (11 - difficulty) *
        :math.pow(stability, -w(scheduler, 9)) *
        (:math.exp((1 - retrievability) * w(scheduler, 10)) - 1) *
        hard_penalty *
        easy_bonus

    stability * (1 + increase)
  end

  # Same-day reviews do not follow the forgetting curve. A successful one never
  # shrinks stability; only Again can.
  defp short_term_stability(stability, rating, scheduler) do
    increase =
      :math.exp(w(scheduler, 17) * (rating_number(rating) - 3 + w(scheduler, 18))) *
        :math.pow(stability, -w(scheduler, 19))

    increase = if rating == :again, do: increase, else: max(increase, 1.0)

    max(stability * increase, @stability_min)
  end

  defp w(scheduler, index), do: Enum.at(scheduler.parameters, index)

  defp rating_number(rating), do: Map.fetch!(@rating_numbers, rating)

  # --- Intervals -------------------------------------------------------------

  @doc """
  The interval in days at which a card of the given stability is expected to
  drop to the scheduler's desired retention.

  At least 1 and at most `maximum_interval`.
  """
  @spec next_interval(float(), t()) :: pos_integer()
  def next_interval(stability, %__MODULE__{} = scheduler) do
    (stability / scheduler.factor *
       (:math.pow(scheduler.desired_retention, 1 / scheduler.decay) - 1))
    |> round()
    |> max(1)
    |> min(scheduler.maximum_interval)
  end

  @doc """
  Randomizes an interval (in days) with FSRS-6's cumulative-delta fuzz.

  Intervals under 2.5 days are returned unchanged. Longer ones are moved a
  few days either way, by more the longer they are, never below 2 days and
  never above `maximum_interval`.
  """
  @spec get_fuzzed_interval(number(), pos_integer()) :: pos_integer()
  def get_fuzzed_interval(interval_days, maximum_interval \\ @maximum_interval)

  def get_fuzzed_interval(interval_days, _maximum_interval) when interval_days < 2.5 do
    round(interval_days)
  end

  def get_fuzzed_interval(interval_days, maximum_interval) do
    delta = fuzz_delta(interval_days)
    max_ivl = min(round(interval_days + delta), maximum_interval)
    min_ivl = interval_days |> Kernel.-(delta) |> round() |> max(2) |> min(max_ivl)

    fuzzed = min_ivl + :rand.uniform(max_ivl - min_ivl + 1) - 1
    min(fuzzed, maximum_interval)
  end

  defp fuzz_delta(interval_days) do
    Enum.reduce(@fuzz_ranges, 1.0, fn range, delta ->
      delta + range.factor * days_within(interval_days, range.start, range.end)
    end)
  end

  defp days_within(interval_days, start, nil), do: max(interval_days - start, 0.0)

  defp days_within(interval_days, start, range_end) do
    max(min(interval_days, range_end) - start, 0.0)
  end

  @doc """
  The probability that the card is still remembered at `datetime`.

  Follows the FSRS-6 forgetting curve
  `(1 + factor × elapsed_days / stability) ^ decay`, where elapsed days are
  counted from the card's last review and floored at 0. A card that has never
  been reviewed, or has no stability, has retrievability `0.0`.
  """
  @spec get_retrievability(ExFsrs.t(), DateTime.t(), t()) :: float()
  def get_retrievability(card, datetime, scheduler)

  def get_retrievability(%ExFsrs{stability: s}, _datetime, _scheduler) when is_nil(s) or s <= 0,
    do: 0.0

  def get_retrievability(%ExFsrs{last_review: nil}, _datetime, _scheduler), do: 0.0

  def get_retrievability(%ExFsrs{} = card, datetime, %__MODULE__{} = scheduler) do
    elapsed_days = max(0, DateTime.diff(datetime, card.last_review, :day))
    :math.pow(1 + scheduler.factor * elapsed_days / card.stability, scheduler.decay)
  end

  # --- Rescheduling ----------------------------------------------------------

  @doc """
  Rebuilds a card by replaying its review history through this scheduler.

  Use it after changing parameters, retention or steps: the result is the card
  as it would be had this scheduler been in use all along. The replay honours
  `enable_fuzzing`, so disable it for a deterministic result.

  `review_logs` may be `ExFsrs.ReviewLog` structs or maps with `:rating`,
  `:review_datetime` and either `:card_id` or a `:card` carrying one. Order
  does not matter. Raises `ArgumentError` if any log belongs to another card.
  """
  @spec reschedule_card(t(), ExFsrs.t(), [ExFsrs.ReviewLog.t() | map()]) :: ExFsrs.t()
  def reschedule_card(%__MODULE__{} = scheduler, %ExFsrs{} = card, review_logs)
      when is_list(review_logs) do
    for log <- review_logs, log_card_id(log) != card.card_id do
      raise ArgumentError,
            "ReviewLog card_id #{log_card_id(log)} does not match Card card_id #{card.card_id}"
    end

    review_logs
    |> Enum.sort_by(&log_datetime/1, DateTime)
    |> Enum.reduce(ExFsrs.new(card_id: card.card_id, due: card.due), fn log, replayed ->
      {updated, _log} = review_card(scheduler, replayed, log_rating(log), log_datetime(log))
      updated
    end)
  end

  defp log_card_id(%{card: %{card_id: id}}), do: id
  defp log_card_id(%{card_id: id}), do: id

  defp log_rating(%{rating: rating}), do: rating
  defp log_datetime(%{review_datetime: datetime}), do: datetime
end
