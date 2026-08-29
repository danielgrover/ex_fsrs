defmodule ExFsrs.Scheduler do
  @moduledoc """
  FSRS-6 Scheduler implementation in Elixir.
  Handles the core spaced repetition algorithm.
  """

  alias ExFsrs

  # 1 minute, 10 minutes
  @learning_steps [1.0, 10.0]
  # 10 minutes
  @relearning_steps [10.0]
  @maximum_interval 36_500
  @stability_min 0.001
  @parameter_count 21
  @default_parameters [
    # w[0]: initial stability for Again rating
    0.212,
    # w[1]: initial stability for Hard rating
    1.2931,
    # w[2]: initial stability for Good rating
    2.3065,
    # w[3]: initial stability for Easy rating
    8.2956,
    # w[4]: initial difficulty base
    6.4133,
    # w[5]: initial difficulty scaling factor
    0.8334,
    # w[6]: difficulty change multiplier per rating deviation
    3.0194,
    # w[7]: difficulty mean reversion strength
    0.001,
    # w[8]: recall stability increase factor
    1.8722,
    # w[9]: recall stability decay exponent
    0.1666,
    # w[10]: recall retrievability sensitivity
    0.796,
    # w[11]: post-lapse stability base multiplier
    1.4835,
    # w[12]: post-lapse difficulty negative power
    0.0614,
    # w[13]: post-lapse previous stability power
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
    # w[20]: power forgetting curve decay constant
    0.1542
  ]

  @fuzz_ranges [
    %{start: 2.5, end: 7.0, factor: 0.15},
    %{start: 7.0, end: 20.0, factor: 0.1},
    %{start: 20.0, end: :infinity, factor: 0.05}
  ]

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

  defstruct parameters: @default_parameters,
            desired_retention: 0.9,
            learning_steps: @learning_steps,
            relearning_steps: @relearning_steps,
            maximum_interval: @maximum_interval,
            enable_fuzzing: true,
            decay: nil,
            factor: nil

  @doc """
  Creates a new scheduler with default parameters.

  ## Parameters
    - opts: Keyword list of options
      - parameters: List of 21 model weights
      - desired_retention: Target retention rate (default: 0.9)
      - learning_steps: List of time intervals for learning state (in minutes)
      - relearning_steps: List of time intervals for relearning state (in minutes)
      - maximum_interval: Maximum days for future scheduling
      - enable_fuzzing: Whether to apply random intervals

  ## Returns
    - A new Scheduler struct
  """
  def new(opts \\ []) do
    parameters = Keyword.get(opts, :parameters, @default_parameters)

    if Enum.count_until(parameters, @parameter_count + 1) != @parameter_count do
      raise ArgumentError,
            "expected #{@parameter_count} parameters, got #{length(parameters)}"
    end

    desired_retention = Keyword.get(opts, :desired_retention, 0.9)
    learning_steps = Keyword.get(opts, :learning_steps, @learning_steps)
    relearning_steps = Keyword.get(opts, :relearning_steps, @relearning_steps)
    maximum_interval = Keyword.get(opts, :maximum_interval, @maximum_interval)
    enable_fuzzing = Keyword.get(opts, :enable_fuzzing, true)

    decay = -Enum.at(parameters, 20)
    factor = :math.pow(0.9, 1 / decay) - 1

    %__MODULE__{
      parameters: parameters,
      desired_retention: desired_retention,
      learning_steps: learning_steps,
      relearning_steps: relearning_steps,
      maximum_interval: maximum_interval,
      enable_fuzzing: enable_fuzzing,
      decay: decay,
      factor: factor
    }
  end

  @doc """
  Reviews a card and returns the updated card and review log.

  ## Parameters
    - scheduler: Scheduler struct
    - card: ExFsrs struct to review
    - rating: Rating given to the card
    - review_datetime: DateTime of the review
    - review_duration: Duration of the review in milliseconds

  ## Returns
    - Tuple containing {updated_card, review_log}
  """
  def review_card(
        %__MODULE__{} = scheduler,
        %ExFsrs{} = card,
        rating,
        review_datetime \\ DateTime.utc_now(),
        review_duration \\ nil
      ) do
    card_state =
      if is_atom(card.state) do
        card.state
      else
        String.to_existing_atom(card.state)
      end

    updated_card =
      case card_state do
        :learning -> update_learning_card(card, rating, review_datetime, scheduler)
        :review -> update_review_card(card, rating, review_datetime, scheduler)
        :relearning -> update_relearning_card(card, rating, review_datetime, scheduler)
      end

    review_log = ExFsrs.ReviewLog.new(card, rating, review_datetime, review_duration)

    {updated_card, review_log}
  end

  defp update_learning_card(card, rating, review_datetime, scheduler) do
    {stability, difficulty} =
      compute_stability_difficulty(card, rating, review_datetime, scheduler)

    {next_state, next_step, next_due} =
      handle_learning_steps(card, rating, stability, review_datetime, scheduler)

    %{
      card
      | state: next_state,
        step: next_step,
        stability: stability,
        difficulty: difficulty,
        due: next_due,
        last_review: review_datetime
    }
  end

  # With no learning steps configured, every rating graduates straight to review.
  defp handle_learning_steps(
         _card,
         _rating,
         stability,
         review_datetime,
         %{learning_steps: []} = scheduler
       ) do
    days = maybe_fuzz(next_interval(stability, scheduler), scheduler)
    {:review, nil, DateTime.add(review_datetime, days, :day)}
  end

  defp handle_learning_steps(_card, :again, _stability, review_datetime, scheduler) do
    {:learning, 0, add_minutes(review_datetime, Enum.at(scheduler.learning_steps, 0, 1))}
  end

  # When step >= length for hard/good/easy, graduate to review
  defp handle_learning_steps(card, rating, stability, review_datetime, scheduler)
       when rating in [:hard, :good, :easy] and
              card.step >= length(scheduler.learning_steps) do
    days = maybe_fuzz(next_interval(stability, scheduler), scheduler)
    {:review, nil, DateTime.add(review_datetime, days, :day)}
  end

  defp handle_learning_steps(card, :hard, _stability, review_datetime, scheduler) do
    interval = hard_interval(card.step, scheduler.learning_steps)
    {:learning, card.step, add_minutes(review_datetime, interval)}
  end

  defp handle_learning_steps(card, :good, stability, review_datetime, scheduler) do
    if card.step + 1 == length(scheduler.learning_steps) do
      days = maybe_fuzz(next_interval(stability, scheduler), scheduler)
      {:review, nil, DateTime.add(review_datetime, days, :day)}
    else
      {:learning, card.step + 1,
       add_minutes(review_datetime, Enum.at(scheduler.learning_steps, card.step + 1, 10))}
    end
  end

  defp handle_learning_steps(_card, :easy, stability, review_datetime, scheduler) do
    days = maybe_fuzz(next_interval(stability, scheduler), scheduler)
    {:review, nil, DateTime.add(review_datetime, days, :day)}
  end

  defp update_review_card(card, rating, review_datetime, scheduler) do
    {stability, difficulty} =
      compute_stability_difficulty(card, rating, review_datetime, scheduler)

    {next_state, next_step, next_due} =
      handle_review_rating(rating, stability, review_datetime, scheduler)

    %{
      card
      | state: next_state,
        step: next_step,
        stability: stability,
        difficulty: difficulty,
        due: next_due,
        last_review: review_datetime
    }
  end

  defp handle_review_rating(
         :again,
         stability,
         review_datetime,
         %{relearning_steps: []} = scheduler
       ) do
    days = maybe_fuzz(next_interval(stability, scheduler), scheduler)
    {:review, nil, DateTime.add(review_datetime, days, :day)}
  end

  defp handle_review_rating(:again, _stability, review_datetime, scheduler) do
    {:relearning, 0, add_minutes(review_datetime, Enum.at(scheduler.relearning_steps, 0))}
  end

  defp handle_review_rating(_rating, stability, review_datetime, scheduler) do
    days = maybe_fuzz(next_interval(stability, scheduler), scheduler)
    {:review, nil, DateTime.add(review_datetime, days, :day)}
  end

  defp update_relearning_card(card, rating, review_datetime, scheduler) do
    {stability, difficulty} =
      compute_stability_difficulty(card, rating, review_datetime, scheduler)

    {next_state, next_step, next_due} =
      handle_relearning_steps(card, rating, stability, review_datetime, scheduler)

    %{
      card
      | state: next_state,
        step: next_step,
        stability: stability,
        difficulty: difficulty,
        due: next_due,
        last_review: review_datetime
    }
  end

  # With no relearning steps configured, every rating graduates straight to review.
  defp handle_relearning_steps(
         _card,
         _rating,
         stability,
         review_datetime,
         %{relearning_steps: []} = scheduler
       ) do
    days = maybe_fuzz(next_interval(stability, scheduler), scheduler)
    {:review, nil, DateTime.add(review_datetime, days, :day)}
  end

  defp handle_relearning_steps(_card, :again, _stability, review_datetime, scheduler) do
    {:relearning, 0, add_minutes(review_datetime, Enum.at(scheduler.relearning_steps, 0, 10))}
  end

  # When step >= length, graduate to review for hard/good/easy
  defp handle_relearning_steps(card, rating, stability, review_datetime, scheduler)
       when rating in [:hard, :good, :easy] and
              card.step >= length(scheduler.relearning_steps) do
    days = maybe_fuzz(next_interval(stability, scheduler), scheduler)
    {:review, nil, DateTime.add(review_datetime, days, :day)}
  end

  defp handle_relearning_steps(card, :hard, _stability, review_datetime, scheduler) do
    interval = hard_interval(card.step, scheduler.relearning_steps)
    {:relearning, card.step, add_minutes(review_datetime, interval)}
  end

  defp handle_relearning_steps(card, :good, stability, review_datetime, scheduler) do
    if card.step + 1 == length(scheduler.relearning_steps) do
      days = maybe_fuzz(next_interval(stability, scheduler), scheduler)
      {:review, nil, DateTime.add(review_datetime, days, :day)}
    else
      {:relearning, card.step + 1,
       add_minutes(review_datetime, Enum.at(scheduler.relearning_steps, card.step + 1, 10))}
    end
  end

  defp handle_relearning_steps(_card, :easy, stability, review_datetime, scheduler) do
    days = maybe_fuzz(next_interval(stability, scheduler), scheduler)
    {:review, nil, DateTime.add(review_datetime, days, :day)}
  end

  # Shared helpers

  defp compute_stability_difficulty(card, rating, review_datetime, scheduler) do
    cond do
      is_nil(card.stability) or is_nil(card.difficulty) or card.stability <= 0 ->
        {initial_stability(rating, scheduler), initial_difficulty(rating, scheduler)}

      days_since_last_review(card, review_datetime) < 1 ->
        {short_term_stability(card.stability, rating, scheduler),
         next_difficulty(card.difficulty, rating, scheduler)}

      true ->
        {next_stability(
           card.difficulty,
           card.stability,
           get_retrievability(card, review_datetime, scheduler),
           rating,
           scheduler
         ), next_difficulty(card.difficulty, rating, scheduler)}
    end
  end

  # Learning/relearning steps are fractional minutes (e.g. 5.5); keep that
  # precision by scheduling in seconds rather than rounding to whole minutes.
  defp add_minutes(datetime, minutes) do
    DateTime.add(datetime, round(minutes * 60), :second)
  end

  defp hard_interval(0, [first]) do
    first * 1.5
  end

  defp hard_interval(0, [first, second | _]) do
    (first + second) / 2.0
  end

  defp hard_interval(step, steps) do
    Enum.at(steps, step, 10)
  end

  defp maybe_fuzz(days, %{enable_fuzzing: true, maximum_interval: max_ivl}) do
    get_fuzzed_interval(days, max_ivl)
  end

  defp maybe_fuzz(days, _scheduler), do: days

  # Core algorithm

  @doc """
  Calculates the next interval in days based on stability and desired retention.
  """
  def next_interval(stability, scheduler) do
    (stability / scheduler.factor *
       (:math.pow(scheduler.desired_retention, 1 / scheduler.decay) - 1))
    |> round()
    |> max(1)
    |> min(scheduler.maximum_interval)
  end

  @doc """
  Applies fuzzing to an interval (in days) using the FSRS-6 cumulative delta algorithm.
  """
  def get_fuzzed_interval(interval_days, maximum_interval \\ @maximum_interval)

  def get_fuzzed_interval(interval_days, _maximum_interval) when interval_days < 2.5 do
    round(interval_days)
  end

  def get_fuzzed_interval(interval_days, maximum_interval) do
    delta = compute_fuzz_delta(interval_days)
    min_ivl = max(2, round(interval_days - delta))
    max_ivl = min(round(interval_days + delta), maximum_interval)
    min_ivl = min(min_ivl, max_ivl)

    fuzzed = min_ivl + :rand.uniform(max_ivl - min_ivl + 1) - 1
    min(fuzzed, maximum_interval)
  end

  defp compute_fuzz_delta(interval_days) do
    Enum.reduce(@fuzz_ranges, 1.0, fn fuzz_range, acc ->
      acc + fuzz_range_contribution(fuzz_range, interval_days)
    end)
  end

  defp fuzz_range_contribution(%{end: :infinity, start: start, factor: factor}, interval_days)
       when interval_days > start do
    factor * (interval_days - start)
  end

  defp fuzz_range_contribution(%{end: :infinity}, _interval_days), do: 0.0

  defp fuzz_range_contribution(%{start: start, end: range_end, factor: factor}, interval_days) do
    factor * max(min(interval_days, range_end) - start, 0.0)
  end

  defp initial_stability(rating, scheduler) do
    max(Enum.at(scheduler.parameters, rating_to_number(rating) - 1), @stability_min)
  end

  defp initial_difficulty(rating, scheduler) do
    w4 = Enum.at(scheduler.parameters, 4)
    w5 = Enum.at(scheduler.parameters, 5)
    difficulty = w4 - :math.exp(w5 * (rating_to_number(rating) - 1)) + 1
    min(max(difficulty, 1.0), 10.0)
  end

  defp initial_difficulty_unclamped(rating, scheduler) do
    w4 = Enum.at(scheduler.parameters, 4)
    w5 = Enum.at(scheduler.parameters, 5)
    w4 - :math.exp(w5 * (rating_to_number(rating) - 1)) + 1
  end

  defp next_stability(difficulty, stability, retrievability, rating, scheduler) do
    result =
      case rating do
        :again -> next_forget_stability(difficulty, stability, retrievability, scheduler)
        _ -> next_recall_stability(difficulty, stability, retrievability, rating, scheduler)
      end

    max(result, @stability_min)
  end

  defp next_forget_stability(difficulty, stability, retrievability, scheduler) do
    w11 = Enum.at(scheduler.parameters, 11)
    w12 = Enum.at(scheduler.parameters, 12)
    w13 = Enum.at(scheduler.parameters, 13)
    w14 = Enum.at(scheduler.parameters, 14)
    w17 = Enum.at(scheduler.parameters, 17)
    w18 = Enum.at(scheduler.parameters, 18)

    long_term =
      w11 *
        :math.pow(difficulty, -w12) *
        (:math.pow(stability + 1, w13) - 1) *
        :math.exp((1 - retrievability) * w14)

    short_term = stability / :math.exp(w17 * w18)

    min(long_term, short_term)
  end

  defp next_recall_stability(difficulty, stability, retrievability, rating, scheduler) do
    w8 = Enum.at(scheduler.parameters, 8)
    w9 = Enum.at(scheduler.parameters, 9)
    w10 = Enum.at(scheduler.parameters, 10)
    w15 = Enum.at(scheduler.parameters, 15)
    w16 = Enum.at(scheduler.parameters, 16)

    hard_penalty = if rating == :hard, do: w15, else: 1.0
    easy_bonus = if rating == :easy, do: w16, else: 1.0

    stability *
      (1 +
         :math.exp(w8) *
           (11 - difficulty) *
           :math.pow(stability, -w9) *
           (:math.exp((1 - retrievability) * w10) - 1) *
           hard_penalty *
           easy_bonus)
  end

  defp next_difficulty(difficulty, rating, scheduler) do
    difficulty = difficulty || 1.0

    linear_damping = fn delta_difficulty, diff ->
      (10.0 - diff) * delta_difficulty / 9.0
    end

    w7 = Enum.at(scheduler.parameters, 7)

    mean_reversion = fn arg1, arg2 ->
      w7 * arg1 + (1 - w7) * arg2
    end

    arg1 = initial_difficulty_unclamped(:easy, scheduler)
    w6 = Enum.at(scheduler.parameters, 6)
    delta_difficulty = -(w6 * (rating_to_number(rating) - 3))
    arg2 = difficulty + linear_damping.(delta_difficulty, difficulty)
    next_difficulty = mean_reversion.(arg1, arg2)

    min(max(next_difficulty, 1.0), 10.0)
  end

  defp short_term_stability(stability, rating, scheduler) do
    w17 = Enum.at(scheduler.parameters, 17)
    w18 = Enum.at(scheduler.parameters, 18)
    w19 = Enum.at(scheduler.parameters, 19)

    increase =
      :math.exp(w17 * (rating_to_number(rating) - 3 + w18)) *
        :math.pow(stability, -w19)

    increase =
      if rating in [:hard, :good, :easy] do
        max(increase, 1.0)
      else
        increase
      end

    max(stability * increase, @stability_min)
  end

  @doc """
  Reschedules a card by replaying its review logs through this scheduler.

  Useful when switching to a scheduler with different parameters — replays
  all reviews as if this scheduler had always been used.

  ## Parameters
    - scheduler: Scheduler struct to use for rescheduling
    - card: ExFsrs struct to reschedule
    - review_logs: List of review logs (maps or ReviewLog structs with :rating,
      :review_datetime, and a card with matching :card_id). Order doesn't matter.

  ## Returns
    - Rescheduled ExFsrs card

  ## Raises
    - ArgumentError if any review log's card_id doesn't match the card's card_id
  """
  def reschedule_card(%__MODULE__{} = scheduler, %ExFsrs{} = card, review_logs)
      when is_list(review_logs) do
    card_id = card.card_id

    Enum.each(review_logs, fn log ->
      log_card_id = extract_card_id(log)

      if log_card_id != card_id do
        raise ArgumentError,
              "ReviewLog card_id #{log_card_id} does not match Card card_id #{card_id}"
      end
    end)

    sorted_logs =
      Enum.sort_by(
        review_logs,
        fn log ->
          extract_review_datetime(log)
        end,
        DateTime
      )

    rescheduled = ExFsrs.new(card_id: card_id, due: card.due)

    Enum.reduce(sorted_logs, rescheduled, fn log, acc ->
      rating = extract_rating(log)
      review_datetime = extract_review_datetime(log)
      {updated, _log} = review_card(scheduler, acc, rating, review_datetime)
      updated
    end)
  end

  defp extract_card_id(%ExFsrs.ReviewLog{card: card}), do: card.card_id
  defp extract_card_id(%{card: %ExFsrs{card_id: id}}), do: id
  defp extract_card_id(%{card: %{card_id: id}}), do: id
  defp extract_card_id(%{card_id: id}), do: id

  defp extract_rating(%ExFsrs.ReviewLog{rating: rating}), do: rating
  defp extract_rating(%{rating: rating}), do: rating

  defp extract_review_datetime(%ExFsrs.ReviewLog{review_datetime: dt}), do: dt
  defp extract_review_datetime(%{review_datetime: dt}), do: dt

  defp rating_to_number(rating) do
    case rating do
      :again -> 1
      :hard -> 2
      :good -> 3
      :easy -> 4
    end
  end

  defp days_since_last_review(card, review_datetime) do
    case card.last_review do
      nil -> -1
      last_review -> DateTime.diff(review_datetime, last_review, :day)
    end
  end

  @doc """
  Calculates the retrievability of a card at a given time.
  """
  def get_retrievability(%{stability: nil}, _review_datetime, _scheduler), do: 0
  def get_retrievability(%{stability: s}, _review_datetime, _scheduler) when s <= 0, do: 0

  def get_retrievability(card, review_datetime, scheduler) do
    case card.last_review do
      nil ->
        0

      last_review ->
        elapsed_days = max(0, DateTime.diff(review_datetime, last_review, :day))
        :math.pow(1 + scheduler.factor * elapsed_days / card.stability, scheduler.decay)
    end
  end
end
