defmodule ExFsrs.AlgorithmValidationTest do
  @moduledoc """
  Algorithm validation tests with golden values sourced from external FSRS-6
  implementations (ts-fsrs and py-fsrs) to cross-validate our implementation.

  Sources:
    - ts-fsrs: packages/fsrs/__tests__/FSRS-6.test.ts, algorithm.test.ts
    - py-fsrs: tests/test_basic.py
    - fsrs-rs: src/inference.rs
  """
  use ExUnit.Case, async: true

  @start_datetime ~U[2022-11-29 12:30:00Z]

  # --- 1. First Review Output Validation ---
  # Source: ts-fsrs FSRS-6.test.ts "first repeat"
  describe "first review output validation (ts-fsrs golden values)" do
    setup do
      scheduler = ExFsrs.Scheduler.new(enable_fuzzing: false)
      card = ExFsrs.new(state: :learning, step: 0)
      {:ok, scheduler: scheduler, card: card, now: @start_datetime}
    end

    test "again: stability=0.212, difficulty=6.4133, state=learning, interval=0 days", ctx do
      {card, _log} = ExFsrs.Scheduler.review_card(ctx.scheduler, ctx.card, :again, ctx.now)

      assert_in_delta card.stability, 0.212, 1.0e-9
      assert_in_delta card.difficulty, 6.4133, 1.0e-9
      assert card.state == :learning
      assert DateTime.diff(card.due, ctx.now, :second) == 60
    end

    test "hard: stability=1.2931, difficulty=5.1122, state=learning, interval=0 days", ctx do
      {card, _log} = ExFsrs.Scheduler.review_card(ctx.scheduler, ctx.card, :hard, ctx.now)

      assert_in_delta card.stability, 1.2931, 1.0e-9
      assert_in_delta card.difficulty, 5.112170705601056, 1.0e-9
      assert card.state == :learning
      # hard at step 0 of [1, 10] -> (1 + 10) / 2 = 5.5 minutes
      assert DateTime.diff(card.due, ctx.now, :second) == 330
    end

    test "good: stability=2.3065, difficulty=2.1181, state=learning, interval=0 days", ctx do
      {card, _log} = ExFsrs.Scheduler.review_card(ctx.scheduler, ctx.card, :good, ctx.now)

      assert_in_delta card.stability, 2.3065, 1.0e-9
      assert_in_delta card.difficulty, 2.118103970459016, 1.0e-9
      assert card.state == :learning
      assert DateTime.diff(card.due, ctx.now, :second) == 600
    end

    test "easy: stability=8.2956, difficulty=1.0 (clamped), state=review, interval=8 days", ctx do
      {card, _log} = ExFsrs.Scheduler.review_card(ctx.scheduler, ctx.card, :easy, ctx.now)

      assert_in_delta card.stability, 8.2956, 1.0e-9
      assert card.difficulty == 1.0
      assert card.state == :review
      assert DateTime.diff(card.due, ctx.now, :day) == 8
    end
  end

  # --- 2. Forgetting Curve ---
  # Source: ts-fsrs algorithm.test.ts "forgetting_curve"
  # Inputs: s=1.0, default w[20]=0.1542, t=[0,1,2,3]
  describe "forgetting curve (fsrs-rs golden values)" do
    test "retrievability at multiple elapsed times with stability=1.0" do
      card =
        ExFsrs.new(
          state: :review,
          stability: 1.0,
          difficulty: 5.0,
          last_review: @start_datetime
        )

      scheduler = ExFsrs.Scheduler.new()

      # ts-fsrs algorithm.test.ts forgetting_curve golden values
      cases = [
        {0, 1.0},
        {1, 0.9},
        {2, 0.8458846447796301},
        {3, 0.8093881028681906}
      ]

      for {elapsed_days, expected_r} <- cases do
        review_at = DateTime.add(@start_datetime, elapsed_days, :day)
        r = ExFsrs.Scheduler.get_retrievability(card, review_at, scheduler)

        assert_in_delta r,
                        expected_r,
                        0.0001,
                        "R at elapsed=#{elapsed_days}: expected #{expected_r}, got #{r}"
      end
    end
  end

  # --- 3. Retrievability for Non-Review States ---
  describe "retrievability for non-review states" do
    test "new card with nil last_review returns 0" do
      card = ExFsrs.new(state: :learning, step: 0)
      assert ExFsrs.get_retrievability(card, @start_datetime) == 0
    end

    test "review dated before last_review clamps elapsed to 0 rather than exceeding 1.0" do
      card =
        ExFsrs.new(
          state: :review,
          stability: 10.0,
          difficulty: 5.0,
          last_review: @start_datetime
        )

      scheduler = ExFsrs.Scheduler.new()
      earlier = DateTime.add(@start_datetime, -5, :day)

      # py-fsrs floors elapsed_days at 0; without that, R would exceed 1.0
      assert ExFsrs.Scheduler.get_retrievability(card, earlier, scheduler) == 1.0
    end

    test "card with last_review at elapsed=0 returns 1.0 (no decay)" do
      card =
        ExFsrs.new(
          state: :learning,
          stability: 2.0,
          difficulty: 5.0,
          last_review: @start_datetime
        )

      scheduler = ExFsrs.Scheduler.new()
      r = ExFsrs.Scheduler.get_retrievability(card, @start_datetime, scheduler)
      assert_in_delta r, 1.0, 0.0001
    end
  end

  # --- 4. next_interval Across Retention Rates ---
  # Source: fsrs-rs test_next_interval (primary), ts-fsrs algorithm.test.ts (secondary)
  # fsrs-rs values at retention 0.1-0.4 use f32 rounding; ts-fsrs agrees at 0.5+
  describe "next_interval across retention rates (fsrs-rs/ts-fsrs golden values)" do
    test "stability=1.0 with varying desired_retention" do
      # Golden values from fsrs-rs test_next_interval and ts-fsrs algorithm.test.ts
      # fsrs-rs: [3116766, 34793, 2508, 387, 90, 27, 9, 3, 1, 1]
      # ts-fsrs: [3116769, 34793, 2508, 387, 90, 27, 9, 3, 1, 1]
      # Difference at 0.1 is f32 vs f64 precision; we use f64 like ts-fsrs
      expected = [
        {0.1, 3_116_769},
        {0.2, 34_793},
        {0.3, 2_508},
        {0.4, 387},
        {0.5, 90},
        {0.6, 27},
        {0.7, 9},
        {0.8, 3},
        {0.9, 1},
        {1.0, 1}
      ]

      for {retention, expected_interval} <- expected do
        scheduler =
          ExFsrs.Scheduler.new(
            desired_retention: retention,
            maximum_interval: 36_500_000
          )

        result = ExFsrs.Scheduler.next_interval(1.0, scheduler)

        assert result == expected_interval,
               "next_interval(1.0) with retention=#{retention}: expected #{expected_interval}, got #{result}"
      end
    end

    test "stability=10.0 at default retention 0.9 returns 10" do
      scheduler = ExFsrs.Scheduler.new()
      assert ExFsrs.Scheduler.next_interval(10.0, scheduler) == 10
    end

    # Source: fsrs-rs test_next_interval, max_limit case
    test "next_interval respects maximum_interval cap" do
      scheduler = ExFsrs.Scheduler.new(maximum_interval: 365)
      # stability=737.47 should produce interval >> 365, but capped
      assert ExFsrs.Scheduler.next_interval(737.47, scheduler) == 365
    end
  end

  # --- 5. Stability Update Ordering ---
  # Source: ts-fsrs algorithm.test.ts "next_ds" and FSRS design invariant
  # For a review-state card: s_again < original < s_hard < s_good < s_easy
  describe "stability update ordering (ts-fsrs)" do
    test "s_again < original < s_hard < s_good < s_easy for review card" do
      scheduler = ExFsrs.Scheduler.new(enable_fuzzing: false)

      card =
        ExFsrs.new(
          state: :review,
          stability: 10.0,
          difficulty: 5.0,
          last_review: DateTime.add(@start_datetime, -10, :day)
        )

      {card_again, _} = ExFsrs.Scheduler.review_card(scheduler, card, :again, @start_datetime)
      {card_hard, _} = ExFsrs.Scheduler.review_card(scheduler, card, :hard, @start_datetime)
      {card_good, _} = ExFsrs.Scheduler.review_card(scheduler, card, :good, @start_datetime)
      {card_easy, _} = ExFsrs.Scheduler.review_card(scheduler, card, :easy, @start_datetime)

      assert card_again.stability < 10.0,
             "again stability #{card_again.stability} should be < 10.0"

      assert card_again.stability < card_hard.stability,
             "again #{card_again.stability} should be < hard #{card_hard.stability}"

      assert card_hard.stability < card_good.stability,
             "hard #{card_hard.stability} should be < good #{card_good.stability}"

      assert card_good.stability < card_easy.stability,
             "good #{card_good.stability} should be < easy #{card_easy.stability}"

      # py-fsrs golden values for the same inputs
      assert_in_delta card_again.stability, 1.3919869729546932, 1.0e-9
      assert_in_delta card_hard.stability, 23.246875110466814, 1.0e-9
      assert_in_delta card_good.stability, 32.02672948198672, 1.0e-9
      assert_in_delta card_easy.stability, 51.253861646812936, 1.0e-9
    end
  end

  # --- 6. Difficulty Update Sequence ---
  # Source: ts-fsrs algorithm.test.ts "next_difficulty" starting from d=5.0
  # Again->8.34176237, Hard->6.66599536, Good->4.99022837, Easy->3.31446137
  describe "difficulty update sequence (ts-fsrs golden values)" do
    test "difficulty changes from d=5.0 match ts-fsrs reference" do
      scheduler = ExFsrs.Scheduler.new(enable_fuzzing: false)

      card =
        ExFsrs.new(
          state: :review,
          stability: 10.0,
          difficulty: 5.0,
          last_review: DateTime.add(@start_datetime, -10, :day)
        )

      {card_again, _} = ExFsrs.Scheduler.review_card(scheduler, card, :again, @start_datetime)
      {card_hard, _} = ExFsrs.Scheduler.review_card(scheduler, card, :hard, @start_datetime)
      {card_good, _} = ExFsrs.Scheduler.review_card(scheduler, card, :good, @start_datetime)
      {card_easy, _} = ExFsrs.Scheduler.review_card(scheduler, card, :easy, @start_datetime)

      # Tolerance is tight on purpose: at 0.01 this passes even if
      # mean-reversion clamps its target to 1.0 instead of using the raw
      # (negative) initial difficulty for Easy, which is the reference behaviour.
      assert_in_delta card_again.difficulty, 8.341762369296838, 1.0e-9
      assert_in_delta card_hard.difficulty, 6.665995369296838, 1.0e-9
      assert_in_delta card_good.difficulty, 4.9902283692968386, 1.0e-9
      assert_in_delta card_easy.difficulty, 3.3144613692968385, 1.0e-9
    end

    test "difficulty stays clamped in [1.0, 10.0] after extreme ratings" do
      scheduler = ExFsrs.Scheduler.new(enable_fuzzing: false)

      card_high =
        ExFsrs.new(
          state: :review,
          stability: 10.0,
          difficulty: 9.5,
          last_review: DateTime.add(@start_datetime, -10, :day)
        )

      {card_again, _} =
        ExFsrs.Scheduler.review_card(scheduler, card_high, :again, @start_datetime)

      assert card_again.difficulty <= 10.0

      card_low =
        ExFsrs.new(
          state: :review,
          stability: 10.0,
          difficulty: 1.5,
          last_review: DateTime.add(@start_datetime, -10, :day)
        )

      {card_easy, _} = ExFsrs.Scheduler.review_card(scheduler, card_low, :easy, @start_datetime)
      assert card_easy.difficulty >= 1.0
    end
  end

  # --- 7. Short-Term vs Long-Term Stability Paths ---
  # FSRS-6 uses short-term stability formula when elapsed < 1 day,
  # long-term formula otherwise. Same card + same rating should differ.
  describe "short-term vs long-term stability paths" do
    test "same card, same rating, different elapsed time produces different stability" do
      scheduler = ExFsrs.Scheduler.new(enable_fuzzing: false)

      base_card =
        ExFsrs.new(
          state: :review,
          stability: 10.0,
          difficulty: 5.0,
          last_review: @start_datetime
        )

      # Short-term: review same day (< 1 day elapsed)
      same_day = DateTime.add(@start_datetime, 6, :hour)
      {short_term_card, _} = ExFsrs.Scheduler.review_card(scheduler, base_card, :good, same_day)

      # Long-term: review after 10 days
      ten_days = DateTime.add(@start_datetime, 10, :day)
      {long_term_card, _} = ExFsrs.Scheduler.review_card(scheduler, base_card, :good, ten_days)

      assert short_term_card.stability != long_term_card.stability,
             "Short-term #{short_term_card.stability} should differ from long-term #{long_term_card.stability}"
    end
  end

  # --- 8. Decay and Factor Computation ---
  # Validates scheduler struct fields computed from w[20]
  describe "decay and factor computation" do
    test "default w[20]=0.1542 produces decay=-0.1542 and valid factor" do
      scheduler = ExFsrs.Scheduler.new()

      # decay = -w[20]
      assert scheduler.decay == -0.1542

      # The defining identity: (1 + factor)^decay must equal 0.9
      assert_in_delta :math.pow(1 + scheduler.factor, scheduler.decay), 0.9, 0.0001
    end

    test "custom w[20]=0.3 produces decay=-0.3 and valid factor" do
      params = [
        0.212,
        1.2931,
        2.3065,
        8.2956,
        6.4133,
        0.8334,
        3.0194,
        0.001,
        1.8722,
        0.1666,
        0.796,
        1.4835,
        0.0614,
        0.2629,
        1.6483,
        0.6014,
        1.8729,
        0.5425,
        0.0912,
        0.0658,
        0.3
      ]

      scheduler = ExFsrs.Scheduler.new(parameters: params)

      assert scheduler.decay == -0.3
      assert_in_delta :math.pow(1 + scheduler.factor, scheduler.decay), 0.9, 0.0001
    end

    test "different w[20] values produce different factors" do
      default_scheduler = ExFsrs.Scheduler.new()

      custom_params = [
        0.212,
        1.2931,
        2.3065,
        8.2956,
        6.4133,
        0.8334,
        3.0194,
        0.001,
        1.8722,
        0.1666,
        0.796,
        1.4835,
        0.0614,
        0.2629,
        1.6483,
        0.6014,
        1.8729,
        0.5425,
        0.0912,
        0.0658,
        0.3
      ]

      custom_scheduler = ExFsrs.Scheduler.new(parameters: custom_params)

      assert default_scheduler.factor != custom_scheduler.factor
    end
  end

  # --- 9. Fuzzing Boundary Tests ---
  # Source: py-fsrs test_fuzz, ts-fsrs algorithm.test.ts apply_fuzz
  describe "fuzzing boundary tests" do
    test "interval < 2.5 returns rounded original with no randomness" do
      results = for _ <- 1..50, do: ExFsrs.Scheduler.get_fuzzed_interval(2.0)
      assert Enum.all?(results, &(&1 == 2))

      results = for _ <- 1..50, do: ExFsrs.Scheduler.get_fuzzed_interval(1.0)
      assert Enum.all?(results, &(&1 == 1))
    end

    test "interval at 2.5 begins fuzzing (values vary)" do
      results = for _ <- 1..100, do: ExFsrs.Scheduler.get_fuzzed_interval(2.5)

      # Should have variation (not all the same value)
      assert MapSet.size(MapSet.new(results)) > 1,
             "Fuzzed values at 2.5 should vary, got: #{inspect(Enum.uniq(results))}"

      # All values should be reasonable (around 2-5)
      assert Enum.all?(results, &(&1 >= 2 and &1 <= 5))
    end

    test "fuzz respects maximum_interval" do
      max_ivl = 5
      results = for _ <- 1..100, do: ExFsrs.Scheduler.get_fuzzed_interval(10.0, max_ivl)
      assert Enum.all?(results, &(&1 <= max_ivl))
    end

    test "fuzzed intervals for interval=10 stay within expected range" do
      # Cumulative delta for interval=10:
      #   base=1.0 + 0.15*(7-2.5) + 0.1*(10-7) = 1.0 + 0.675 + 0.3 = 1.975
      # min_ivl = max(2, round(10-1.975)) = 8
      # max_ivl = round(10+1.975) = 12
      results = for _ <- 1..100, do: ExFsrs.Scheduler.get_fuzzed_interval(10.0)

      {min_val, max_val} = Enum.min_max(results)

      assert min_val >= 8,
             "Min fuzzed value should be >= 8, got #{min_val}"

      assert max_val <= 13,
             "Max fuzzed value should be <= 13, got #{max_val}"
    end
  end

  # --- 10. Parameter Validation ---
  describe "parameter validation" do
    test "20 parameters raises ArgumentError" do
      assert_raise ArgumentError, ~r/expected 21 parameters, got 20/, fn ->
        ExFsrs.Scheduler.new(parameters: List.duplicate(1.0, 20))
      end
    end

    test "22 parameters raises ArgumentError" do
      assert_raise ArgumentError, ~r/expected 21 parameters, got 22/, fn ->
        ExFsrs.Scheduler.new(parameters: List.duplicate(1.0, 22))
      end
    end

    test "exactly 21 parameters succeeds" do
      scheduler = ExFsrs.Scheduler.new(parameters: List.duplicate(1.0, 21))
      assert scheduler.parameters == List.duplicate(1.0, 21)
    end
  end

  # --- 11. Short-Term Stability ---
  # Source: ts-fsrs algorithm.test.ts "next_short_term_stability"
  # Inputs: s=5 for all ratings, default parameters
  # Expected: [1.596818, 5, 5, 8.12960956]
  # fsrs-rs applies max(sinc, 1.0) for rating >= 2 (Hard, Good, Easy)
  # Only Again (rating=1) can decrease stability in short-term reviews
  describe "short-term stability (ts-fsrs golden values)" do
    setup do
      scheduler = ExFsrs.Scheduler.new(enable_fuzzing: false)

      # Card in review state, reviewed same day (elapsed < 1 day → short-term path)
      card =
        ExFsrs.new(
          state: :review,
          stability: 5.0,
          difficulty: 5.0,
          last_review: @start_datetime
        )

      {:ok, scheduler: scheduler, card: card, now: @start_datetime}
    end

    test "again decreases stability", ctx do
      {card, _} = ExFsrs.Scheduler.review_card(ctx.scheduler, ctx.card, :again, ctx.now)
      assert_in_delta card.stability, 1.5968179979869215, 1.0e-9
    end

    test "hard preserves stability via clamp", ctx do
      {card, _} = ExFsrs.Scheduler.review_card(ctx.scheduler, ctx.card, :hard, ctx.now)
      assert card.stability == 5.0
    end

    test "good preserves stability via clamp", ctx do
      {card, _} = ExFsrs.Scheduler.review_card(ctx.scheduler, ctx.card, :good, ctx.now)
      assert card.stability == 5.0
    end

    test "easy increases stability", ctx do
      {card, _} = ExFsrs.Scheduler.review_card(ctx.scheduler, ctx.card, :easy, ctx.now)
      assert_in_delta card.stability, 8.129609559916112, 1.0e-9
    end
  end

  # --- 12. Forget Stability (Lapse) ---
  # Source: ts-fsrs algorithm.test.ts "next_forget_stability"
  # Inputs: s=5, d=1, r=0.9 (achieved by elapsed_days=stability=5)
  # Expected: s_fail = 1.05253961
  # Formula: w[11] * d^(-w[12]) * ((s+1)^w[13] - 1) * exp((1-r)*w[14])
  #   floored by min(result, s / exp(w[17]*w[18]))
  describe "forget stability after lapse (ts-fsrs golden values)" do
    test "again on review card at r=0.9 produces expected forget stability" do
      scheduler = ExFsrs.Scheduler.new(enable_fuzzing: false)

      card =
        ExFsrs.new(
          state: :review,
          stability: 5.0,
          difficulty: 1.0,
          last_review: @start_datetime
        )

      # Review 5 days later → elapsed=stability → r=0.9 exactly
      review_at = DateTime.add(@start_datetime, 5, :day)
      {card, _} = ExFsrs.Scheduler.review_card(scheduler, card, :again, review_at)

      assert_in_delta card.stability, 1.0525396080650788, 1.0e-9
    end
  end
end
