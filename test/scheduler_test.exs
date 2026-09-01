defmodule ExFsrs.SchedulerTest do
  use ExUnit.Case, async: true

  describe "new/1" do
    test "creates new scheduler with default parameters" do
      scheduler = ExFsrs.Scheduler.new()

      assert Enum.count_until(scheduler.parameters, 22) == 21
      assert scheduler.desired_retention == 0.9
      assert scheduler.learning_steps == [1.0, 10.0]
      assert scheduler.relearning_steps == [10.0]
      assert scheduler.maximum_interval == 36_500
      assert scheduler.enable_fuzzing == true
      assert is_float(scheduler.decay)
      assert is_float(scheduler.factor)
    end

    test "creates new scheduler with custom parameters" do
      custom_params = List.duplicate(1.0, 21)

      scheduler =
        ExFsrs.Scheduler.new(
          parameters: custom_params,
          desired_retention: 0.8,
          learning_steps: [5.0, 15.0],
          relearning_steps: [20.0],
          maximum_interval: 1000,
          enable_fuzzing: false
        )

      assert scheduler.parameters == custom_params
      assert scheduler.desired_retention == 0.8
      assert scheduler.learning_steps == [5.0, 15.0]
      assert scheduler.relearning_steps == [20.0]
      assert scheduler.maximum_interval == 1000
      assert scheduler.enable_fuzzing == false
    end

    test "raises on wrong parameter count" do
      assert_raise ArgumentError, ~r/expected 21 parameters, got 5/, fn ->
        ExFsrs.Scheduler.new(parameters: [1.0, 2.0, 3.0, 4.0, 5.0])
      end
    end
  end

  describe "next_interval/2" do
    test "calculates proper interval based on stability" do
      scheduler = ExFsrs.Scheduler.new()

      # Test with different stability values — next_interval returns days
      intervals = [
        {1.0, 1},
        {5.0, 5},
        {25.0, 25},
        {100.0, 100}
      ]

      Enum.each(intervals, fn {stability, expected_days} ->
        result = ExFsrs.Scheduler.next_interval(stability, scheduler)
        assert result == expected_days
      end)
    end

    test "respects maximum interval" do
      scheduler = ExFsrs.Scheduler.new(maximum_interval: 100)

      # Even with very high stability, should not exceed maximum
      result = ExFsrs.Scheduler.next_interval(1000.0, scheduler)
      assert result == 100
    end
  end

  describe "get_fuzzed_interval/2" do
    test "does not change intervals below 2.5" do
      :rand.seed(:exsss, {1, 2, 3})

      result = ExFsrs.Scheduler.get_fuzzed_interval(2)
      assert result == 2
    end

    test "correctly fuzzes intervals using cumulative delta" do
      :rand.seed(:exsss, {1, 2, 3})

      # For interval=5 days: delta = 1.0 + 0.15*(5-2.5) = 1.375
      # min_ivl = max(2, round(5 - 1.375)) = 4
      # max_ivl = round(5 + 1.375) = 6
      result = ExFsrs.Scheduler.get_fuzzed_interval(5)
      assert result >= 4
      assert result <= 6
    end

    test "correctly fuzzes larger intervals" do
      :rand.seed(:exsss, {1, 2, 3})

      # For interval=30 days: delta = 1.0 + 0.15*4.5 + 0.1*13 + 0.05*10 = 3.475
      result = ExFsrs.Scheduler.get_fuzzed_interval(30)
      assert result >= 26
      assert result <= 34
    end

    test "respects maximum interval" do
      :rand.seed(:exsss, {1, 2, 3})

      result = ExFsrs.Scheduler.get_fuzzed_interval(100, 100)
      assert result <= 100
    end
  end

  describe "review_card/5 default datetime" do
    test "uses DateTime.utc_now() when review_datetime is not provided" do
      scheduler = ExFsrs.Scheduler.new(enable_fuzzing: false)
      card = ExFsrs.new(state: :learning, step: 0)

      {updated_card, log} = ExFsrs.Scheduler.review_card(scheduler, card, :good)

      assert updated_card.state == :learning
      assert is_number(updated_card.stability)
      assert %DateTime{} = log.review_datetime
    end
  end

  describe "review_card/5 with a learning card that has no step" do
    # py-fsrs treats a learning card with `step = None` as being at step 0.
    test "starts at step 0 rather than graduating" do
      scheduler = ExFsrs.Scheduler.new(enable_fuzzing: false)
      now = DateTime.utc_now()
      card = ExFsrs.new(state: :learning, step: nil)

      {updated_card, _log} = ExFsrs.Scheduler.review_card(scheduler, card, :good, now)

      assert updated_card.state == :learning
      assert updated_card.step == 1
      assert DateTime.diff(updated_card.due, now, :minute) == 10
    end
  end

  describe "review_card/5 with an invalid rating" do
    test "raises rather than scheduling" do
      scheduler = ExFsrs.Scheduler.new()
      rating = String.to_atom("meh")

      assert_raise FunctionClauseError, fn ->
        ExFsrs.Scheduler.review_card(scheduler, ExFsrs.new(), rating)
      end
    end
  end

  describe "review_card/5 with partial nil stability/difficulty" do
    test "treats card with only stability nil as initial" do
      scheduler = ExFsrs.Scheduler.new(enable_fuzzing: false)
      now = DateTime.utc_now()

      card = %ExFsrs{
        card_id: 1,
        state: :learning,
        step: 0,
        stability: nil,
        difficulty: 5.0,
        due: now,
        last_review: nil
      }

      {updated_card, _log} = ExFsrs.Scheduler.review_card(scheduler, card, :good, now)

      # a nil on either side means "new card": both fall back to the initial values
      assert updated_card.stability == 2.3065
      assert_in_delta updated_card.difficulty, 2.118103970459016, 1.0e-9
    end

    test "treats card with only difficulty nil as initial" do
      scheduler = ExFsrs.Scheduler.new(enable_fuzzing: false)
      now = DateTime.utc_now()

      card = %ExFsrs{
        card_id: 1,
        state: :learning,
        step: 0,
        stability: 2.0,
        difficulty: nil,
        due: now,
        last_review: nil
      }

      {updated_card, _log} = ExFsrs.Scheduler.review_card(scheduler, card, :good, now)

      assert updated_card.stability == 2.3065
      assert_in_delta updated_card.difficulty, 2.118103970459016, 1.0e-9
    end
  end

  describe "learning step overflow graduation" do
    test "good graduates when step >= length of learning_steps" do
      scheduler = ExFsrs.Scheduler.new(enable_fuzzing: false, learning_steps: [1.0])
      now = DateTime.utc_now()

      card = %ExFsrs{
        card_id: 1,
        state: :learning,
        step: 1,
        stability: 2.0,
        difficulty: 5.0,
        due: now,
        last_review: now
      }

      {updated_card, _log} = ExFsrs.Scheduler.review_card(scheduler, card, :good, now)
      assert updated_card.state == :review
      assert updated_card.step == nil
    end

    test "hard graduates when step >= length of learning_steps" do
      scheduler = ExFsrs.Scheduler.new(enable_fuzzing: false, learning_steps: [1.0])
      now = DateTime.utc_now()

      card = %ExFsrs{
        card_id: 1,
        state: :learning,
        step: 1,
        stability: 2.0,
        difficulty: 5.0,
        due: now,
        last_review: now
      }

      {updated_card, _log} = ExFsrs.Scheduler.review_card(scheduler, card, :hard, now)
      assert updated_card.state == :review
      assert updated_card.step == nil
    end
  end

  describe "relearning step overflow graduation" do
    # Mirrors the learning-step case: a card parked at a step index the current
    # scheduler no longer has. Goldens from py-fsrs on the same inputs.
    setup do
      scheduler = ExFsrs.Scheduler.new(enable_fuzzing: false, relearning_steps: [10.0])
      start = ~U[2022-11-29 12:30:00Z]

      card = %ExFsrs{
        card_id: 1,
        state: :relearning,
        step: 1,
        stability: 5.0,
        difficulty: 5.0,
        due: start,
        last_review: start
      }

      {:ok, scheduler: scheduler, card: card, now: DateTime.add(start, 2, :day)}
    end

    test "hard/good/easy graduate when step >= length of relearning_steps", ctx do
      expected = [
        {:hard, 8.623545704277777, 6.665995369296838, 9},
        {:good, 11.025184077615195, 4.9902283692968386, 11},
        {:easy, 16.284567258965495, 3.3144613692968385, 16}
      ]

      for {rating, stability, difficulty, days} <- expected do
        {updated, _} = ExFsrs.Scheduler.review_card(ctx.scheduler, ctx.card, rating, ctx.now)

        assert updated.state == :review, "#{rating} should graduate"
        assert updated.step == nil
        assert_in_delta updated.stability, stability, 1.0e-9
        assert_in_delta updated.difficulty, difficulty, 1.0e-9
        assert DateTime.diff(updated.due, ctx.now, :day) == days
      end
    end

    test "again still restarts relearning at step 0", ctx do
      {updated, _} = ExFsrs.Scheduler.review_card(ctx.scheduler, ctx.card, :again, ctx.now)

      assert updated.state == :relearning
      assert updated.step == 0
      assert_in_delta updated.stability, 0.8776886959219631, 1.0e-9
      assert DateTime.diff(updated.due, ctx.now, :second) == 600
    end
  end

  describe "enable_fuzzing affects interval calculations" do
    test "fuzzing changes the due date for review cards" do
      scheduler_no_fuzz = ExFsrs.Scheduler.new(enable_fuzzing: false)
      scheduler_with_fuzz = ExFsrs.Scheduler.new(enable_fuzzing: true)
      now = DateTime.utc_now()

      card =
        ExFsrs.new(
          state: :review,
          stability: 25.0,
          difficulty: 5.0,
          last_review: DateTime.add(now, -30, :day)
        )

      :rand.seed(:exsss, {1, 2, 3})

      {card_no_fuzz, _} = ExFsrs.Scheduler.review_card(scheduler_no_fuzz, card, :good, now)
      {card_with_fuzz, _} = ExFsrs.Scheduler.review_card(scheduler_with_fuzz, card, :good, now)

      no_fuzz_days = DateTime.diff(card_no_fuzz.due, now, :day)
      with_fuzz_days = DateTime.diff(card_with_fuzz.due, now, :day)

      assert no_fuzz_days != with_fuzz_days, "Fuzzing should change the interval"
    end
  end

  describe "reschedule_card/3 with different log formats" do
    test "accepts ReviewLog structs" do
      scheduler = ExFsrs.Scheduler.new(enable_fuzzing: false)
      now = DateTime.utc_now()
      card = ExFsrs.new(card_id: 100, state: :learning, step: 0)

      reviewed_card = ExFsrs.new(card_id: 100)
      log = ExFsrs.ReviewLog.new(reviewed_card, :good, now)

      rescheduled = ExFsrs.Scheduler.reschedule_card(scheduler, card, [log])
      assert rescheduled.card_id == 100
      assert rescheduled.state == :learning
      assert rescheduled.step == 1
      assert rescheduled.stability == 2.3065
    end

    test "accepts maps with ExFsrs card struct" do
      scheduler = ExFsrs.Scheduler.new(enable_fuzzing: false)
      now = DateTime.utc_now()
      card = ExFsrs.new(card_id: 100, state: :learning, step: 0)

      log = %{
        card: ExFsrs.new(card_id: 100),
        rating: :good,
        review_datetime: now
      }

      rescheduled = ExFsrs.Scheduler.reschedule_card(scheduler, card, [log])
      assert rescheduled.card_id == 100
      assert rescheduled.state == :learning
      assert rescheduled.step == 1
      assert rescheduled.stability == 2.3065
    end

    test "accepts maps with plain card_id field" do
      scheduler = ExFsrs.Scheduler.new(enable_fuzzing: false)
      now = DateTime.utc_now()
      card = ExFsrs.new(card_id: 100, state: :learning, step: 0)

      log = %{
        card_id: 100,
        rating: :good,
        review_datetime: now
      }

      rescheduled = ExFsrs.Scheduler.reschedule_card(scheduler, card, [log])
      assert rescheduled.card_id == 100
      assert rescheduled.state == :learning
      assert rescheduled.step == 1
      assert rescheduled.stability == 2.3065
    end
  end
end
