defmodule ExFsrs.SchedulerTest do
  use ExUnit.Case, async: true
  doctest ExFsrs.Scheduler

  describe "new/1" do
    test "creates new scheduler with default parameters" do
      scheduler = ExFsrs.Scheduler.new()

      assert length(scheduler.parameters) == 21
      assert scheduler.desired_retention == 0.9
      assert scheduler.learning_steps == [1.0, 10.0]
      assert scheduler.relearning_steps == [10.0]
      assert scheduler.maximum_interval == 36500
      assert scheduler.enable_fuzzing == true
      assert scheduler.decay != nil
      assert scheduler.factor != nil
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
  end

  describe "review_card/5" do
    setup do
      scheduler = ExFsrs.Scheduler.new(enable_fuzzing: false)
      now = DateTime.utc_now()

      {:ok, scheduler: scheduler, now: now}
    end

    test "reviews new learning card with 'again' rating", %{scheduler: scheduler, now: now} do
      card = ExFsrs.new(state: :learning, step: 0)

      {updated_card, log} = ExFsrs.Scheduler.review_card(scheduler, card, :again, now)

      assert updated_card.state == :learning
      assert updated_card.step == 0
      assert updated_card.stability != nil
      assert updated_card.difficulty != nil
      assert updated_card.last_review == now
      assert DateTime.diff(updated_card.due, now, :minute) == 1

      assert log.rating == :again
      assert log.review_datetime == now
    end

    test "reviews new learning card with 'hard' rating", %{scheduler: scheduler, now: now} do
      card = ExFsrs.new(state: :learning, step: 0)

      {updated_card, _log} = ExFsrs.Scheduler.review_card(scheduler, card, :hard, now)

      assert updated_card.state == :learning
      assert updated_card.step == 0
      assert updated_card.stability != nil
      assert updated_card.difficulty != nil

      # Due in 1 to 6 minutes (depending on learning_steps and calculation)
      minutes_until_due = DateTime.diff(updated_card.due, now, :minute)
      assert minutes_until_due >= 1
      assert minutes_until_due <= 6
    end

    test "reviews new learning card with 'good' rating", %{scheduler: scheduler, now: now} do
      card = ExFsrs.new(state: :learning, step: 0)

      {updated_card, _log} = ExFsrs.Scheduler.review_card(scheduler, card, :good, now)

      assert updated_card.state == :learning
      assert updated_card.step == 1
      assert updated_card.stability != nil
      assert updated_card.difficulty != nil

      # Due in approximately 10 minutes
      assert DateTime.diff(updated_card.due, now, :minute) == 10
    end

    test "reviews new learning card with 'easy' rating", %{scheduler: scheduler, now: now} do
      card = ExFsrs.new(state: :learning, step: 0)

      {updated_card, _log} = ExFsrs.Scheduler.review_card(scheduler, card, :easy, now)

      assert updated_card.state == :review
      assert updated_card.step == nil
      assert updated_card.stability != nil
      assert updated_card.difficulty != nil

      # Due in at least 1 day
      assert DateTime.diff(updated_card.due, now, :day) >= 1
    end
  end

  describe "review_card/5 for review state" do
    setup do
      scheduler = ExFsrs.Scheduler.new(enable_fuzzing: false)
      now = DateTime.utc_now()

      card =
        ExFsrs.new(
          state: :review,
          stability: 10.0,
          difficulty: 5.0,
          last_review: DateTime.add(now, -10, :day),
          due: now
        )

      {:ok, scheduler: scheduler, now: now, card: card}
    end

    test "reviews review card with 'again' rating", %{scheduler: scheduler, now: now, card: card} do
      {updated_card, _log} = ExFsrs.Scheduler.review_card(scheduler, card, :again, now)

      assert updated_card.state == :relearning
      assert updated_card.step == 0
      assert updated_card.stability != nil
      # Difficulty should increase
      assert updated_card.difficulty > card.difficulty

      # Due in approximately 10 minutes
      assert DateTime.diff(updated_card.due, now, :minute) == 10
    end

    test "reviews review card with 'hard' rating", %{scheduler: scheduler, now: now, card: card} do
      {updated_card, _log} = ExFsrs.Scheduler.review_card(scheduler, card, :hard, now)

      assert updated_card.state == :review
      assert updated_card.step == nil
      assert updated_card.stability != nil
      # Difficulty should increase
      assert updated_card.difficulty > card.difficulty

      # Due in future days (depends on stability calculation)
      assert DateTime.diff(updated_card.due, now, :day) > 0
    end

    test "reviews review card with 'good' rating", %{scheduler: scheduler, now: now, card: card} do
      {updated_card, _log} = ExFsrs.Scheduler.review_card(scheduler, card, :good, now)

      assert updated_card.state == :review
      assert updated_card.step == nil
      # Stability should increase
      assert updated_card.stability > card.stability

      # Due in future days (more than hard rating)
      assert DateTime.diff(updated_card.due, now, :day) > 0
    end

    test "reviews review card with 'easy' rating", %{scheduler: scheduler, now: now, card: card} do
      {updated_card, _log} = ExFsrs.Scheduler.review_card(scheduler, card, :easy, now)

      assert updated_card.state == :review
      assert updated_card.step == nil
      # Stability should increase significantly
      assert updated_card.stability > card.stability

      # Due in future days (more than good rating)
      days_until_due = DateTime.diff(updated_card.due, now, :day)
      assert days_until_due > 0
    end
  end

  describe "review_card/5 for relearning state" do
    setup do
      scheduler = ExFsrs.Scheduler.new(enable_fuzzing: false)
      now = DateTime.utc_now()

      card =
        ExFsrs.new(
          state: :relearning,
          step: 0,
          stability: 5.0,
          difficulty: 7.0,
          last_review: DateTime.add(now, -1, :day),
          due: now
        )

      {:ok, scheduler: scheduler, now: now, card: card}
    end

    test "reviews relearning card with 'again' rating", %{
      scheduler: scheduler,
      now: now,
      card: card
    } do
      {updated_card, _log} = ExFsrs.Scheduler.review_card(scheduler, card, :again, now)

      assert updated_card.state == :relearning
      assert updated_card.step == 0
      # Stability should decrease
      assert updated_card.stability < card.stability
      # Difficulty should increase
      assert updated_card.difficulty > card.difficulty

      # Due in approximately 10 minutes
      assert DateTime.diff(updated_card.due, now, :minute) == 10
    end

    test "reviews relearning card with 'hard' rating", %{
      scheduler: scheduler,
      now: now,
      card: card
    } do
      {updated_card, _log} = ExFsrs.Scheduler.review_card(scheduler, card, :hard, now)

      assert updated_card.state == :relearning
      assert updated_card.step == 0
      assert updated_card.stability != nil
      # Difficulty should increase
      assert updated_card.difficulty > card.difficulty

      # Due in 15 minutes (10 * 1.5)
      assert DateTime.diff(updated_card.due, now, :minute) == 15
    end

    test "reviews relearning card with 'good' rating", %{
      scheduler: scheduler,
      now: now,
      card: card
    } do
      {updated_card, _log} = ExFsrs.Scheduler.review_card(scheduler, card, :good, now)

      assert updated_card.state == :review
      assert updated_card.step == nil
      assert updated_card.stability != nil

      # Due in future days
      assert DateTime.diff(updated_card.due, now, :day) > 0
    end

    test "reviews relearning card with 'easy' rating", %{
      scheduler: scheduler,
      now: now,
      card: card
    } do
      {updated_card, _log} = ExFsrs.Scheduler.review_card(scheduler, card, :easy, now)

      assert updated_card.state == :review
      assert updated_card.step == nil
      # Stability should increase significantly
      assert updated_card.stability > card.stability

      # Due in future days (more than good rating)
      assert DateTime.diff(updated_card.due, now, :day) > 0
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

  # Tests for private functions using function capture
  describe "internal utility functions" do
    test "rating_to_number/1 converts rating atoms to numbers" do
      assert ExFsrs.Scheduler.rating_to_number(:again) == 1
      assert ExFsrs.Scheduler.rating_to_number(:hard) == 2
      assert ExFsrs.Scheduler.rating_to_number(:good) == 3
      assert ExFsrs.Scheduler.rating_to_number(:easy) == 4
    end
  end
end
