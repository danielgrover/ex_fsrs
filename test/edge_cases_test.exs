defmodule ExFsrs.EdgeCasesTest do
  use ExUnit.Case, async: true

  @start_datetime ~U[2022-11-29 12:30:00Z]

  describe "edge cases in scheduler" do
    test "very high stability is capped at maximum_interval" do
      scheduler = ExFsrs.Scheduler.new(enable_fuzzing: false)

      card =
        ExFsrs.new(
          state: :review,
          stability: 1_000_000.0,
          difficulty: 5.0,
          last_review: @start_datetime
        )

      {updated_card, _} = ExFsrs.Scheduler.review_card(scheduler, card, :good, @start_datetime)

      days_until_due = DateTime.diff(updated_card.due, @start_datetime, :day)
      assert days_until_due == scheduler.maximum_interval
    end

    test "very low stability produces minimum 1-day interval" do
      scheduler = ExFsrs.Scheduler.new(enable_fuzzing: false)

      card =
        ExFsrs.new(
          state: :review,
          stability: 0.1,
          difficulty: 5.0,
          last_review: @start_datetime
        )

      {updated_card, _} = ExFsrs.Scheduler.review_card(scheduler, card, :good, @start_datetime)

      days_until_due = DateTime.diff(updated_card.due, @start_datetime, :day)
      assert days_until_due == 1
    end

    test "nil stability and difficulty initializes to valid values" do
      scheduler = ExFsrs.Scheduler.new(enable_fuzzing: false)

      card =
        ExFsrs.new(
          state: :learning,
          stability: nil,
          difficulty: nil
        )

      {updated_card, _} = ExFsrs.Scheduler.review_card(scheduler, card, :good, @start_datetime)

      assert updated_card.stability >= 0.001
      assert updated_card.difficulty >= 1.0
      assert updated_card.difficulty <= 10.0
    end

    test "nil stability in review state initializes instead of crashing" do
      scheduler = ExFsrs.Scheduler.new(enable_fuzzing: false)

      card =
        ExFsrs.new(
          state: :review,
          stability: nil,
          difficulty: nil,
          last_review: @start_datetime
        )

      {updated_card, _} = ExFsrs.Scheduler.review_card(scheduler, card, :good, @start_datetime)

      assert is_number(updated_card.stability)
      assert updated_card.stability >= 0.001
      assert updated_card.difficulty >= 1.0
      assert updated_card.difficulty <= 10.0
    end

    test "nil stability in relearning state initializes instead of crashing" do
      scheduler = ExFsrs.Scheduler.new(enable_fuzzing: false)

      card =
        ExFsrs.new(
          state: :relearning,
          step: 0,
          stability: nil,
          difficulty: nil,
          last_review: @start_datetime
        )

      {updated_card, _} = ExFsrs.Scheduler.review_card(scheduler, card, :good, @start_datetime)

      assert is_number(updated_card.stability)
      assert updated_card.stability >= 0.001
      assert updated_card.difficulty >= 1.0
      assert updated_card.difficulty <= 10.0
    end

    test "get_retrievability returns 0 for nil stability" do
      scheduler = ExFsrs.Scheduler.new()

      card =
        ExFsrs.new(
          state: :review,
          stability: nil,
          difficulty: 5.0,
          last_review: @start_datetime
        )

      r = ExFsrs.Scheduler.get_retrievability(card, @start_datetime, scheduler)
      assert r == 0
    end
  end

  describe "short-term vs long-term boundary" do
    test "review at exactly 1 day elapsed uses long-term stability path" do
      scheduler = ExFsrs.Scheduler.new(enable_fuzzing: false)
      review_at = DateTime.add(@start_datetime, 1, :day)

      card =
        ExFsrs.new(
          state: :review,
          stability: 10.0,
          difficulty: 5.0,
          last_review: @start_datetime
        )

      {card_at_1d, _} = ExFsrs.Scheduler.review_card(scheduler, card, :good, review_at)

      # Same-day review (short-term path)
      {card_at_0d, _} = ExFsrs.Scheduler.review_card(scheduler, card, :good, @start_datetime)

      # The two paths should produce different stability values
      # because they use different formulas
      assert card_at_1d.stability != card_at_0d.stability
    end

    test "review at 23 hours uses short-term path (< 1 day)" do
      scheduler = ExFsrs.Scheduler.new(enable_fuzzing: false)
      review_at = DateTime.add(@start_datetime, 23 * 60, :minute)

      card =
        ExFsrs.new(
          state: :review,
          stability: 10.0,
          difficulty: 5.0,
          last_review: @start_datetime
        )

      {card_23h, _} = ExFsrs.Scheduler.review_card(scheduler, card, :good, review_at)

      # Same-day review (short-term path)
      {card_0d, _} = ExFsrs.Scheduler.review_card(scheduler, card, :good, @start_datetime)

      # Both use short-term path, so stability should be identical
      # (short-term stability doesn't depend on elapsed time)
      assert card_23h.stability == card_0d.stability
    end
  end

  describe "zero stability crash (pow(0.0, negative))" do
    test "stability 0.0 in short-term path does not crash" do
      scheduler = ExFsrs.Scheduler.new(enable_fuzzing: false)

      card =
        ExFsrs.new(
          state: :learning,
          step: 0,
          stability: 0.0,
          difficulty: 5.0,
          last_review: @start_datetime
        )

      {updated_card, _} = ExFsrs.Scheduler.review_card(scheduler, card, :good, @start_datetime)

      assert is_number(updated_card.stability)
      assert updated_card.stability >= 0.001
    end

    test "stability 0.0 in long-term (review state) path does not crash" do
      scheduler = ExFsrs.Scheduler.new(enable_fuzzing: false)
      review_at = DateTime.add(@start_datetime, 2, :day)

      card =
        ExFsrs.new(
          state: :review,
          stability: 0.0,
          difficulty: 5.0,
          last_review: @start_datetime
        )

      {updated_card, _} = ExFsrs.Scheduler.review_card(scheduler, card, :good, review_at)

      assert is_number(updated_card.stability)
      assert updated_card.stability >= 0.001
    end
  end
end
