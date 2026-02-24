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
  end
end
