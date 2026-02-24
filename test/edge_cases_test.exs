defmodule ExFsrs.EdgeCasesTest do
  use ExUnit.Case, async: true

  describe "edge cases in scheduler" do
    test "handles very high stability values" do
      scheduler = ExFsrs.Scheduler.new()
      now = DateTime.utc_now()

      card =
        ExFsrs.new(
          state: :review,
          stability: 1_000_000.0,
          difficulty: 5.0
        )

      {updated_card, _} = ExFsrs.Scheduler.review_card(scheduler, card, :good, now)

      days_until_due = DateTime.diff(updated_card.due, now, :day)
      assert days_until_due <= scheduler.maximum_interval
    end

    test "handles very low stability values" do
      scheduler = ExFsrs.Scheduler.new()
      now = DateTime.utc_now()

      card =
        ExFsrs.new(
          state: :review,
          stability: 0.1,
          difficulty: 5.0
        )

      {updated_card, _} = ExFsrs.Scheduler.review_card(scheduler, card, :good, now)

      days_until_due = DateTime.diff(updated_card.due, now, :day)
      assert days_until_due >= 1
    end

    test "handles nil stability and difficulty for learning card" do
      scheduler = ExFsrs.Scheduler.new()
      now = DateTime.utc_now()

      card =
        ExFsrs.new(
          state: :learning,
          stability: nil,
          difficulty: nil
        )

      {updated_card, _} = ExFsrs.Scheduler.review_card(scheduler, card, :good, now)

      assert is_number(updated_card.stability)
      assert is_number(updated_card.difficulty)
    end
  end
end
