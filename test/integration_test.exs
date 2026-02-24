defmodule ExFsrs.IntegrationTest do
  use ExUnit.Case, async: true

  @now ~U[2022-11-29 12:30:00Z]

  describe "serialization and deserialization" do
    test "round-trip card through map conversion" do
      scheduler = ExFsrs.Scheduler.new(enable_fuzzing: false)
      card = ExFsrs.new(due: @now)
      {reviewed_card, _} = ExFsrs.Scheduler.review_card(scheduler, card, :good, @now)

      map = ExFsrs.to_map(reviewed_card)
      restored_card = ExFsrs.from_map(map)

      assert restored_card.card_id == reviewed_card.card_id
      assert restored_card.state == reviewed_card.state
      assert restored_card.step == reviewed_card.step
      assert restored_card.stability == reviewed_card.stability
      assert restored_card.difficulty == reviewed_card.difficulty
      assert restored_card.due == reviewed_card.due
      assert restored_card.last_review == reviewed_card.last_review
    end

    test "round-trip review log through map conversion" do
      scheduler = ExFsrs.Scheduler.new(enable_fuzzing: false)
      card = ExFsrs.new(due: @now)
      {reviewed_card, _} = ExFsrs.Scheduler.review_card(scheduler, card, :good, @now)

      log_struct = ExFsrs.ReviewLog.new(reviewed_card, :good, @now, 1000)
      map = ExFsrs.ReviewLog.to_map(log_struct)
      restored_log = ExFsrs.ReviewLog.from_map(map)

      assert restored_log.card.card_id == log_struct.card.card_id
      assert restored_log.rating == log_struct.rating
      assert restored_log.review_datetime == log_struct.review_datetime
      assert restored_log.review_duration == log_struct.review_duration
    end
  end
end
