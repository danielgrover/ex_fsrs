defmodule ExFsrs.IntegrationTest do
  use ExUnit.Case, async: true

  describe "serialization and deserialization" do
    test "round-trip card through map conversion" do
      card = ExFsrs.new()
      {reviewed_card, _} = ExFsrs.review_card(card, :good)

      map = ExFsrs.to_map(reviewed_card)
      restored_card = ExFsrs.from_map(map)

      assert restored_card.card_id == reviewed_card.card_id
      assert restored_card.state == reviewed_card.state
      assert restored_card.step == reviewed_card.step
      assert restored_card.stability == reviewed_card.stability
      assert restored_card.difficulty == reviewed_card.difficulty
      assert DateTime.to_iso8601(restored_card.due) == DateTime.to_iso8601(reviewed_card.due)

      assert DateTime.to_iso8601(restored_card.last_review) ==
               DateTime.to_iso8601(reviewed_card.last_review)
    end

    test "round-trip review log through map conversion" do
      card = ExFsrs.new()
      {reviewed_card, log} = ExFsrs.review_card(card, :good)

      log_struct = ExFsrs.ReviewLog.new(reviewed_card, :good, log.review_datetime, 1000)
      map = ExFsrs.ReviewLog.to_map(log_struct)
      restored_log = ExFsrs.ReviewLog.from_map(map)

      assert restored_log.card.card_id == log_struct.card.card_id
      assert restored_log.rating == log_struct.rating

      assert DateTime.to_date(restored_log.review_datetime) ==
               DateTime.to_date(log_struct.review_datetime)

      assert DateTime.to_time(restored_log.review_datetime) |> Time.truncate(:second) ==
               DateTime.to_time(log_struct.review_datetime) |> Time.truncate(:second)

      assert restored_log.review_duration == log_struct.review_duration
    end
  end
end
