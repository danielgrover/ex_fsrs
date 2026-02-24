defmodule ExFsrsTest do
  use ExUnit.Case, async: true
  doctest ExFsrs

  describe "new/1" do
    test "creates a new card with default values" do
      card = ExFsrs.new()

      assert card.state == :learning
      assert card.step == 0
      assert card.stability == nil
      assert card.difficulty == nil
      assert card.last_review == nil
      assert is_integer(card.card_id)
      assert %DateTime{} = card.due
    end

    test "creates a new card with custom values" do
      now = DateTime.utc_now()

      card =
        ExFsrs.new(
          card_id: 12_345,
          state: :review,
          step: 2,
          stability: 10.5,
          difficulty: 4.2,
          due: now,
          last_review: now
        )

      assert card.card_id == 12_345
      assert card.state == :review
      assert card.step == 2
      assert card.stability == 10.5
      assert card.difficulty == 4.2
      assert card.due == now
      assert card.last_review == now
    end
  end

  describe "to_map/1" do
    test "converts card to map" do
      now = DateTime.utc_now()

      card =
        ExFsrs.new(
          card_id: 12_345,
          state: :review,
          step: 2,
          stability: 10.5,
          difficulty: 4.2,
          due: now,
          last_review: now
        )

      map = ExFsrs.to_map(card)

      assert map["card_id"] == 12_345
      assert map["state"] == "review"
      assert map["step"] == 2
      assert map["stability"] == 10.5
      assert map["difficulty"] == 4.2
      assert map["due"] == DateTime.to_iso8601(now)
      assert map["last_review"] == DateTime.to_iso8601(now)
    end

    test "handles nil last_review" do
      now = DateTime.utc_now()

      card =
        ExFsrs.new(
          card_id: 12_345,
          state: :review,
          step: 2,
          stability: 10.5,
          difficulty: 4.2,
          due: now,
          last_review: nil
        )

      map = ExFsrs.to_map(card)

      assert map["last_review"] == nil
    end
  end

  describe "from_map/1" do
    test "creates card from map with string keys" do
      now = DateTime.utc_now()
      now_iso = DateTime.to_iso8601(now)

      map = %{
        "card_id" => 12_345,
        "state" => "review",
        "step" => 2,
        "stability" => 10.5,
        "difficulty" => 4.2,
        "due" => now_iso,
        "last_review" => now_iso
      }

      card = ExFsrs.from_map(map)

      assert card.card_id == 12_345
      assert card.state == :review
      assert card.step == 2
      assert card.stability == 10.5
      assert card.difficulty == 4.2
      assert DateTime.to_iso8601(card.due) == now_iso
      assert DateTime.to_iso8601(card.last_review) == now_iso
    end

    test "creates card from map with atom keys" do
      now = DateTime.utc_now()
      now_iso = DateTime.to_iso8601(now)

      map = %{
        card_id: 12_345,
        state: "review",
        step: 2,
        stability: 10.5,
        difficulty: 4.2,
        due: now_iso,
        last_review: now_iso
      }

      card = ExFsrs.from_map(map)

      assert card.card_id == 12_345
      assert card.state == :review
      assert card.step == 2
      assert card.stability == 10.5
      assert card.difficulty == 4.2
      assert DateTime.to_iso8601(card.due) == now_iso
      assert DateTime.to_iso8601(card.last_review) == now_iso
    end

    test "handles nil last_review" do
      now = DateTime.utc_now()
      now_iso = DateTime.to_iso8601(now)

      map = %{
        "card_id" => 12_345,
        "state" => "review",
        "step" => 2,
        "stability" => 10.5,
        "difficulty" => 4.2,
        "due" => now_iso,
        "last_review" => nil
      }

      card = ExFsrs.from_map(map)

      assert card.last_review == nil
    end

    test "raises error on invalid due date" do
      map = %{
        "card_id" => 12_345,
        "state" => "review",
        "step" => 2,
        "stability" => 10.5,
        "difficulty" => 4.2,
        "due" => "invalid date",
        "last_review" => nil
      }

      assert_raise RuntimeError, ~r/Invalid ISO8601 datetime format/, fn ->
        ExFsrs.from_map(map)
      end
    end

    test "raises error on invalid last_review date" do
      now = DateTime.utc_now()
      now_iso = DateTime.to_iso8601(now)

      map = %{
        "card_id" => 12_345,
        "state" => "review",
        "step" => 2,
        "stability" => 10.5,
        "difficulty" => 4.2,
        "due" => now_iso,
        "last_review" => "invalid date"
      }

      assert_raise RuntimeError, ~r/Invalid ISO8601 datetime format/, fn ->
        ExFsrs.from_map(map)
      end
    end

    test "defaults to :learning when state is neither atom nor string" do
      now_iso = DateTime.to_iso8601(DateTime.utc_now())

      map = %{
        "card_id" => 1,
        "state" => 42,
        "step" => 0,
        "stability" => nil,
        "difficulty" => nil,
        "due" => now_iso,
        "last_review" => nil
      }

      card = ExFsrs.from_map(map)
      assert card.state == :learning
    end
  end

  describe "get_retrievability/2" do
    test "returns 0 for card with nil last_review" do
      card = ExFsrs.new()

      assert ExFsrs.get_retrievability(card) == 0
    end
  end

  describe "review_card/4" do
    test "delegates to ExFsrs.Scheduler.review_card/5" do
      card = ExFsrs.new()
      now = DateTime.utc_now()

      {updated_card, _log} = ExFsrs.review_card(card, :good, now, 1000)

      assert updated_card.state == :learning
      assert is_number(updated_card.stability)
      assert is_number(updated_card.difficulty)
      assert updated_card.last_review == now
    end
  end

  describe "reschedule_card/2" do
    test "delegates to Scheduler.reschedule_card with default scheduler" do
      card = ExFsrs.new(card_id: 42, state: :learning, step: 0)
      now = DateTime.utc_now()

      logs = [
        %{card: %{card_id: 42}, rating: :good, review_datetime: now},
        %{card: %{card_id: 42}, rating: :good, review_datetime: DateTime.add(now, 10, :minute)}
      ]

      rescheduled = ExFsrs.reschedule_card(card, logs)
      assert rescheduled.card_id == 42
      assert is_number(rescheduled.stability)
      assert is_number(rescheduled.difficulty)
    end
  end
end
