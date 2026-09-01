defmodule ExFsrs.OptimizerTest do
  @moduledoc """
  Checks the optimizer against a recorded run of py-fsrs's own optimizer over
  py-fsrs's own review-log fixture (see `test/fixtures/generate_trace.py`).

  The recorded run reproduces py-fsrs's published `test_optimal_parameters` to
  8.9e-16, so the trace is ground truth rather than a reimplementation.

  Exact agreement depends on replaying py-fsrs's card ordering, which sets the
  minibatch boundaries and therefore the whole Adam trajectory. The trace
  records that ordering per epoch and the tests inject it via `:card_orders`.
  """
  use ExUnit.Case, async: true

  alias ExFsrs.Fixtures
  alias ExFsrs.Optimizer
  alias ExFsrs.Optimizer.Data

  setup_all do
    trace = Fixtures.trace()
    logs = Fixtures.review_log_tuples()

    {:ok, trace: trace, logs: logs, sequences: Data.build_sequences(logs)}
  end

  describe "data pipeline" do
    test "reproduces py-fsrs's card and scored-review counts", ctx do
      assert Enum.count_until(ctx.sequences, 1206) == 1205
      assert Data.num_reviews(ctx.sequences) == ctx.trace["num_reviews"]
    end

    test "review log structs and tuples produce identical sequences", ctx do
      from_tuples = ctx.sequences
      from_structs = Data.build_sequences(Fixtures.review_logs())

      assert from_tuples == from_structs
    end

    test "a review is scored only when it follows an earlier one by a full day" do
      start = ~U[2024-01-01 09:00:00Z]

      logs = [
        {1, :good, start},
        # same day: updates state, not scored
        {1, :good, DateTime.add(start, 3600)},
        {1, :again, DateTime.add(start, 3 * 86_400)}
      ]

      assert [{1, [first, same_day, next_day]}] = Data.build_sequences(logs)

      refute first.counts_for_loss?
      assert first.elapsed_days == -1

      refute same_day.counts_for_loss?
      assert same_day.elapsed_days == 0

      assert next_day.counts_for_loss?
      # Measured from the previous review at 10:00 on day 1, not from the first.
      assert next_day.elapsed_days == 2
      assert next_day.label == 0.0
    end
  end

  describe "minibatch partitioning" do
    test "matches py-fsrs's minibatch sizes for every epoch", ctx do
      expected =
        Enum.map(ctx.trace["epochs"], fn epoch ->
          ctx.trace["steps"]
          |> Enum.slice(epoch["first_step"]..epoch["last_step"])
          |> Enum.map(& &1["n_losses"])
        end)

      actual =
        Enum.map(ctx.trace["epochs"], fn epoch ->
          Optimizer.minibatch_sizes(ctx.sequences, epoch["card_order"])
        end)

      assert actual == expected
    end
  end

  describe "batch_loss/2" do
    test "matches py-fsrs at the default parameters", ctx do
      loss = Optimizer.batch_loss(ctx.sequences, ExFsrs.Scheduler.new().parameters)

      assert_in_delta loss, ctx.trace["batch_loss_at_defaults"], 1.0e-12
    end

    test "matches py-fsrs at its optimized parameters", ctx do
      loss = Optimizer.batch_loss(ctx.sequences, ctx.trace["final_parameters"])

      assert_in_delta loss, ctx.trace["batch_loss_at_expected"], 1.0e-12
    end

    test "accepts raw review logs as well as built sequences", ctx do
      from_sequences = Optimizer.batch_loss(ctx.sequences, ExFsrs.Scheduler.new().parameters)
      from_logs = Optimizer.batch_loss(ctx.logs, ExFsrs.Scheduler.new().parameters)

      assert from_sequences == from_logs
    end

    test "raises when nothing can be scored" do
      logs = [{1, :good, ~U[2024-01-01 09:00:00Z]}]

      assert_raise ArgumentError, ~r/no reviews to score/, fn ->
        Optimizer.batch_loss(logs, ExFsrs.Scheduler.new().parameters)
      end
    end

    test "optimized parameters beat the defaults", ctx do
      optimized = Optimizer.batch_loss(ctx.sequences, ctx.trace["final_parameters"])
      defaults = Optimizer.batch_loss(ctx.sequences, ExFsrs.Scheduler.new().parameters)

      assert optimized < defaults
    end
  end

  describe "predictions/2 and evaluate/2" do
    test "score every review that counts for the loss, and only those", ctx do
      predictions = Optimizer.predictions(ctx.sequences, ExFsrs.Scheduler.new().parameters)

      assert length(predictions) == Data.num_reviews(ctx.sequences)

      for {prediction, outcome, delta_t, index, lapses} <- predictions do
        assert prediction > 0.0 and prediction <= 1.0
        assert outcome in [0.0, 1.0]
        assert delta_t >= 1
        assert index >= 2
        assert lapses >= 0
      end
    end

    test "counts a lapse only once it has happened" do
      logs = [
        {1, :good, ~U[2024-01-01 09:00:00Z]},
        {1, :again, ~U[2024-01-03 09:00:00Z]},
        {1, :good, ~U[2024-01-05 09:00:00Z]}
      ]

      assert [{_p1, +0.0, 2, 2, 0}, {_p2, 1.0, 2, 3, 1}] =
               Optimizer.predictions(logs, ExFsrs.Scheduler.new().parameters)
    end

    test "evaluate reports log loss consistent with batch_loss", ctx do
      parameters = ExFsrs.Scheduler.new().parameters
      report = Optimizer.evaluate(ctx.sequences, parameters)

      assert report.reviews == Data.num_reviews(ctx.sequences)
      assert_in_delta report.log_loss, Optimizer.batch_loss(ctx.sequences, parameters), 1.0e-9
      assert report.rmse_bins > 0.0 and report.rmse_bins < 1.0
    end

    test "optimized parameters improve both metrics on the reference collection", ctx do
      defaults = Optimizer.evaluate(ctx.sequences, ExFsrs.Scheduler.new().parameters)
      optimized = Optimizer.evaluate(ctx.sequences, ctx.trace["final_parameters"])

      assert optimized.log_loss < defaults.log_loss
      assert optimized.rmse_bins < defaults.rmse_bins
    end
  end

  # A small collection — just over the training threshold — so a full run with
  # every option switched on stays fast enough to sit in the default suite.
  defp small_collection do
    seed = :rand.seed_s(:exsss, {4, 8, 15})

    {logs, _seed} = Enum.map_reduce(1..70, seed, &card_history/2)

    List.flatten(logs)
  end

  # Twelve reviews of one card, a random rating and a 1-6 day gap each.
  defp card_history(card_id, seed) do
    start = ~U[2024-01-01 09:00:00Z]

    {reviews, {_day, seed}} =
      Enum.map_reduce(0..11, {0, seed}, fn index, {day, seed} ->
        {roll, seed} = :rand.uniform_s(10, seed)
        {gap, seed} = :rand.uniform_s(6, seed)
        day = if index == 0, do: 0, else: day + gap

        {{card_id, rating_for(roll), DateTime.add(start, day, :day)}, {day, seed}}
      end)

    {reviews, seed}
  end

  defp rating_for(roll) when roll <= 2, do: :again
  defp rating_for(roll) when roll <= 4, do: :hard
  defp rating_for(roll) when roll <= 8, do: :good
  defp rating_for(_roll), do: :easy

  describe "compute_optimal_parameters/2 with the upgrades enabled" do
    @tag timeout: 300_000
    test "initialize, regularization and recency all take effect and stay in bounds" do
      logs = small_collection()
      sequences = Data.build_sequences(logs)
      assert Data.num_reviews(sequences) >= 512

      # Pin the card order so the two runs differ only in their options.
      card_ids = Enum.map(sequences, fn {card_id, _reviews} -> card_id end)
      orders = List.duplicate(card_ids, 5)

      plain = Optimizer.compute_optimal_parameters(logs, card_orders: orders)

      upgraded =
        Optimizer.compute_optimal_parameters(logs,
          card_orders: orders,
          initialize: true,
          regularization: 1.0,
          recency: true
        )

      assert Enum.count_until(upgraded, 22) == 21
      assert upgraded != plain
      assert upgraded != ExFsrs.Scheduler.new().parameters

      for {{value, lower}, upper} <-
            Enum.zip(Enum.zip(upgraded, Optimizer.lower_bounds()), Optimizer.upper_bounds()) do
        assert value >= lower and value <= upper
      end

      # The first four weights start from a fit of the data rather than the
      # defaults, so initialization alone must move them.
      initialized =
        Optimizer.compute_optimal_parameters(logs, card_orders: orders, initialize: true)

      assert Enum.take(initialized, 4) != Enum.take(plain, 4)
    end

    test "the scalar and batched models agree with recency weights applied" do
      logs = small_collection()
      sequences = Data.build_sequences(logs, recency: true)
      card_ids = Enum.map(sequences, fn {card_id, _reviews} -> card_id end)
      orders = List.duplicate(card_ids, 5)

      steps = fn model ->
        {:ok, agent} = Agent.start_link(fn -> [] end)

        Optimizer.compute_optimal_parameters(logs,
          card_orders: orders,
          recency: true,
          model: model,
          on_step: fn step -> Agent.update(agent, &[step.loss | &1]) end
        )

        losses = agent |> Agent.get(& &1) |> Enum.reverse()
        Agent.stop(agent)
        losses
      end

      batched = steps.(:batched)
      scalar = steps.(:scalar)

      assert length(batched) == length(scalar)

      for {b, s} <- Enum.zip(batched, scalar) do
        assert_in_delta b, s, abs(s) * 1.0e-10
      end
    end
  end

  describe "compute_optimal_parameters/2" do
    test "returns the defaults unchanged when there is too little data" do
      start = ~U[2024-01-01 09:00:00Z]

      logs =
        for card_id <- 1..10, review <- 0..4 do
          {card_id, :good, DateTime.add(start, review * 86_400)}
        end

      assert Optimizer.compute_optimal_parameters(logs) ==
               ExFsrs.Scheduler.new().parameters
    end

    # One full optimization run takes ~90s, so this covers every property of the
    # result in a single pass rather than re-running it per assertion.
    @tag timeout: 600_000
    test "reproduces py-fsrs's optimized parameters", ctx do
      orders = Enum.map(ctx.trace["epochs"], & &1["card_order"])

      parameters = Optimizer.compute_optimal_parameters(ctx.logs, card_orders: orders)

      assert Enum.count_until(parameters, 22) == 21

      for {{actual, expected}, index} <-
            Enum.with_index(Enum.zip(parameters, ctx.trace["final_parameters"])) do
        assert_in_delta actual,
                        expected,
                        1.0e-9,
                        "w[#{index}] diverged from py-fsrs: #{actual} vs #{expected}"
      end

      # Properties py-fsrs's own test suite asserts.
      assert parameters != ExFsrs.Scheduler.new().parameters

      assert Optimizer.batch_loss(ctx.sequences, parameters) <
               Optimizer.batch_loss(ctx.sequences, ExFsrs.Scheduler.new().parameters)

      for {{value, lower}, upper} <-
            Enum.zip(Enum.zip(parameters, Optimizer.lower_bounds()), Optimizer.upper_bounds()) do
        assert value >= lower
        assert value <= upper
      end

      # The result must be usable as scheduler parameters.
      scheduler = ExFsrs.Scheduler.new(parameters: parameters, enable_fuzzing: false)

      {reviewed, _log} =
        ExFsrs.Scheduler.review_card(
          scheduler,
          ExFsrs.new(state: :learning, step: 0),
          :good,
          ~U[2024-01-01 09:00:00Z]
        )

      assert reviewed.stability > 0
      assert reviewed.difficulty >= 1.0 and reviewed.difficulty <= 10.0
    end
  end
end
