defmodule ExFsrs.Optimizer.ModelTest do
  @moduledoc """
  Verifies the differentiable forward model against the already-validated
  `ExFsrs.Scheduler`, and its gradients against central finite differences.

  The scheduler is itself pinned to ts-fsrs/py-fsrs/fsrs-rs by
  `ExFsrs.ReferenceTest` and `ExFsrs.AlgorithmValidationTest`, so agreeing with
  it transitively pins the Nx model to the reference implementations.
  """
  use ExUnit.Case, async: true

  alias ExFsrs.Optimizer.Loss
  alias ExFsrs.Optimizer.Model

  @start ~U[2022-11-29 12:30:00Z]
  @ratings [:again, :hard, :good, :easy]
  @rating_numbers %{again: 1, hard: 2, good: 3, easy: 4}

  defp default_params, do: Nx.tensor(ExFsrs.Scheduler.new().parameters, type: :f64)

  defp num(tensor), do: Nx.to_number(tensor)

  # Replays the same {rating, elapsed_days} sequence through the scheduler and
  # the Nx model, returning both stability/difficulty traces.
  defp replay(sequence) do
    scheduler = ExFsrs.Scheduler.new(enable_fuzzing: false)
    params = default_params()

    initial = {ExFsrs.new(state: :learning, step: 0), @start, nil, [], []}

    {_card, _now, _state, scheduler_trace, model_trace} =
      Enum.reduce(sequence, initial, fn {rating, elapsed_days},
                                        {card, now, state, sched_acc, model_acc} ->
        # Adding whole days keeps DateTime.diff/3 in :day exact; the 2-hour
        # offset for a 0-day gap exercises the same-day branch.
        now = DateTime.add(now, elapsed_days * 86_400 + if(elapsed_days == 0, do: 7200, else: 0))

        {card, _log} = ExFsrs.Scheduler.review_card(scheduler, card, rating, now)

        state = Model.step(params, state, @rating_numbers[rating], elapsed_days)
        {stability, difficulty} = state

        {card, now, state, [{card.stability, card.difficulty} | sched_acc],
         [{num(stability), num(difficulty)} | model_acc]}
      end)

    {Enum.reverse(scheduler_trace), Enum.reverse(model_trace)}
  end

  describe "agreement with ExFsrs.Scheduler" do
    test "first review matches for every rating" do
      params = default_params()
      scheduler = ExFsrs.Scheduler.new(enable_fuzzing: false)

      for rating <- @ratings do
        {card, _log} =
          ExFsrs.Scheduler.review_card(
            scheduler,
            ExFsrs.new(state: :learning, step: 0),
            rating,
            @start
          )

        {stability, difficulty} = Model.step(params, nil, @rating_numbers[rating], -1)

        assert_in_delta num(stability), card.stability, 1.0e-12
        assert_in_delta num(difficulty), card.difficulty, 1.0e-12
      end
    end

    test "same-day reviews match (short-term stability branch)" do
      sequence = [{:good, -1}, {:good, 0}, {:again, 0}, {:hard, 0}, {:easy, 0}]
      {scheduler_trace, model_trace} = replay(sequence)

      for {{s1, d1}, {s2, d2}} <- Enum.zip(scheduler_trace, model_trace) do
        assert_in_delta s1, s2, 1.0e-12
        assert_in_delta d1, d2, 1.0e-12
      end
    end

    test "multi-day reviews match (long-term stability branch)" do
      sequence = [{:good, -1}, {:good, 3}, {:again, 10}, {:hard, 2}, {:easy, 30}, {:good, 100}]
      {scheduler_trace, model_trace} = replay(sequence)

      for {{s1, d1}, {s2, d2}} <- Enum.zip(scheduler_trace, model_trace) do
        assert_in_delta s1, s2, 1.0e-11
        assert_in_delta d1, d2, 1.0e-11
      end
    end

    test "randomized sequences match across all three branches" do
      seed = :rand.seed_s(:exsss, {1, 2, 3})

      {sequences, _seed} =
        Enum.map_reduce(1..40, seed, fn _, acc ->
          {sequence, acc} =
            Enum.map_reduce(1..12, acc, fn i, acc ->
              {rating_index, acc} = :rand.uniform_s(4, acc)
              # 0 exercises the same-day branch; -1 marks a card's first review.
              {elapsed, acc} = :rand.uniform_s(60, acc)
              elapsed = if i == 1, do: -1, else: rem(elapsed, 8)
              {{Enum.at(@ratings, rating_index - 1), elapsed}, acc}
            end)

          {sequence, acc}
        end)

      for sequence <- sequences do
        {scheduler_trace, model_trace} = replay(sequence)

        for {{s1, d1}, {s2, d2}} <- Enum.zip(scheduler_trace, model_trace) do
          assert_in_delta s1, s2, 1.0e-11
          assert_in_delta d1, d2, 1.0e-11
        end
      end
    end

    test "retrievability matches the scheduler's forgetting curve" do
      params = default_params()
      scheduler = ExFsrs.Scheduler.new(enable_fuzzing: false)

      for elapsed <- [0, 1, 3, 10, 365] do
        card = %ExFsrs{
          card_id: 1,
          state: :review,
          stability: 2.3065,
          difficulty: 5.0,
          due: @start,
          last_review: @start
        }

        expected =
          ExFsrs.Scheduler.get_retrievability(
            card,
            DateTime.add(@start, elapsed * 86_400),
            scheduler
          )

        actual = Model.retrievability(params, Nx.tensor(2.3065, type: :f64), elapsed)

        assert_in_delta num(actual), expected, 1.0e-12
      end
    end
  end

  describe "gradients" do
    # Loss over a fixed review sequence, as a function of the 21 parameters.
    defp sequence_loss(params) do
      sequence = [{3, -1}, {3, 2}, {1, 5}, {2, 1}, {4, 12}, {3, 40}]

      {total, _state} =
        Enum.reduce(sequence, {Nx.tensor(0.0, type: :f64), nil}, fn {rating, elapsed},
                                                                    {acc, state} ->
          {accumulate(params, acc, state, rating, elapsed),
           Model.step(params, state, rating, elapsed)}
        end)

      total
    end

    # A card's first review has nothing to predict from, so it is not scored.
    defp accumulate(_params, acc, nil, _rating, _elapsed), do: acc

    defp accumulate(params, acc, {stability, _difficulty}, rating, elapsed) do
      prediction = Model.retrievability(params, stability, elapsed)
      label = if rating == 1, do: 0.0, else: 1.0

      Nx.add(acc, Loss.binary_cross_entropy(prediction, label))
    end

    test "autodiff matches central finite differences for all 21 parameters" do
      params = default_params()

      {value, gradient} = Nx.Defn.value_and_grad(params, &sequence_loss/1)

      assert num(value) > 0.0
      assert Nx.shape(gradient) == {21}

      epsilon = 1.0e-6

      for index <- 0..20 do
        bump =
          0..20
          |> Enum.map(&if(&1 == index, do: epsilon, else: 0.0))
          |> Nx.tensor(type: :f64)

        forward = num(sequence_loss(Nx.add(params, bump)))
        backward = num(sequence_loss(Nx.subtract(params, bump)))
        numerical = (forward - backward) / (2 * epsilon)

        analytical = num(gradient[index])

        assert_in_delta analytical,
                        numerical,
                        1.0e-5 + abs(numerical) * 1.0e-4,
                        "gradient mismatch at w[#{index}]: " <>
                          "analytical=#{analytical} numerical=#{numerical}"
      end
    end

    test "gradient is non-zero for the parameters this sequence exercises" do
      params = default_params()
      {_value, gradient} = Nx.Defn.value_and_grad(params, &sequence_loss/1)

      # w[20] drives the forgetting curve and must always receive signal.
      assert abs(num(gradient[20])) > 0.0

      refute Enum.all?(0..20, fn i -> num(gradient[i]) == 0.0 end)
    end
  end

  describe "binary cross-entropy" do
    test "matches the closed form" do
      for {prediction, label} <- [{0.9, 1.0}, {0.9, 0.0}, {0.1, 1.0}, {0.5, 1.0}] do
        actual =
          Loss.binary_cross_entropy(Nx.tensor(prediction, type: :f64), label)

        expected = -(label * :math.log(prediction) + (1 - label) * :math.log(1 - prediction))

        assert_in_delta num(actual), expected, 1.0e-12
      end
    end

    test "clamps log outputs at -100 like torch" do
      loss = Loss.binary_cross_entropy(Nx.tensor(0.0, type: :f64), 1.0)

      assert num(loss) == 100.0
    end
  end
end
