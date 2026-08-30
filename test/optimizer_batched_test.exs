defmodule ExFsrs.Optimizer.Model.BatchedTest do
  @moduledoc """
  Diffs the batched model against the scalar one, which is the stricter of the
  two available checks: it compares the exact quantities the optimizer consumes
  (a minibatch's summed loss, its gradient, and the state carried to the next
  minibatch) on the exact chunks the optimizer would build, rather than only the
  end of a full training run.

  The scalar model is itself pinned to `ExFsrs.Scheduler` and to py-fsrs's
  recorded trace, so agreement here inherits both.
  """
  use ExUnit.Case, async: true

  alias ExFsrs.Fixtures
  alias ExFsrs.Optimizer
  alias ExFsrs.Optimizer.Data
  alias ExFsrs.Optimizer.Model.Batched

  setup_all do
    trace = Fixtures.trace()
    sequences = Data.build_sequences(Fixtures.review_log_tuples())
    params = Nx.tensor(ExFsrs.Scheduler.new().parameters, type: :f64)

    chunks = Optimizer.minibatches(sequences, hd(trace["epochs"])["card_order"])

    {:ok, trace: trace, sequences: sequences, params: params, chunks: chunks}
  end

  defp num(tensor), do: Nx.to_number(tensor)

  # Walks an epoch's minibatches through both models, threading each one's carry
  # state into the next exactly as the optimizer does.
  defp walk(params, chunks) do
    {_scalar_carry, _batched_carry, results} =
      Enum.reduce(chunks, {nil, nil, []}, fn chunk, {scalar_carry, batch_carry, results} ->
        {scalar_loss, scalar_state} = Optimizer.scalar_chunk_loss(params, chunk, scalar_carry)

        {batch_loss, stability, difficulty} =
          Batched.run(params, Batched.prepare(chunk, batch_carry))

        batch_state = {stability, difficulty}

        {scalar_state, batch_state,
         [
           %{
             scalar: scalar_loss,
             batched: batch_loss,
             scalar_state: scalar_state,
             batched_state: batch_state
           }
           | results
         ]}
      end)

    Enum.reverse(results)
  end

  describe "agreement with the scalar model" do
    test "loss and carry state match on every minibatch of an epoch", ctx do
      results = walk(ctx.params, ctx.chunks)

      assert Enum.count(results) == Enum.count(ctx.chunks)

      for {result, index} <- Enum.with_index(results) do
        assert_in_delta num(result.batched),
                        num(result.scalar),
                        abs(num(result.scalar)) * 1.0e-12,
                        "minibatch #{index} loss diverged"

        {expected_stability, expected_difficulty} = result.scalar_state
        {stability, difficulty} = result.batched_state

        assert_in_delta num(stability),
                        num(expected_stability),
                        1.0e-11,
                        "minibatch #{index} carry stability diverged"

        assert_in_delta num(difficulty),
                        num(expected_difficulty),
                        1.0e-11,
                        "minibatch #{index} carry difficulty diverged"
      end
    end

    test "gradients match, and none are NaN", ctx do
      # Only the first few minibatches: a gradient pass is the expensive part,
      # and a per-element divergence shows up immediately.
      for {chunk, index} <- ctx.chunks |> Enum.take(3) |> Enum.with_index() do
        carry = if index == 0, do: nil, else: seed_carry()

        {_loss, expected} =
          Nx.Defn.value_and_grad(ctx.params, fn p ->
            {loss, _carry} = Optimizer.scalar_chunk_loss(p, chunk, carry)
            loss
          end)

        batch = Batched.prepare(chunk, carry)

        {_loss, actual} =
          Nx.Defn.value_and_grad(ctx.params, fn p ->
            {loss, _stability, _difficulty} = Batched.run(p, batch)
            loss
          end)

        # A NaN in a discarded branch survives Nx.select/3 into the gradient
        # even when the value itself came out clean, so check explicitly.
        for value <- Nx.to_flat_list(actual) do
          assert is_float(value), "non-finite gradient in minibatch #{index}: #{inspect(value)}"
        end

        for i <- 0..20 do
          expected_i = num(expected[i])

          assert_in_delta num(actual[i]),
                          expected_i,
                          1.0e-9 + abs(expected_i) * 1.0e-10,
                          "minibatch #{index} gradient w[#{i}] diverged"
        end
      end
    end

    test "carried-in state reaches a continuing segment", ctx do
      # The second minibatch always opens with a continuing segment, because the
      # first one cuts mid-card.
      chunk = Enum.at(ctx.chunks, 1)
      assert hd(chunk).continues?

      carry = seed_carry()
      other = {Nx.tensor(1.5, type: :f64), Nx.tensor(9.0, type: :f64)}

      {expected, _} = Optimizer.scalar_chunk_loss(ctx.params, chunk, carry)
      {actual, _stability, _difficulty} = Batched.run(ctx.params, Batched.prepare(chunk, carry))

      assert_in_delta num(actual), num(expected), abs(num(expected)) * 1.0e-12

      # And the carry must actually be read, not silently dropped: a different
      # carry has to produce a different loss.
      {different, _} = Optimizer.scalar_chunk_loss(ctx.params, chunk, other)
      refute_in_delta num(expected), num(different), 1.0e-9
    end

    test "padding does not affect the result", ctx do
      ragged =
        ctx.sequences
        |> Enum.filter(fn {_id, reviews} -> length(reviews) in [2, 8, 20] end)
        |> Enum.take(12)
        |> Enum.map(fn {_id, reviews} -> %{reviews: reviews, continues?: false} end)

      lengths = ragged |> Enum.map(&length(&1.reviews)) |> Enum.uniq()
      assert match?([_, _ | _], lengths), "expected ragged lengths, got #{inspect(lengths)}"

      {expected, _} = Optimizer.scalar_chunk_loss(ctx.params, ragged, nil)
      {actual, _stability, _difficulty} = Batched.run(ctx.params, Batched.prepare(ragged, nil))

      assert_in_delta num(actual), num(expected), abs(num(expected)) * 1.0e-12
    end

    test "agreement holds at parameters far from the defaults", ctx do
      # Guards against agreement that only holds near the defaults.
      params = Nx.tensor(ctx.trace["final_parameters"], type: :f64)
      chunk = hd(ctx.chunks)

      {expected, _} = Optimizer.scalar_chunk_loss(params, chunk, nil)
      {actual, _stability, _difficulty} = Batched.run(params, Batched.prepare(chunk, nil))

      assert_in_delta num(actual), num(expected), abs(num(expected)) * 1.0e-12
    end
  end

  defp seed_carry, do: {Nx.tensor(7.5, type: :f64), Nx.tensor(4.25, type: :f64)}
end
