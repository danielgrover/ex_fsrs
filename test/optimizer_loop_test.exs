defmodule ExFsrs.Optimizer.Model.LoopTest do
  @moduledoc """
  Diffs the `while`-loop model against the scalar reference on real minibatches,
  the same way `ExFsrs.Optimizer.Model.BatchedTest` checks the unrolled one.

  The loop model only works on an Nx whose gradients through `while` are
  correct for f64, which `Loop.supported?/0` detects. On an Nx without that fix
  the optimizer must refuse to use it rather than train on wrong gradients, so
  that path is asserted too.
  """
  use ExUnit.Case, async: true

  alias ExFsrs.Fixtures
  alias ExFsrs.Optimizer
  alias ExFsrs.Optimizer.Data
  alias ExFsrs.Optimizer.Model.Loop

  # Decided once, when the module compiles, so the agreement tests can be
  # skipped as a group on an Nx without the fix.
  @supported Loop.supported?()

  setup_all do
    trace = Fixtures.trace()
    sequences = Data.build_sequences(Fixtures.review_log_tuples())
    params = Nx.tensor(ExFsrs.Scheduler.new().parameters, type: :f64)
    chunks = Optimizer.minibatches(sequences, hd(trace["epochs"])["card_order"])

    {:ok, params: params, sequences: sequences, chunks: Enum.take(chunks, 2)}
  end

  defp num(tensor), do: Nx.to_number(tensor)

  test "supported?/0 answers with a boolean" do
    assert is_boolean(@supported)
  end

  # The check runs before any training, so a full-size collection costs nothing
  # here; a collection under the 512-review threshold would return the defaults
  # before ever reaching it.
  test "the optimizer refuses model: :loop on an Nx without the gradient fix", ctx do
    if @supported do
      # Nothing to assert against; the refusal only fires on an unpatched Nx.
      assert Loop.supported?()
    else
      assert_raise ArgumentError, ~r/model: :loop/, fn ->
        Optimizer.compute_optimal_parameters(ctx.sequences, model: :loop)
      end
    end
  end

  describe "agreement with the scalar model" do
    unless @supported do
      @describetag skip: "installed Nx cannot differentiate f64 through while"
    end

    test "forward loss and carry state match on consecutive minibatches", ctx do
      Enum.reduce(ctx.chunks, {nil, nil}, fn chunk, {scalar_carry, loop_carry} ->
        {expected, scalar_state} = Optimizer.scalar_chunk_loss(ctx.params, chunk, scalar_carry)

        {actual, stability, difficulty} = Loop.run(ctx.params, Loop.prepare(chunk, loop_carry))

        assert_in_delta num(actual), num(expected), abs(num(expected)) * 1.0e-12

        {expected_stability, expected_difficulty} = scalar_state
        assert_in_delta num(stability), num(expected_stability), 1.0e-11
        assert_in_delta num(difficulty), num(expected_difficulty), 1.0e-11

        {scalar_state, {stability, difficulty}}
      end)
    end

    test "gradients match the scalar model and are finite", ctx do
      chunk = hd(ctx.chunks)

      {_loss, expected} =
        Nx.Defn.value_and_grad(ctx.params, fn p ->
          {loss, _carry} = Optimizer.scalar_chunk_loss(p, chunk, nil)
          loss
        end)

      {_loss, actual} = Loop.loss_and_grad(ctx.params, Loop.prepare(chunk, nil))

      for value <- Nx.to_flat_list(actual) do
        assert is_float(value), "non-finite gradient: #{inspect(value)}"
      end

      for i <- 0..20 do
        expected_i = num(expected[i])
        assert_in_delta num(actual[i]), expected_i, 1.0e-9 + abs(expected_i) * 1.0e-10
      end
    end

    @tag :exla
    test "the same holds when compiled with EXLA", ctx do
      if Code.ensure_loaded?(EXLA) do
        chunk = hd(ctx.chunks)
        prepared = Loop.prepare(chunk, nil)

        {plain_loss, plain_gradient} = Loop.loss_and_grad(ctx.params, prepared)
        {loss, gradient} = Loop.loss_and_grad(ctx.params, prepared, EXLA)

        assert_in_delta num(loss), num(plain_loss), abs(num(plain_loss)) * 1.0e-10

        for i <- 0..20 do
          expected_i = num(plain_gradient[i])
          assert_in_delta num(gradient[i]), expected_i, 1.0e-8 + abs(expected_i) * 1.0e-8
        end
      end
    end
  end
end
