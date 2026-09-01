if Code.ensure_loaded?(Nx) do
  defmodule ExFsrs.Optimizer.Model.Loop do
    @moduledoc """
    The batched model expressed as a real loop inside `defn`.

    `ExFsrs.Optimizer.Model.Batched` steps timesteps with `Enum.reduce`, which
    runs at trace time and therefore *inlines* one copy of the body per
    timestep. At 16 timesteps that is already 9,373 operations in a single
    straight-line graph, and XLA's compile time grows superlinearly in graph
    size — enough to make it unusable past a handful of timesteps.

    Here the timestep loop is a `while` generator, so the graph is a fixed size
    regardless of sequence length and a compiler has something small to work on.
    The per-timestep arithmetic is the batched model's `step_columns`, called
    from inside the loop; only the loop itself lives here.

    Two rules make this work, both learned the hard way:

    * Everything the loop touches must be a loop argument. `defn` cannot close
      over outside variables inside `while`.
    * The gradient must be taken *inside* `defn`. Calling
      `Nx.Defn.value_and_grad/2` from ordinary Elixir on a function containing
      `while` silently returns zeros rather than raising.

    Columns are stored transposed as `{timesteps, cards}` so that indexing by
    the loop variable yields one `{cards}` slice per step.
    """

    import Nx.Defn

    alias ExFsrs.Optimizer.Model.Batched

    @doc """
    Whether the installed Nx computes f64 gradients through `while` correctly.

    Nx returns silently wrong gradients — often zeros — when an f64 adjoint has
    to be scaled inside a `while`, which is exactly what this model relies on.
    Nothing raises, so an unpatched Nx would quietly train on garbage gradients.
    The check differentiates a loop with a known answer.

    See `bench/NX_WHILE_GRAD_F64.md`.
    """
    def supported? do
      grad = Nx.to_number(support_probe(Nx.tensor(2.0, type: :f64)))

      abs(grad - 6.0) < 1.0e-9
    rescue
      _ -> false
    end

    defn support_probe(x) do
      grad(x, fn x ->
        {acc, _i, _x} =
          while {acc = Nx.tensor(0.0, type: :f64), i = 0, x = x}, Nx.less(i, 3) do
            {acc + x, i + 1, x}
          end

        acc * Nx.tensor(2.0, type: :f64)
      end)
    end

    @doc """
    Prepares a minibatch, reusing the batched model's padding and bucketing.

    Column tensors are transposed to `{timesteps, cards}` for loop indexing.
    """
    def prepare(chunk, carry) do
      prepared = Batched.prepare(chunk, carry)

      %{prepared | buckets: Enum.map(prepared.buckets, &transpose_bucket/1)}
    end

    defp transpose_bucket(bucket) do
      %{
        bucket
        | ratings: Nx.transpose(bucket.ratings),
          elapsed: Nx.transpose(bucket.elapsed),
          labels: Nx.transpose(bucket.labels),
          scored: Nx.transpose(bucket.scored),
          weights: Nx.transpose(bucket.weights),
          valid: Nx.transpose(bucket.valid)
      }
    end

    @doc """
    Loss and gradient for a minibatch, summed over its buckets.

    Buckets are independent, so the gradient of the summed loss is the sum of
    their gradients, and each can be compiled for its own shape.
    """
    def loss_and_grad(params, prepared, compiler \\ nil) do
      zero_gradient = Nx.broadcast(Nx.tensor(0.0, type: :f64), Nx.shape(params))

      Enum.reduce(prepared.buckets, {Nx.tensor(0.0, type: :f64), zero_gradient}, fn bucket,
                                                                                    {total,
                                                                                     gradient} ->
        {loss, bucket_gradient} =
          apply_bucket(&bucket_value_and_grad/10, params, bucket, compiler)

        {Nx.add(total, loss), Nx.add(gradient, bucket_gradient)}
      end)
    end

    @doc """
    Runs a minibatch forward, returning `{loss, stability, difficulty}` where the
    state tensors are the carry for the next minibatch.
    """
    def run(params, prepared, compiler \\ nil) do
      Enum.reduce(prepared.buckets, {Nx.tensor(0.0, type: :f64), nil, nil}, fn bucket,
                                                                               {total, stability,
                                                                                difficulty} ->
        {loss, final_stability, final_difficulty} =
          apply_bucket(&bucket_forward/10, params, bucket, compiler)

        case Enum.find_index(bucket.indices, &(&1 == prepared.last_index)) do
          nil ->
            {Nx.add(total, loss), stability, difficulty}

          row ->
            # Bring the carry back to the default backend. A compiler returns
            # device-backed tensors, and letting those seed the next minibatch's
            # `prepare/2` makes every tensor in it device-backed too, which turns
            # a 33ms compiled minibatch into a 2s one. It is two scalars.
            {Nx.add(total, loss), to_default(final_stability[row]),
             to_default(final_difficulty[row])}
        end
      end)
    end

    defp to_default(tensor), do: Nx.backend_copy(tensor, Nx.BinaryBackend)

    defp apply_bucket(fun, params, bucket, compiler) do
      args = [
        params,
        bucket.ratings,
        bucket.elapsed,
        bucket.labels,
        bucket.scored,
        bucket.weights,
        bucket.valid,
        bucket.stability0,
        bucket.difficulty0,
        bucket.has_state0
      ]

      if compiler do
        Nx.Defn.jit_apply(fun, args, compiler: compiler)
      else
        apply(fun, args)
      end
    end

    # --- everything below runs inside defn ---

    defnp bucket_value_and_grad(
            params,
            ratings,
            elapsed,
            labels,
            scored,
            weights,
            valid,
            s0,
            d0,
            h0
          ) do
      value_and_grad(params, fn p ->
        {loss, _stability, _difficulty} =
          sequence(p, ratings, elapsed, labels, scored, weights, valid, s0, d0, h0)

        loss
      end)
    end

    defnp bucket_forward(params, ratings, elapsed, labels, scored, weights, valid, s0, d0, h0) do
      sequence(params, ratings, elapsed, labels, scored, weights, valid, s0, d0, h0)
    end

    defnp sequence(params, ratings, elapsed, labels, scored, weights, valid, s0, d0, h0) do
      {_params, _ratings, _elapsed, _labels, _scored, _weights, _valid, total, stability,
       difficulty, _has_state} =
        while {params, ratings, elapsed, labels, scored, weights, valid,
               total = Nx.tensor(0.0, type: :f64), stability = s0, difficulty = d0,
               has_state = h0},
              i <- 0..(Nx.axis_size(ratings, 0) - 1) do
          {total, stability, difficulty, has_state} =
            step(
              params,
              {ratings[i], elapsed[i], labels[i], scored[i], weights[i], valid[i]},
              {total, stability, difficulty, has_state}
            )

          {params, ratings, elapsed, labels, scored, weights, valid, total, stability, difficulty,
           has_state}
        end

      {total, stability, difficulty}
    end

    # `Batched.step_columns/3` is ordinary Elixir built from `Nx` calls, which
    # is exactly what `defn` traces, so the loop body is shared rather than
    # rewritten in operator syntax.
    deftransformp(step(params, columns, state), do: Batched.step_columns(params, columns, state))
  end
end
