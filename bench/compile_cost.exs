alias ExFsrs.Optimizer.Model.Batched

params = Nx.tensor(ExFsrs.Scheduler.new().parameters, type: :f64)
sequences = ExFsrs.Optimizer.Data.build_sequences(ExFsrs.Fixtures.review_log_tuples())

# Build synthetic single-bucket chunks of a given sequence length.
bucket_for = fn len ->
  segs =
    sequences
    |> Enum.filter(fn {_id, r} -> length(r) >= len end)
    |> Enum.take(4)
    |> Enum.map(fn {_id, r} -> %{reviews: Enum.take(r, len), continues?: false} end)

  Batched.prepare(segs, nil)
end

for len <- [2, 4, 8, 16] do
  batch = bucket_for.(len)
  shape = batch.buckets |> hd() |> Map.get(:ratings) |> Nx.shape()

  {t_cold, _} = :timer.tc(fn -> Batched.value_and_grad(params, batch, EXLA) end)
  {t_warm, _} = :timer.tc(fn -> Batched.value_and_grad(params, batch, EXLA) end)
  {t_bin, _} = :timer.tc(fn -> Batched.value_and_grad(params, batch, nil) end)

  IO.puts("shape=#{inspect(shape)}  EXLA cold=#{Float.round(t_cold/1_000, 0)}ms  warm=#{Float.round(t_warm/1_000, 1)}ms  BinaryBackend=#{Float.round(t_bin/1_000, 1)}ms")
end
