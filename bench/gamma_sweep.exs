# Needs an Nx carrying the `while` f64 gradient fix (bench/NX_WHILE_GRAD_F64.md):
# this script trains with `model: :loop` for speed, and stock Nx computes wrong
# gradients through `while`. Point EX_FSRS_NX_PATH at a patched checkout before
# `mix deps.get`; on stock Nx the optimizer refuses `:loop` rather than train on
# bad gradients, and the script stops with that error.
#
# Sweeps the L2 gamma on held-out cards. gamma = 0 is L2 off.
#
#     MIX_ENV=test mix run bench/gamma_sweep.exs
alias ExFsrs.AnkiDataset
alias ExFsrs.Optimizer
alias ExFsrs.Optimizer.Data

base = [model: :loop, compiler: EXLA, initialize: true]
gammas = [0.0, 0.25, 0.5, 1.0, 2.0, 4.0]

split = fn sequences ->
  {train, test} =
    sequences
    |> Enum.with_index()
    |> Enum.split_with(fn {_sequence, index} -> rem(index, 5) != 0 end)

  {Enum.map(train, &elem(&1, 0)), Enum.map(test, &elem(&1, 0))}
end

IO.puts("HELD-OUT loss by L2 gamma (all runs use initialize: true)\n")
IO.puts("  user | " <> Enum.map_join(gammas, " | ", &String.pad_leading("γ=#{&1}", 8)))

rows =
  AnkiDataset.summary()
  |> Enum.sort_by(& &1.scored)
  |> Enum.flat_map(fn %{user_id: id} ->
    {train, test} = split.(AnkiDataset.sequences(id))

    if Data.num_reviews(train) < 512 do
      []
    else
      losses =
        Enum.map(gammas, fn gamma ->
          parameters =
            Optimizer.compute_optimal_parameters(train, [regularization: gamma] ++ base)

          Optimizer.batch_loss(test, parameters)
        end)

      IO.puts(
        "#{String.pad_leading("#{id}", 6)} | " <>
          Enum.map_join(
            losses,
            " | ",
            &String.pad_leading(Float.to_string(Float.round(&1, 6)), 8)
          )
      )

      [losses]
    end
  end)

count = length(rows)
means = rows |> Enum.zip() |> Enum.map(fn t -> Enum.sum(Tuple.to_list(t)) / count end)

IO.puts("\n#{count} collections. Mean held-out loss:")

for {gamma, mean} <- Enum.zip(gammas, means) do
  wins =
    Enum.count(rows, fn r ->
      Enum.min(r) == Enum.at(r, Enum.find_index(gammas, &(&1 == gamma)))
    end)

  IO.puts(
    "  γ=#{String.pad_trailing("#{gamma}", 5)} #{Float.round(mean, 6)}   best on #{wins}/#{count}"
  )
end
