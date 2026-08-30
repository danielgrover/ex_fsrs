defmodule WhileSpike do
  import Nx.Defn

  defn accumulate(params, columns) do
    zero = Nx.tensor(0.0, type: :f64)
    ones = Nx.broadcast(Nx.tensor(1.0, type: :f64), {4})

    {_t, total, _s, _p, _c} =
      while {t = 0, total = zero, s = ones, p = params, c = columns},
            t < Nx.axis_size(c, 1) do
        col = Nx.squeeze(Nx.slice_along_axis(c, t, 1, axis: 1), axes: [1])
        s = Nx.multiply(s, Nx.add(1, Nx.multiply(p[0], col)))
        {t + 1, Nx.add(total, Nx.sum(s)), s, p, c}
      end

    total
  end
end

params = Nx.tensor([0.01, 0.02], type: :f64)
columns = Nx.broadcast(Nx.tensor(1.0, type: :f64), {4, 8})

IO.puts("forward: #{Nx.to_number(WhileSpike.accumulate(params, columns))}")

try do
  {v, g} = Nx.Defn.value_and_grad(params, fn p -> WhileSpike.accumulate(p, columns) end)
  IO.puts("grad through while WORKS: value=#{Nx.to_number(v)} grad=#{inspect(Nx.to_flat_list(g))}")
rescue
  e -> IO.puts("grad through while FAILS: #{Exception.message(e) |> String.slice(0, 200)}")
end

eps = 1.0e-7
fd =
  for i <- 0..1 do
    bump = Nx.tensor(Enum.map(0..1, &if(&1 == i, do: eps, else: 0.0)), type: :f64)
    f = Nx.to_number(WhileSpike.accumulate(Nx.add(params, bump), columns))
    b = Nx.to_number(WhileSpike.accumulate(Nx.subtract(params, bump), columns))
    (f - b) / (2 * eps)
  end

IO.puts("finite differences: #{inspect(fd)}")
