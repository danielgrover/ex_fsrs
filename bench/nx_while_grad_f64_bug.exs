# Minimal reproduction: Nx's gradient through `while` is wrong for f64 tensors.
#
# The same loop, differing only in dtype, computes x^n. The analytic gradient is
# n * x^(n-1). f32 is correct at every trip count; f64 is wrong at every trip
# count, including n = 1. It does not raise — it returns near-zero garbage.
#
# Reproduced on Nx 0.13.1 (hex) and on main @ 37901d7. Nx's own while-gradient
# tests in nx/test/nx/defn/grad_test.exs are all f32, which is why this passes CI.
#
#     mix run bench/nx_while_grad_f64_bug.exs
defmodule WhileGradDtype do
  import Nx.Defn

  defn pow_f32(x, opts \\ []) do
    {acc, _i, _x} =
      while {acc = Nx.tensor(1.0, type: :f32), i = 0, x = x}, Nx.less(i, opts[:n]) do
        {acc * x, i + 1, x}
      end

    acc
  end

  defn pow_f64(x, opts \\ []) do
    {acc, _i, _x} =
      while {acc = Nx.tensor(1.0, type: :f64), i = 0, x = x}, Nx.less(i, opts[:n]) do
        {acc * x, i + 1, x}
      end

    acc
  end

  defn grad_f32(x, opts \\ []), do: grad(x, fn x -> pow_f32(x, n: opts[:n]) end)
  defn grad_f64(x, opts \\ []), do: grad(x, fn x -> pow_f64(x, n: opts[:n]) end)
end

IO.puts("  n |        f32 grad |        f64 grad |        analytic")

for n <- [1, 2, 3, 4, 8, 16] do
  expected = n * :math.pow(1.02, n - 1)
  f32 = Nx.to_number(WhileGradDtype.grad_f32(Nx.tensor(1.02, type: :f32), n: n))
  f64 = Nx.to_number(WhileGradDtype.grad_f64(Nx.tensor(1.02, type: :f64), n: n))

  verdict = fn value ->
    if abs(value - expected) / expected < 1.0e-4, do: "ok ", else: "BAD"
  end

  IO.puts(
    "#{String.pad_leading("#{n}", 3)} |" <>
      " #{String.pad_leading(:erlang.float_to_binary(f32, decimals: 4), 11)} #{verdict.(f32)} |" <>
      " #{String.pad_leading(:erlang.float_to_binary(f64, decimals: 4), 11)} #{verdict.(f64)} |" <>
      " #{:erlang.float_to_binary(expected, decimals: 4)}"
  )
end
