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
defmodule AdjointScaling do
  import Nx.Defn

  # The loop body is identical in both. Only what happens downstream differs.
  defn loop(x, opts \\ []) do
    {acc, _i, _x} =
      while {acc = Nx.tensor(0.0, type: opts[:type]), i = 0, x = x}, Nx.less(i, 3) do
        {acc + x, i + 1, x}
      end

    acc
  end

  defn plain(x, opts \\ []), do: grad(x, fn x -> loop(x, type: opts[:type]) end)

  defn scaled(x, opts \\ []) do
    grad(x, fn x -> loop(x, type: opts[:type]) * Nx.tensor(2.0, type: opts[:type]) end)
  end
end

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

IO.puts("An additive loop, scaled outside. Body identical; only the adjoint differs.\n")
IO.puts("dtype | grad(loop) | grad(loop * 2.0) | expected 3.0 / 6.0")

for type <- [:f32, :f64] do
  x = Nx.tensor(2.0, type: type)
  plain = Nx.to_number(AdjointScaling.plain(x, type: type))
  scaled = Nx.to_number(AdjointScaling.scaled(x, type: type))

  IO.puts(
    " #{type} | #{String.pad_leading(:erlang.float_to_binary(plain, decimals: 4), 10)} |" <>
      " #{String.pad_leading(:erlang.float_to_binary(scaled, decimals: 4), 16)} |" <>
      " #{if abs(scaled - 6.0) < 1.0e-6, do: "ok", else: "BAD"}"
  )
end

IO.puts("\nSame loop computing x^n, adjoint scaled by x inside the loop.\n")
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
