# Nx: `grad` through `while` returns zeros for f64 multiplicative bodies

Handoff notes for filing upstream at [elixir-nx/nx](https://github.com/elixir-nx/nx)
and attempting a fix. Everything below was reproduced locally; nothing is inferred.

Runnable reproduction: [`bench/nx_while_grad_f64_bug.exs`](nx_while_grad_f64_bug.exs)

---

## Summary

`grad` through a `while` loop silently returns zeros (or near-zero garbage) when
the loop's accumulator is **f64** and the body **multiplies**. The same loop in
f32 is correct. Nothing raises.

```elixir
defn pow(x, opts \\ []) do
  {acc, _i, _x} =
    while {acc = Nx.tensor(1.0, type: opts[:type]), i = 0, x = x}, Nx.less(i, 3) do
      {acc * x, i + 1, x}
    end

  acc
end

defn g(x, opts \\ []), do: grad(x, fn x -> pow(x, type: opts[:type]) end)

g(Nx.tensor(2.0, type: :f32), type: :f32)  #=> 12.0   correct (d/dx x³ at 2)
g(Nx.tensor(2.0, type: :f64), type: :f64)  #=>  0.0   wrong
```

## Evidence

Same loop computing `x^n`, analytic gradient `n·x^(n-1)`, `x = 1.02`:

```
  n |        f32 grad |        f64 grad |        analytic
  1 |      1.0000 ok  |      0.0000 BAD | 1.0000
  2 |      2.0400 ok  |     -0.0000 BAD | 2.0400
  3 |      3.1212 ok  |     -0.0000 BAD | 3.1212
  4 |      4.2448 ok  |      0.0557 BAD | 4.2448
  8 |      9.1895 ok  |      0.0001 BAD | 9.1895
 16 |     21.5339 ok  |      0.0384 BAD | 21.5339
```

f32 is correct at every trip count. f64 is wrong at every trip count, including
`n = 1`.

## Scope — what does and does not fail

| loop body | dtype | gradient | correct? |
|---|---|---|---|
| `acc + x` | f64 | 3.0 | ✅ |
| `acc * 2.0` (constant Jacobian) | f64 | 0.0 | ❌ |
| `acc * x` (state-dependent Jacobian) | f64 | 0.0 | ❌ |
| `acc * x` | f32 | 12.0 | ✅ |
| `x * x * x`, no loop, hand-unrolled | f64 | 12.0 | ✅ |

Two things this rules out:

- **Not f64 autodiff generally** — the hand-unrolled f64 product is correct, so
  only the `while` path is affected.
- **Not the reverse-order VJP issue** ([#1747](https://github.com/elixir-nx/nx/issues/1747))
  — that is about state-dependent Jacobians, but here a *constant* Jacobian
  (`acc * 2.0`) fails too, while an additive body succeeds.

The discriminator is **addition versus multiplication**. Addition's VJP passes
the adjoint through unchanged; multiplication's VJP must multiply the adjoint by
the other operand's **forward value**. So the defect is in recovering or typing
that forward value in f64.

That the result is exactly `0.0` — not merely inaccurate — suggests the adjoint
is being multiplied by a zeroed tensor rather than accumulated wrongly.

## Versions

Reproduced on both:

- `nx 0.13.1` (hex)
- `nx` main @ `37901d749105076e2882fdea89a6e18393eecc0f`

Elixir 1.20.4, Erlang/OTP 29, macOS arm64 (Apple M2), `Nx.BinaryBackend`.

Note that the [#1747](https://github.com/elixir-nx/nx/issues/1747) reverse-order
fix **is** present on main and does work — verified by reproducing the failure on
0.13.1 (Nx's own expected `448.0` comes out as `5461.0`; `square_via_while`
expects `1.0` and gives `1.25`) and confirming both pass on main. This f64 issue
is separate and survives that fix.

## Why it has not been caught

Every gradient test in the `describe "while"` block of
`nx/test/nx/defn/grad_test.exs` — including the whole #1747 regression suite —
uses f32. The f64 path has no coverage.

## Where to look

`nx/lib/nx/defn/grad.ex`, `defp update_grads(:while, [initial, arg, condition, body], ...)`
(~86 lines on main).

Working hypothesis: a type mismatch in the reverse loop's state carry. Candidates
in that function, in rough order of suspicion:

1. `zero = Expr.tensor(0)` and the `k_arg` / `j_arg` counters built from it —
   these seed the rematerialization loop, and an untyped-or-default constant
   flowing into an f64 composite could coerce state to zeros.
2. `select_composite(remat?, body, s0_arg)` and
   `select_composite(remat?, grad_args_tuple, grad_body_tuple)` — if the two
   branches disagree on type, the f64 branch may be lost.
3. `gs = Enum.zip_with(gs, flatten_initial, &Nx.broadcast/2)` — `Nx.broadcast/2`
   copies *shape* from the template but not *type*, so an f32 incoming adjoint
   would stay f32 against an f64 initial.

The failure at `n = 1` is informative: with a single iteration the
rematerialization phase is skipped entirely (`remat?` is false immediately), so
whatever is wrong is present even on the simplest path through the reverse loop,
not only in the O(n²) replay.

## Suggested fix shape

1. Add f64 variants of the existing `describe "while"` gradient tests. They
   should fail before the fix and pass after. This is probably the highest-value
   part of the change regardless of the root cause.
2. Ensure every constant and carry constructed inside `update_grads(:while, ...)`
   adopts the dtype of the corresponding `initial` element rather than a default.

## A second, separate bug

Calling `Nx.Defn.value_and_grad/2` from **outside** `defn` on a function
containing `while` returns `0.0` regardless of dtype, rather than raising:

```elixir
defmodule M do
  import Nx.Defn

  defn f(x) do
    {_i, acc} = while {i = 0, acc = x}, i < 3 do
      {i + 1, acc * 2.0}
    end
    acc
  end
end

Nx.Defn.value_and_grad(Nx.tensor(3.0), &M.f/1)  #=> {24.0, 0.0}, expected grad 8.0
M.grad_f(Nx.tensor(3.0))                        #=> 8.0 when grad is taken inside defn
```

Present on both 0.13.1 and main. Arguably this should raise, or be documented, if
taking gradients across the defn boundary is unsupported. Silent zeros are the
worst outcome.

## Context

Found while building an FSRS parameter optimizer in Elixir. The optimizer trains
21 weights on review history, and its verification depends on matching py-fsrs
(torch float64) to 1e-9 — so f32 is not an available workaround.

The `while` form is needed because the alternative, stepping the sequence with
`Enum.reduce`, unrolls the body into the graph at trace time: ~151 operations per
timestep, giving 9,373 operations at 16 timesteps and ~33,000 at the sequence
lengths the real data needs. XLA's compile time grows superlinearly in graph size
(measured: 0.45s → 1.6s → 10s → 67s for 545 → 1,801 → 4,325 → 9,373 nodes), which
makes EXLA unusable for this workload. `while` would make the graph a fixed size.

Forward-pass agreement is already confirmed: the `defn`/`while` model matches a
scalar reference implementation to 8.5e-14. Only the gradient is affected.
