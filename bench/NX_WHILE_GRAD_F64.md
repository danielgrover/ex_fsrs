# Nx: `grad` through `while` is wrong for f64 adjoints

Handoff notes for filing upstream at [elixir-nx/nx](https://github.com/elixir-nx/nx)
and attempting a fix. Everything below was reproduced locally; nothing is inferred.

Runnable reproduction: [`bench/nx_while_grad_f64_bug.exs`](nx_while_grad_f64_bug.exs)

---

## Summary

`grad` through a `while` loop silently returns wrong values — usually zero — when
the gradient flowing into the loop is **f64** and has to be scaled by anything
other than 1. The identical loop in f32 is correct. Nothing raises.

The clearest demonstration keeps the loop body byte-identical and changes only
what happens *downstream* of it:

```elixir
defn loop(x) do
  {acc, _i, _x} =
    while {acc = Nx.tensor(0.0, type: :f64), i = 0, x = x}, Nx.less(i, 3) do
      {acc + x, i + 1, x}
    end

  acc
end

defn plain(x),  do: grad(x, &loop/1)                                        #=> 3.0  correct
defn scaled(x), do: grad(x, fn x -> loop(x) * Nx.tensor(2.0, type: :f64) end) #=> 0.0  expected 6.0
```

The loop is purely additive and correct on its own. Multiplying its *result* by
two — entirely outside the loop — collapses the gradient to zero.

## Evidence

### The incoming gradient is what matters

Loop body identical in every row; only the downstream scaling differs:

| carry | downstream | incoming adjoint | gradient | correct? |
|---|---|---|---|---|
| f64 | none | 1 | 3.0 | ✅ |
| f64 | `* 1.0` | 1 | 3.0 | ✅ |
| f64 | `* 2.0` | 2 | **0.0** | ❌ (want 6.0) |
| f32 | none | 1 | 3.0 | ✅ |
| f32 | `* 2.0` | 2 | 6.0 | ✅ |

Scaling by 1.0 is fine; scaling by 2.0 is not. An adjoint of 1 needs no
multiplication, which is what masks the defect.

### f64 anywhere is enough

| carry type | scaling constant | correct? |
|---|---|---|
| f64 | bare literal `2.0` | ❌ |
| f64 | f64 | ❌ |
| f64 | f32 | ❌ |
| f32 | f64 | ❌ |
| f32 | f32 / bare literal | ✅ |

An f64 carry *or* an f64 adjoint is sufficient to break it. Only the fully-f32
path is correct.

### Trip counts

Same loop computing `x^n`, analytic gradient `n·x^(n-1)`, `x = 1.02`. Here the
adjoint is scaled by `x` inside the loop, so f64 fails at every trip count:

```
  n |        f32 grad |        f64 grad |        analytic
  1 |      1.0000 ok  |      0.0000 BAD | 1.0000
  2 |      2.0400 ok  |     -0.0000 BAD | 2.0400
  4 |      4.2448 ok  |      0.0557 BAD | 4.2448
  8 |      9.1895 ok  |      0.0001 BAD | 9.1895
 16 |     21.5339 ok  |      0.0384 BAD | 21.5339
```

Note the result is not always exactly zero (`0.0557` at n = 4), so this is
corrupted arithmetic rather than a simple drop to zero.

## Scope — what does and does not fail

| construct | dtype | correct? |
|---|---|---|
| `acc + x`, `acc - x`, `acc + (x + x)` | f64 | ✅ |
| `acc + 0.0`, `acc * 1.0` (identities) | f64 | ✅ |
| `acc + x * 2.0` | f64 | ❌ |
| `acc * x` | f64 | ❌ |
| `(additive loop) * 2.0` | f64 | ❌ |
| all of the above | f32 | ✅ |
| `x * x * x`, no loop at all | f64 | ✅ |

Two things this rules out:

- **Not f64 autodiff generally.** The hand-unrolled f64 product is correct, so
  only the `while` path is affected.
- **Not the loop body, and not multiplication per se.** `acc * 1.0` is fine, and
  a purely additive loop still fails once its result is scaled outside.
- **Not the reverse-order VJP issue** ([#1747](https://github.com/elixir-nx/nx/issues/1747)).
  That concerns state-dependent Jacobians; here a constant-Jacobian body fails
  and an additive one succeeds.

The unifying rule: **an f64 adjoint that must be multiplied by a non-unit value
inside the while-gradient comes out corrupted.**

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

The failure is in how the incoming adjoints (`gs`) are typed and carried into the
reverse loop, not in the body's own differentiation. Candidates in that function:

1. `gs = Enum.zip_with(gs, flatten_initial, &Nx.broadcast/2)` — `Nx.broadcast/2`
   takes *shape* from the template but leaves the type alone, so an adjoint whose
   type disagrees with its corresponding `initial` element stays mismatched.
2. `zero = Expr.tensor(0)` and the `index_arg` / `k_arg` / `j_arg` parameters
   built from it. These are created without an explicit type and are threaded
   through the same composite as the f64 state.
3. `select_composite(remat?, grad_args_tuple, grad_body_tuple)` — if the two
   branches disagree on type, the f64 branch may be coerced or dropped.

That an adjoint of exactly 1 works while 2 does not points at the multiplication
of the adjoint specifically, rather than at its transport through the loop.

Failure at `n = 1` is also informative: with a single iteration the
rematerialization phase is skipped entirely (`remat?` is false immediately), so
the defect is on the simplest path through the reverse loop, not in the O(n²)
replay added by the #1747 fix.

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
