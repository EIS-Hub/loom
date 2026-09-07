# Tasks

A task is a truth table: every input pattern, and the bits demanded of each output line.

- `inputs(n)` enumerates all `2**n` patterns as bits, least significant first, shape `[2**n, n]`.
- `k_junta(key, n_in, n_out, k)` draws a task in which every output bit depends on `k` random
  inputs through a random balanced table (exactly half ones, so no output is a constant). Solving
  it means discovering which inputs matter for each output and what function of them is asked.
- `add(n_in)` adds the two halves of the input as unsigned integers, with `n_in//2 + 1` output
  bits so the carry is demanded too.

For recipes, a task is a function of a key returning `(x, y)`: `junta(n_in, n_out, k)` draws a
fresh junta per key, `addition(n_in)` is the same table whatever the key. Every task is exact-width;
when several tasks share one fabric (step 5, step 8) a task will say which output lines it uses,
as a mask rather than zero padding, which would demand a constant 0 of the spare lines.
