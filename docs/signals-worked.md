# Signals, worked

The equations of [`signals.md`](signals.md) with every symbol named and one example, a 2-input XOR
gate, carried through the read and the gradient. Read this once; the reference stays there.

## The read

$$r(u) = \sum_a P(a \mid u)\, T[a], \qquad P(a \mid u) = \prod_j u_j^{a_j} (1-u_j)^{1-a_j}.$$

In words, for one gate with $k$ inputs:

| symbol | what it is | ranges over |
|---|---|---|
| $T$ | the gate's table, one entry in $[0,1]$ per input pattern; bits when hard, probabilities when soft | $2^k$ entries |
| $a$ | the table's index, an integer from $0$ to $2^k - 1$; its binary digits are the $a_j$, input $j$ being bit $j$ of the index, least significant first (the convention `read` uses) | the sum: all $2^k$ entries |
| $u$ | what is actually at the inputs, $k$ numbers in $[0,1]$; $u_j$ is input $j$ | the product: the $k$ inputs |
| $P(a \mid u)$ | the probability that the inputs together spell the pattern of index $a$, reading each $u_j$ as the probability that input $j$ is a 1, independently of the others (on bits, 0 or 1) | the product: one factor per input $j$, the probability that input $j$ takes the bit $a_j$: $u_j$ if that bit is 1, $1-u_j$ if it is 0 (the exponents are that switch) |
| $T[a]$ | the table's entry at index $a$ | |
| $r(u)$ | the gate's output: every entry weighted by how likely its index is to be the one addressed | |

- On bits, exactly one index has $P = 1$ and the read is the plain lookup $T[a]$.
- On soft inputs, $P$ is a distribution over the $2^k$ indices, summing to 1, and the read is the table's
  expectation under it. One formula covers both passes.

Example, $k = 2$, the XOR table $T = (0, 1, 1, 0)$ over indices $a = (00, 01, 10, 11)$, inputs
$u = (0.9, 0.2)$:

```
P(00 | u) = 0.1 · 0.8 = 0.08
P(01 | u) = 0.1 · 0.2 = 0.02
P(10 | u) = 0.9 · 0.8 = 0.72
P(11 | u) = 0.9 · 0.2 = 0.18        (sum 1)
r(u) = 0.08·0 + 0.02·1 + 0.72·1 + 0.18·0 = 0.74
```

With bits $u = (1, 0)$ only index $10$ survives and $r = T[10] = 1$. In code, `signals.address`
builds $P$ one input at a time, input $i$ being address bit $i$, the convention the halving `read` uses.

## The gradient, three factors

$$\frac{\partial L}{\partial z[a]} \;=\; e \cdot P(a \mid u) \cdot \sigma'(z[a]).$$

In words, per case:

| symbol | what it is | where it comes from |
|---|---|---|
| $z[a]$ | the logit behind entry $a$; $T[a] = \sigma(z[a])$ | the tile's parameters |
| $L$ | half the squared error at the circuit's outputs, over cases and output bits | the task |
| $e = \partial L / \partial r$ | the **error at this gate's output**: how much the loss moves per unit move of $r$. At an output gate it is the residual, read minus demanded (over the mean's $N$); inside, it is the residual carried back through the gates downstream | elsewhere: the only factor the transport brings |
| $P(a \mid u)$ | how much entry $a$ is addressed by the current inputs: the share of the output this entry is responsible for | the gate's own inputs, as above |
| $\sigma'(z[a]) = T[a]\,(1 - T[a])$ | how much the entry moves when its logit moves: the sigmoid's slope, at most $1/4$, near $0$ once the entry is saturated at $0$ or $1$ | the gate's own logit |

Example, the same gate as an output gate with a soft table $T = (0.1, 0.9, 0.9, 0.1)$ (so
$\sigma' = 0.09$ at every entry), inputs $u = (0.9, 0.2)$, hence $P = (0.08, 0.02, 0.72, 0.18)$,
one case demanding $y = 1$:

```
r     = 0.08·0.1 + 0.02·0.9 + 0.72·0.9 + 0.18·0.1 = 0.692
e     = r − y = −0.308                          (one case, one output: L = ½(r − y)²)
∂L/∂z = e · P · σ′
  00:  −0.308 · 0.08 · 0.09 = −0.0022
  01:  −0.308 · 0.02 · 0.09 = −0.0006
  10:  −0.308 · 0.72 · 0.09 = −0.0200
  11:  −0.308 · 0.18 · 0.09 = −0.0050
```

Descent subtracts the gradient, so every entry rises toward this case's target, entry $10$ by far
the most: a case pulls the entries it addressed, in proportion to how much it addressed them.
Entry $11$ is pulled the wrong way for XOR; the case $(1, 1)$ pulls it back, and a table is the
compromise between the cases that address it. On bits, $u = (1, 0)$ and $H = (0, 1, 1, 0)$ give
$r = 1 = y$, $e = 0$ and no gradient at all: a right case teaches nothing on the bits. Had $H[10]$
been $0$, $e = -1$ and only entry $10$ moves, by $-\sigma'(z[10])$ with the surrogate and by $-1$
without.

## The transport, one hop

$$e_g = \sum_{h \text{ fed by } g} e_h \cdot \frac{\partial r_h}{\partial u_{h,j}}, \qquad
\frac{\partial r_h}{\partial u_{h,j}} = r_h(u_{h,j}{=}1) - r_h(u_{h,j}{=}0).$$

| symbol | what it is |
|---|---|
| $g$ | the XOR gate above, now a hidden gate: its output $r_g = 0.692$ is a line |
| $h$ | a gate downstream, whose input $0$ is that line and whose input $1$ is another line, $v = 0.3$ |
| $\partial r_h / \partial u_{h,0}$ | $h$'s sensitivity to $g$: the read with $g$'s line forced to $1$ minus the read with it forced to $0$, $v$ as it is |
| $e_h$ | the error at $h$'s output, here the residual since $h$ is an output gate |
| $e_g$ | the error at $g$'s output: what $h$ sends back, summed over every gate $g$ feeds |

Let $h$'s table be $T_h = (0.2, 0.9, 0.6, 0.3)$, bit $0$ of the index being $g$'s line, and one case
demanding $y = 0$ at $h$:

```
P_h(· | 0.692, 0.3) = (0.2156, 0.4844, 0.0924, 0.2076)
r_h                = 0.2156·0.2 + 0.4844·0.9 + 0.0924·0.6 + 0.2076·0.3 = 0.5968
e_h = r_h − y      = 0.5968

sensitivity of h to g's line, two reads with v = 0.3:
  r_h(u_0 = 1) = 0.7·0.9 + 0.3·0.3 = 0.72
  r_h(u_0 = 0) = 0.7·0.2 + 0.3·0.6 = 0.32
  ∂r_h/∂u_0    = 0.72 − 0.32 = +0.40        (check: 0.692·0.72 + 0.308·0.32 = 0.5968 = r_h)

message to g:   relay    e_h · 0.40 = +0.2387
                uniform  e_h · 1    = +0.5968
```

$g$ feeds only $h$, so $e_g$ is that one message. Its gradient is then the three factors again,
$e_g \cdot P_g \cdot \sigma'_g$ with $P_g = (0.08, 0.02, 0.72, 0.18)$ and $\sigma'_g = 0.09$:

```
relay:    ∂L/∂z_g = (+0.0017, +0.0004, +0.0155, +0.0039)
uniform:  ∂L/∂z_g = (+0.0043, +0.0011, +0.0387, +0.0097)
```

Both positive: descent lowers $g$'s addressed entry, which lowers $r_g$, which lowers $r_h$ toward
$0$, since $h$ increases with $g$'s line. The two transports agree because the sensitivity happened
to be positive; the blind split assumed it.

Now flip $h$'s table to $T_h = (0.9, 0.2, 0.3, 0.6)$, so that $h$ *decreases* with $g$'s line:

```
r_h = 0.4432,  e_h = 0.4432,  ∂r_h/∂u_0 = 0.32 − 0.72 = −0.40
message to g:   relay    −0.1773      → raise g's entries (lowers r_h through the negative path)
                uniform  +0.4432      → lower them: the wrong way, and consistently so
```

The relay follows the path's sign; the blind split pushes $g$ the wrong way and keeps pushing.
That is the first hop of the chance-level sign agreement in the hidden layers, and the push that
$h$ will absorb by drifting toward a table that increases with $g$'s line: alignment.

Had $g$ fed two gates, $e_g$ would be the sum of their two messages, one per wire, which is what
the wiring's fan-out costs in hardware.

**On bits.** $u_g = (1, 0)$ and $H_g = (0, 1, 1, 0)$ give $r_g = 1$; with $v = 0$, $h$ addresses
index $01$ of $H_h = (0, 1, 1, 0)$ and reads $1$; $y = 0$, so $e_h = 1$. Sensitivity by two reads:
$H_h[01] - H_h[00] = 1 - 0 = 1$; the message is $1$; $g$'s addressed entry $10$ gets
$1 \cdot 1 \cdot \sigma'$, or $1$ with the surrogate dropped: lower it, and this case is fixed. Had
$H_h$ been $(0, 0, 1, 1)$, a gate that copies $v$ and ignores $g$, the two reads would both give
$0$: sensitivity $0$, message $0$, a dead edge, and $g$ hears nothing about this case however wrong
$h$ is. The blind split would still send $1$.
