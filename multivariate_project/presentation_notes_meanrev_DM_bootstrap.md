# Presentation & Q&A notes: Newey–West mean-reversion test, Diebold–Mariano, and the wild bootstrap

These notes explain three pieces of the FX cointegration / VECM notebook in
enough depth to defend them in a Q&A:

1. The **Newey–West (HAC) mean-reversion test** — the predictive regression in §12(a).
2. The **Diebold–Mariano test** — the forecast-accuracy comparison in §12(b).
3. The **wild bootstrap** of the Johansen rank test in §5 — *why* we bootstrapped
   at all, and *why the specific random-number process (Rademacher ±1 weights) is
   the right one here.*

Throughout, $z_t$ is the (standardised) **error-correction term** — the deviation
of the basket from its estimated long-run equilibrium $\beta' y_t$. If the basket
is cointegrated, $z_t$ should be mean-reverting (stationary); if it is not, $z_t$
behaves like a random walk and nothing is forecastable.

---

## 1. The Newey–West mean-reversion test

### 1.1 What hypothesis it actually tests

Mean reversion is the *economic* claim that when the basket is stretched away from
its equilibrium, it tends to snap back. We turn that into a *testable* regression
by asking: does today's deviation $z_t$ predict tomorrow's **change** in the
spread $\Delta s_{t+1}$?

$$
\Delta s_{t+1} = a + b\,z_t + u_{t+1}.
$$

- $b < 0$ **is** error-correction: a positive deviation today ($z_t>0$) predicts a
  *fall* tomorrow ($\Delta s_{t+1}<0$) — i.e. the spread is pulled back toward
  equilibrium. This is exactly the discrete-time signature of mean reversion.
- $b = 0$ means the spread is a martingale — no reversion, no predictability.
- $b > 0$ would be explosive / trending, the opposite of reversion.

So the null and alternative are:

$$
H_0:\; b = 0 \quad\text{(no mean reversion)} \qquad
H_1:\; b < 0 \quad\text{(mean reversion / error-correction)}.
$$

We reject $H_0$ in favour of reversion only if $\hat b$ is **negative *and*
statistically significant.**

> **Where the slope comes from physically.** This is a reduced-form version of the
> VECM adjustment equation $\Delta y_t = \alpha\,(\beta' y_{t-1}) + \dots$ — the
> slope $b$ plays the role of the adjustment speed $\alpha$ projected onto the
> spread. That is why §9's **half-life**, $\text{HL}=\ln(0.5)/\ln(\phi)$ from the
> AR(1) $z_t=c+\phi z_{t-1}+e_t$, is the same phenomenon measured a different way:
> $\phi$ close to 1 ⇒ slow reversion ⇒ long half-life ⇒ small $|b|$.

### 1.2 Why Newey–West (HAC) standard errors are *mandatory* here

The point estimate $\hat b$ from OLS is fine. The **standard error** is the
problem. Ordinary OLS standard errors assume the regression errors $u_{t+1}$ are
**i.i.d.** — homoskedastic and serially uncorrelated. In this regression *both*
assumptions fail, for structural reasons:

1. **Serial correlation / overlap.** $z_t$ is a highly persistent (near-unit-root)
   series, and the left-hand side is a one-step change in a related persistent
   series. Consecutive $u_{t+1}$ are autocorrelated. Persistent regressors with
   autocorrelated errors are the textbook case where OLS SEs are biased.
2. **Volatility clustering (conditional heteroskedasticity).** This is daily FX:
   variance comes in bursts (calm periods, then crises). $\operatorname{Var}(u_{t+1})$
   is not constant.

Both effects make the *true* sampling variance of $\hat b$ **larger** than the OLS
formula reports. If we used plain OLS SEs we would understate the SE, **overstate**
the $t$-statistic, and over-reject $H_0$ — i.e. we would "find" mean reversion that
is really just an artefact of autocorrelation. As the notebook puts it: *"Without
HAC the $t$-stat would be overstated."*

**Newey–West (1987)** is a **HAC** estimator — *Heteroskedasticity- and
Autocorrelation-Consistent*. It estimates the long-run variance of the score by
summing the autocovariances of $x_t u_t$ up to a lag truncation $L$, with
**Bartlett (triangular) weights** $1-\tfrac{k}{L+1}$ that downweight distant lags:

$$
\widehat{\operatorname{LRV}} = \hat\gamma_0 + 2\sum_{k=1}^{L}\Big(1-\tfrac{k}{L+1}\Big)\hat\gamma_k .
$$

The Bartlett weights are not cosmetic: they guarantee the estimated variance is
**positive semi-definite** (a raw truncated sum can come out negative). In the
notebook we set `cov_type="HAC", cov_kwds={"maxlags": 21}` — a ~one-month window
(21 trading days), long enough to absorb the autocorrelation, short enough to keep
the estimator stable.

### 1.3 How to read the output / likely questions

- *"Why is the slope tiny?"* Because reversion in FX is slow — a small $|b|$ with a
  long half-life is exactly the unstable-cointegration thesis of the project, not a
  bug. Daily error-correction is a *weak* force.
- *"Why test it out-of-sample (on the OOS error-correction term)?"* §2–§9 are
  descriptive in-sample inference; §12 is the predictive test, so the spread is the
  **frozen, walk-forward** $z_t$ that uses only past information — no look-ahead.
- *"HAC vs HAR / which lag length?"* The Newey–West Bartlett kernel is the standard,
  most-defensible choice; 21 lags is a principled prior (one trading month). One
  could use an automatic Andrews/Newey–West bandwidth, but the conclusion is not
  knife-edge on $L$.
- *"Is a significant $b<0$ enough to trade?"* No — that is *statistical*
  predictability. §12(b) (Diebold–Mariano) and §13 (benchmarks) ask whether it is
  *economically* enough to beat a random walk net of the noise.

---

## 2. The Diebold–Mariano test

### 2.1 What it tests and why we need it

A significant mean-reversion slope says the spread is *predictable in-sample*. It
does **not** prove the model produces **better forecasts than the trivial
benchmark**. In FX the benchmark to beat is the **random walk** (Meese–Rogoff
1983: exchange-rate changes are famously close to unforecastable, and most "clever"
models fail to beat $\hat s_{t+1}=s_t$ out-of-sample).

Diebold–Mariano (1995) is a formal test of **equal predictive accuracy** between
two competing forecasts, using whatever loss function we choose. Here we compare:

- **Forecast 1 — random walk:** $\hat s_{t+1}=s_t$, with error $e^{\text{RW}}_t$.
- **Forecast 2 — the error-correction model:** one-step forecast of the spread
  from the fitted AR(1)/ECT dynamics, with error $e^{\text{VECM}}_t$.

### 2.2 The loss differential and the statistic

We use **squared-error loss** and form the per-period **loss differential**

$$
d_t = \big(e^{\text{RW}}_t\big)^2 - \big(e^{\text{VECM}}_t\big)^2 .
$$

If the model forecasts better, its squared errors are smaller, so $d_t$ is on
average **positive**. The hypotheses are about the *mean* of that differential:

$$
H_0:\; \mathbb{E}[d_t]=0 \;\;(\text{equal accuracy}),\qquad
H_1:\; \mathbb{E}[d_t]\ne 0 .
$$

The DM statistic is just a $t$-test on $\bar d$, but with a **HAC long-run
variance** in the denominator (because the loss differentials are themselves
autocorrelated — especially for multi-step forecasts):

$$
\mathrm{DM} = \frac{\bar d}{\sqrt{\widehat{\operatorname{Var}}(\bar d)}}
            = \frac{\bar d}{\sqrt{\widehat{\operatorname{LRV}}(d)/T}}
            \ \xrightarrow{\ d\ }\ \mathcal N(0,1).
$$

In the notebook's `diebold_mariano`:
- `d = e1**2 - e2**2` with `e1`=RW, `e2`=model, so **$\mathrm{DM}>0$ favours the
  model**.
- The long-run variance reuses the **same Newey–West / Bartlett machinery** as in
  §1.2, with the standard automatic truncation
  $L=\lfloor 4(T/100)^{2/9}\rfloor$ (the Schwert rule).
- We report the **two-sided** p-value `2 * norm.sf(|DM|)`; we then read the *sign*
  to decide direction.

### 2.3 Interpreting it / likely questions

- *"What does DM>0 but insignificant mean?"* The model's errors are slightly
  smaller on average, but not by enough to rule out luck — **we cannot reject equal
  accuracy, so Meese–Rogoff stands.** For this basket on the full sample that is the
  honest, expected outcome and is reported as such.
- *"Why DM rather than just comparing RMSEs?"* Because a raw RMSE comparison has no
  notion of sampling uncertainty. DM gives the comparison a **distribution**, so we
  can say whether a smaller RMSE is statistically real or noise.
- *"Why squared-error loss?"* It is the natural loss for a mean/variance forecasting
  problem and matches how we'd actually penalise the spread forecast. DM is
  agnostic — you could plug in absolute or directional loss — but squared error is
  the convention and keeps the comparison clean.
- *"Is DM valid even when both models are wrong?"* Yes — that is its strength. DM
  compares forecasts, not models; neither needs to be the true DGP. (One caveat:
  the basic DM is for **non-nested** forecasts; RW vs ECT are effectively
  non-nested here. For deeply nested models one would prefer Clark–West.)
- *"Connection to §12(a)?"* §12(a) asks *is there in-sample predictability?* (the
  $b<0$ test). §12(b) asks *does that predictability survive as out-of-sample
  forecast value against the toughest naïve benchmark?* You need both; the first
  without the second is the classic way to fool yourself in FX.

---

## 3. The wild bootstrap — what we did and why the RNG is appropriate

This is the part most likely to draw a hard Q&A question, so it is worth being able
to explain from first principles.

### 3.1 Why bootstrap at all? (The motivation)

The Johansen **trace test** for cointegration rank compares an observed trace
statistic to **asymptotic critical values**. Those critical values are derived
under two assumptions that are *badly violated* by daily FX:

1. **Homoskedastic, roughly Gaussian errors.** The asymptotic null distribution of
   the trace statistic (a functional of Brownian motion) assumes constant-variance
   innovations. FX has heavy **volatility clustering (ARCH/GARCH)** and fat tails.
   Conditional heteroskedasticity **inflates the trace statistic**, so the
   asymptotic test **over-rejects** $H_0:r=0$ — it "finds" cointegration that isn't
   there. (§3's normality and whiteness diagnostics already flagged this: residuals
   are non-Gaussian and carry ARCH that no lag length removes.)
2. **Large sample / finite-sample size distortion.** Even with nice errors, the
   trace test is known to over-reject in finite samples; that is the *separate*
   Reinsel–Ahn $T\to T-nk$ correction. The bootstrap addresses the *distributional*
   problem, the correction addresses the *finite-sample* one — they are
   complementary.

So the bootstrap exists to get a **valid critical value / p-value for $H_0:r=0$
that is robust to the actual error structure of the data**, rather than trusting an
asymptotic table that the data demonstrably violate. This follows
**Cavaliere, Rahbek & Taylor (CRT, 2010/2012)**, who showed the wild bootstrap
restores correct size for the cointegration rank test under heteroskedasticity.

### 3.2 What we actually did (step by step)

The function `wild_bootstrap_rank0` implements a **restricted residual bootstrap**:
we simulate data **under the null hypothesis $r=0$** and see how extreme our
real trace statistic is in that null world.

1. **Impose the null DGP.** Under $H_0:r=0$ there is *no* cointegration, so the
   correct data-generating process is a **VAR in first differences** (no
   error-correction term). We fit it: regress $\Delta y_t$ on a constant and
   $k$ lagged differences, giving coefficients $\hat B$ and residuals
   $\hat\varepsilon_t = \Delta y_t - X_t\hat B$.
2. **Generate bootstrap residuals by *wild* weighting** (the crucial step):
   $$
   \varepsilon^*_t = w_t\,\hat\varepsilon_t,\qquad w_t \in \{-1,+1\}\ \text{i.i.d., } \Pr(\pm1)=\tfrac12 .
   $$
   Each *whole residual vector* at time $t$ is multiplied by a single random sign
   $w_t$ (`w = g.choice([-1,1], size=Tn)` in the code).
3. **Rebuild a bootstrap sample under the null.** Recursively propagate the fitted
   VAR-in-differences with the wild residuals to get $\Delta y^*_t$, then
   cumulate back to levels $y^*_t$ (starting from the real initial values). This
   $y^*$ is, by construction, a series with **no cointegration** but with the *same
   short-run dynamics and the same volatility pattern* as the real data.
4. **Recompute the statistic.** Run the same from-scratch Johansen routine on
   $y^*$ and store $\text{trace}^*(0)$. Repeat $B=399$ times to build the null
   distribution.
5. **Bootstrap p-value.**
   $$
   p = \frac{\#\{\,\text{trace}^*_b(0)\ge \text{trace}^{\text{obs}}(0)\,\}+1}{B+1}.
   $$
   A large $p$ (the notebook's ≈0.345) means our observed trace statistic is
   *unremarkable* in a world with no cointegration ⇒ **cannot reject $r=0$.**

> $B=399$ (rather than 400) makes $(B+1)=400$ divide the p-value cleanly and is a
> standard Monte-Carlo-test convention; the `+1` in numerator and denominator is
> the finite-sample correction that keeps the test exact under the null.

### 3.3 Why the Rademacher (±1) random process is the *right* RNG here

This is the key "why" question. The choice of $w_t\in\{-1,+1\}$ — the **Rademacher
distribution** — is not arbitrary; it is what makes the bootstrap *heteroskedasticity-robust*.

**(a) It preserves the conditional heteroskedasticity / volatility clustering.**
Because we multiply by a *sign only*, $|\varepsilon^*_t| = |\hat\varepsilon_t|$:
the **magnitude** of each residual — and therefore the local variance, the ARCH
bursts, the fat tails — is **kept exactly where it was in calendar time.** A big
crisis-day shock stays a big shock; a calm-day shock stays small. Contrast this
with the **i.i.d. pairs/residual bootstrap**, which *reshuffles* residuals across
time and so destroys the volatility-clustering structure — precisely the structure
that distorts the trace test. The wild bootstrap leaves the variance pattern intact
and only randomises the **sign**, which is exactly the robustness CRT prove you
need.

**(b) The weights have the right first two moments to be valid.** For the bootstrap
to reproduce the correct null distribution, the weights must satisfy
$$
\mathbb{E}[w_t]=0,\qquad \mathbb{E}[w_t^2]=1 .
$$
- $\mathbb{E}[w_t]=0$ ⇒ the bootstrap residuals are **mean-zero**, so we are
  genuinely simulating under the null (no spurious drift/error-correction injected).
- $\mathbb{E}[w_t^2]=1$ ⇒ the bootstrap innovations have the **same (conditional)
  variance** as the real residuals, so the resampled data match the data's scale.

Rademacher $\pm1$ satisfies both *exactly*: mean 0, variance 1. (Gaussian
$w_t\sim N(0,1)$ also satisfies them and is a legitimate alternative — the
"Mammen" two-point distribution is another.)

**(c) Why ±1 specifically, over a Gaussian or Mammen weight.** For this test the
Rademacher weight is the standard, recommended choice because:
- It is the **most stable / lowest-variance** two-point scheme: every weight has the
  same magnitude, so no single observation is randomly amplified, which matters with
  the heavy-tailed FX residuals (a Gaussian $w_t$ can occasionally draw a large
  multiplier and inflate an already-fat-tailed shock).
- It keeps $|\varepsilon^*_t|$ literally equal to $|\hat\varepsilon_t|$, giving the
  cleanest possible preservation of the variance path — point (a) in its strongest
  form.
- Mammen's asymmetric weights are designed to also match the **third** moment
  (skewness), which matters for some regression bootstraps; for the *symmetric*
  trace-statistic problem under CRT, third-moment matching buys little and
  Rademacher is the convention.

**(d) Why it is appropriate *for this problem* and not over-engineering.** The thing
that breaks the asymptotic Johansen test is **time-varying variance**, not the mean
dynamics (which we've already fitted). The wild bootstrap is the minimal device that
(i) imposes the null exactly, (ii) keeps the short-run mean dynamics via $\hat B$,
and (iii) keeps the variance pattern via sign-only weighting. That is a precise
match between the *failure mode of the test* and the *robustness of the method.*

### 3.4 Likely Q&A on the bootstrap

- *"Why not just use more data / the asymptotic test?"* More data doesn't fix the
  *distributional* assumption — heteroskedasticity biases the asymptotic test at any
  sample size. The bootstrap rebuilds the null distribution from the data's own
  variance structure.
- *"Doesn't multiplying by ±1 just flip signs — how is that random enough?"* The
  randomness is in the **independent sign of every period**, giving $2^{T}$ possible
  resamples; the statistic is a nonlinear functional of the whole path, so different
  sign patterns produce genuinely different trace values — enough to trace out the
  null distribution.
- *"Why resample residuals of the differenced VAR rather than the levels?"* Because
  the **null is $r=0$**: the correct restricted model under the null is a VAR in
  differences with no error-correction term. Bootstrapping from that model is what
  makes the p-value a test *of that null.*
- *"Is the bootstrap result consistent with the others?"* Yes — trace, max-eig,
  the Reinsel–Ahn correction, Engle–Granger and Phillips–Ouliaris all fail to reject
  $r=0$, and the bootstrap (p≈0.345) agrees. The robustness check confirms the
  asymptotic verdict rather than overturning it, which is the reassuring outcome:
  the "no stable cointegration on the full sample" finding is **not** an artefact of
  heteroskedasticity.

---

## 4. One-paragraph summary for the slide

> Mean reversion is tested as a predictive regression $\Delta s_{t+1}=a+b z_t+u$;
> a significant **negative $b$** is error-correction, and we use **Newey–West (HAC)**
> standard errors because the persistent regressor and FX volatility clustering make
> ordinary OLS standard errors too small and the $t$-stat overstated. Statistical
> predictability is not forecasting value, so the **Diebold–Mariano** test compares
> the model's one-step squared-error loss against the **random-walk** benchmark
> (Meese–Rogoff); $\mathrm{DM}>0$ and significant means the model genuinely beats the
> random walk. Finally, the **wild bootstrap** gives a valid Johansen rank-test
> p-value when the asymptotic critical values can't be trusted: FX volatility
> clustering inflates the trace statistic, so we resample residuals of the
> null-imposed (differenced) VAR with **Rademacher ±1 weights**, which keep each
> residual's magnitude — and hence the variance/ARCH pattern — fixed in time while
> randomising only the sign (mean 0, variance 1, the conditions for validity). The
> bootstrap agrees with the asymptotic tests that $r=0$, confirming the finding is
> real and not a heteroskedasticity artefact.
