# When Does Currency Diversification Break Down?
## The Story, Interleaved With the Tests That Tell It

### A Cointegration Lens on a Multinational's Cash-Reserve Currencies

*This document fuses two companions to `fx_cointegration_vecm.ipynb`. It keeps the
narrative spine of `STORY.md` — the diversification question, who cares, and why "no
stable relationship most of the time" is a complete result — but at every point where
the story leans on a finding, it stops and explains the **actual test**: what it asks,
the hypothesis, the statistic, the decision rule, and the numbers that came back. Read
it as the story with the machinery exposed. For the bare narrative see `STORY.md`; for
the test catalogue in isolation see `PRESENTER_GUIDE.md`.*

---

## The setting

Imagine a multinational corporation that earns revenue and pays costs across a basket
of commodity / risk-sensitive currencies — **AUD, NZD, CAD, NOK, SEK, ZAR, MXN** — and
holds cash reserves spread across them. Part of the rationale for spreading reserves
across several currencies is **diversification**: if the currencies move independently,
holding seven of them is genuinely less risky than holding one.

That rationale has a hidden assumption — that the currencies *stay* independent. They
do not always. Under certain macro regimes these currencies stop behaving like seven
separate bets and collapse into **a single shared risk factor**. When that happens, the
treasury's apparent diversification is an illusion: it is holding one bet wearing seven
labels, and a shock that hits the common factor hits the whole reserve at once.

**The data behind the story.** Everything below runs on a Bloomberg terminal pull
(`src/fx_data_pull.py`) of daily `PX_LAST`, ~2,600 trading days, 2016–2026. Every series
is quoted as **USD per 1 unit of the currency** (USD-quoted pairs inverted) and then
**logged**, so we model $y_t = \log(\text{price})$. The common numéraire plus logs is
what makes a cross-currency combination $\beta'y_t$ interpretable, and the basket was
chosen **ex-ante on economic theme** (the commodity / risk bloc), not by screening for
what happens to cointegrate — so we are not data-snooping the basket into existence.

---

## The question

**When does the diversification across these reserve currencies break down — and when
does it hold?**

This is not a question about *which* currency to hold (a directional, yield-driven
decision this analysis deliberately does not make — see the scope note below). It is a
**risk-monitoring** question: identifying the time periods in which the basket behaves
as one factor rather than many.

Cointegration is exactly the right instrument for it. A set of $I(1)$ currencies is
**cointegrated** when some linear combination of them is stationary — i.e. when they are
tied together by a common long-run equilibrium and cannot drift apart:

$$
z_t = \beta' y_t \sim I(0).
$$

*Cointegration is therefore the formal signature of diversification failure:* the
periods in which such a stationary combination exists are precisely the periods in which
the currencies have become a single risk factor. So the project's thesis is:

> **Use cointegration to identify the regimes in which a multinational's currency
> diversification breaks down — and report honestly that, most of the time, it holds.**

---

## A note on scope (read this before the "why not just pick a currency?" question)

This project is **descriptive risk monitoring, not reserve allocation.** It does *not*
recommend which currency to hold, for three deliberate reasons:

- A reserve-allocation decision is driven by **liability matching** (hold what you owe),
  **yield / carry**, and **liquidity** — none of which cointegration measures.
- A cointegrating combination is **dollar-neutral and net-zero** (long some currencies,
  short others); it is not something a treasury can *hold* as a positive cash balance.
- Carry is **explicitly excluded** throughout (the benchmark P&L is price-only), which is
  appropriate for a co-movement study but would be disqualifying for an allocation claim.

What cointegration *does* answer — and answers well — is the diversification-risk
question above. Keeping that boundary sharp is part of the rigour.

---

## The arc

The analysis is a funnel — each test earns the right to run the next:

> are the series $I(1)$? → how many lags, residuals clean? → which deterministic case?
> → cointegration rank → VECM $\beta,\alpha$ → error-correction term → **rolling-window
> re-estimation**.

The early stages establish that the machinery is *valid* to run at all (the series are
$I(1)$, the lag order is justified, the deterministic case fits the visible drift). The
payoff stage is the walk-forward, where we re-ask the diversification question inside
every rolling window and watch the answer change with the macro regime.

One framing point keeps everything honest. The notebook does two kinds of work to two
different standards:

| Regime | Sections | Claim type | Standard |
|---|---|---|---|
| In-sample structural inference | §2–§9 | *descriptive* ("were these cointegrated over 2016–26?") | full sample is legitimate — no forecast is made |
| Out-of-sample evaluation | §11–§13 | *predictive* ("could you have traded it?") | strictly causal — any quantity used at $t$ uses only data $\le t$ |

That distinction pre-empts the "isn't that look-ahead bias?" objection: the whole-sample
work makes no forecast, and the forecasting work never touches future data.

---

## Step 1 — Earning the right to ask the question: are the currencies $I(1)$?

Before "are they cointegrated?" can even be posed, each series must be **integrated of
the same order**, conventionally $I(1)$ — non-stationary in levels but stationary in
first differences. A series that were already stationary ($I(0)$) cannot belong to a
cointegrating relation; one that were $I(2)$ would break the standard Johansen setup. So
we pin the integration order first, with a battery of tests using *opposite nulls* so
they can confirm one another.

**The tests, and what each one asks:**

- **Augmented Dickey–Fuller (ADF)** — the workhorse. Regression
  $\Delta y_t = \mu + \delta t + \rho y_{t-1} + \sum_i \gamma_i \Delta y_{t-i} + \varepsilon_t$;
  the lagged differences "augment" it to whiten the errors.
  - $H_0$: $\rho = 0$, a **unit root** is present ($y_t$ non-stationary, $I(1)$).
  - $H_1$: $\rho < 0$, $y_t$ is (trend-)stationary, $I(0)$.
  - Statistic $\mathrm{ADF}=\hat\rho/\mathrm{se}(\hat\rho)$ — a $t$-ratio, but under $H_0$
    it follows the non-standard **Dickey–Fuller distribution**, not Student's $t$.
  - Decision: reject $H_0$ (conclude stationary) if ADF is **more negative** than the
    critical value, i.e. $p<0.05$.
- **KPSS** — the complementary test, with the **null reversed**: $H_0$ is *stationary*,
  $H_1$ is *unit root*. Reject (conclude non-stationary) if the statistic exceeds its
  critical value.
- **DF-GLS** (Elliott–Rothenberg–Stock) — same hypotheses as ADF but GLS-detrends the
  data first, giving substantially more power near the unit-root boundary (ADF alone is
  notoriously weak).
- **Phillips–Perron (PP)** — same unit-root null as ADF, but applies a non-parametric HAC
  correction instead of adding lags; apt for FX, where volatility clustering violates the
  i.i.d.-error assumption.
- **Zivot–Andrews** — break-robust. $H_0$: unit root with no break; $H_1$: stationary
  around **one endogenously chosen break**, the break date picked to be maximally
  favourable to rejecting. This matters because an unmodelled structural break makes a
  *stationary* series look like a unit root (Perron 1989), and we *expect* breaks (COVID,
  2022 hiking cycle) — so a plain-ADF $I(1)$ verdict would otherwise be self-contradictory.

> **Why both ADF and KPSS:** ADF has low power and often fails to reject even for
> stationary series. Pairing it with KPSS's opposite null gives a **confirmatory** read.
> On the **levels** we want ADF to *fail to reject* **and** KPSS to *reject* → both point
> to non-stationarity → confident $I(1)$. On the **differences** the pattern flips.

**A critical-values note to file away** (it returns, reversed, in Step 5): here the
series are **directly observed**, so the standard Dickey–Fuller critical values are
correct. That is deliberately *not* the case when we later test an *estimated* residual.

**What came back:**

- **Levels:** the unit-root nulls (ADF / DF-GLS / PP / ZA) are generally **not rejected**,
  and KPSS **rejects** stationarity ($p\approx 0$) for all seven currencies. → **$I(1)$.**
- **Differences:** ADF / DF-GLS / PP all give $p\approx 0.000$ → stationary → this
  **confirms $I(1)$ and rules out $I(2)$.**
- Two minor cross-test disagreements (MXN DF-GLS $p=0.014$; NZD Zivot–Andrews $p=0.019$)
  are exactly what you expect when running five tests on seven series; the
  **preponderance** says $I(1)$, and the notebook flags the exceptions honestly rather
  than burying them.

The machinery is now licensed to run: seven $I(1)$ series, all the same order.

---

## Step 2 — How much short-run dynamics, and is the residual clean enough?

Johansen's rank test rests on an underlying VAR($p$) in levels,
$y_t = \nu + \sum_{i=1}^p A_i y_{t-i} + \varepsilon_t$. We must choose the lag length
$p$, and then check that the residuals are well enough behaved for the rank test's
asymptotics to mean anything.

**Lag selection.** We pick $p$ to minimise an information criterion (each trades fit
$\ln|\hat\Sigma|$ against parameter count $k=pn^2$): AIC $=\ln|\hat\Sigma| + 2k/T$,
BIC with penalty $k\ln T/T$, HQIC with $2k\ln\ln T/T$. The VECM lag *in differences* is
$k_{\text{ar diff}} = p-1$.

- **Choice: AIC, floored at $p=2$** (so $k_{\text{ar diff}}=1$). *Why AIC and not BIC?*
  For cointegration the consistent criteria (BIC / HQIC) tend to **under-select**
  short-run dynamics, leaving residual autocorrelation that **biases the trace test**.
  Here AIC picks $2$ while BIC / HQIC pick $1$ — we take the safer, less-biased $2$.

**The whiteness gate — Ljung–Box / portmanteau.**

- $H_0$: the VAR residuals are **white** (no autocorrelation up to lag $h$).
- Statistic (multivariate, adjusted):
  $Q_h = T^2\sum_{j=1}^h (T-j)^{-1}\operatorname{tr}(\hat C_j' \hat C_0^{-1}\hat C_j \hat C_0^{-1}) \sim \chi^2$,
  where $\hat C_j$ is the residual autocovariance at lag $j$.
- This is a **gate**, not an afterthought: the trace test's asymptotic distribution
  assumes serially-uncorrelated errors, so whiteness is a precondition.

**What came back — and how to read the "failure":** the whiteness test **rejects**
($p\approx 0$). We do not hide this; we own it.

> At **daily** frequency strict whiteness is essentially unattainable: the portmanteau
> statistic also absorbs **ARCH / volatility clustering**, which no lag length removes.
> Adding lags would chase conditional heteroskedasticity, not mean dynamics, and merely
> over-parameterise the system. So we keep the AIC order and get valid rank inference
> from the **heteroskedasticity-robust wild bootstrap** (Step 4) rather than from the
> asymptotic test alone.

This turns a nuisance into a rigour point: the failed whiteness test is precisely what
*motivates* the bootstrap that makes the headline result robust.

---

## Step 3 — Which deterministic world are we in?

The VECM can carry deterministic terms split between the levels and the cointegrating
relation:

$$
\Delta y_t = \alpha\bigl(\beta' y_{t-1} + \rho_0 + \rho_1 t\bigr) + \gamma_0 + \gamma_1 t + \sum_{i=1}^{k}\Gamma_i \Delta y_{t-i} + \varepsilon_t .
$$

Johansen formalises five cases (no deterministics; restricted constant; **unrestricted
constant** = linear trend allowed in levels; restricted trend; unrestricted trend). The
choice is substantive because **it changes the critical values** of the rank test.

- **Choice: Case 3 (unrestricted constant, `deterministic='co'`).** The EDA shows EM
  currencies (ZAR, MXN, NOK) **drift persistently** versus the dollar. We confirm it with
  a regression $y_t = a + bt$ and HAC standard errors: the slope $t$-statistics are
  **huge** (NZD $t=-27$, NOK / ZAR $t\approx-15$) → a genuine deterministic trend the
  model must accommodate → unrestricted constant. Case 2 (restricted constant) is run as a
  robustness check, not the headline spec.

This matters to the story: that same persistent dollar drift is what makes "is the
breakdown just shared dollar exposure?" a real question later (Step 8).

---

## Step 4 — The core test: does a stable equilibrium exist over the whole decade?

This is the heart of the in-sample inference. Re-parameterise the VAR as the
**Vector Error-Correction Model**:

$$
\Delta y_t = \Pi y_{t-1} + \sum_{i=1}^{k}\Gamma_i \Delta y_{t-i} + (\text{determ.}) + \varepsilon_t,
\qquad \Pi = \alpha\beta' .
$$

$\Pi$ is the **long-run impact matrix**, and its **rank** is the cointegration rank,
$r=\operatorname{rank}(\Pi)$. If $r=0$ there is no cointegration ($\Pi=0$, a VAR in pure
differences — the currencies wander independently). If $0<r<n$ there are $r$ equilibria,
with $\beta$ ($n\times r$) the cointegrating vectors and $\alpha$ ($n\times r$) the
adjustment speeds. **In the language of the treasury question, $r\ge 1$ is exactly
diversification breaking down: a stationary combination means the currencies have fused.**

Johansen's MLE solves a generalised eigenvalue problem on the residual moment matrices,
yielding $\hat\lambda_1 > \cdots > \hat\lambda_n$ (each a squared canonical correlation
between levels and differences; a large $\hat\lambda_i$ signals a mean-reverting
direction). Two tests then read off the rank:

- **Trace test:** $\mathrm{LR}_{\text{trace}}(r) = -T\sum_{i=r+1}^{n}\ln(1-\hat\lambda_i)$.
  $H_0$: at most $r$ relations; $H_1$: more than $r$. Start at $r=0$; if rejected try
  $r=1$, … stop at the first non-rejection.
- **Maximum-eigenvalue test:** $\mathrm{LR}_{\max}(r) = -T\ln(1-\hat\lambda_{r+1})$.
  $H_0$: rank $=r$; $H_1$: rank $=r+1$. Sharper, can disagree with trace — so we report
  both and reconcile.

Both have **non-standard** asymptotic distributions (functionals of Brownian motion)
depending on $n-r$ and the deterministic case; critical values come from
MacKinnon–Haug–Michelis tables.

**Two robustness layers — this is what elevates the result from "a test said no" to
"every robust route says no":**

- **(a) Reinsel–Ahn finite-sample correction.** The trace test over-rejects in finite
  samples; replace $T$ by the effective sample $T-nk$:
  $\mathrm{LR}^{*}_{\text{trace}}(r) = -(T-nk)\sum_{i>r}\ln(1-\hat\lambda_i)$.
- **(b) Wild bootstrap (Cavaliere–Rahbek–Taylor).** Asymptotic critical values assume
  homoskedastic Gaussian errors, but FX has heavy volatility clustering that **inflates**
  the trace statistic. To test $H_0: r=0$: estimate the restricted model (VAR in
  differences), keep residuals $\hat\varepsilon_t$; draw **Rademacher signs**
  $w_t\in\{-1,+1\}$ and form $\varepsilon^*_t = w_t\hat\varepsilon_t$ (this preserves the
  conditional heteroskedasticity); rebuild a bootstrap series and recompute
  $\mathrm{LR}_{\text{trace}}(0)$; repeat $B$ times; the bootstrap $p$-value is
  $(\#\{\mathrm{LR}^*\ge\mathrm{LR}^{\text{obs}}\}+1)/(B+1)$. (The notebook's from-scratch
  reduced-rank routine is validated against `statsmodels.coint_johansen` to the decimal.)

**What came back — the headline for the full sample:**

| Quantity | Value |
|---|---|
| Trace rank @95% | **0** |
| Finite-sample-corrected trace rank | **0** |
| Max-eigenvalue rank @95% | **0** |
| Observed $\mathrm{LR}_{\text{trace}}(0)$ | 115.58 |
| Asymptotic 95% crit. value | 125.62 |
| **Bootstrap 95% crit. value** | **135.62** |
| **Bootstrap $p$-value** | **0.345** |

Read this the way theory predicts: the bootstrap critical value (135.6) sits **above**
the asymptotic one (125.6) — exactly the direction you expect when errors are
heteroskedastic and the asymptotic test over-rejects. The observed statistic (115.6) is
below *both*, and the bootstrap $p$-value of **0.345** is nowhere near rejection. All four
routes agree: **no cointegration in the full 2016–26 sample.**

**In the language of the treasury question, this is reassuring.** Over the decade as a
whole, the reserve currencies do *not* collapse into a single factor — the
diversification rationale is sound on average, and because the bootstrap critical value
sits well above the asymptotic one, "diversification holds" is **not a near miss**.

---

## Step 5 — A second opinion: residual-based cointegration tests

Relying on one test family is fragile, so we confirm with a completely different
construction: regress one currency on the others,
$y_{1t} = \theta' y_{-1,t} + u_t$, and test the residual $\hat u_t$ for a unit root.

- $H_0$: $\hat u_t$ has a unit root → **no cointegration.** $H_1$: $\hat u_t$ stationary →
  cointegration.
- **Engle–Granger** runs an ADF on $\hat u_t$; **Phillips–Ouliaris** uses a PP-type HAC
  statistic on $\hat u_t$.

**The critical-values point, now reversed from Step 1.** Because $\hat u_t$ is the
residual of an *estimated* regression whose coefficients were chosen to **minimise its
variance**, $\hat u_t$ is biased toward looking stationary. So we **must** use the
stricter, $N$-dependent Engle–Granger / Phillips–Ouliaris critical values (more
regressors ⇒ more stringent). Feeding $\hat u_t$ into a plain ADF would silently use
too-lenient values and **over-detect** cointegration — the `arch` package applies the
correct values automatically. (This is the mirror image of Step 1, where directly
observed series correctly used standard DF values.)

**What came back:** Engle–Granger $p=0.61$, Phillips–Ouliaris $p=0.69$ → **no
cointegration**, independently confirming Johansen by a different route.

So the full-sample picture is now triangulated from two independent test families plus
two robustness layers, all saying the same thing: **on average over the decade,
diversification holds.** But a single decade-long number hides the dynamics — a
relationship that *fails intermittently* and recovers averages out to "intact" even when
it collapsed badly in particular sub-periods. To see those collapses we have to look
window by window. First, two short detours that sharpen how we read any breakdown we find.

---

## Step 6 — Demonstrating the VECM machinery (rank set to 1 on purpose)

Since the honest full-sample rank is **0**, there is no equilibrium to estimate. But the
walk-forward *will* find windows with rank $\ge 1$, so we need the estimation and
diagnostic machinery ready and demonstrated. The notebook therefore sets rank $=1$
**only to exhibit the machinery** — the rank-0 verdict still stands — and fits:

$$
\Delta y_t = \alpha\beta' y_{t-1} + \sum_{i=1}^{k}\Gamma_i\Delta y_{t-i} + (\text{determ.}) + \varepsilon_t ,
$$

with three diagnostic checks ("a model we have not checked is a model we cannot trust"):

1. **Stationarity of the error-correction term $\hat\beta'y_t$** (ADF / KPSS) — directly
   verifies the combination is $I(0)$, closing the loop. Flagged *indicative only*,
   because $\beta$ is estimated (the Step 5 caveat); the formal evidence is the
   properly-valued residual tests.
2. **Companion-matrix stability** — the VAR companion form must have exactly $n-r$ unit
   roots and all other eigenvalue moduli $<1$.
3. **Residual whiteness / normality** on the fitted VECM.

This is plumbing for the headline section, stated plainly so no one mistakes the
demonstration rank for a claim that $r=1$ on the full sample.

---

## Step 7 — Roles within the basket and the speed of reversion

Two by-products of the VECM feed directly into how we run and read the walk-forward.

**Weak exogeneity — anchors vs adjusters.** A currency whose loading $\alpha_j\approx 0$
does **not** error-correct: it *pushes* the system but is not pulled back — a long-run
**anchor**. A large, significant $\alpha_j$ marks an **adjuster** that reverts toward the
basket.

- $H_0$: $\alpha_j = 0$ (currency $j$ is weakly exogenous). Formally an LR test; we report
  the asymptotically-equivalent $t$-tests on each $\alpha_j$. Reject ($p<0.05$) ⇒
  adjuster; fail to reject ⇒ anchor.

| Currency | $\alpha$ | $t$ | $p$ | Role |
|---|---|---|---|---|
| AUD | −0.0013 | −2.07 | 0.038 | **Adjuster** |
| NZD | −0.0013 | −2.08 | 0.038 | **Adjuster** |
| CAD | −0.0006 | −1.34 | 0.179 | Anchor |
| NOK | −0.0026 | −3.55 | 0.000 | **Adjuster** |
| SEK | −0.0023 | −3.60 | 0.000 | **Adjuster** |
| ZAR | +0.0018 | 1.92 | 0.055 | Anchor |
| MXN | −0.0001 | −0.18 | 0.858 | Anchor |

→ **CAD, ZAR, MXN are anchors** (what the basket is pinned to); **AUD / NZD / NOK / SEK
are adjusters** (where any mean reversion lives).

**Half-life of mean reversion.** Model the error-correction term as an AR(1),
$z_t = c + \phi z_{t-1} + e_t$; the half-life of a shock is
$H = \ln(0.5)/\ln\phi$ trading days. We found $\phi = 0.9789 \Rightarrow H = 32.4$ days.
This is not a curiosity: we **set the walk-forward refit cadence $M=32$ from this
half-life** rather than picking it arbitrarily — a derived parameter, not an assumed one.

---

## Step 8 — The payoff: re-asking the diversification question window by window

Now the headline. We re-run the whole rank question inside rolling windows and watch the
answer change with the macro regime — re-asking, at each point in time, "have the
currencies fused *right now*?"

**Mechanics — said slowly, because this is the part people misread.** We are **not**
forecasting exchange rates or "predicting cointegration." At each rebalance point $t_0$:

1. Take the **trailing** window $[t_0-N,\,t_0)$ with $N=504$ days ($\approx 2$ years) —
   past data only.
2. Run the finite-sample-corrected Johansen trace test. **If $r\ge 1$**, fit the VECM and
   **freeze** $\beta,\mu,\sigma$. **If $r=0$, hold cash** for the next block — encode the
   instability rather than force a trade.
3. For the next $M=32$ days (the half-life from Step 7), trade the frozen signal.
4. Advance by $M$ and **refit**, stitching the out-of-sample blocks into one continuous,
   **fully-causal** series.

We use **rolling (fixed-$N$)**, not expanding, windows on purpose: we *want* the model to
forget pre-regime data, because that adaptivity is the right choice for a relationship we
already know is unstable. A leakage assertion checks that every train index precedes the
returns it is used to trade.

### What exactly is the transaction when we get a signal?

"Trade the frozen signal" in step 3 above is doing a lot of quiet work, so here is the
concrete chain from a Johansen result to an actual set of positions.

**1. There is only a transaction at all when the window says rank $\ge 1$.** If the
trailing window returns $r=0$ (diversification intact — no stationary combination), there
is **no trade**: we hold USD cash for the whole next 32-day block. About 85% of days fall
here. A transaction exists *only* in the ~15% of windows where the basket has fused into a
tradeable equilibrium. So the "signal" is fundamentally a by-product of a diversification
*breakdown* — when the currencies are behaving independently there is nothing to do.

**2. The frozen $\hat\beta$ defines the one combination that is mean-reverting.** When
$r\ge 1$ we freeze the cointegrating vector $\hat\beta$ (a fixed set of seven weights, one
per currency) and the mean $\mu$ and standard deviation $\sigma$ of the in-window
equilibrium error. $\hat\beta$ is the *recipe* for the synthetic, stationary portfolio:
the specific long-some / short-others mix of the seven currencies whose value
$s_t=\hat\beta'y_t$ wanders around a fixed level instead of drifting. Nothing is traded
yet — this just names which combination to watch.

**3. Each day we measure how stretched that combination is.** Mark the frozen recipe to
the current (log) prices and standardise it:

$$
s_t=\hat\beta'y_t,\qquad z_t=\frac{s_t-\mu}{\sigma}.
$$

$z_t$ is a *z-score of disequilibrium*: $z_t=0$ means the fused basket sits at its
equilibrium (no edge, no position); $z_t=+2$ means the combination is two standard
deviations rich relative to where the just-estimated equilibrium says it belongs;
$z_t=-2$, two standard deviations cheap. Because the combination is stationary, a stretched
$z_t$ is the prediction that it will partly revert.

**4. The position leans against the stretch, dollar-neutral.** We hold the cheap legs and
fund them by shorting the rich legs, with weights

$$
w_t=-\,z_t\,\hat\beta\ \big/\ \lVert z_t\hat\beta\rVert_1 .
$$

Read literally: take the recipe $\hat\beta$, scale it by **minus** the stretch $-z_t$ (so
we bet *on* reversion — sell what is rich, buy what is cheap), then divide by the
$\ell_1$-norm so the gross book sums to 1. **The transaction is therefore a single
long/short FX basket trade**, rebalanced daily within the block. Concretely, if on a given
day $z_t>0$ and $\hat\beta=(\beta_{\text{AUD}},\dots,\beta_{\text{MXN}})$, the position is
short the currencies $\hat\beta$ loads positively, long the ones it loads negatively, sized
proportionally to $|w_t|$ — e.g. "short ~30% AUD, ~10% NZD; long ~25% NOK, ~20% SEK …"
with the magnitudes growing as $|z_t|$ grows and collapsing toward zero as the basket
returns to equilibrium. The legs are simultaneous spot FX positions (each funded in USD),
so the book is **dollar-neutral by construction**: it strips out the common USD direction
and bets only on the *relative* value among the seven currencies.

**5. The P&L is mechanical from there.** A position formed at the close of day $t$ earns
the $t\!\to\!t\!+\!1$ currency returns; we never use a price we could not have observed.
When the block ends after 32 days we re-run Johansen and either get a fresh recipe, or —
if the window has reverted to $r=0$ — flatten back to cash.

**What this is *not*.** It is not a directional bet that, say, AUD will rise; the
dollar-neutral construction deliberately removes that. It is not a reserve allocation a
treasury could literally hold — the book is net-zero and includes short legs (the scope
note). And it is not, on the evidence of Step 10, a money-maker: it is the *operational
read-out* of the diversification signal — "the basket has fused and is currently stretched
this way" expressed as the trade that would lean against it.

**What came back:** the leakage self-check **PASSES**; the OOS span is 2018-05 → 2026-06.
Across the **66 refit windows**, the basket reaches **rank $\ge 1$ in 10 of them
(~15%)** — meaning diversification broke down about 15% of the time — and the system is
in-market only ~15% of days (cash 85%). Crucially those rank-$\ge1$ windows are **not
scattered randomly**: they cluster into **four episodes**, each aligned with a
recognisable macro stress.

### Episodes 1–3 — brief, single-refit breakdowns

| Window | trace(0) / 95% crit. | Macro context |
|---|---|---|
| 2020-03-20 → 2020-05-04 | 1.12 | COVID crash / global dash-for-dollars |
| 2021-04-29 → 2021-06-11 | 1.04 | post-vaccine reflation trade |
| 2022-01-24 → 2022-03-08 | 1.12 | onset of the Fed hiking cycle / Ukraine |

Each is a single window in which the currencies briefly fused into one factor, then
separated again at the next refit — diversification failing under acute, shared
risk-repricing and recovering once the shock passed. These are exactly the moments a
treasurer would *most* want to know their spread-out reserves had quietly become one bet.

### Episode 4 — a sustained breakdown (2025-05 → 2026-03)

| Window | trace(0) / 95% crit. |
|---|---|
| 2025-05-19 → 2025-07-01 | 1.05 |
| 2025-07-02 → 2025-08-14 | 1.11 |
| 2025-08-15 → 2025-09-29 | 1.10 |
| 2025-09-30 → 2025-11-12 | 1.10 |
| 2025-11-13 → 2025-12-26 | 1.16 |
| 2025-12-29 → 2026-02-10 | 1.12 |
| 2026-02-11 → 2026-03-26 | 1.11 |

Seven consecutive rank-$\ge1$ windows — the one stretch where the breakdown *persisted*
across refits rather than reverting immediately. A prolonged regime in which the
diversification benefit of holding these currencies separately was materially reduced.

**Where to see this in the notebook.** Cell 31 plots it directly: the lower panel
("Johansen rank per refit window") is a step function that jumps $0\to1$ exactly at the
windows above, while the upper panel shows the cointegrating vector $\beta$ — the
*direction* of the shared factor — drifting across them. Cell 33 shows the recursive
$\mathrm{trace}(0)/\text{critical-value}$ ratio crossing 1.0 in these same periods.

**A descriptive companion, firewalled from the signal.** Alongside the causal
walk-forward, a recursive (expanding-window) trace statistic scaled by its 95% critical
value wanders above and below 1, confirming the relationship's *existence* is not
constant; and a Bai–Perron / PELT break detector (via `ruptures`) on the ECT dates the
regime shifts. These full-sample break dates are **never fed back into the signal** —
that would be look-ahead — they are there to characterise, not to trade.

---

## Step 9 — Reading the breakdowns honestly

Two caveats keep the interpretation disciplined — and both reinforce the thesis rather
than undercut it.

**1. The breakdowns are marginal.** Every rank-1 window has
$\mathrm{trace}(0)/\text{crit.}$ between **1.04 and 1.16** — just over the line. Given
that Step 4 showed the asymptotic threshold itself *understates* the true (bootstrap)
critical value by roughly 8%, the weakest windows (the 1.04–1.05 cases) are close to
threshold noise. Read these as **diversification degrading, not vanishing entirely** —
which is exactly what regime-dependence looks like.

**2. Is the breakdown just shared dollar exposure?** Because every series is quoted vs
USD, a "cointegrating relation" might be nothing but **common dollar exposure** — which
for a treasury is hedgeable with a single USD position, whereas a genuine cross-currency
factor is not. We tested this directly on the full sample rather than asserting it. The
purest dollar relation is the equal-weight direction $b=(1,\dots,1)'$; restricting
$\beta$ to it gives a likelihood-ratio statistic:

$$
\lambda^{*} = \frac{b' S_{10} S_{00}^{-1} S_{01}\, b}{b' S_{11}\, b},
\qquad
\mathrm{LR} = T\,\ln\!\frac{1-\lambda^{*}}{1-\hat\lambda_1} \;\sim\; \chi^2_{\,n-1}.
$$

- $H_0$: the cointegrating vector **equals** the dollar direction (a pure, untradeable
  dollar factor). $H_1$: $\beta\ne b$, a genuine cross-currency relationship. Small $p$ ⇒
  reject ⇒ *not* just the dollar.
- **What came back:** $\hat\lambda_1 = 0.0146$, $\lambda^* = 0.0047$,
  $\mathrm{LR} = 26.07$, $\chi^2_6$ $p = 0.0002$ → **reject $H_0$:** the dominant
  direction is statistically distinguishable from a pure dollar factor.

The honest limit: this LR test was run on the **full sample**, not re-run inside each
window, so the rolling regimes are identified **descriptively** — we stop short of
classifying each episode as dollar-driven versus genuinely cross-currency.

Neither caveat weakens the project. The deliverable is the **identification of the
regimes in which diversification breaks down, plus the honest finding that it holds the
rest of the time** — not a claim that any single window is a clean, exploitable signal.

---

## Step 10 — Could you have *traded* the breakdowns? (the discipline check)

The diversification answer stands on its own, but the notebook also asks the harder
predictive question — and reports the answer that keeps everyone honest. The benchmark
any FX model must clear is the **random walk** (Meese–Rogoff 1983: exchange-rate changes
are close to unforecastable).

**Predictive regression — is there local mean reversion at all?**

$$
\Delta s_{t+1} = a + b\,z_t + u_{t+1}, \qquad \text{Newey–West (HAC) standard errors.}
$$

- $H_0$: $b=0$ (the equilibrium error has no predictive content). $H_1$: $b<0$
  (deviations are partly reversed — error correction). HAC is mandatory because
  autocorrelation and volatility clustering would otherwise overstate the $t$-stat.
- **What came back:** $b=-0.00398$, **HAC $t=-3.47$, $p=0.001$** → **statistically
  significant mean reversion.**

**Diebold–Mariano — but does it beat a random walk?** Compare one-step forecasts of $s_t$
from the error-correction model against the random-walk forecast $\hat s_{t+1}=s_t$, with
loss differential $d_t = e^{\text{RW}\,2}_t - e^{\text{VECM}\,2}_t$:

$$
\mathrm{DM} = \frac{\bar d}{\sqrt{\widehat{\operatorname{Var}}(\bar d)}} \xrightarrow{d} \mathcal N(0,1).
$$

- $H_0$: equal forecast accuracy. $H_1$: the model forecasts better ($\mathrm{DM}>0$).
- **What came back:** $\mathrm{DM}=1.34$, $p=0.18$ → **cannot reject equal accuracy: the
  model does not beat the random walk.**

> **The punchline:** there is *statistically significant local mean reversion* in the
> equilibrium error, yet it **does not translate into beating a random walk**. That is
> precisely the Meese–Rogoff nuance — predictability $\ne$ forecast superiority — and it
> is exactly why this project is risk *monitoring*, not a trading claim.

**For completeness, the price-only benchmark table** (over the identical OOS window):

| Strategy | Total ret | Ann. vol | Sharpe | Max DD |
|---|---|---|---|---|
| **Cointegration (walk-fwd)** | **+0.030** | 0.011 | **+0.34** | **−0.02** |
| Hold AUD | −0.070 | 0.100 | −0.08 | −0.29 |
| Hold NZD | −0.170 | 0.098 | −0.21 | −0.30 |
| Hold CAD | −0.084 | 0.064 | −0.16 | −0.19 |
| Hold NOK | −0.156 | 0.120 | −0.16 | −0.38 |
| Hold SEK | −0.084 | 0.103 | −0.10 | −0.33 |
| Hold ZAR | −0.276 | 0.141 | −0.23 | −0.47 |
| Hold MXN | +0.122 | 0.120 | +0.12 | −0.32 |
| USD cash | 0.000 | 0.000 | – | 0.00 |
| Equal-weight basket | −0.103 | 0.086 | −0.14 | −0.25 |

Two honesty caveats: the strategy is **long/short, dollar-neutral** (not apples-to-apples
with the directional buy-and-holds — by sitting in cash 85% of the time it simply
sidestepped the EM drawdowns), and **carry is omitted**, so the high-yielder lines
(ZAR, MXN) are understated. The table demonstrates behaviour; it is not the deliverable.

---

## The takeaway

> A multinational that spreads cash reserves across AUD, NZD, CAD, NOK, SEK, ZAR and MXN
> is relying on those currencies behaving independently. Cointegration is the formal test
> of when that independence fails. The full sample — triangulated across Johansen trace
> and max-eigenvalue, a Reinsel–Ahn finite-sample correction, a heteroskedasticity-robust
> wild bootstrap ($p=0.345$), and the Engle–Granger / Phillips–Ouliaris residual tests
> ($p=0.61$, $0.69$) — says diversification **holds on average over the decade**. The
> rolling-window analysis shows it **breaks down in ~15% of windows**, concentrated in
> four macro-stress episodes (COVID 2020, the 2021 reflation trade, the 2022 hiking cycle,
> and a sustained 2025–26 stretch). The mean reversion inside the equilibrium error is
> statistically significant ($t=-3.5$) yet does **not** beat a random walk ($\mathrm{DM}$
> $p=0.18$). Knowing *when* the reserve basket collapses into a single risk factor — and
> confirming it mostly does not — is the result.

The grade lives in the rigour of the identification, not in any tradeable signal — and
every step above is a test, with a hypothesis, a statistic, and a number that earned the
right to the next one.

---

### Map of tests to story steps

| Story step | Question | Test(s) | Verdict |
|---|---|---|---|
| 1 | Are the currencies $I(1)$? | ADF, KPSS, DF-GLS, PP, Zivot–Andrews | all seven **$I(1)$** |
| 2 | How many lags / residuals clean? | AIC + portmanteau (Ljung–Box) | $p=2$; whiteness fails → motivates bootstrap |
| 3 | Which deterministic case? | HAC drift regression $y_t=a+bt$ | **Case 3** (unrestricted constant) |
| 4 | Stable equilibrium over the decade? | Johansen trace & max-eig, Reinsel–Ahn, **wild bootstrap** | **rank 0** — diversification holds on average |
| 5 | Second opinion? | Engle–Granger, Phillips–Ouliaris | **no cointegration** (confirms) |
| 6 | Machinery demonstration | VECM fit + companion / whiteness diagnostics | rank set to 1 *for demo only* |
| 7 | Roles & reversion speed | weak-exogeneity $t$-tests; AR(1) half-life | anchors CAD/ZAR/MXN; $H=32$ days |
| 8 | When does it break down? | rolling Johansen, recursive trace, Bai–Perron | **regime-dependent**, ~15% of windows, 4 episodes |
| 9 | Read the breakdowns honestly | $\beta$-restriction LR ($\chi^2_6$) | **not** a pure dollar factor (full sample) |
| 10 | Could you trade it? | predictive regression (HAC); Diebold–Mariano | reversion significant; **no** RW beat |
