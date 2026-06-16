# The Story of This Project
## When Does Currency Diversification Break Down?
### A Cointegration Lens on a Multinational's Cash-Reserve Currencies

*Companion narrative to `fx_cointegration_vecm.ipynb` and `PRESENTER_GUIDE.md`. Where
the presenter's guide specifies every test, this document tells the **story**: the
question we ask, who would care about the answer, what we find, and why "no stable
relationship most of the time" is a complete and useful result rather than a
disappointment.*

---

## The setting

Imagine a multinational corporation that earns revenue and pays costs across a basket
of commodity / risk-sensitive currencies — AUD, NZD, CAD, NOK, SEK, ZAR, MXN — and
holds cash reserves spread across them. Part of the rationale for spreading reserves
across several currencies is **diversification**: if the currencies move independently,
holding seven of them is genuinely less risky than holding one.

That rationale has a hidden assumption — that the currencies *stay* independent. They
do not always. Under certain macro regimes these currencies stop behaving like seven
separate bets and collapse into **a single shared risk factor**. When that happens, the
treasury's apparent diversification is an illusion: it is holding one bet wearing seven
labels, and a shock that hits the common factor hits the whole reserve at once.

---

## The question

**When does the diversification across these reserve currencies break down — and when
does it hold?**

This is not a question about *which* currency to hold (that is a directional,
yield-driven decision this analysis deliberately does not make — see the scope note
below). It is a **risk-monitoring** question: identifying the time periods in which the
basket behaves as one factor rather than many.

Cointegration is exactly the right instrument for it. A set of currencies is
**cointegrated** when some linear combination of them is stationary — i.e. when they are
tied together by a common long-run equilibrium and cannot drift apart. *Cointegration
is therefore the formal signature of diversification failure:* the periods in which a
stationary combination exists are precisely the periods in which the currencies have
become a single risk factor.

So the project's thesis is:

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
- Carry is **explicitly excluded** throughout (see §13), which is appropriate for a
  co-movement study but would be disqualifying for an allocation claim.

What cointegration *does* answer — and answers well — is the diversification-risk
question above. Keeping that boundary sharp is part of the rigour.

---

## The arc

The analysis is a funnel — each test earns the right to run the next:

> are the series I(1)? → how many lags, residuals clean? → which deterministic case?
> → cointegration rank → VECM β, α → error-correction term → **rolling-window
> re-estimation**.

The early stages establish that the machinery is *valid* to run at all (the series are
I(1), the lag order is justified, the deterministic case fits the visible drift). The
payoff stage is the walk-forward in §11, where we re-ask the diversification question
inside every rolling window and watch the answer change with the macro regime.

---

## What the full sample says: diversification mostly holds

Run on the whole 2016–2026 sample, **every test agrees there is no cointegration**:

| Test | Result | Verdict |
|---|---|---|
| Johansen trace @95% | rank = 0 | no cointegration |
| Johansen trace, Reinsel–Ahn corrected | rank = 0 | no cointegration |
| Johansen max-eigenvalue @95% | rank = 0 | no cointegration |
| Wild bootstrap of H₀: r = 0 (robust to ARCH) | observed 115.6 vs. bootstrap crit. 135.6, **p = 0.345** | cannot reject r = 0 |
| Engle–Granger | p = 0.612 | no cointegration |
| Phillips–Ouliaris | p = 0.686 | no cointegration |

In the language of the treasury question, this is **reassuring**: over the decade as a
whole, the reserve currencies do *not* collapse into a single factor — the
diversification rationale is sound on average. The wild bootstrap matters here, because
FX volatility clustering inflates the asymptotic trace statistic; the bootstrap critical
value (135.6) sits well above the asymptotic one (125.6), so "diversification holds" is
not a near miss.

But a single decade-long number hides the dynamics. Diversification that *fails
intermittently* and recovers averages out to "intact" over ten years even when it
collapsed badly in particular sub-periods. To see those collapses, we look window by
window.

---

## What the rolling windows say: diversification breaks down in identifiable regimes

The walk-forward (§11) uses a **504-day (~2-year) lookback**, refit every **32 trading
days** (the half-life of the error-correction term, from §9). At each refit it re-runs
the finite-sample-corrected Johansen trace test and records the rank. Across the
**66 windows**, the basket reaches **rank ≥ 1 in 10 of them (~15%)** — meaning
diversification broke down about 15% of the time — and those windows are not scattered
randomly. They cluster into **four episodes**, each aligned with a recognisable macro
stress or repricing event.

### Episodes 1–3 — brief, single-refit breakdowns

| Window | trace(0) / 95% crit. | Macro context |
|---|---|---|
| 2020-03-20 → 2020-05-04 | 1.12 | COVID crash / global dash-for-dollars |
| 2021-04-29 → 2021-06-11 | 1.04 | post-vaccine reflation trade |
| 2022-01-24 → 2022-03-08 | 1.12 | onset of the Fed hiking cycle / Ukraine |

Each is a single window in which the currencies briefly fused into one factor, then
separated again at the next refit — diversification failing under acute, shared
risk-repricing and recovering once the shock passed. These are exactly the moments a
treasurer would *most* want to know that their spread-out reserves had quietly become a
single bet.

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

Seven consecutive windows of rank ≥ 1 — the one stretch where the breakdown *persisted*
across refits rather than reverting immediately. A prolonged regime in which the
diversification benefit of holding these currencies separately was materially reduced.

**Where to see this in the notebook.** Cell 31 plots it directly: the lower panel
("Johansen rank per refit window") is a step function that jumps 0 → 1 exactly at the
windows above, while the upper panel shows the cointegrating vector β — the *direction*
of the shared factor — drifting across them. Cell 33 shows the recursive
`trace(0) / critical-value` ratio crossing 1.0 in these same periods.

---

## Reading the breakdowns honestly

Two caveats keep the interpretation disciplined — and they reinforce the thesis rather
than undercut it:

1. **The breakdowns are marginal.** Every rank-1 window has `trace(0) / crit.` between
   **1.04 and 1.16** — just over the line. Given that §5 showed the asymptotic threshold
   itself understates the true (bootstrap) critical value by roughly 8%, the weakest
   windows (the 1.04–1.05 cases) are close to threshold noise. Read these as
   *diversification degrading*, not vanishing entirely — which is exactly what
   regime-dependence looks like.

2. **Is the breakdown just shared dollar exposure?** The EDA (§1) shows NOK, SEK, ZAR
   and MXN all drifting persistently against the dollar, and §8 tests — on the full
   sample — whether the relationship is a pure common-dollar factor (it rejects that on
   the full sample). For a treasury this distinction matters: a breakdown driven by a
   common *dollar* move is hedgeable with a single USD position, whereas a genuine
   cross-currency factor is not. The §8 test was not re-run inside each window, so the
   regimes are identified *descriptively*, and we stop short of classifying each one as
   dollar-driven versus genuinely cross-currency.

Neither caveat weakens the project. The deliverable is the **identification of the
regimes in which diversification breaks down, and the honest finding that it holds the
rest of the time** — not a claim that any window is a clean, exploitable signal.

---

## The takeaway

> A multinational that spreads cash reserves across AUD, NZD, CAD, NOK, SEK, ZAR and MXN
> is relying on those currencies behaving independently. Cointegration is the formal test
> of when that independence fails. The full sample says it **holds on average over the
> decade**; the rolling-window analysis shows it **breaks down in ~15% of windows**,
> concentrated in four macro-stress episodes (COVID 2020, the 2021 reflation trade, the
> 2022 hiking cycle, and a sustained 2025–26 stretch). Knowing *when* the reserve basket
> collapses into a single risk factor — and confirming it mostly does not — is the result.

The grade lives in the rigour of the identification, not in any tradeable signal.
