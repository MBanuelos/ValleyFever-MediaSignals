# Pesticide (Fungicide / Rodenticide) Feature Test — 2026-10-26

**Question:** do fungicide and rodenticide use (CalPIP/PUR) improve development-period forecasts when used only as they would actually be available?

**Setup**
- Development period: targets 2017-01 to 2020-12 (48 per county), 8 counties.
- Case lag 1. Ridge, plus XGBoost with seeds 42–44 (predictions averaged over seeds).
- Configs: `configs/pesticide/*.json`. Analysis: `scripts/env_pesticide_check.py`, which writes `results/env/pesticide_check/`.

Every variant is compared with `none` (no pesticide features) on identical targets.

| Variant | Pesticide information at origin t |
|---|---|
| `none` | — |
| `stale36` | Value at t−36 and its 12-month mean (conservative publication delay; see data-availability note §1b) |
| `ets_fill` | Data up to t−36, then the gap to t−1 forecast by additive Holt–Winters on the last 72 available months. Features: the filled t−1 value and the 12-month mean ending at t−1. Refitted for every row's own origin, so training mimics deployment. |
| `oracle1` | **Retrospective only:** value at t−1 and 12-month mean. Not available in practice. |

## Results [V]

The first table counts, out of 8 counties, where adding pesticides lowered RMSE ("better"), how many of those were significant (DM p < 0.05), and the median RMSE change (negative means pesticides help).

| Model / inputs | Variant | h=1 better (sig) / median | h=3 better (sig) / median | h=6 better (sig) / median |
|---|---|---|---|---|
| XGBoost E | stale36 | 8 (5) / −21% | 7 (2) / −17% | 7 (1) / −17% |
| XGBoost E | ets_fill | 8 (6) / −17% | 7 (3) / −13% | 8 (1) / −9% |
| XGBoost E | oracle1 | 6 (4) / −6% | 6 (2) / −6% | 4 (1) / −1% |
| Ridge E | stale36 | 6 (4) / −11% | 8 (2) / −11% | 8 (1) / −13% |
| XGBoost RE | any | 3–7 (0–2) / −4% to +1% | | |
| Ridge RE | any | 2–4 (0–1) / −3% to +3% | | |

Median skill vs seasonal naive for XGBoost (h = 1 / 3 / 6):

| Inputs | No pesticides | With stale36 pesticides |
|---|---|---|
| E | 0.03 / −0.07 / −0.03 | **0.21 / 0.15 / 0.12** |
| RE | 0.24 / 0.10 / 0.14 | 0.23 / 0.14 / 0.14 |

For XGBoost, the improvements are consistent across seeds: "better" and "better with all 3 seeds agreeing" differ by at most 2 counties in any cell.

## Interpretation

1. **In combined models (RE), pesticides add nothing.** The median change is within ±4% and rarely significant. Dropping them from the primary RE model costs nothing.
2. **In environment-only models (E), pesticides help a lot.** They bring XGBoost E from no skill to almost RE-level skill.
3. **The pattern argues against a direct pesticide effect.** Values from three years earlier help *more* than values from last month (stale36 better than oracle1). The ETS fill, which with no trend term essentially carries forward the last released level and seasonal shape, behaves like stale36.
   - [H1] The pesticide features act as a slowly varying **level/trend proxy**. Without case history, E has no other way to track a county's rising incidence level (2016–2020). Any slowly drifting county-level series would help.
   - [H2] 3-year-old agricultural activity proxies **multi-year antecedent conditions**, such as the 2012–2016 drought, which the current precipitation features cannot see (they reach back at most 12 months).
4. **This is predictive, not causal.** Do not report it as evidence that pesticide use affects Valley fever incidence.

## Recommended follow-up (development period only)

Rerun XGBoost/Ridge E with:

- (a) a simple trend proxy (e.g., years since the start of the training window);
- (b) longer precipitation windows (24- and 36-month means), to capture antecedent drought;
- each with and without stale36 pesticides.

If the pesticide benefit disappears once (a) or (b) is present, the paper can state that the apparent pesticide signal reflects long-term trend or antecedent climate, not pesticide use.
