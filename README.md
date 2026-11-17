# GTA VI Launch-Day Twitch Forecast

[![make.fun](https://make.fun/badge/gta6-twitch-ccu-forecast-01.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01)
[![rank](https://make.fun/badge/gta6-twitch-ccu-forecast-01/FadliAR/gta6-twitch-ccu-forecast-01/rank.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01)
[![reward](https://make.fun/badge/gta6-twitch-ccu-forecast-01/FadliAR/gta6-twitch-ccu-forecast-01/reward.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01)

Entry for the make.fun competition
[gta6-twitch-ccu-forecast-01](https://make.fun/bounty/gta6-twitch-ccu-forecast-01):
forecast Grand Theft Auto VI's Twitch concurrent viewers, hour by hour,
for the 23 hours after the first hour of release day (19 Nov 2026, UTC).

## Program

`main.py` — Python 3.13, standard library only, no network, fully
deterministic. It reads every `*.in` case from `$MAKEFUN_INPUT_DIR`
(or `/input/`) and writes one `<id>.out` per case to
`$MAKEFUN_OUTPUT_DIR` (or `/output/`), each with exactly 23 rows
(hours 2–24) of five quantiles (0.05 / 0.25 / 0.5 / 0.75 / 0.95).

## Method

1. **Comparable launch curves.** For each of the 38–42 comparable
   launches, align its hourly series on its own release day and measure
   the log-ratio curve against a blended anchor: the hour-1 print
   averaged in log space with the mean log viewers of the 6 hours
   before release day (weight 0.75 on hour 1). Blending tames the
   single-hour noise and the size effect — big hour-1 prints grow
   less in log terms over the day.
2. **Weighted quantiles.** Per forecast hour, take the weighted
   distribution of log-ratios across comparables. Clean multi-platform
   AAA launches weigh most; early-access-contaminated and free-to-play
   launches weigh less; when the target unlocks on a rolling midnight,
   rolling-midnight comparables weigh 3x. The target's own category
   history (Grand Theft Auto VI itself) is never used as a comparable.
3. **Calibrated spread.** Each quantile blends the empirical weighted
   quantile with a normal-in-log quantile (median ± z · IQR/1.349),
   then shrinks the distance from the median by 0.70.
4. **Fallbacks.** Minute-level history (the final case) is aggregated
   into hourly means before use; missing comparables fall back to the
   persistence-baseline shape. `streams` and `categories_hourly` are
   accepted whether null or populated.

## Local verification

- Runs on all four practice cases in ~0.1 s each; outputs are
  well-formed (23 rows, non-decreasing quantiles, ≤ 1,600 bytes) and
  byte-identical across repeated runs.
- Comparable hold-out evaluation (train on 37 launches, predict a held
  out rolling AAA launch from its hour-1 print only): **mean pinball
  loss 0.290 vs 0.406 for the persistence baseline** (8 targets;
  median 0.24 vs 0.26). The model beats the baseline on 5 of 8
  targets and loses narrowly on 3 flat-curve launches.

## License

MIT. See [LICENSE](LICENSE).
