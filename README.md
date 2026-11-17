# Size-Conditioned Launch Curve Forecaster

[![Competition](https://make.fun/badge/gta6-twitch-ccu-forecast-01.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01)
[![Rank](https://make.fun/badge/gta6-twitch-ccu-forecast-01/FadliAR/gta6-twitch-ccu-forecast-01/rank.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01)
[![Reward](https://make.fun/badge/gta6-twitch-ccu-forecast-01/FadliAR/gta6-twitch-ccu-forecast-01/reward.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01)

Entry for the make.fun competition
[gta6-twitch-ccu-forecast-01](https://make.fun/bounty/gta6-twitch-ccu-forecast-01):
forecast Grand Theft Auto VI's launch-day Twitch viewers, hour by hour,
as five quantiles (q05, q25, q50, q75, q95) for each of hours 2-24.

## How it works

`main.py` (Python 3, standard library only) reads each case from
`$MAKEFUN_INPUT_DIR` and writes one output file per case to
`$MAKEFUN_OUTPUT_DIR`. The model transfers launch curves from the
case's comparable launches to the target:

- Every launch is summarised by a blended anchor: 75% of its hour-1
  viewer print plus 25% of the mean log viewers of the six pre-launch
  hours. All curves are compared as log ratios to their own anchor.
- Comparables are weighted by launch-pattern tier (clean comparable
  1.0, early-access contaminated 0.4, weak 0.25), with a x3 boost for
  rolling-midnight launches when the target itself rolls out at
  midnight local time.
- The pooled ratios are conditioned on launch size. Small launches
  multiply many-fold during the day while huge launches start near
  their full audience, so a per-hour weighted regression of log ratio
  on log anchor (shrinkage 0.40, slope clipped to [-1.2, +0.3],
  smoothed across neighbouring hours) tilts the curve to the target's
  size. Each hour's distribution is built from the regression
  residuals and re-centred accordingly.
- Quantiles blend the empirical weighted distribution with a normal
  approximation whose sigma is the residuals' IQR / 1.349, and
  deviations from the median are shrunk by a 0.65 spread factor.

The program is a pure function of its input, so repeated runs are
byte-identical. On the local hold-out protocol (eight comparable
rolling-midnight launches scored through the real code path) it reaches
a mean pinball loss of 0.2458, against 0.4055 for the persistence
baseline.

## License

MIT — see `LICENSE`.
