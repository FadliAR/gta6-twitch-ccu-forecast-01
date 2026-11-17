# Size-Conditioned Launch Curve Forecaster

[![Competition](https://make.fun/badge/gta6-twitch-ccu-forecast-01.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01)
[![Rank](https://make.fun/badge/gta6-twitch-ccu-forecast-01/FadliAR/gta6-twitch-ccu-forecast-01/rank.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01)
[![Reward](https://make.fun/badge/gta6-twitch-ccu-forecast-01/FadliAR/gta6-twitch-ccu-forecast-01/reward.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01)

This repository is an entry for the make.fun competition
[gta6-twitch-ccu-forecast-01](https://make.fun/bounty/gta6-twitch-ccu-forecast-01).
The task is to forecast Grand Theft Auto VI's launch-day audience on Twitch,
hour by hour. For each hour from hour 2 through hour 24 the program emits five
quantiles of concurrent viewers — q05, q25, q50, q75 and q95 — so the forecast
is a full predictive distribution for the day, not just a point estimate.

## How it works

The forecaster lives in `main.py` (Python 3, standard library only). It reads
each case from `$MAKEFUN_INPUT_DIR` and writes one output file per case to
`$MAKEFUN_OUTPUT_DIR`. There is no training step and no external data: the
model transfers launch curves from comparable past launches onto the target.

Every launch — comparables and target alike — is first summarised by a blended
anchor: 75% of its hour-1 viewer print plus 25% of the mean log viewers over
the six pre-launch hours. Each launch's hourly curve is then expressed as log
ratios to its own anchor, which puts launches of very different sizes on a
common scale.

Comparables are weighted by launch-pattern tier: clean comparables count 1.0,
early-access contaminated launches 0.4, and weak matches 0.25. When the target
itself rolls out at midnight local time, comparables with the same
rolling-midnight pattern receive an additional ×2.5 boost.

The core of the model is a per-hour weighted regression of log ratio on log
anchor (shrinkage 0.40, slope clipped to [−1.2, +0.3], smoothed across
neighbouring hours). This tilts the transferred curve to the target's size,
because launch behaviour is strongly size-dependent: small launches multiply
many-fold over the day, while huge launches start close to their full audience.

Each hour's predictive distribution is built from the regression residuals,
re-centred on the regression line. The final quantiles blend the empirical
weighted residual distribution with a normal approximation whose sigma is a
robust MAD estimate of the residuals: `max(1.4826 × weighted median of
|r − weighted median(r)|, 0.05)`. Deviations from the median are scaled by a
0.70 spread factor before the quantiles are read off.

Version 3 changed three things relative to the previous entry: the MAD-based
sigma replaced an IQR-based estimate, the rolling-midnight boost was softened
from ×3 to ×2.5, and the spread factor moved from 0.65 to 0.70.

## Validation

The program is a pure function of its input, so repeated runs on the same case
are byte-identical. On the local hold-out protocol — eight comparable
rolling-midnight launches scored through the real code path — the model reaches
a mean pinball loss of 0.2404, against 0.4055 for the persistence baseline and
0.2458 for the previous version of this entry.

## License

MIT — see [LICENSE](LICENSE).
