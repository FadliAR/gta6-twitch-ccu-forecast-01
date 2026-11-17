#!/usr/bin/env python3
"""Forecast a game's launch-day Twitch viewers for hours 2-24.

The forecast is a set of five quantiles per hour, built by transferring
launch curves from comparable game launches to the target game.

Method, in outline (everything is in log space):

* Every launch, the target included, is summarised by a blended anchor:
  mostly its hour-1 viewer print, partly the mean log viewers of the
  six hours before launch.  Working with ratios to that anchor removes
  the huge scale differences between launches.
* A comparable contributes, for each launch hour, the log ratio between
  its viewers that hour and its own anchor, weighted by how trustworthy
  its launch pattern is (clean comparable, early-access contaminated,
  or weak) and boosted when its unlock style matches a rolling-midnight
  target.
* The pooled log ratios are conditioned on launch size.  Measured on
  the pool, small launches multiply many-fold during the day while
  huge launches are already near their full audience at hour 1, so a
  single pooled curve over-forecasts big launches late in the day and
  under-forecasts small ones.  A per-hour weighted regression of the
  log ratio on the log anchor (heavily shrunk, then clipped) captures
  that tilt; the distribution for an hour is built from the regression
  residuals and re-centred on launches of the target's size.
* Each hour's quantiles blend the empirical weighted distribution of
  the residuals with a normal approximation whose sigma also comes
  from the residuals, and deviations from the median are shrunk by a
  spread factor before the size shift is applied.

Only the Python standard library is used, and the program is a pure
function of its input files, so repeated runs are byte-identical.
"""

import json
import math
import os
import re
import sys
from datetime import datetime, timedelta

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z_OF_LEVEL = [-1.645, -0.674, 0.0, 0.674, 1.645]

ANCHOR_ALPHA = 0.75      # hour-1 print share of the blended anchor
PRE_WINDOW = 6           # pre-launch hours feeding the anchor
PRE_MIN = 3              # minimum pre-launch points for the blend
MIN_DAY_HOURS = 20       # a comparable needs this many launch-day hours
TIER_WEIGHT = {"comparable": 1.0, "early-access-contaminated": 0.4, "weak": 0.25}
ROLL_BOOST = 3.0         # weight boost for rolling launches, rolling target
REG_SHRINK = 0.40        # shrinkage applied to the raw size slope
SLOPE_LO = -1.2          # clip bounds for the applied slope
SLOPE_HI = 0.3
SIGMA_FLOOR = 0.05       # smallest normal-approximation sigma
BLEND = 0.5              # empirical vs normal mix for each quantile
SPREAD = 0.65            # shrink of (quantile - median) in log space
MAX_VIEWERS = 1_000_000_000.0


def parse_time(text):
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


_TIER_RE = re.compile(r"Tier:\s*([a-z\-]+)")


def tier_of(unlock_text):
    m = _TIER_RE.search(unlock_text or "")
    return m.group(1) if m else "comparable"


def kind_of(text):
    low = (text or "").lower()
    if "simultaneous" in low or "single global" in low:
        return "sim"
    if "rolling" in low or "midnight local" in low:
        return "roll"
    return "unk"


def blended_anchor(hour1_viewers, pre_viewers):
    """Anchor in viewers: alpha on the hour-1 print, the rest on the
    mean log of the pre-launch hours, when enough of them are known."""
    hour1 = max(float(hour1_viewers), 1.0)
    if len(pre_viewers) >= PRE_MIN:
        mean_log_pre = sum(math.log(v) for v in pre_viewers) / len(pre_viewers)
        return math.exp(ANCHOR_ALPHA * math.log(hour1)
                        + (1.0 - ANCHOR_ALPHA) * mean_log_pre)
    return hour1


def prepare_comparable(comp, target_kind):
    """Reduce one comparable to its launch-day curve and anchor.

    Returns None when the comparable cannot be used: no hour-1 row, or
    too few launch-day hours to describe a curve."""
    day_start = parse_time(comp["day_start"])
    by_offset = {}
    for row in comp.get("hourly") or []:
        viewers = row.get("viewers")
        if viewers is None or viewers <= 0:
            continue
        offset = (parse_time(row["from"]) - day_start).total_seconds() / 3600.0
        if offset != int(offset):
            continue
        by_offset[int(offset)] = float(viewers)
    day = {o: v for o, v in by_offset.items() if 0 <= o <= 23}
    if 0 not in day or len(day) < MIN_DAY_HOURS:
        return None
    pre = [by_offset[o] for o in range(-PRE_WINDOW, 0) if o in by_offset]
    anchor = blended_anchor(day[0], pre)
    weight = TIER_WEIGHT.get(tier_of(comp.get("unlock", "")), 1.0)
    if target_kind == "roll" and kind_of(comp.get("unlock", "")) == "roll":
        weight *= ROLL_BOOST
    return {"anchor_log": math.log(anchor), "day": day, "weight": weight}


def target_anchor_log(case):
    day_start = parse_time(case["day_start"])
    hour1 = case["known"][0]["viewers"]
    by_time = {}
    for row in case.get("history_hourly") or []:
        if row.get("viewers"):
            by_time[parse_time(row["from"])] = float(row["viewers"])
    pre = []
    for back in range(PRE_WINDOW, 0, -1):
        v = by_time.get(day_start - timedelta(hours=back))
        if v is not None and v > 0:
            pre.append(v)
    return math.log(blended_anchor(hour1, pre))


def weighted_quantile(sorted_pairs, tau):
    """Step quantile of (value, weight) pairs sorted by value: the
    first value whose cumulative weight reaches tau of the total."""
    total = sum(w for _, w in sorted_pairs)
    if total <= 0:
        return sorted_pairs[0][0]
    threshold = tau * total
    acc = 0.0
    for value, w in sorted_pairs:
        acc += w
        if acc >= threshold:
            return value
    return sorted_pairs[-1][0]


def hour_pairs(pool, hour):
    """(log ratio, log anchor, weight) for comparables present at hour."""
    offset = hour - 1
    pairs = []
    for comp in pool:
        if offset in comp["day"]:
            ratio = math.log(comp["day"][offset]) - comp["anchor_log"]
            pairs.append((ratio, comp["anchor_log"], comp["weight"]))
    return pairs


def applied_slope(pairs):
    """Shrunk, clipped weighted slope of log ratio on log anchor."""
    total_w = sum(w for _, _, w in pairs)
    if total_w <= 0:
        return 0.0
    x_bar = sum(w * x for _, x, w in pairs) / total_w
    y_bar = sum(w * v for v, _, w in pairs) / total_w
    num = sum(w * (x - x_bar) * (v - y_bar) for v, x, w in pairs)
    den = sum(w * (x - x_bar) ** 2 for _, x, w in pairs)
    if den <= 0:
        return 0.0
    slope = REG_SHRINK * (num / den)
    return max(SLOPE_LO, min(SLOPE_HI, slope))


def hour_log_quantiles(pairs, slope, tgt_anchor_log):
    """The hour's five log-viewer quantiles for the target."""
    total_w = sum(w for _, _, w in pairs)
    x_bar = sum(w * x for _, x, w in pairs) / total_w
    residuals = sorted((v - slope * (x - x_bar), w) for v, x, w in pairs)
    median = weighted_quantile(residuals, 0.5)
    q25 = weighted_quantile(residuals, 0.25)
    q75 = weighted_quantile(residuals, 0.75)
    sigma = max((q75 - q25) / 1.349, SIGMA_FLOOR)
    shift = slope * (tgt_anchor_log - x_bar)
    out = []
    for tau, z in zip(LEVELS, Z_OF_LEVEL):
        empirical = weighted_quantile(residuals, tau)
        normal = median + z * sigma
        q = BLEND * empirical + (1.0 - BLEND) * normal
        q = median + SPREAD * (q - median)
        out.append(tgt_anchor_log + q + shift)
    return out


def fallback_log_quantiles(case, hour):
    """Persistence fallback for an hour with no comparable data: the
    hour-1 print with a slowly widening log spread."""
    p = max(float(case["known"][0]["viewers"]), 1.0)
    width = 0.15 * math.sqrt(hour - 1)
    return [math.log(p) + z * width for z in Z_OF_LEVEL]


def clean_quantiles(log_values):
    """Exponentiate, round, and enforce 0 < q1 <= ... <= q5 <= 1e9."""
    values = []
    for lv in log_values:
        v = round(math.exp(lv), 2)
        v = min(max(v, 0.01), MAX_VIEWERS)
        if values and v < values[-1]:
            v = values[-1]
        values.append(v)
    return values


def forecast_case(case):
    target_kind = kind_of(case.get("notes", ""))
    pool = []
    for comp in case.get("comparables") or []:
        prepared = prepare_comparable(comp, target_kind)
        if prepared is not None:
            pool.append(prepared)
    tgt_anchor_log = target_anchor_log(case)

    hours = sorted(fh["hour"] for fh in case.get("forecast") or [])
    pairs_by_hour = {h: hour_pairs(pool, h) for h in hours}

    # Pass 1: one applied slope per hour, smoothed over its neighbours.
    raw_slopes = {h: applied_slope(p) for h, p in pairs_by_hour.items() if p}
    slopes = {}
    for h in hours:
        near = [raw_slopes[n] for n in (h - 1, h, h + 1) if n in raw_slopes]
        slopes[h] = sum(near) / len(near) if near else 0.0

    # Pass 2: quantiles from the residuals of each hour's regression.
    rows = []
    for h in hours:
        pairs = pairs_by_hour[h]
        if pairs:
            log_q = hour_log_quantiles(pairs, slopes[h], tgt_anchor_log)
        else:
            log_q = fallback_log_quantiles(case, h)
        rows.append({"hour": h, "q": clean_quantiles(log_q)})
    return {"hours": rows}


def main():
    in_dir = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
    out_dir = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")
    os.makedirs(out_dir, exist_ok=True)
    for name in sorted(os.listdir(in_dir)):
        if not name.endswith(".in"):
            continue
        with open(os.path.join(in_dir, name), encoding="utf-8") as fh:
            case = json.load(fh)
        result = forecast_case(case)
        out_path = os.path.join(out_dir, name[:-3] + ".out")
        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump(result, fh, separators=(",", ":"))


if __name__ == "__main__":
    sys.exit(main())
