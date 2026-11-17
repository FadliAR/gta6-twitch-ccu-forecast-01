#!/usr/bin/env python3
"""Size-conditioned launch-curve forecaster for Twitch CCU cases.

Each case describes one game's launch day with the hour-1 viewer print
already known; the job is to forecast hourly viewer quantiles for
hours 2..24 at the levels 0.05 / 0.25 / 0.5 / 0.75 / 0.95.

Method. Every game is summarised by a blended launch anchor: mostly
its hour-1 print, partly the mean log viewers of its last pre-launch
hours. For each forecast hour, the pool of comparable launches gives
pairs (log anchor size, log growth ratio versus the comparable's own
anchor). Big launches sit near their audience from hour 1 and grow
little; small launches multiply during the day, so a single pooled
curve misprices both ends. The model therefore regresses the log
growth ratio on log anchor size per hour, shrinks and clips the
slope, smooths it across neighbouring hours, and reads the forecast
distribution off the regression residuals, re-centred to the target's
own size through the slope. Quantiles blend the empirical step
quantiles of the residuals with a normal-in-log component whose
sigma is a robust MAD estimate of the same residuals, and the final
(quantile - median) distances are shrunk by a spread factor.

Python 3.13 standard library only. Deterministic: no randomness, no
clock, no network, no state between cases.
"""

import json
import math
import os
import re
from datetime import datetime, timedelta
from statistics import NormalDist

TAUS = [0.05, 0.25, 0.5, 0.75, 0.95]
ZS = [NormalDist().inv_cdf(t) for t in TAUS]

ALPHA = 0.75          # anchor weight on the hour-1 print (log space)
PRE_HOURS = 6         # pre-launch hours feeding the anchor blend
PRE_MIN = 3           # minimum pre-launch points for the blend
TIER_WEIGHTS = {
    'comparable': 1.0,
    'early-access-contaminated': 0.4,
    'weak': 0.25,
}
ROLL_BOOST = 2.5      # weight multiplier for rolling comparables
MIN_VALID_HOURS = 20  # launch-day hours a comparable must show
REG_SHRINK = 0.40     # shrinkage applied to the per-hour size slope
SLOPE_MIN = -1.2
SLOPE_MAX = 0.3
BLEND = 0.5           # empirical vs normal-in-log mixture weight
SPREAD = 0.70         # final shrink of (quantile - median), log space
SIGMA_FLOOR = 0.05

TIER_RE = re.compile(r'Tier:\s*([a-z\-]+)')


def parse_ts(text):
    return datetime.fromisoformat(text.replace('Z', '+00:00'))


def tier_of(unlock_text):
    """Research tier declared in a comparable's unlock notes."""
    m = TIER_RE.search(unlock_text or '')
    return m.group(1) if m else 'comparable'


def kind_of(text):
    """Unlock pattern of a launch: simultaneous global, rolling
    midnight local, or unknown."""
    low = (text or '').lower()
    if 'simultaneous' in low or 'single global' in low:
        return 'sim'
    if 'rolling' in low or 'midnight local' in low:
        return 'roll'
    return 'unk'


def blended_anchor(hour1_viewers, pre_viewers):
    """Launch anchor in viewers: ALPHA on the hour-1 print and the
    rest on the mean log of the pre-launch hours, when at least
    PRE_MIN of them exist; otherwise the plain hour-1 print."""
    p = max(float(hour1_viewers), 1.0)
    pre = [float(v) for v in pre_viewers if v and float(v) > 0]
    if len(pre) >= PRE_MIN:
        mean_log = sum(math.log(v) for v in pre) / len(pre)
        return math.exp(ALPHA * math.log(p) + (1.0 - ALPHA) * mean_log)
    return p


def weighted_quantile(pairs, tau):
    """Step quantile of (value, weight) pairs: the first value whose
    cumulative weight reaches tau * total weight."""
    ordered = sorted(pairs)
    total = sum(w for _, w in ordered)
    if total <= 0:
        return ordered[0][0]
    threshold = tau * total
    cum = 0.0
    for value, w in ordered:
        cum += w
        if cum >= threshold:
            return value
    return ordered[-1][0]


def prepare_comparable(comp, target_kind):
    """Pool entry for one comparable launch, or None when it cannot
    be used (no hour-1 print, or too few valid launch-day hours)."""
    try:
        ds = parse_ts(comp['day_start'])
    except Exception:
        return None
    rows = {}
    for row in comp.get('hourly') or []:
        try:
            rows[parse_ts(row['from'])] = float(row['viewers'])
        except Exception:
            continue
    hour1 = rows.get(ds)
    if not hour1 or hour1 <= 0:
        return None
    valid = sum(1 for k in range(24) if ds + timedelta(hours=k) in rows)
    if valid < MIN_VALID_HOURS:
        return None
    pre = [rows[ds - timedelta(hours=k)] for k in range(PRE_HOURS, 0, -1)
           if ds - timedelta(hours=k) in rows]
    anchor = blended_anchor(hour1, pre)
    unlock = comp.get('unlock', '')
    weight = TIER_WEIGHTS.get(tier_of(unlock), 1.0)
    if target_kind == 'roll' and kind_of(unlock) == 'roll':
        weight *= ROLL_BOOST
    return {
        'day_start': ds,
        'rows': rows,
        'log_anchor': math.log(anchor),
        'weight': weight,
    }


def persistence_quantiles(p1, hour):
    """Fallback for an hour with no pool data: the hour-1 print with
    a slowly widening log-normal band."""
    base = max(float(p1), 1.0)
    return [base * math.exp(z * 0.15 * math.sqrt(hour - 1)) for z in ZS]


def clean_quantiles(qs):
    """Enforce the output contract: finite, positive, capped, and
    non-decreasing across the five levels."""
    out = []
    prev = 0.0
    for q in qs:
        if not math.isfinite(q) or q <= 0:
            q = prev if prev > 0 else 1.0
        q = round(min(max(q, 0.01), 1e9), 2)
        if q < prev:
            q = prev
        out.append(q)
        prev = q
    return out


def forecast_case(case):
    day_start = parse_ts(case['day_start'])
    known = case.get('known') or []
    p1 = float(known[0]['viewers']) if known else 1.0

    # Target anchor: hour-1 print blended with the pre-launch hours
    # that precede day_start in the case's hourly history.
    pre_rows = []
    for row in case.get('history_hourly') or []:
        try:
            t = parse_ts(row['from'])
        except Exception:
            continue
        if t < day_start and row.get('viewers'):
            pre_rows.append((t, float(row['viewers'])))
    pre_rows.sort()
    anchor_t = blended_anchor(p1, [v for _, v in pre_rows[-PRE_HOURS:]])
    log_anchor_t = math.log(anchor_t)
    target_kind = kind_of(case.get('notes', ''))

    pool = []
    for comp in case.get('comparables') or []:
        entry = prepare_comparable(comp, target_kind)
        if entry is not None:
            pool.append(entry)

    # Pass 1: per-hour (size, log growth ratio, weight) pairs and the
    # shrunk, clipped size slope of each hour, smoothed over its
    # neighbours.
    hour_pairs = {}
    slopes = {}
    for h in range(2, 25):
        pairs = []
        for e in pool:
            v = e['rows'].get(e['day_start'] + timedelta(hours=h - 1))
            if v and v > 0:
                pairs.append((e['log_anchor'],
                              math.log(v) - e['log_anchor'],
                              e['weight']))
        if not pairs:
            continue
        hour_pairs[h] = pairs
        wsum = sum(w for _, _, w in pairs)
        xbar = sum(w * x for x, _, w in pairs) / wsum
        ybar = sum(w * y for _, y, w in pairs) / wsum
        sxx = sum(w * (x - xbar) ** 2 for x, _, w in pairs)
        sxy = sum(w * (x - xbar) * (y - ybar) for x, y, w in pairs)
        raw = (sxy / sxx) if sxx > 0 else 0.0
        slopes[h] = min(max(REG_SHRINK * raw, SLOPE_MIN), SLOPE_MAX)
    smooth = {}
    for h, b in slopes.items():
        neigh = [slopes[k] for k in (h - 1, h, h + 1) if k in slopes]
        smooth[h] = sum(neigh) / len(neigh)

    # Pass 2: quantiles from the size-regression residuals, shifted
    # to the target's size, blended with a MAD-sigma normal component.
    out_hours = []
    for h in range(2, 25):
        pairs = hour_pairs.get(h)
        if not pairs:
            out_hours.append({'hour': h,
                              'q': clean_quantiles(persistence_quantiles(p1, h))})
            continue
        b = smooth[h]
        wsum = sum(w for _, _, w in pairs)
        xbar = sum(w * x for x, _, w in pairs) / wsum
        resid = [(y - b * (x - xbar), w) for x, y, w in pairs]
        med = weighted_quantile(resid, 0.5)
        dev = [(abs(r - med), w) for r, w in resid]
        sigma = max(1.4826 * weighted_quantile(dev, 0.5), SIGMA_FLOOR)
        shift = b * (log_anchor_t - xbar)
        centre = log_anchor_t + shift + med
        logq = []
        for tau, z in zip(TAUS, ZS):
            emp = weighted_quantile(resid, tau)
            norm = med + z * sigma
            blended = BLEND * emp + (1.0 - BLEND) * norm
            logq.append(centre + SPREAD * (blended - med))
        out_hours.append({'hour': h,
                          'q': clean_quantiles([math.exp(v) for v in logq])})
    return {'hours': out_hours}


def persistence_case(case):
    """Whole-case fallback: persistence quantiles for every hour,
    built only from the hour-1 print."""
    known = case.get('known') or []
    p1 = float(known[0]['viewers']) if known else 1.0
    return {'hours': [{'hour': h,
                       'q': clean_quantiles(persistence_quantiles(p1, h))}
                      for h in range(2, 25)]}


def main():
    in_dir = os.environ.get('MAKEFUN_INPUT_DIR', '/input')
    out_dir = os.environ.get('MAKEFUN_OUTPUT_DIR', '/output')
    os.makedirs(out_dir, exist_ok=True)
    names = sorted(n for n in os.listdir(in_dir) if n.endswith('.in'))
    for name in names:
        with open(os.path.join(in_dir, name), 'r', encoding='utf-8') as fh:
            case = json.load(fh)
        # Guard the run: an unexpected input shape must degrade to
        # the persistence forecast, never crash — a crashed run makes
        # the case invalid, a degraded one still scores.
        try:
            result = forecast_case(case)
        except Exception:
            result = persistence_case(case)
        out_name = name[:-len('.in')] + '.out'
        with open(os.path.join(out_dir, out_name), 'w', encoding='utf-8') as fh:
            json.dump(result, fh)


if __name__ == '__main__':
    main()
