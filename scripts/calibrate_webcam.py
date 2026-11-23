"""Recalibrate the liveness UNCERTAIN band on this scanner's own webcam data.

Step 1 of docs/domain_adaptation_plan.md (no change to model weights), approved by
the project owner on 2026-11-23 for a single-scanner calibration.

Reads data/domain_adaptation/manifest.csv written by Admin -> Test Mode ->
Collection mode, and uses the 5-frame average P(spoof) that the scanner itself
decides on. The new lower edge of the UNCERTAIN band is set BELOW the lowest score
any recorded attack reached (with a safety margin), so on the recorded data no
photo, phone or screen would be accepted as LIVE. It then reports how many real-face
frames would pass. Session 2 crops, if any, are only used as a held-out check.

Nothing is written unless you pass --write, and gray_zone_v2.json is never touched:
the result goes to liveness/weights/gray_zone_v3.json, which the app loads instead
of v2 when it exists. Delete gray_zone_v3.json to go back to v2.

    python scripts/calibrate_webcam.py            # report only
    python scripts/calibrate_webcam.py --write    # report, then save gray_zone_v3.json
"""
import argparse
import csv
import json
import os
import sys
from datetime import datetime, timezone

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MANIFEST = os.path.join(ROOT, "data", "domain_adaptation", "manifest.csv")
V2 = os.path.join(ROOT, "liveness", "weights", "gray_zone_v2.json")
V3 = os.path.join(ROOT, "liveness", "weights", "gray_zone_v3.json")

MIN_REAL = 40        # real-face crops needed in session 1
MIN_ATTACK = 40      # attack crops needed in session 1
SAFETY = 0.8         # band_low = SAFETY x lowest attack score seen


def load(session):
    rows = []
    with open(MANIFEST, newline="") as fh:
        for r in csv.DictReader(fh):
            if r["session"] != str(session) or not r["p_spoof_avg"]:
                continue
            rows.append((r["label"], r["bona_fide"] == "1", float(r["p_spoof_avg"]), r["person_id"]))
    return rows


def rates(rows, band_low, band_high, threshold):
    def decide(p):
        if band_low <= p <= band_high:
            return "UNCERTAIN"
        return "SPOOF" if p >= threshold else "LIVE"
    real = [decide(p) for _, bona, p, _ in rows if bona]
    att = [decide(p) for _, bona, p, _ in rows if not bona]
    return {
        "real_n": len(real), "real_live": real.count("LIVE"),
        "real_pass_rate": real.count("LIVE") / len(real) if real else None,
        "attack_n": len(att), "attack_accepted": att.count("LIVE"),
        "attack_accept_rate": att.count("LIVE") / len(att) if att else None,
    }


def pct(x):
    return "n/a" if x is None else f"{x * 100:.1f}%"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true", help="save liveness/weights/gray_zone_v3.json")
    args = ap.parse_args()

    if not os.path.isfile(MANIFEST):
        sys.exit("No collected data yet. Record in Admin -> Test Mode -> Collection mode first.")
    v2 = json.load(open(V2))
    threshold, band_high, old_low = v2["threshold"], v2["band_high"], v2["band_low"]

    s1 = load(1)
    real = sorted(p for _, bona, p, _ in s1 if bona)
    attacks = sorted(p for _, bona, p, _ in s1 if not bona)
    print(f"Session 1: {len(real)} real-face crops, {len(attacks)} attack crops "
          f"({len({pid for *_, pid in s1})} person ID(s)).")
    if len(real) < MIN_REAL or len(attacks) < MIN_ATTACK:
        sys.exit(f"Need at least {MIN_REAL} real and {MIN_ATTACK} attack crops in session 1. Record more first.")

    by_label = {}
    for label, _, p, _ in s1:
        by_label.setdefault(label, []).append(p)
    print("\nP(spoof) (5-frame average) per label, session 1:")
    for label, ps in sorted(by_label.items()):
        ps.sort()
        print(f"  {label:18s} n={len(ps):4d}  min={ps[0]:.4f}  median={ps[len(ps)//2]:.4f}  max={ps[-1]:.4f}")

    new_low = min(SAFETY * attacks[0], threshold)
    print(f"\nLowest attack score: {attacks[0]:.4f}  ->  proposed band_low = {SAFETY} x that = {new_low:.4f}")
    print(f"(current band_low from v2 = {old_low:.4f}; threshold {threshold}, band_high {band_high:.4f} unchanged)")

    before, after = rates(s1, old_low, band_high, threshold), rates(s1, new_low, band_high, threshold)
    print("\nOn session 1 (the data the cut-off was chosen from):")
    print(f"  real faces accepted as LIVE:  v2 {pct(before['real_pass_rate'])}  ->  v3 {pct(after['real_pass_rate'])}")
    print(f"  attacks accepted as LIVE:     v2 {pct(before['attack_accept_rate'])}  ->  v3 {pct(after['attack_accept_rate'])}")

    s2 = load(2)
    held = None
    if s2:
        held = {"v2": rates(s2, old_low, band_high, threshold), "v3": rates(s2, new_low, band_high, threshold)}
        print(f"\nHeld-out session 2 ({len(s2)} crops, never used to choose the cut-off):")
        print(f"  real faces accepted as LIVE:  v2 {pct(held['v2']['real_pass_rate'])}  ->  v3 {pct(held['v3']['real_pass_rate'])}")
        print(f"  attacks accepted as LIVE:     v2 {pct(held['v2']['attack_accept_rate'])}  ->  v3 {pct(held['v3']['attack_accept_rate'])}")
    else:
        print("\nNo session 2 data: these numbers come from the same recordings the cut-off was chosen on, so real-world"
              " attack acceptance can be higher. Record a session 2 (another day, other photos) to check.")

    if after["real_pass_rate"] is not None and after["real_pass_rate"] < 0.5:
        print("\nWarning: real faces and attacks overlap a lot on this webcam; recalibration alone helps little. "
              "Step 2 of the plan (fine-tuning) is the next option.")

    if not args.write:
        print("\nReport only. Run again with --write to save gray_zone_v3.json.")
        return
    out = dict(v2)
    out.update({
        "band_low": new_low,
        "band_selected_on": "webcam collection session 1 (scripts/calibrate_webcam.py): "
                            f"band_low = {SAFETY} x lowest attack P(spoof) average",
        "calibrated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "previous_band_low_v2": old_low,
        "webcam_session1": after, "webcam_session1_before_v2": before,
        "webcam_session2_heldout": held,
        "people": sorted({pid for *_, pid in s1}),
    })
    with open(V3, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\nSaved {V3}. Restart the app to use it. Delete that file to go back to v2.")


if __name__ == "__main__":
    main()
