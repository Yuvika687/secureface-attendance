"""
build_summary.py
------------------
TASK K: builds results/summary.json, the single source of truth for
every number the rebuilt dashboard will show. This script does NOT run
any new inference -- it only reads and re-labels the existing result
JSON files this project's other scripts already produce, so it is
regenerable in one command any time those are updated:

    python build_summary.py

CANONICAL HEADLINE PROTOCOL
----------------------------
Every number under "headline", "v1_vs_v2", and "backbone_comparison"
below uses ONE protocol, so they are directly comparable to each other:

    Full 3,900-image CelebA-Spoof test set; deployment-style face crops
    (BlazeFace detection + 25% margin, falling back to the dataset's own
    official bounding box when BlazeFace misses -- see
    build_face_cropped_dataset.py); plain 0.5 decision threshold.

Every number OUTSIDE those three sections (gray zone, masked eval,
BlazeFace-only detection-rate figures) uses a DIFFERENT, explicitly
labeled protocol and n -- never silently compared to the headline
numbers. See CONSISTENCY_NOTES below for why two pairs of numbers for
"the same thing" have appeared in this project's reports, and which one
is canonical going forward.
"""

import json
import os

import config

CANONICAL_PROTOCOL = (
    "Full 3,900-image CelebA-Spoof test set; deployment-style face crops "
    "(BlazeFace detection + 25% margin, falling back to the dataset's own "
    "official bounding box when BlazeFace misses -- see "
    "build_face_cropped_dataset.py); plain 0.5 decision threshold. n=3900 "
    "for every model under this protocol (fallback crops mean no image is "
    "ever excluded)."
)

BLAZEFACE_ONLY_PROTOCOL = (
    "Same 3,900-image test set, but using ONLY live BlazeFace detection -- "
    "no bbox fallback. Images where BlazeFace finds no face get NO "
    "decision at all (matching true live-webcam behavior, where no "
    "official bounding box exists to fall back to). n < 3900; see "
    "'n' and 'coverage' fields for the exact count."
)

CONSISTENCY_NOTES = [
    {
        "numbers": "v1 deployment-crop accuracy reported as both 95.94% (TASK F) and 94.54% (TASK G v1-vs-v2 table)",
        "explanation": (
            "95.94% used BLAZEFACE_ONLY_PROTOCOL (n=3839/3900 -- the 61 images "
            "where live BlazeFace detection missed were excluded entirely, matching "
            "true deployment behavior where there is no fallback). 94.54% used "
            "CANONICAL_PROTOCOL (n=3900 -- those same 61 images got a crop via the "
            "dataset's own official bounding box, which only exists for this "
            "labeled benchmark, not for a live webcam frame). Both are real "
            "numbers computed correctly; they simply answer different questions. "
            "94.54% is the one reported under 'v1_vs_v2' below (canonical)."
        ),
    },
    {
        "numbers": "v2 accuracy reported as both 99.04% (masked-eval's unmasked baseline) and 98.69% (TASK G headline)",
        "explanation": (
            "Same root cause as above. 99.04% is BLAZEFACE_ONLY_PROTOCOL (n=3839), "
            "deliberately chosen as the masked-eval baseline because bbox fallback "
            "isn't a fair stand-in for a MASKED face -- the dataset's official "
            "bounding box was computed on the unmasked face, so using it when a "
            "mask causes a miss would hide exactly the Stage-1 effect masked-eval "
            "exists to measure. 98.69% is CANONICAL_PROTOCOL (n=3900) and is the "
            "number reported under 'headline' and 'backbone_comparison' below. "
            "The masked-eval section is intentionally NOT on the canonical "
            "protocol and is labeled as such."
        ),
    },
]


def load_json(path):
    if not os.path.isfile(path):
        return None
    with open(path) as f:
        return json.load(f)


def build_summary(results_dir: str, predictions_dir: str):
    """Returns (summary_dict, missing_source_names). Pure function, no I/O
    other than reading the source JSON files -- doesn't write anything,
    so it's directly testable without touching the real results/ dir."""
    v1_vs_v2 = load_json(os.path.join(results_dir, "v1_vs_v2.json"))
    if v1_vs_v2 is None:
        raise SystemExit("results/v1_vs_v2.json not found -- run evaluate_v1_vs_v2.py first.")

    resnet18_metrics = load_json(os.path.join(predictions_dir, "v2_cropped", "resnet18", "metrics_v2_resnet18_test.json"))
    efficientnet_metrics = load_json(os.path.join(predictions_dir, "v2_cropped", "efficientnet_b0", "metrics_v2_efficientnet_b0_test.json"))

    breakdown_mobilenet = load_json(os.path.join(results_dir, "breakdown_v2.json"))
    breakdown_resnet18 = load_json(os.path.join(results_dir, "breakdown_v2_resnet18.json"))
    breakdown_efficientnet = load_json(os.path.join(results_dir, "breakdown_v2_efficientnet_b0.json"))

    gray_zone_v2 = load_json(os.path.join(results_dir, "gray_zone_v2.json"))
    masked_eval = load_json(os.path.join(results_dir, "masked_eval.json"))
    deployment_crop_eval_v2 = load_json(os.path.join(results_dir, "deployment_crop_eval_v2.json"))
    crop_build_stats = load_json(os.path.join(results_dir, "crop_dataset_build_stats.json"))
    backbone_latency = load_json(os.path.join(results_dir, "backbone_latency.json"))  # TASK L, optional

    v2_metrics = v1_vs_v2["v2_metrics_on_deployment_crops"]
    v1_metrics = v1_vs_v2["v1_metrics_on_deployment_crops"]

    summary = {
        "canonical_protocol": CANONICAL_PROTOCOL,
        "blazeface_only_protocol": BLAZEFACE_ONLY_PROTOCOL,
        "consistency_notes": CONSISTENCY_NOTES,

        "headline": {
            "protocol": CANONICAL_PROTOCOL,
            "n": 3900,
            "active_model": "v2 (mobilenet_v3_small, face-crop-trained; see TASK G)",
            "accuracy": v2_metrics["accuracy"],
            "apcer": v2_metrics["apcer"],
            "bpcer": v2_metrics["bpcer"],
            "hter": v2_metrics["hter"],
        },

        "v1_vs_v2": {
            "protocol": CANONICAL_PROTOCOL,
            "n": 3900,
            "v1": {"checkpoint": v1_vs_v2["v1_checkpoint"], **v1_metrics},
            "v2": {"checkpoint": v1_vs_v2["v2_checkpoint"], **v2_metrics},
            "per_attack_type_apcer": {
                "v1": v1_vs_v2["v1_per_attack_type_apcer"],
                "v2": v1_vs_v2["v2_per_attack_type_apcer"],
            },
        },

        "backbone_comparison": {
            "protocol": CANONICAL_PROTOCOL,
            "n": 3900,
            "deployed_backbone": "mobilenet_v3_small",
            "mobilenet_v3_small": v2_metrics,
            "resnet18": resnet18_metrics,
            "efficientnet_b0": efficientnet_metrics,
            "latency_and_size": backbone_latency,  # TASK L; null until that script has run
        },

        "breakdowns": {
            "protocol": CANONICAL_PROTOCOL + " Per-attack-type/illumination/environment APCER.",
            "mobilenet_v3_small": breakdown_mobilenet,
            "resnet18": breakdown_resnet18,
            "efficientnet_b0": breakdown_efficientnet,
        },

        "gray_zone": {
            "protocol": "Band selected on the VALIDATION set only (never test); applied to the "
                        "canonical-protocol test set (n=3900). v2 (mobilenet_v3_small) only.",
            **(gray_zone_v2 or {}),
        },

        "masked_eval": {
            "protocol": "RAW test images (live+spoof) with a synthetic surgical-mask overlay "
                        "(nose-to-chin, white/light_blue/black), then BLAZEFACE-ONLY detection "
                        "(no bbox fallback -- see consistency_notes). The unmasked baseline here "
                        "uses the SAME BlazeFace-only methodology for a fair comparison, and is "
                        "NOT the canonical headline number.",
            "n_total_raw_images": masked_eval["n_total"] if masked_eval else None,
            "mask_synthesis_failures": masked_eval["mask_synthesis_failures"] if masked_eval else None,
            "stage1_on_masked": masked_eval["stage1_on_masked"] if masked_eval else None,
            "masked_metrics": masked_eval["stage2_masked_detected_subset"] if masked_eval else None,
            "unmasked_baseline_metrics": masked_eval["stage2_unmasked_baseline"] if masked_eval else None,
            "unmasked_baseline_protocol": BLAZEFACE_ONLY_PROTOCOL,
            "bpcer_flag_triggered": masked_eval["bpcer_live_masked_flag"] if masked_eval else None,
            "v3_retraining_estimate_if_flagged": masked_eval["v3_retraining_estimate_if_flagged"] if masked_eval else None,
            "per_spoof_type_apcer_masked": masked_eval["per_spoof_type_apcer_masked"] if masked_eval else None,
            "gray_zone_on_masked": masked_eval["gray_zone_masked"] if masked_eval else None,
        },

        "deployment_crop_eval_v2_blazeface_only": {
            "protocol": BLAZEFACE_ONLY_PROTOCOL,
            **(deployment_crop_eval_v2 or {}),
        },

        "crop_dataset_build_stats": crop_build_stats,
    }

    missing = [name for name, val in [
        ("v1_vs_v2.json", v1_vs_v2), ("resnet18 metrics", resnet18_metrics),
        ("efficientnet_b0 metrics", efficientnet_metrics), ("breakdown_v2.json", breakdown_mobilenet),
        ("breakdown_v2_resnet18.json", breakdown_resnet18), ("breakdown_v2_efficientnet_b0.json", breakdown_efficientnet),
        ("gray_zone_v2.json", gray_zone_v2), ("masked_eval.json", masked_eval),
        ("deployment_crop_eval_v2.json", deployment_crop_eval_v2),
    ] if val is None]
    if backbone_latency is None:
        missing.append("backbone_latency.json (TASK L, optional)")

    return summary, missing


def main():
    results_dir = os.path.join(config.PROJECT_ROOT, "results")
    predictions_dir = os.path.join(config.PROJECT_ROOT, "outputs", "predictions")

    summary, missing = build_summary(results_dir, predictions_dir)

    out_path = os.path.join(results_dir, "summary.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved summary to: {out_path}")

    if missing:
        print(f"NOTE: missing/optional source files, summary has null sections for: {missing}")


if __name__ == "__main__":
    main()
