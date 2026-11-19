"""
generate_report.py
------------------
Builds SecureFace_Attendance_Report.pdf describing the system as it is now.

Every number in the report is read at generation time from a real file:
  - training/results/summary.json   (dataset results of the liveness model)
  - docs/pipeline_latency_2026-11-07.txt   (measured end-to-end latency)
Nothing numeric is typed into this script. If a file is missing, the section
says so instead of showing a number.

Usage:
    ./venv/bin/python generate_report.py
"""

import json
import os
from datetime import date
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (Image, PageBreak, Paragraph, Preformatted, SimpleDocTemplate, Spacer, Table,
                                TableStyle, KeepTogether)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SUMMARY_PATH = os.path.join(BASE_DIR, "training", "results", "summary.json")
LATENCY_PATH = os.path.join(BASE_DIR, "docs", "pipeline_latency_2026-11-07.txt")
FIGURES = {
    "training_curves": ("training/outputs/plots/v2_cropped/training_curves.png", "Training curves of the deployed v2 model."),
    "confusion_test": ("training/outputs/plots/v2_cropped/confusion_matrix_v2_cropped_test.png", "Confusion matrix, test set (v2)."),
    "gray_zone": ("training/results/plots/v2/gray_zone_coverage_vs_error.png", "UNCERTAIN band: coverage vs. error on the test set."),
}

ACCENT = colors.HexColor("#0f766e")


def pct(x):
    return "—" if x is None else f"{x * 100:.2f}%"


def styles():
    s = getSampleStyleSheet()
    return {
        "title": ParagraphStyle("title", parent=s["Title"], fontSize=26, leading=32, textColor=ACCENT, alignment=TA_CENTER),
        "subtitle": ParagraphStyle("subtitle", parent=s["Normal"], fontSize=13, leading=18, alignment=TA_CENTER,
                                   textColor=colors.HexColor("#334155")),
        "h1": ParagraphStyle("h1", parent=s["Heading1"], fontSize=16, textColor=ACCENT, spaceBefore=14, spaceAfter=8),
        "h2": ParagraphStyle("h2", parent=s["Heading2"], fontSize=12.5, textColor=colors.HexColor("#1e293b"),
                             spaceBefore=10, spaceAfter=5),
        "body": ParagraphStyle("body", parent=s["Normal"], fontSize=10.2, leading=14.5, alignment=TA_JUSTIFY, spaceAfter=6),
        "small": ParagraphStyle("small", parent=s["Normal"], fontSize=8.5, leading=11, textColor=colors.HexColor("#475569")),
        "cell": ParagraphStyle("cell", parent=s["Normal"], fontSize=8.8, leading=11),
        "mono": ParagraphStyle("mono", parent=s["Code"], fontSize=7.4, leading=9.5),
    }


def table(rows, col_widths, st, header=True):
    data = [[c if isinstance(c, Paragraph) else Paragraph(escape(str(c)), st["cell"]) for c in r] for r in rows]
    t = Table(data, colWidths=col_widths, repeatRows=1 if header else 0)
    style = [
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#cbd5e1")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 4), ("RIGHTPADDING", (0, 0), (-1, -1), 4),
    ]
    if header:
        style += [("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e2e8f0"))]
    t.setStyle(TableStyle(style))
    return t


def figure(key, st, width=15 * cm):
    rel, caption = FIGURES[key]
    path = os.path.join(BASE_DIR, rel)
    if not os.path.isfile(path):
        return [Paragraph(f"<i>Figure not found: {escape(rel)}</i>", st["small"])]
    img = Image(path)
    ratio = img.imageHeight / float(img.imageWidth)
    img.drawWidth, img.drawHeight = width, width * ratio
    return [KeepTogether([img, Paragraph(f"{escape(caption)} Source: <font face='Courier'>{escape(rel)}</font>", st["small"]),
                          Spacer(1, 8)])]


def create_pdf(filename="SecureFace_Attendance_Report.pdf"):
    st = styles()
    summary = None
    if os.path.isfile(SUMMARY_PATH):
        with open(SUMMARY_PATH) as fh:
            summary = json.load(fh)
    latency_text = open(LATENCY_PATH).read() if os.path.isfile(LATENCY_PATH) else None

    doc = SimpleDocTemplate(os.path.join(BASE_DIR, filename), pagesize=A4,
                            leftMargin=2 * cm, rightMargin=2 * cm, topMargin=1.8 * cm, bottomMargin=1.8 * cm,
                            title="SecureFace Attendance: Project Report")
    W = A4[0] - 4 * cm
    story = []
    P = lambda text, style="body": story.append(Paragraph(text, st[style]))

    # ---------------- cover ----------------
    story += [Spacer(1, 5 * cm), Paragraph("SecureFace Attendance", st["title"]), Spacer(1, 10),
              Paragraph("Face-recognition attendance with a trained liveness (anti-spoofing) model", st["subtitle"]),
              Spacer(1, 2.5 * cm),
              Paragraph("Python 3.13 · Flask · PyTorch (MobileNetV3-Small) · MediaPipe · ONNX Runtime (ArcFace)", st["subtitle"]),
              Spacer(1, 10), Paragraph(f"Report generated {date.today().isoformat()}", st["subtitle"]), PageBreak()]

    # ---------------- 1. overview ----------------
    P("1. What the system does", "h1")
    P("SecureFace Attendance marks student attendance with a webcam. Before it checks <i>who</i> a person is, "
      "it checks that a real, live person is in front of the camera and not a printed photo, a phone or a screen. "
      "That check is done by a liveness model trained for this project. Recognition only runs on faces the model "
      "judges LIVE.")
    P("The system has three parts:")
    story.append(table([
        ["Part", "Who uses it", "What it does"],
        ["Common scanner", "Anyone, on an admin-approved device only",
         "Camera -> liveness -> recognition -> natural blink -> attendance. Shows name, student ID and "
         "\"Attendance marked\" or \"Already marked today\"."],
        ["Student pages", "Logged-in students", "Create an account, register their face once, view their own attendance."],
        ["Admin pages", "Logged-in admins",
         "All attendance, students (including Reset face), settings, scanner devices, scan and spoof logs, "
         "Results, Test Mode, Pipeline View with Grad-CAM."],
    ], [3.2 * cm, 4.3 * cm, W - 7.5 * cm], st))

    # ---------------- 2. pipeline ----------------
    P("2. The scan pipeline", "h1")
    P("One request per camera frame to <font face='Courier'>/api/recognize_face</font>. Each stage can stop the request:")
    steps = [
        ["1", "Device check", "The request must come from a browser an admin approved as a scanner (random token in a cookie; only its SHA-256 hash is stored). Otherwise: \"This device is not an approved scanner.\""],
        ["2", "Face detection", "BlazeFace. No face or more than one face stops here."],
        ["3", "Same-face tracking", "IoU between this frame's face box and the previous frame's. Below 0.25 counts as a different person and resets the session state."],
        ["4", "Crop", "Face box plus 25% margin, resized to 224×224 with ImageNet normalization, byte-identical to the training crop (checked by tests/test_crop_identical.py)."],
        ["5", "Liveness model", "MobileNetV3-Small gives P(spoof) for the frame; the decision uses the average of the last 5 frames for this browser session."],
        ["6", "Three-way decision", "Inside the UNCERTAIN band -> UNCERTAIN (retry); otherwise SPOOF if >= the admin threshold, else LIVE. SPOOF and UNCERTAIN stop here, so recognition never runs on them."],
        ["7", "Recognition", "MediaPipe Face Mesh aligns the face; ArcFace (w600k_r50) gives a 512-number embedding, compared by cosine similarity with the registered students."],
        ["8", "Natural blink", "The server checks its own Face Mesh eye readings for open -> closed -> open eyes of the same student within 8 s. A blink signal sent by the browser is ignored."],
        ["9", "Attendance", "Present or Late (admin cut-off time), once per day. Every stage reached is logged with the frame score, the 5-frame average and the IoU."],
    ]
    story.append(table([["#", "Stage", "What happens"]] + steps, [0.7 * cm, 3.2 * cm, W - 3.9 * cm], st))

    # ---------------- 3. models ----------------
    P("3. Models", "h1")
    story.append(table([
        ["Model", "Job", "Source"],
        ["BlazeFace (short range)", "Find faces", "Pretrained (Google MediaPipe)"],
        ["MobileNetV3-Small + small head (1 logit)", "Real face or spoof", "Trained for this project (training/)"],
        ["MediaPipe Face Landmarker (478 points)", "Alignment and eye openness for the blink", "Pretrained (Google MediaPipe)"],
        ["ArcFace w600k_r50 (ONNX)", "Identity embedding", "Pretrained (InsightFace buffalo_l)"],
    ], [5.5 * cm, 5 * cm, W - 10.5 * cm], st))
    P("<b>Why a trained model for anti-spoofing.</b> Hand-written rules (texture, screen-edge detection, depth guesses) "
      "depend on lighting, camera and attack type, and break when any of these change. A model trained on many "
      "labeled real and spoof images learns the differences instead. An earlier rule-based engine was removed "
      "completely and is not part of this system.")

    # ---------------- 4. results ----------------
    P("4. Liveness model results (dataset)", "h1")
    if not summary:
        P(f"<i>training/results/summary.json was not found, so no results are shown.</i>")
    else:
        h = summary["headline"]
        P(f"Test set: {h['n']} CelebA-Spoof images. Protocol: {escape(h['protocol'])}", "small")
        story.append(Spacer(1, 4))
        story.append(table([["Accuracy", "APCER", "BPCER", "HTER"],
                            [pct(h["accuracy"]), pct(h["apcer"]), pct(h["bpcer"]), pct(h["hter"])]],
                           [W / 4] * 4, st))
        P("APCER = attacks accepted as real; BPCER = real faces rejected; HTER = their average.", "small")

        vv = summary["v1_vs_v2"]
        P("The v1 -> v2 crop fix", "h2")
        P("v1 was trained on face crops framed differently from the ones the live pipeline produces. v2 was retrained on "
          "crops made exactly as the live pipeline makes them. On the same test set:")
        story.append(table([["Model", "Accuracy", "APCER", "BPCER", "HTER"]] +
                           [[name, pct(m["accuracy"]), pct(m["apcer"]), pct(m["bpcer"]), pct(m["hter"])]
                            for name, m in (("v1", vv["v1"]), ("v2 (deployed)", vv["v2"]))],
                           [4 * cm] + [(W - 4 * cm) / 4] * 4, st))

        bc = summary["backbone_comparison"]
        P("Backbone comparison", "h2")
        rows = [["Backbone", "Accuracy", "HTER", "Params", "Size (MB)", "CPU ms", "ONNX CPU ms"]]
        for b in ("mobilenet_v3_small", "efficientnet_b0", "resnet18"):
            m, l = bc[b], bc["latency_and_size"][b]
            rows.append([b + (" (deployed)" if b == bc["deployed_backbone"] else ""), pct(m["accuracy"]), pct(m["hter"]),
                         f"{l['param_count']:,}", f"{l['model_file_size_mb']:.2f}",
                         f"{l['latency_cpu_ms_median']:.2f}", f"{l['cpu_onnx_ms']:.2f}"])
        story.append(table(rows, [4.6 * cm] + [(W - 4.6 * cm) / 6] * 6, st))
        P("Single-image inference medians from the training project's benchmark, not full-pipeline times. "
          "On this test set EfficientNet-B0 scored higher; MobileNetV3-Small is the smallest model and the one deployed.",
          "small")

        gz = summary["gray_zone"]
        P("UNCERTAIN band", "h2")
        P(f"The band [{gz['band_low']:.4f}, {gz['band_high']:.4f}] was chosen on the validation set only. "
          f"On the test set it leaves {pct(1 - gz['test_gray_zone_decided_only']['coverage'])} of images UNCERTAIN, and "
          f"the HTER on the decided images is {pct(gz['test_gray_zone_decided_only']['hter'])}, compared with "
          f"{pct(gz['test_baseline_plain_threshold']['hter'])} with a plain 0.5 threshold.")

        me = summary["masked_eval"]
        P("Masked faces", "h2")
        P(f"With a synthetic surgical-mask overlay ({me['masked_metrics']['n']} test images where the face was still "
          f"detected): accuracy {pct(me['masked_metrics']['accuracy'])}, HTER {pct(me['masked_metrics']['hter'])}. "
          f"Same detection protocol without masks: HTER {pct(me['unmasked_baseline_metrics']['hter'])}. "
          "Liveness detection works with masks. Face recognition requires the mask to be removed.")

        for key in ("training_curves", "confusion_test", "gray_zone"):
            story += figure(key, st, width=13 * cm)

    # ---------------- 5. latency ----------------
    P("5. Measured pipeline latency", "h1")
    if latency_text:
        P("Output of <font face='Courier'>scripts/measure_pipeline_latency.py</font> against the running app "
          "(45 requests on real sample images), saved in <font face='Courier'>docs/pipeline_latency_2026-11-07.txt</font>:")
        story.append(Preformatted(latency_text.strip(), st["mono"]))
    else:
        P("<i>No saved latency measurement found.</i>")

    # ---------------- 6. admin tools ----------------
    P("6. Admin tools for checking the model", "h1")
    P("<b>Pipeline View</b> shows, for the current camera frame, the BlazeFace box and confidence, the crop, the exact "
      "224×224 model input, P(spoof) and the decision. <b>Explain</b> computes Grad-CAM on demand (not every frame) on the "
      "last convolutional block of MobileNetV3-Small, <font face='Courier'>feature_extractor.features[12]</font> "
      "(576 channels, 7×7), and overlays it on the crop: highlighted areas influenced the model's decision most. "
      "Nothing is saved to disk.")
    P("<b>Test Mode</b> records the scanner's real decisions while the admin shows the camera a real face, a masked face, "
      "a printed photo, a phone photo, a phone video or a laptop screen, and computes APCER, BPCER and HTER per category "
      "from those counts. It never runs recognition and never marks attendance. Its collection mode saves labeled crops "
      "for the domain-adaptation plan (docs/domain_adaptation_plan.md).")
    P("<b>Spoof Log</b> lists SPOOF/UNCERTAIN events with scores, filters and CSV export. <b>Results</b> shows the dataset "
      "numbers above (read from summary.json) and, separately, live Test Mode results.")

    # ---------------- 7. security ----------------
    P("7. Security and privacy", "h1")
    for item in [
        "CSRF protection on every state-changing request (Flask-WTF).",
        "One login page with a Student/Admin toggle; the account's real role is checked, so a wrong toggle gives \"Invalid credentials\".",
        "Only admin-approved scanner devices can mark attendance; approvals and removals are audit-logged.",
        "Face embeddings are stored encrypted (Fernet).",
        "Registration rejects samples that don't match the account's earlier samples or that match another registered person.",
        "Wiping all face data requires typing a confirmation word and is audit-logged with the admin's identity.",
        "No hardcoded credentials; admins are created with scripts/create_admin.py.",
    ]:
        P("• " + item)

    # ---------------- 8. testing ----------------
    P("8. Testing", "h1")
    P("<font face='Courier'>pytest</font> runs the crop-identity regression test plus app tests for login and role access "
      "(including a wrong role toggle), blocking students from admin pages, CSRF, scanner device approval, attendance "
      "marking and \"Already marked today\", the wipe confirmation, registration owner checks, and Test Mode never marking "
      "attendance. Tests use a temporary database, never the real one.")

    # ---------------- 9. limitations ----------------
    P("9. Known limitations", "h1")
    for item in [
        "<b>Domain shift.</b> The liveness model was trained on dataset photos. On a real webcam, a real face often lands "
        "just inside the UNCERTAIN band, and one phone photo was judged LIVE during testing. A data-collection tool and a "
        "plan to recalibrate (and only if needed fine-tune) are in place; no recalibration has been done yet.",
        "<b>Silent blink.</b> The scanner waits for a natural blink without a visible prompt. A video of the real person "
        "would also blink, so the blink is not a defense against video replay; the liveness model is the main defense.",
        "<b>Postgres.</b> The Postgres code path exists but has never been run against a live database.",
        "<b>Dataset numbers are not webcam numbers.</b> Section 4 describes the CelebA-Spoof test set only.",
    ]:
        P("• " + item)

    doc.build(story)
    print(f"Wrote {filename}")


if __name__ == "__main__":
    create_pdf()
