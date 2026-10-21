// page-overview.js -- landing page, adapted from the Stitch design export
// in design/stitch/. Every number here is pulled live from /api/summary
// (results/summary.json) or from one real sample-image inference call --
// none are hardcoded mockup values. The Stitch mockup's own placeholder
// numbers (99.15% / 0.85% HTER / 26.7 FPS / a fabricated "LBP+SVM 61.2%"
// baseline) are not reproduced; see design/stitch/DESIGN.md for the
// original.
import { api } from "./api.js";
import { pct, ms, statusClass } from "./utils.js";

const PREVIEW_SAMPLE_ID = "live";

export async function initOverviewPage(root) {
  root.innerHTML = `
    <section class="ov-hero">
      <div class="ov-hero-copy">
        <div class="ov-eyebrow"><span class="ov-eyebrow-dot"></span>Mask-Aware Anti-Spoofing</div>
        <h1 class="ov-h1">Stops photo and screen-replay attacks before attendance &mdash; <span class="ov-accent">even with a mask on.</span></h1>
        <p class="ov-sub">Checks that a real, live person is physically in front of the camera before attendance is marked. Spoofs are rejected and logged in real time.</p>
        <div class="ov-ctas">
          <button class="btn btn-primary" data-nav="live">Try Live Detection &rarr;</button>
          <button class="btn" data-nav="results">See Full Results</button>
        </div>
        <div class="ov-specs mono" id="ov-specs">loading measured spec strip&hellip;</div>
      </div>

      <div class="ov-preview panel">
        <div class="ov-preview-head">
          <span class="ov-preview-label mono">SAMPLE DETECTION &middot; REAL MODEL INFERENCE, RUN NOW</span>
        </div>
        <div class="ov-preview-body" id="ov-preview-body">
          <div class="empty-state">running the real pipeline on a sample image&hellip;</div>
        </div>
      </div>
    </section>

    <section class="grid grid-3 ov-stats" id="ov-stats">
      <div class="kpi"><div class="figure">&hellip;</div><div class="label">loading</div></div>
      <div class="kpi"><div class="figure">&hellip;</div><div class="label">loading</div></div>
      <div class="kpi"><div class="figure">&hellip;</div><div class="label">loading</div></div>
    </section>

    <section class="panel ov-section">
      <h3>How a frame is judged</h3>
      <p class="page-sub">Four real stages run on every frame before an attendance record is written.</p>
      <div class="ov-pipeline">
        <div class="ov-stage">
          <div class="ov-stage-top"><span class="ov-stage-num mono">STAGE 01</span></div>
          <h4>Webcam Frame</h4>
          <div class="ov-stage-tag mono">Raw browser camera frame</div>
          <p>Captured via <span class="mono">getUserMedia</span>, JPEG-encoded, sent to the backend for every detection cycle.</p>
        </div>
        <div class="ov-stage">
          <div class="ov-stage-top"><span class="ov-stage-num mono">STAGE 02</span></div>
          <h4>Detect + Crop</h4>
          <div class="ov-stage-tag mono">MediaPipe BlazeFace</div>
          <p>Locates the face, then crops with a 25% margin to a 224&times;224 image &mdash; the exact crop the model was trained on.</p>
        </div>
        <div class="ov-stage">
          <div class="ov-stage-top"><span class="ov-stage-num mono">STAGE 03</span></div>
          <h4>Classify</h4>
          <div class="ov-stage-tag mono">MobileNetV3-Small, fine-tuned</div>
          <p>Outputs P(spoof) &isin; [0,1]. BlazeFace only finds faces &mdash; it cannot tell live from spoofed; this model is what decides that.</p>
        </div>
        <div class="ov-stage">
          <div class="ov-stage-top"><span class="ov-stage-num mono">STAGE 04</span></div>
          <h4>Decision Gate</h4>
          <div class="ov-stage-tag mono">Three-way LIVE / SPOOF / UNCERTAIN</div>
          <p>A gray-zone band (selected on validation only) flags low-confidence frames as UNCERTAIN instead of guessing.</p>
        </div>
      </div>
    </section>

    <section class="panel ov-section" id="ov-compare-section">
      <span class="ov-tagline mono">EMPIRICAL FINDING &middot; v1 vs v2</span>
      <h3 style="margin-top:6px">Why this model was retrained from scratch</h3>
      <p class="page-sub" id="ov-compare-sub">loading&hellip;</p>
      <div class="ov-compare-bars" id="ov-compare-bars"></div>
    </section>

    <section class="ov-stack-row">
      <span class="ov-stack-label mono">CORE ENGINEERING STACK</span>
      <div class="ov-stack-badges">
        <span class="ov-stack-badge mono">Python 3.11</span>
        <span class="ov-stack-badge mono">PyTorch</span>
        <span class="ov-stack-badge mono">OpenCV</span>
        <span class="ov-stack-badge mono">MediaPipe</span>
        <span class="ov-stack-badge mono">Flask</span>
      </div>
    </section>

    <section class="panel ov-cta">
      <h2>Ready to test spoof resilience?</h2>
      <p>Launch the live detection console with your own camera &mdash; no install needed.</p>
      <button class="btn btn-primary" data-nav="live">Try Live Detection</button>
    </section>
  `;

  root.querySelectorAll("[data-nav]").forEach((btn) => {
    btn.addEventListener("click", () => {
      document.querySelector(`.tab-btn[data-view="${btn.dataset.nav}"]`)?.click();
    });
  });

  runSamplePreview();

  let summary;
  try {
    summary = await api.summary();
  } catch (e) {
    document.getElementById("ov-stats").innerHTML = '<div class="panel">Could not load results/summary.json.</div>';
    document.getElementById("ov-specs").textContent = "measured spec strip unavailable (backend unreachable)";
    document.getElementById("ov-compare-sub").textContent = "Could not load v1-vs-v2 comparison data.";
    return;
  }

  renderStats(summary);
  renderSpecs(summary);
  renderCompare(summary);
}

async function runSamplePreview() {
  const body = document.getElementById("ov-preview-body");
  try {
    const [samples, result] = await Promise.all([api.sampleImages(), api.sampleDetect(PREVIEW_SAMPLE_ID)]);
    const meta = samples.images.find((s) => s.id === PREVIEW_SAMPLE_ID);
    if (!result.face_detected) {
      body.innerHTML = '<div class="empty-state">No face detected in this sample image.</div>';
      return;
    }
    const cls = statusClass(result.label);
    const totalMs = Object.values(result.timings).reduce((a, b) => a + b, 0);
    const pSpoof = result.label === "LIVE" ? 1 - result.confidence : result.confidence;
    body.innerHTML = `
      <div class="ov-preview-img-wrap">
        <img src="${api.sampleImageUrl(PREVIEW_SAMPLE_ID)}" alt="${meta ? meta.label : PREVIEW_SAMPLE_ID}">
        <span class="badge ${cls} ov-preview-badge">${result.label} &middot; <span class="prob mono">P(spoof)=${pct(pSpoof)}</span></span>
      </div>
      <div class="ov-preview-caption mono">${meta ? meta.label : PREVIEW_SAMPLE_ID} &middot; total pipeline: ${ms(totalMs)} on this request</div>
    `;
  } catch (e) {
    body.innerHTML = '<div class="empty-state">Sample inference unavailable (backend unreachable).</div>';
  }
}

function renderStats(summary) {
  const h = summary.headline;
  const latency = summary.backbone_comparison?.latency_and_size?.mobilenet_v3_small;
  document.getElementById("ov-stats").innerHTML = `
    <div class="kpi">
      <div class="figure mono">${pct(h.accuracy)}</div>
      <div class="label">Accuracy</div>
      <div class="meaning">Canonical protocol, full 3,900-image CelebA-Spoof test set, deployment-style crops.</div>
    </div>
    <div class="kpi">
      <div class="figure mono">${pct(h.hter)}</div>
      <div class="label">HTER</div>
      <div class="meaning">APCER ${pct(h.apcer)} &middot; BPCER ${pct(h.bpcer)}, plain 0.5 threshold.</div>
    </div>
    <div class="kpi">
      <div class="figure mono">${latency ? ms(latency.cpu_onnx_ms) : "—"}</div>
      <div class="label">Inference latency</div>
      <div class="meaning">MobileNetV3-Small, ONNX Runtime, CPU, median of ${latency ? latency.n_timed_runs : "—"} timed runs. (PyTorch/CPU: ${latency ? ms(latency.latency_cpu_ms_median) : "—"}; MPS: ${latency ? ms(latency.latency_mps_ms_median) : "—"}.)</div>
    </div>
  `;
}

function renderSpecs(summary) {
  const latency = summary.backbone_comparison?.latency_and_size?.mobilenet_v3_small;
  document.getElementById("ov-specs").innerHTML = `
    <span>Latency: <strong>${latency ? ms(latency.cpu_onnx_ms) : "—"}</strong> (ONNX/CPU)</span>
    <span>Model: <strong>MobileNetV3-Small</strong></span>
    <span>Threshold: <strong>0.50</strong></span>
  `;
}

function renderCompare(summary) {
  const vv = summary.v1_vs_v2;
  if (!vv) {
    document.getElementById("ov-compare-sub").textContent = "v1-vs-v2 comparison data not available.";
    return;
  }
  document.getElementById("ov-compare-sub").textContent =
    `v1 was trained on raw, uncropped CelebA-Spoof frames. The deployed pipeline feeds the model a tight BlazeFace+25%-margin crop, a real train/deploy mismatch. v2 was retrained on crops built the same way the deployment pipeline crops -- same backbone, same data, only the crop changed.`;
  document.getElementById("ov-compare-bars").innerHTML = `
    <div class="ov-bar-row">
      <div class="ov-bar-head"><span>v1 &mdash; raw-frame-trained</span><span class="mono">${pct(vv.v1.accuracy)} acc &middot; HTER ${pct(vv.v1.hter)}</span></div>
      <div class="ov-bar-track"><div class="ov-bar-fill ov-bar-v1" style="width:${(vv.v1.accuracy * 100).toFixed(2)}%"></div></div>
    </div>
    <div class="ov-bar-row">
      <div class="ov-bar-head"><span>v2 &mdash; deployment-crop-trained <span class="badge live" style="margin-left:4px">deployed</span></span><span class="mono">${pct(vv.v2.accuracy)} acc &middot; HTER ${pct(vv.v2.hter)}</span></div>
      <div class="ov-bar-track"><div class="ov-bar-fill ov-bar-v2" style="width:${(vv.v2.accuracy * 100).toFixed(2)}%"></div></div>
    </div>
  `;
}
