import { api } from "./api.js";
import { initOverviewPage } from "./page-overview.js";
import { initLivePage } from "./page-live.js";
import { initResultsPage } from "./page-results.js";
import { initRobustnessPage } from "./page-robustness.js";
import { initModelsPage } from "./page-models.js";

// ---- modal (shared image-enlarge, used by the Robustness gallery) ----
const modalBackdrop = document.getElementById("modal-backdrop");
const modalBox = document.getElementById("modal-box");
export function openModal(html) {
  modalBox.innerHTML = `<span class="modal-close" id="modal-close-btn">&times; close</span>${html}`;
  modalBackdrop.classList.add("show");
  document.getElementById("modal-close-btn").addEventListener("click", closeModal);
}
export function closeModal() {
  modalBackdrop.classList.remove("show");
  modalBox.innerHTML = "";
}
modalBackdrop.addEventListener("click", (e) => {
  if (e.target === modalBackdrop) closeModal();
});

// ---- header: model pill + connection status ----
async function loadHeader() {
  try {
    const info = await api.modelInfo();
    const name = info.backbone === "mobilenet_v3_small" ? "MobileNetV3-Small"
      : info.backbone === "resnet18" ? "ResNet-18"
      : info.backbone === "efficientnet_b0" ? "EfficientNet-B0" : info.backbone;
    document.getElementById("model-pill").innerHTML =
      `<b>${name}</b> &middot; ${info.model_version} &middot; ${info.device}`;
    document.getElementById("conn-dot").className = "conn-dot ok";
    document.getElementById("conn-text").textContent = "backend connected";
  } catch (err) {
    document.getElementById("model-pill").textContent = "model info unavailable";
    document.getElementById("conn-dot").className = "conn-dot bad";
    document.getElementById("conn-text").textContent = "backend unreachable";
  }
}
loadHeader();

// ---- tab navigation ----
const pageInit = {
  overview: { fn: initOverviewPage, done: false },
  live: { fn: initLivePage, done: false },
  results: { fn: initResultsPage, done: false },
  robustness: { fn: initRobustnessPage, done: false },
  models: { fn: initModelsPage, done: false },
};

function activateTab(name) {
  document.querySelectorAll(".tab-btn").forEach((b) => b.classList.toggle("active", b.dataset.view === name));
  document.querySelectorAll(".view").forEach((v) => v.classList.toggle("active", v.id === `view-${name}`));
  const entry = pageInit[name];
  if (entry && !entry.done) {
    entry.done = true;
    entry.fn(document.getElementById(`view-${name}`));
  }
}

document.getElementById("tabs").addEventListener("click", (e) => {
  const btn = e.target.closest(".tab-btn");
  if (!btn) return;
  activateTab(btn.dataset.view);
});

// Overview renders first (it's the landing page).
activateTab("overview");
