(() => {
    const byId = id => document.getElementById(id);
    const root = byId("scanner");
    const surface = root.querySelector(".surface");
    const studio = root.querySelector(".studio");
    const video = byId("scanVideo");
    const canvas = byId("preview");
    const ctx = canvas.getContext("2d");
    const steps = Array.from(root.querySelectorAll(".steps li"));
    const fullScreen = byId("fullscreen");
    const ready = root.dataset.approved === "1" && root.dataset.enrolled === "1";
    let box = null, restart = true, attemptStart = null, busy = false;

    function status(state, title, detail, step = 0) {
        surface.dataset.state = state;
        steps.forEach((item, index) => {
            if (index + 1 === step) item.setAttribute("aria-current", "step");
            else item.removeAttribute("aria-current");
            item.classList.toggle("done", index + 1 < step || state === "success");
        });
        if (byId("title").textContent !== title) byId("title").textContent = title;
        if (byId("description").textContent !== detail) byId("description").textContent = detail;
    }
    if (document.fullscreenEnabled) {
        fullScreen.hidden = false;
        fullScreen.addEventListener("click", async () => {
            try {
                if (document.fullscreenElement) await document.exitFullscreen();
                else await document.documentElement.requestFullscreen();
            } catch {
                fullScreen.textContent = "Full screen unavailable";
            }
        });
        document.addEventListener("fullscreenchange", () => {
            fullScreen.textContent = document.fullscreenElement ? "Exit full screen" : "Full screen";
        });
    }

    // The hidden video supplies original frames to the existing server pipeline.
    // Only the portrait crop is ever painted on screen, including before detection.
    function paint() {
        if (surface.dataset.camera !== "on") return;
        const w = video.videoWidth, h = video.videoHeight;
        if (w && h && video.readyState >= 2) {
            const ratio = canvas.width / canvas.height;
            const cx = box ? (box.x + box.width / 2) * w / box.frame_width : w / 2;
            const cy = box ? (box.y + box.height / 2) * h / box.frame_height : h / 2;
            const desired = box ? Math.max(box.height * h / box.frame_height,
                box.width * w / box.frame_width / ratio) * 1.12 : h * .7;
            const sh = Math.min(h, w / ratio, desired), sw = sh * ratio;
            const sx = Math.min(w - sw, Math.max(0, cx - sw / 2));
            const sy = Math.min(h - sh, Math.max(0, cy - sh / 2));
            ctx.drawImage(video, sx, sy, sw, sh, 0, 0, canvas.width, canvas.height);
        }
        requestAnimationFrame(paint);
    }
    if (matchMedia("(hover:hover) and (prefers-reduced-motion:no-preference)").matches) {
        studio.addEventListener("pointermove", e => {
            const rect = studio.getBoundingClientRect();
            studio.style.setProperty("--px", (e.clientX - rect.left) + "px");
            studio.style.setProperty("--py", (e.clientY - rect.top) + "px");
        });
    }
    document.addEventListener("keydown", e => {
        if (e.key.toLowerCase() === "d" && !e.target.matches("input,select,textarea")) byId("debug").hidden = !byId("debug").hidden;
    });

    function finishScan(state, title, detail) {
        surface.dataset.camera = "off";
        stopCamera();
        video.srcObject = null;
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        byId("cameraTag").textContent = "Camera off";
        status(state, title, detail);
    }
    async function scan() {
        if (busy || !ready) return;
        const image = captureFrame("scanVideo");
        if (!image) { finishScan("blocked","Camera unavailable", "Allow camera access and refresh this page."); return; }
        busy = true;
        let delay = 150;
        const abort = new AbortController();
        const timeout = setTimeout(() => abort.abort(), 8000);
        try {
            const response = await fetch("/api/recognize_face", {
                method:"POST", headers:{"Content-Type":"application/json"}, signal:abort.signal,
                body:JSON.stringify({image, restart_attempt:restart, test_label:window.getDevTestLabel?.()})
            });
            const res = await response.json();
            restart = false;
            box = res.face_box || null;
            byId("debug").textContent = JSON.stringify({state:res.state, average:res.p_spoof,
                similarity:res.confidence, eyeRatio:res.liveness_metrics?.ear, timings:res.timings});
            if (res.device_not_approved) {
                finishScan("setup","Scanner approval required","Open Admin → Settings → Scanner Devices to approve this browser.");
                return;
            }
            if (!response.ok || res.error) throw new Error("scanner unavailable");
            if (res.success) {
                status("success", res.msg, [res.user,res.emp_id,res.status,res.time].filter(Boolean).join(" · "), 3);
                delay = 3000; attemptStart = null;
            } else if (res.state === "UNCERTAIN") {
                attemptStart ??= performance.now();
                if (performance.now() - attemptStart >= 6000) {
                    finishScan("enroll","We couldn’t verify your live face",
                        "Attendance was not recorded. If this is your first visit, select Register your face to get started.");
                    return;
                } else {
                    status("checking","Checking for a live face","Look straight ahead with light on your face.", 2);
                }
            } else if (res.state === "MASK") {
                attemptStart = null;
                status("blocked","Face mask detected", res.msg, 2);
            } else {
                attemptStart = null;
                if (res.spoofing_detected) {
                    finishScan("blocked","Liveness check failed","A photo or screen replay may have been detected. Attendance was not recorded.");
                    return;
                }
                else if (res.recognized || res.blink_pending) status("checking","Verifying your blink","Blink naturally to finish checking in.", 2);
                else if (res.face_found) {
                    finishScan(res.account_mismatch ? "blocked" : "enroll",res.account_mismatch ? "Different account" : "Register your face",
                        res.account_mismatch ? res.msg : "Select Register your face to sign in and capture your enrollment samples.");
                    return;
                } else status("searching",res.multi_face ? "One person at a time" : "Center your face",res.msg, 1);
            }
        } catch {
            finishScan("blocked","Scanner connection interrupted","Check that the local server is running and refresh this page.");
            return;
        } finally {
            clearTimeout(timeout);
            busy = false;
        }
        setTimeout(scan, delay);
    }
    async function begin() {
        if (!ready) return;
        byId("cameraTag").textContent = "Opening camera";
        if (!video.srcObject?.active) await startCamera("scanVideo", false);
        if (!video.srcObject?.active) {
            finishScan("blocked","Camera unavailable","Allow camera access in your browser and refresh this page.");
            return;
        }
        surface.dataset.camera = "on";
        byId("cameraTag").textContent = "Camera active";
        paint();
        status("searching","Center your face","Look at the camera and blink naturally.", 1);
        scan();
    }
    window.addEventListener("pageshow", e => { if (e.persisted) location.reload(); });
    begin();
})();
