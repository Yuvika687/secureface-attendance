/* Attendance scanner. Sends the full camera frame to /api/recognize_face (the server
   runs BlazeFace -> liveness model -> ArcFace -> blink -> attendance) and only shows
   what the server decided. The square face view is display-only (camera.js). */
(() => {
    const root = document.getElementById("scanner");
    const statusEl = document.getElementById("status");
    const result = document.getElementById("result");
    const idle = document.getElementById("idle");
    const faceView = document.getElementById("faceView");
    const video = document.getElementById("scanVideo");
    const view = createFaceView("scanVideo", "preview");
    const ready = root.dataset.approved === "1" && root.dataset.enrolled === "1";
    const COLORS = { LIVE: "#10b981", SPOOF: "#f43f5e", UNCERTAIN: "#38bdf8", NO_FACE: "#f59e0b", MULTI_FACE: "#f59e0b", MASK: "#f59e0b" };
    let restart = true, busy = false;

    function setStatus(text, state) {
        if (statusEl.textContent !== text) statusEl.textContent = text;
        statusEl.dataset.state = state || "idle";
    }
    function showResult(res) {
        document.getElementById("resultMsg").textContent = res.msg;
        document.getElementById("resultName").textContent = res.user || "";
        document.getElementById("resultId").textContent = res.emp_id ? "Student ID " + res.emp_id : "";
        result.hidden = false;
        setStatus("", "LIVE");
    }

    async function scan() {
        if (busy || !ready) return;
        const image = captureFrame("scanVideo");
        if (!image) { setTimeout(scan, 300); return; }
        busy = true;
        let delay = 150;
        try {
            const response = await fetch("/api/recognize_face", {
                method: "POST", headers: { "Content-Type": "application/json" },
                signal: AbortSignal.timeout(8000),
                body: JSON.stringify({ image, restart_attempt: restart, test_label: window.getDevTestLabel?.() })
            });
            const res = await response.json();
            restart = false;
            document.getElementById("debug").textContent = JSON.stringify({ state: res.state, p_spoof: res.p_spoof, timings: res.timings });
            if (res.device_not_approved) { setStatus(res.msg, "ERROR"); stop(); return; }
            if (!response.ok) throw new Error("server");

            const state = res.state || "NO_FACE";
            faceView.style.borderColor = res.face_box ? (COLORS[state] || "") : "";
            // several faces: zoom out (no single box) so everyone in view can see why
            const box = state === "MULTI_FACE" ? null : res.face_box;
            view.setBox(box, COLORS[state], box ? state.replace("_", " ") : "");

            if (res.success) {
                showResult(res);
                delay = 3500; restart = true;               // hold the result, then start fresh
            } else {
                result.hidden = true;
                if (state === "SPOOF") { setStatus(res.msg, "SPOOF"); delay = 2000; }
                else if (state === "UNCERTAIN") setStatus(res.msg, "UNCERTAIN");
                else if (state === "MULTI_FACE" || state === "MASK") setStatus(res.msg, "WAIT");
                else if (state === "NO_FACE") setStatus(res.msg || "Step into the frame.", "idle");
                else if (res.recognized || res.blink_pending) setStatus("Blink once to check in.", "LIVE");
                else setStatus(res.msg, "ERROR");          // not recognized / different account / engine error
            }
        } catch {
            setStatus("Connection problem. Retrying…", "ERROR");
            delay = 2000;
        } finally {
            busy = false;
        }
        setTimeout(scan, delay);
    }

    function stop() {
        stopCamera(); video.srcObject = null; view.stop(); idle.hidden = false;
    }

    async function begin() {
        if (!ready) return;
        await startCamera("scanVideo", false);
        if (!video.srcObject?.active) {
            setStatus("Camera unavailable. Allow camera access and refresh.", "ERROR");
            return;
        }
        idle.hidden = true;
        view.start();
        setStatus("Step into the frame.", "idle");
        scan();
    }
    window.addEventListener("pageshow", e => { if (e.persisted) location.reload(); });
    begin();
})();
