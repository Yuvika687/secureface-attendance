let cameraRequest = 0;

async function startCamera(videoElementId, showAlert = true) {
    stopCamera();
    const requestId = cameraRequest;
    const video = document.getElementById(videoElementId);
    let stream;
    try {
        stream = await navigator.mediaDevices.getUserMedia({video: {
            facingMode: "user", width: {ideal: 1280}, height: {ideal: 720}
        }});
        if (cameraRequest !== requestId) {
            stream.getTracks().forEach(track => track.stop());
            return;
        }
        video.srcObject = stream;
        window.localStream = stream;
        await video.play();
    } catch (error) {
        if (cameraRequest === requestId) {
            stopCamera();
            if (showAlert) alert("Camera unavailable. Allow camera access and open this page on localhost or HTTPS.");
        }
    }
}

function stopCamera() {
    cameraRequest++;
    if (window.localStream) window.localStream.getTracks().forEach(track => track.stop());
    window.localStream = null;
}

window.addEventListener("pagehide", stopCamera);

function captureFrame(videoElementId) {
    const video = document.getElementById(videoElementId);
    if (!video.videoWidth) return null;
    
    const canvas = document.createElement('canvas');
    
    // Higher resolution capture for better face recognition accuracy
    // 640px provides excellent LBPH matching while keeping API payload reasonable
    const MAX_WIDTH = 640;
    const scale = Math.min(MAX_WIDTH / video.videoWidth, 1.0);
    
    canvas.width = video.videoWidth * scale;
    canvas.height = video.videoHeight * scale;
    
    const ctx = canvas.getContext('2d');
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    
    // Higher JPEG quality (0.85) — the extra detail significantly improves
    // LBPH matching accuracy, especially with glasses/beards/varying lighting.
    // Payload increase is ~30KB which is negligible for modern networks.
    return canvas.toDataURL('image/jpeg', 0.85);
}

/* Face view (display only). Paints a square, zoomed crop of the live camera onto a
   canvas, centred on the last face box the server returned, and draws that box on
   top. The video element itself stays hidden; captureFrame() still sends the FULL
   camera frame to the server, so detection and liveness see exactly what they did
   before. Box coords arrive in the captured frame's pixels (face_box.frame_width/
   frame_height) and are mapped through the same crop, then mirrored like a selfie. */
function createFaceView(videoElementId, canvasElementId) {
    const video = document.getElementById(videoElementId);
    const canvas = document.getElementById(canvasElementId);
    const ctx = canvas.getContext('2d');
    const ZOOM = 1.9;            // crop side = 1.9 x the face size, so the face fills most of the view
    let box = null, color = '#10b981', label = '', running = false, crop = null;
    const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));

    function targetCrop(w, h) {
        let side = Math.min(w, h), cx = w / 2, cy = h / 2;      // no face: widest square view
        if (box && box.frame_width) {
            const k = w / box.frame_width;
            cx = (box.x + box.width / 2) * k;
            cy = (box.y + box.height / 2) * k;
            side = Math.max(box.width, box.height) * k * ZOOM;
        }
        side = clamp(side, Math.min(w, h) * 0.35, Math.min(w, h));
        return { x: clamp(cx - side / 2, 0, w - side), y: clamp(cy - side / 2, 0, h - side), s: side };
    }

    function paint() {
        if (!running) return;
        const w = video.videoWidth, h = video.videoHeight;
        if (w && h && video.readyState >= 2) {
            const t = targetCrop(w, h);
            crop = crop ? { x: crop.x + (t.x - crop.x) * 0.2, y: crop.y + (t.y - crop.y) * 0.2, s: crop.s + (t.s - crop.s) * 0.2 } : t;
            const W = canvas.width, H = canvas.height;
            ctx.save();
            ctx.translate(W, 0); ctx.scale(-1, 1);              // mirror like a selfie camera
            ctx.drawImage(video, crop.x, crop.y, crop.s, crop.s, 0, 0, W, H);
            ctx.restore();
            if (box && box.frame_width) {
                const k = w / box.frame_width, f = W / crop.s;
                const bw = box.width * k * f, bh = box.height * k * f;
                const bx = W - ((box.x * k - crop.x) * f + bw);  // mirrored x
                const by = (box.y * k - crop.y) * f;
                ctx.strokeStyle = color; ctx.lineWidth = 3;
                ctx.strokeRect(bx, by, bw, bh);
                if (label) {
                    ctx.font = '600 14px Inter, sans-serif';
                    const tw = ctx.measureText(label).width + 12;
                    const ly = by > 24 ? by - 24 : by + bh;
                    ctx.fillStyle = color; ctx.fillRect(bx, ly, tw, 24);
                    ctx.fillStyle = '#04110c'; ctx.fillText(label, bx + 6, ly + 17);
                }
            }
        }
        requestAnimationFrame(paint);
    }

    return {
        start() { if (!running) { running = true; crop = null; paint(); } },
        stop() { running = false; box = null; crop = null; ctx.clearRect(0, 0, canvas.width, canvas.height); },
        freeze() { running = false; },   // keep showing the last painted frame (call before stopping the camera)
        setBox(faceBox, boxColor, boxLabel) { box = faceBox || null; if (boxColor) color = boxColor; label = boxLabel || ''; },
    };
}

async function sendFaceToAPI(endpoint, base64Image, btnElement) {
    const originalText = btnElement.innerHTML;
    btnElement.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Processing...';
    btnElement.disabled = true;

    try {
        const response = await fetch(endpoint, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ image: base64Image })
        });
        const result = await response.json();
        
        btnElement.innerHTML = originalText;
        btnElement.disabled = false;
        
        return result;
    } catch (error) {
        console.error("API Error:", error);
        btnElement.innerHTML = originalText;
        btnElement.disabled = false;
        return { success: false, msg: "Network error occurred. Please try again." };
    }
}
