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
