/**
 * SmartFace AI — Camera Module v2.0
 * ==================================
 * HD webcam capture optimized for face recognition at scale (500+ employees).
 * - Requests 1280×720 HD resolution from webcam
 * - Higher JPEG quality (0.85) for better LBPH matching accuracy
 * - 640px max width for optimal detail-to-speed balance
 * - Adaptive resolution fallback for older/low-end cameras
 */

function startCamera(videoElementId) {
    const video = document.getElementById(videoElementId);
    if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
        // Request HD resolution for maximum face detail
        const constraints = {
            video: {
                facingMode: "user",
                width: { ideal: 1280, min: 640 },
                height: { ideal: 720, min: 480 },
                frameRate: { ideal: 30, max: 30 }
            }
        };

        navigator.mediaDevices.getUserMedia(constraints)
            .then(function(stream) {
                video.srcObject = stream;
                video.play();
                window.localStream = stream;

                // Log actual resolution achieved
                video.addEventListener('loadedmetadata', () => {
                    console.log(`[Camera] Resolution: ${video.videoWidth}×${video.videoHeight}`);
                });
            })
            .catch(function(err) {
                console.warn("[Camera] HD request failed, trying basic:", err);
                // Fallback to basic constraints
                navigator.mediaDevices.getUserMedia({ video: { facingMode: "user" } })
                    .then(function(stream) {
                        video.srcObject = stream;
                        video.play();
                        window.localStream = stream;
                    })
                    .catch(function(err2) {
                        console.error("[Camera] All attempts failed:", err2);
                        alert("Cannot access the camera. Please check permissions.");
                    });
            });
    } else {
        alert("Camera not supported on this browser.");
    }
}

function stopCamera() {
    if (window.localStream) {
        window.localStream.getTracks().forEach(track => track.stop());
    }
}

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
