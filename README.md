<div align="center">

<!-- Animated Hero Banner -->
<img src="https://capsule-render.vercel.app/api?type=waving&color=0:0a0e1a,50:064e3b,100:10b981&height=220&section=header&text=SmartFace%20AI&fontSize=72&fontColor=ffffff&fontAlignY=35&desc=Enterprise%20Biometric%20Engine%20v3.0&descSize=18&descAlignY=55&descColor=94a3b8&animation=fadeIn" width="100%" />

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Flask](https://img.shields.io/badge/Flask-2.3+-000000?style=for-the-badge&logo=flask&logoColor=white)](https://flask.palletsprojects.com)
[![ONNX](https://img.shields.io/badge/ONNX_ArcFace-99.8%25-005CED?style=for-the-badge&logo=onnx&logoColor=white)](https://onnxruntime.ai/)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-478_Pt_Mesh-4285F4?style=for-the-badge&logo=google&logoColor=white)](https://mediapipe.dev)
[![PostgreSQL](https://img.shields.io/badge/Neon_DB-Supported-336791?style=for-the-badge&logo=postgresql&logoColor=white)](https://neon.tech)

<br/>

<p>
  <img src="https://readme-typing-svg.demolab.com?font=Inter&weight=800&size=20&duration=3000&pause=1000&color=10B981&center=true&vCenter=true&multiline=true&repeat=true&width=750&height=60&lines=ArcFace+ONNX+512-D+Embeddings+%7C+99.8%25+LFW+Accuracy;Multi-Layer+Anti-Spoofing+%7C+3D+Depth+%2B+Texture;MediaPipe+478-Point+Mesh+%7C+Real-Time+Liveness" alt="Typing SVG" />
</p>

<p><em>Enterprise-grade face recognition attendance system powered by ArcFace ONNX and MediaPipe, featuring real-time liveness detection and active anti-spoofing. Built by <strong>Sofzenix Technologies</strong>.</em></p>

</div>

---

## 🌟 Overview

**SmartFace AI v3.0** is a completely upgraded, ultra-fast biometric attendance system. It abandons traditional legacy models (like Haar Cascades or LBPH) in favor of the state-of-the-art **ArcFace w600k_r50 ONNX model**, generating dense 512-dimensional embeddings for unmatched 99.8% accuracy. 

It pairs this deep learning recognition vector with **MediaPipe Face Mesh (478 landmarks)** to execute complex spatial liveness checks, 3D depth variance analysis, and intelligent screen-spoof detection in real-time.

```text
┌───────────────────────────────────────────────────────────────────┐
│                          SMARTFACE v3.0                           │
│                                                                   │
│   Webcam Frame → MediaPipe Mesh Alignment → ArcFace ONNX Model    │
│                            ↓                                      │
│                  512-D Vector Cosine Similarity                   │
│                            ↓                                      │
│                  Multi-Factor Liveness Pipeline:                  │
│       (3D Depth Analysis + LBP Texture + Screen Detection)        │
│                            ↓                                      │
│                ✅ Access Granted & Attendance Logged              │
└───────────────────────────────────────────────────────────────────┘
```

---

## 🚀 Key Upgrades in v3.0

- **ArcFace ONNX Engine:** Replaced LBPH with a quantized `w600k_r50` ArcFace ONNX model for enterprise-level face matching. Matches are done via blazing-fast vectorized cosine similarity.
- **Advanced 3D Liveness:** Uses 478-point MediaPipe mesh to calculate Z-axis depth variance, easily distinguishing a flat 2D photo from a real 3D human face.
- **Hardware-Agnostic:** Pure Python + ONNX runtime means it runs on standard CPUs natively without requiring complex C++ compiler setups or dedicated GPUs.
- **Cloudflare Tunnel Deployment:** Instantly expose the local high-performance recognition server to the internet securely via HTTPS without port forwarding.

---

## 🛡️ Anti-Spoofing Architecture

SmartFace AI uses a deterministic multi-signal physical pipeline to defeat spoofing attempts:

1. **3D Depth Variance (MediaPipe):** Measures Z-coordinate spread across the nose ridge, cheeks, and eyes. Flat screens return <0.05 variance; real faces return >0.15.
2. **Screen Bezel Detection:** Uses Canny Edge and Hough Lines to identify straight, rectangular Bezels surrounding the face (typical of phone/tablet presentation).
3. **Brightness & Contrast Anomaly:** Measures the delta between the face ROI luminosity outshining the background environment (indicative of a backlit digital display).
4. **Color Temperature Shift:** Detects the unnatural blue-channel emission signature of OLED/LCD screens.
5. **Randomized Liveness Challenge:** The system randomly requests dynamic actions (e.g., "Blink", "Turn Head Left", "Turn Head Right") monitored via instantaneous Head Pose estimation and EAR calculation.

---

## ⚙️ Quick Start Guide

### 1. Prerequisites
- **Python 3.10 or 3.11** (recommended for ONNX compatibility)
- Standard USB Webcam

### 2. Local Installation

```bash
git clone https://github.com/Sofzenix/Smart-Attendance-System.git
cd Smart-Attendance-System

# Create Virtual Environment
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Mac/Linux

# Install strictly verified dependencies
pip install -r requirements.txt

# Run the app (It will automatically download the ~300MB ONNX model on first run)
python app.py
```

### 3. Expose to Internet via Cloudflare (Recommended)
You don't need expensive GPU cloud servers. Run the AI engine natively on your local machine and expose it to the web securely via zero-trust tunnels.

```bash
winget install --id Cloudflare.cloudflared
cloudflared tunnel login
cloudflared tunnel --url http://localhost:5000
```
*(Reference `cloudflare_tunnel.md` for full custom domain setup)*

---

## 👨‍💼 Administrator Guide

**Default Admin Status:**
Go to `/admin` and use the default credentials:
- **Email:** `admin@sofzenix.com`
- **Password:** `Admin@123`

### Face Registration Process
1. A new user registers an account at `/register`.
2. Upon first login, they visit the scanner.
3. The system captures **up to 15 unique 512-D embeddings** representing different angles and ambient lighting for maximum resilience.
4. From then on, scanning involves mere millisecond vectorized comparisons.

---

## 🧰 Technology Stack

| Layer | Tools Used | Purpose |
|:------|:-----------|:--------|
| **AI Inference** | ONNX Runtime | Runs ArcFace `w600k_r50` model locally |
| **Spatial Mapping** | Google MediaPipe | 478-point mesh, head yaw, blink EAR |
| **Backend API** | Flask + Werkzeug | Serves UI and processes JSON base64 frames |
| **Database** | SQLite / Neon Postgres | Multi-environment ORM equivalent | 
| **Job Scheduling** | APScheduler | Triggers nightly absentee HR emails |

---

## 📝 License & Contact

This project is licensed under the **MIT License**.
Developed & Maintained by [Sofzenix Technologies](https://github.com/Sofzenix).

<div align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=0:0a0e1a,50:064e3b,100:10b981&height=120&section=footer" width="100%" />
</div>
