# SmartFace AI — Cloudflare Tunnel Deployment Guide

## Overview

Run SmartFace AI on your local PC and expose it securely to the internet via Cloudflare Tunnel.  
Anyone with the URL can access the scanner from any device, anywhere — as long as your PC is on.

---

## Prerequisites

1. **Your PC** running Windows 10/11
2. **Python 3.10+** with all dependencies installed
3. **A free Cloudflare account** — [Sign up here](https://dash.cloudflare.com/sign-up)
4. **A domain on Cloudflare** — You need a domain managed by Cloudflare DNS  
   (You can transfer an existing domain or buy a cheap one like `.xyz` for ~$2/year)

---

## Step 1: Install Cloudflared

Open PowerShell as Administrator:

```powershell
winget install --id Cloudflare.cloudflared
```

Or download manually from: https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/downloads/

Verify installation:
```powershell
cloudflared --version
```

---

## Step 2: Authenticate with Cloudflare

```powershell
cloudflared tunnel login
```

This opens your browser. Log in to Cloudflare and authorize `cloudflared`.  
A certificate is saved to `%USERPROFILE%\.cloudflared\cert.pem`.

---

## Step 3: Create a Named Tunnel

```powershell
cloudflared tunnel create smartface
```

This creates a tunnel with a fixed UUID. Note the tunnel ID (e.g., `a1b2c3d4-...`).  
Credentials are saved to `%USERPROFILE%\.cloudflared\<TUNNEL_ID>.json`.

---

## Step 4: Configure DNS Route

Replace `yourdomain.com` with your actual domain:

```powershell
cloudflared tunnel route dns smartface attendance.yourdomain.com
```

This creates a CNAME record pointing `attendance.yourdomain.com` → your tunnel.

---

## Step 5: Create Config File

Create `%USERPROFILE%\.cloudflared\config.yml`:

```yaml
tunnel: smartface
credentials-file: C:\Users\<YOUR_USERNAME>\.cloudflared\<TUNNEL_ID>.json

ingress:
  - hostname: attendance.yourdomain.com
    service: http://localhost:5000
  - service: http_status:404
```

Replace `<YOUR_USERNAME>` and `<TUNNEL_ID>` with your actual values.

---

## Step 6: Start Everything

### Terminal 1 — Start SmartFace AI:
```powershell
cd A:\Smart-Attendance-System
python app.py
```

### Terminal 2 — Start Cloudflare Tunnel:
```powershell
cloudflared tunnel run smartface
```

Your app is now live at: **https://attendance.yourdomain.com** 🎉

---

## Quick Start (No Domain — Temporary URL)

If you just want a quick test without a domain:

```powershell
# Terminal 1: Start Flask
cd A:\Smart-Attendance-System
python app.py

# Terminal 2: Quick tunnel (random URL)
cloudflared tunnel --url http://localhost:5000
```

This gives you a temporary URL like `https://random-words.trycloudflare.com`.  
The URL changes each time you restart the tunnel.

---

## Run as Windows Service (Auto-start)

To make the tunnel start automatically with Windows:

```powershell
cloudflared service install
```

This installs `cloudflared` as a Windows service that starts on boot.

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `cloudflared` not found | Restart PowerShell after installing |
| Tunnel won't connect | Check `config.yml` paths are correct |
| Camera not working | HTTPS is required for camera access — Cloudflare provides this automatically |
| Slow response | Ensure Flask is running on `localhost:5000` |

---

## Architecture

```
┌──────────────┐     HTTPS     ┌──────────────┐     HTTP      ┌──────────────┐
│   Browser    │ ───────────── │  Cloudflare  │ ───────────── │  Your PC     │
│   (Phone/    │               │  Edge        │               │  Flask :5000 │
│    Laptop)   │               │  Network     │               │  InsightFace │
│              │               │              │               │  MediaPipe   │
└──────────────┘               └──────────────┘               └──────────────┘
```

- **HTTPS** is automatic (Cloudflare provides SSL)
- **Camera access** works because the URL is HTTPS
- **No port forwarding** needed — tunnel punches through NAT
- **DDoS protection** included via Cloudflare
