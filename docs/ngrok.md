# Reaching the app from another device with ngrok (macOS)

Replaces the old Cloudflare Tunnel guide.

## Do you need it?

**Usually not.** Browsers allow camera access on `http://localhost` without HTTPS, so
the scanner, face registration, Pipeline View and Test Mode all work at
`http://localhost:5000` on the Mac running the app.

You need HTTPS only to use the camera from **another device**, such as a phone or a
second laptop used as the scanner. Browsers block the camera on plain
`http://192.168.x.x:5000` addresses. ngrok gives you an `https://` URL that forwards to
your local app.

## Install (once)

```bash
brew install ngrok
ngrok config add-authtoken <your-token>
```

Get the token from https://dashboard.ngrok.com/get-started/your-authtoken (free account).

## Run

Terminal 1: start the app.

```bash
cd ~/Documents/project/secureface-attendance
./venv/bin/python app.py
```

Terminal 2: start the tunnel.

```bash
ngrok http 5000
```

ngrok prints a `Forwarding` line such as `https://<random-name>.ngrok-free.app -> http://localhost:5000`.
Open that `https://` URL on the other device.

## Notes for this app

- **Approve the scanner on the URL you will actually use.** Scanner approval is a cookie
  stored for one address. Approving at `localhost:5000` does not approve the ngrok URL, and
  a free ngrok URL changes every time you restart ngrok. On the scanner device: open the
  ngrok URL, log in as admin, go to **Settings → Scanner Devices**, and click **Set this
  device as scanner**. Log out afterwards; the approval stays.
- ngrok's free plan may show a one-time "You are about to visit…" page first. Click
  through it.
- The tunnel makes the app reachable from the internet while it runs. Login, CSRF
  protection and scanner approval still apply, but stop ngrok (Ctrl+C) when you're done.
- If port 5000 is taken by AirPlay Receiver, turn it off in System Settings, or free the
  port with `lsof -ti:5000 | xargs kill -9`.
