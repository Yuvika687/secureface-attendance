/* SecureFace FX: interaction layer. Purely visual: no network calls,
   no changes to form values, IDs or the scan pipeline. */
(() => {
    "use strict";
    const doc = document.documentElement;
    const reduced = matchMedia("(prefers-reduced-motion: reduce)").matches;
    const finePointer = matchMedia("(hover: hover) and (pointer: fine)").matches;
    const lerp = (a, b, t) => a + (b - a) * t;

    /* ---------- ambient background ---------- */
    function mountBackground() {
        if (document.querySelector(".fx-bg") || document.body.dataset.fxBg === "off") return;
        const bg = document.createElement("div");
        bg.className = "fx-bg";
        bg.setAttribute("aria-hidden", "true");
        bg.innerHTML = '<div class="blob b1"></div><div class="blob b2"></div><div class="blob b3"></div>' +
            '<div class="grid"></div><div class="glow"></div><div class="noise"></div>';
        document.body.prepend(bg);
    }

    /* ---------- spotlight on cards (sets --mx/--my) ---------- */
    const SPOT = ".glass-card, .stat-card, .table-panel, .settings-section, .r-card, .auth-card, .today-status-card, .card, .step, .feature-card, .surface";
    function mountSpotlight() {
        if (!finePointer) return;
        let pending = null, lastEvt = null;
        addEventListener("pointermove", e => {
            lastEvt = e;
            if (pending) return;
            pending = requestAnimationFrame(() => {
                pending = null;
                const el = lastEvt.target instanceof Element ? lastEvt.target.closest(SPOT) : null;
                if (!el) return;
                const r = el.getBoundingClientRect();
                el.style.setProperty("--mx", (lastEvt.clientX - r.left) + "px");
                el.style.setProperty("--my", (lastEvt.clientY - r.top) + "px");
            });
        }, { passive: true });
    }

    /* ---------- magnetic buttons ---------- */
    function mountMagnetic() {
        if (!finePointer || reduced) return;
        const sel = ".btn-primary, .btn-sm, .actions a, .sidebar-logo .logo-icon, .mast nav a, .auth-brand .mark";
        document.querySelectorAll(sel).forEach(el => {
            const strength = el.classList.contains("btn-primary") && el.offsetWidth > 240 ? 4 : 8;
            const base = getComputedStyle(el).transitionProperty === "all" ? "" : getComputedStyle(el).transition;
            el.style.transition = (base ? base + ", " : "") + "translate .5s cubic-bezier(.34,1.56,.64,1)";
            el.addEventListener("pointermove", e => {
                const r = el.getBoundingClientRect();
                const dx = (e.clientX - (r.left + r.width / 2)) / (r.width / 2);
                const dy = (e.clientY - (r.top + r.height / 2)) / (r.height / 2);
                el.style.translate = `${(dx * strength).toFixed(2)}px ${(dy * strength * .7).toFixed(2)}px`;
            });
            el.addEventListener("pointerleave", () => { el.style.translate = "0px 0px"; });
        });
    }

    /* ---------- 3D tilt on stat cards ---------- */
    function mountTilt() {
        if (!finePointer || reduced) return;
        document.querySelectorAll(".stat-card").forEach(el => {
            el.addEventListener("pointermove", e => {
                const r = el.getBoundingClientRect();
                const px = (e.clientX - r.left) / r.width - .5;
                const py = (e.clientY - r.top) / r.height - .5;
                el.style.transform = `perspective(900px) rotateX(${(-py * 7).toFixed(2)}deg) rotateY(${(px * 9).toFixed(2)}deg) translateY(-4px)`;
            });
            el.addEventListener("pointerleave", () => { el.style.transform = ""; });
        });
    }

    /* ---------- ripple on press ---------- */
    function mountRipple() {
        if (reduced) return;
        document.addEventListener("pointerdown", e => {
            const host = e.target instanceof Element ? e.target.closest(".btn-primary, .btn-sm, .btn-filter, .action-btn, .seg-toggle label, .actions a") : null;
            if (!host || host.disabled) return;
            host.classList.add("fx-ripple-host");
            const r = host.getBoundingClientRect();
            const s = document.createElement("span");
            s.className = "fx-ripple";
            s.style.left = (e.clientX - r.left) + "px";
            s.style.top = (e.clientY - r.top) + "px";
            s.style.setProperty("--fx-rs", (Math.max(r.width, r.height) / 4).toFixed(0));
            host.appendChild(s);
            s.addEventListener("animationend", () => s.remove());
        });
    }

    /* ---------- reveal on scroll (staggered) ---------- */
    function mountReveal() {
        if (reduced || !("IntersectionObserver" in window)) return;
        const root = document.querySelector(".main-wrapper");
        const items = document.querySelectorAll(
            ".page-content .stat-card, .page-content .glass-card, .page-content .table-panel, .page-content .settings-section, " +
            ".page-content .r-card, .page-content .today-status-card, .page-content .r-section, .hw .step, .hw .card, .hw h2, .hw .outcomes"
        );
        if (!items.length) return;
        const show = el => {
            if (!el.classList.contains("fx-reveal") || el.classList.contains("fx-in")) return;
            el.classList.add("fx-in");
            // drop the reveal classes once done so they never fight hover transforms
            setTimeout(() => el.classList.remove("fx-reveal", "fx-in"), 1400);
        };
        const io = new IntersectionObserver(entries => {
            entries.forEach(en => { if (en.isIntersecting) { show(en.target); io.unobserve(en.target); } });
        }, { root: root && root.contains(items[0]) ? root : null, rootMargin: "0px 0px -4% 0px", threshold: 0.01 });
        const groups = new Map();
        const viewH = (root && root.contains(items[0]) ? root.clientHeight : innerHeight) || innerHeight;
        items.forEach(el => {
            const key = el.parentElement;
            const i = groups.get(key) || 0;
            groups.set(key, i + 1);
            el.style.setProperty("--fx-delay", Math.min(i * 70, 420) + "ms");
            el.classList.add("fx-reveal");
            // anything already on screen animates in right away; only below-the-fold waits for scroll
            if (el.getBoundingClientRect().top < viewH) requestAnimationFrame(() => requestAnimationFrame(() => show(el)));
            else io.observe(el);
        });
        // safety net: content must never stay hidden if the observer does not fire
        setTimeout(() => items.forEach(show), 2500);
    }

    /* ---------- count-up for real numeric stats ---------- */
    function mountCounters() {
        if (reduced) return;
        document.querySelectorAll(".stat-value, .stat-number, [data-fx-count]").forEach(el => {
            if (el.children.length) return;
            const raw = el.textContent.trim();
            const m = raw.match(/^(\d{1,7})(\.\d+)?(%?)$/);
            if (!m) return;
            const target = parseFloat(m[1] + (m[2] || ""));
            const decimals = m[2] ? m[2].length - 1 : 0;
            const suffix = m[3];
            if (!target) return;
            const t0 = performance.now(), dur = 900 + Math.min(target, 400);
            el.textContent = (0).toFixed(decimals) + suffix;
            (function step(now) {
                const p = Math.min(1, (now - t0) / dur);
                const eased = 1 - Math.pow(1 - p, 4);
                el.textContent = (target * eased).toFixed(decimals) + suffix;
                if (p < 1) requestAnimationFrame(step);
                else el.textContent = raw;   // always end on the exact server value
            })(t0);
        });
    }

    /* ---------- scroll progress + topbar state ---------- */
    function mountScrollState() {
        const scroller = document.querySelector(".main-wrapper") || document.querySelector(".auth-wrapper");
        const topbar = document.querySelector(".topbar");
        if (!scroller) return;
        const bar = document.createElement("div");
        bar.className = "fx-progress";
        bar.setAttribute("aria-hidden", "true");
        document.body.appendChild(bar);
        const update = () => {
            const max = scroller.scrollHeight - scroller.clientHeight;
            const p = max > 0 ? scroller.scrollTop / max : 0;
            bar.style.setProperty("--fx-p", p.toFixed(4));
            if (topbar) topbar.classList.toggle("fx-scrolled", scroller.scrollTop > 6);
        };
        scroller.addEventListener("scroll", update, { passive: true });
        update();
    }

    /* ---------- login/register showcase: sculpted 3D particle head ---------- */
    function mountFaceMesh() {
        const canvas = document.getElementById("fxFaceMesh");
        if (!canvas) return;
        const ctx = canvas.getContext("2d");
        const chips = Array.from(document.querySelectorAll(".auth-pipeline li"));
        const hudState = document.getElementById("fxHudState");

        // Face as a height field: ellipsoid + nose, sockets, brows, cheeks, lips, chin.
        // Face space: x right, y down, z toward the viewer; roughly x in [-.8,.8], y in [-1,1].
        const g = (x, y, cx, cy, sx, sy) => Math.exp(-(((x - cx) / sx) ** 2 + ((y - cy) / sy) ** 2));
        function halfWidth(y) {                                   // head outline: cranium, cheekbones, jaw, chin
            if (y < -0.98 || y > 1.0) return 0;
            if (y < -0.15) return 0.66 * Math.sqrt(Math.max(0, 1 - ((y + 0.15) / 0.85) ** 2));
            if (y < 0.3) return 0.66 + 0.04 * Math.sin((y + 0.15) / 0.45 * Math.PI);
            let w = 0.66 - 0.34 * Math.pow((y - 0.3) / 0.7, 1.25);
            if (y > 0.86) w *= Math.sqrt(Math.max(0, 1 - ((y - 0.86) / 0.14) ** 2));
            return w;
        }
        function faceZ(x, y) {
            const a = halfWidth(y);
            if (a <= 0.01) return null;
            const e = 1 - (x / a) ** 2;
            if (e <= 0) return null;
            let z = Math.pow(e, 0.42) * 0.56 * Math.sqrt(Math.max(0, 1 - 0.7 * y * y));   // flatter front, steep sides
            z += 0.27 * g(x, y, 0, 0.07, 0.075, 0.24);               // nose ridge
            z += 0.13 * g(x, y, 0, 0.25, 0.09, 0.07);             // nose tip
            z -= 0.03 * g(x, y, -0.09, 0.30, 0.04, 0.03) + 0.03 * g(x, y, 0.09, 0.30, 0.04, 0.03); // nostrils
            z -= 0.17 * (g(x, y, -0.29, -0.12, 0.15, 0.09) + g(x, y, 0.29, -0.12, 0.15, 0.09));   // eye sockets
            z += 0.08 * (g(x, y, -0.29, -0.12, 0.08, 0.04) + g(x, y, 0.29, -0.12, 0.08, 0.04));   // eyeballs
            z += 0.08 * (g(x, y, -0.30, -0.27, 0.20, 0.05) + g(x, y, 0.30, -0.27, 0.20, 0.05));   // brows
            z += 0.07 * (g(x, y, -0.43, 0.10, 0.15, 0.13) + g(x, y, 0.43, 0.10, 0.15, 0.13));
            z -= 0.05 * (g(x, y, -0.36, 0.36, 0.12, 0.14) + g(x, y, 0.36, 0.36, 0.12, 0.14));       // cheek hollows       // cheekbones
            z += 0.08 * g(x, y, 0, 0.44, 0.17, 0.035) + 0.075 * g(x, y, 0, 0.53, 0.15, 0.04);       // lips
            z -= 0.045 * g(x, y, 0, 0.485, 0.19, 0.014);                                              // mouth line
            z += 0.08 * g(x, y, 0, 0.80, 0.17, 0.08);
            z -= 0.03 * g(x, y, 0, 0.66, 0.14, 0.03);                                               // chin crease                                               // chin
            return z;
        }

        // Hex-grid sample of the surface with a lighting normal per point.
        const pts = [];
        const step = 0.031;
        for (let row = 0, y = -0.98; y <= 0.98; y += step * 0.866, row++) {
            for (let x = -0.8 + (row % 2 ? step / 2 : 0); x <= 0.8; x += step) {
                const z = faceZ(x, y);
                if (z === null) continue;
                const h = 0.004;
                const zx = faceZ(x + h, y), zy = faceZ(x, y + h);
                const dzdx = zx === null ? 0 : (zx - z) / h, dzdy = zy === null ? 0 : (zy - z) / h;
                const nl = Math.hypot(dzdx, dzdy, 1);
                const eye = Math.min(1, g(x, y, -0.29, -0.12, 0.13, 0.05) + g(x, y, 0.29, -0.12, 0.13, 0.05));
                const iris = Math.min(1, g(x, y, -0.29, -0.12, 0.045, 0.03) + g(x, y, 0.29, -0.12, 0.045, 0.03));
                const lip = g(x, y, 0, 0.485, 0.17, 0.05);
                pts.push({ x, y, z, nx: -dzdx / nl, ny: -dzdy / nl, nz: 1 / nl, eye, iris, lip, row, r: Math.hypot(x, y) });
            }
        }
        const LM = {
            eLo: [-0.44, -0.12], eLi: [-0.15, -0.12], eRi: [0.15, -0.12], eRo: [0.44, -0.12],
            bL: [-0.32, -0.30], bR: [0.32, -0.30], fh: [0, -0.62], nt: [0, 0.26],
            mL: [-0.19, 0.48], mR: [0.19, 0.48], ch: [0, 0.86], jL: [-0.6, 0.38], jR: [0.6, 0.38],
        };
        Object.values(LM).forEach(p => p.push(faceZ(p[0], p[1]) + 0.02));
        const LM_EDGES = [["fh", "bL"], ["fh", "bR"], ["bL", "eLo"], ["bL", "eLi"], ["bR", "eRi"], ["bR", "eRo"],
            ["eLi", "nt"], ["eRi", "nt"], ["eLi", "eRi"], ["eLo", "jL"], ["eRo", "jR"], ["nt", "mL"], ["nt", "mR"],
            ["mL", "mR"], ["mL", "ch"], ["mR", "ch"], ["jL", "mL"], ["jR", "mR"], ["jL", "ch"], ["jR", "ch"]];

        // a few drifting motes for depth
        const motes = Array.from({ length: 36 }, (_, i) => ({ x: (i * 0.6180339) % 1, y: (i * 0.3819660 + 0.13) % 1, s: 0.4 + (i % 5) * 0.25, v: 0.00004 + (i % 7) * 0.00001 }));

        let w = 0, h = 0, targetYaw = 0, targetPitch = 0, yaw = 0, pitch = 0;
        const box = { x0: 0, y0: 0, x1: 0, y1: 0, init: false };
        const resize = () => {
            const r = canvas.getBoundingClientRect();
            const dpr = Math.min(devicePixelRatio || 1, 2);
            w = r.width; h = r.height;
            canvas.width = Math.round(w * dpr); canvas.height = Math.round(h * dpr);
            ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
        };
        resize();
        addEventListener("resize", resize);
        addEventListener("pointermove", e => {
            targetYaw = (e.clientX / innerWidth - 0.5) * 0.8;
            targetPitch = (e.clientY / innerHeight - 0.5) * -0.35;
        }, { passive: true });

        const phases = [
            { at: 0.00, label: "Searching", color: [150, 175, 230] },
            { at: 0.20, label: "Liveness check", color: [110, 168, 255] },
            { at: 0.46, label: "Identity match", color: [166, 139, 255] },
            { at: 0.68, label: "Blink", color: [110, 168, 255] },
            { at: 0.84, label: "Marked", color: [79, 240, 196] },
        ];
        const MINT = [79, 240, 196];
        const CYCLE = 8500;
        const start = performance.now();
        let lastPhase = -1;
        const mix = (a, b, k) => [a[0] + (b[0] - a[0]) * k, a[1] + (b[1] - a[1]) * k, a[2] + (b[2] - a[2]) * k];
        const rgba = (c, al) => `rgba(${c[0] | 0},${c[1] | 0},${c[2] | 0},${al.toFixed(3)})`;
        const L = (() => { const v = [-0.62, -0.45, 0.64], n = Math.hypot(...v); return v.map(c => c / n); })();

        function roundRect(x, y, ww, hh, r) {
            ctx.beginPath(); ctx.moveTo(x + r, y); ctx.arcTo(x + ww, y, x + ww, y + hh, r); ctx.arcTo(x + ww, y + hh, x, y + hh, r);
            ctx.arcTo(x, y + hh, x, y, r); ctx.arcTo(x, y, x + ww, y, r); ctx.closePath();
        }

        function frame(now) {
            const t = ((now - start) % CYCLE) / CYCLE;
            let phase = 0;
            phases.forEach((p, i) => { if (t >= p.at) phase = i; });
            if (phase !== lastPhase) {
                lastPhase = phase;
                chips.forEach((c, i) => c.classList.toggle("on", i <= phase));
                if (hudState) hudState.textContent = phases[phase].label;
            }
            const local = (t - phases[phase].at) / ((phases[phase + 1]?.at ?? 1) - phases[phase].at);
            const tint = phases[phase].color;

            yaw = lerp(yaw, targetYaw + 0.12 + Math.sin(now / 2900) * 0.24, 0.045);
            pitch = lerp(pitch, targetPitch + Math.sin(now / 4100) * 0.07, 0.045);
            const cy = Math.cos(yaw), sy = Math.sin(yaw), cp = Math.cos(pitch), sp = Math.sin(pitch);
            const scale = Math.min(w * 0.3, h * 0.37);
            const ox = w / 2, oy = h / 2 + Math.sin(now / 1800) * 3;
            const rot = (x, y, z) => {
                const x1 = x * cy + z * sy, z1 = -x * sy + z * cy;
                return [x1, y * cp - z1 * sp, y * sp + z1 * cp];
            };
            const project = (x, y, z) => {
                const [a, b, c] = rot(x, y, z);
                const f = 3 / (3 - c);
                return [ox + a * scale * f, oy + b * scale * f, c];
            };

            const scanning = phase <= 1;
            const scanY = -1.05 + ((now % 2400) / 2400) * 2.1;
            const blinkK = phase === 3 ? Math.max(0, 1 - Math.abs(local - 0.35) / 0.12) : 0;  // eyes close then open
            const waveR = phase === 4 ? local * 2.2 : -1;

            ctx.clearRect(0, 0, w, h);

            // motes
            for (const m of motes) {
                const yy = ((m.y - now * m.v) % 1 + 1) % 1;
                ctx.fillStyle = `rgba(160,190,255,${(0.08 + m.s * 0.12).toFixed(3)})`;
                ctx.fillRect(m.x * w, yy * h, m.s * 1.4, m.s * 1.4);
            }
            // soft halo behind the head
            const halo = ctx.createRadialGradient(ox, oy, scale * 0.2, ox, oy, scale * 1.5);
            halo.addColorStop(0, rgba(tint, 0.10)); halo.addColorStop(1, "rgba(0,0,0,0)");
            ctx.fillStyle = halo; ctx.fillRect(0, 0, w, h);

            let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
            let bandMin = Infinity, bandMax = -Infinity, bandY = 0, bandN = 0;
            const proj = new Array(pts.length);
            for (let pi = 0; pi < pts.length; pi++) {
                const p = pts[pi];
                proj[pi] = null;
                let y0 = p.y;
                if (blinkK && p.eye > 0.25) y0 = p.y + (-0.12 - p.y) * blinkK * p.eye;
                const [nx, ny, nz] = rot(p.nx, p.ny, p.nz);
                if (nz < 0.05) continue;                               // facing away
                const [sx, syy, z] = project(p.x, y0, p.z);
                if (sx < minX) minX = sx; if (sx > maxX) maxX = sx;
                if (syy < minY) minY = syy; if (syy > maxY) maxY = syy;
                const light = Math.max(0, nx * L[0] + ny * L[1] + nz * L[2]);
                const rim = Math.pow(1 - nz, 3) * 0.6;                 // edge glow keeps the silhouette crisp
                let col = tint, glow = 0;
                if (scanning) {
                    const d = (scanY - p.y);
                    glow = d >= 0 ? Math.max(0, 1 - d / 0.22) : Math.max(0, 1 + d / 0.03);
                    if (Math.abs(d) < 0.025) { bandMin = Math.min(bandMin, sx); bandMax = Math.max(bandMax, sx); bandY += syy; bandN++; }
                }
                if (waveR > 0) {
                    const k = Math.max(0, Math.min(1, (waveR - p.r) / 0.25));
                    col = mix(tint, MINT, k);
                    glow = Math.max(glow, Math.max(0, 1 - Math.abs(waveR - p.r) / 0.12));
                }
                if (p.iris > 0.2) col = mix(col, [120, 230, 255], p.iris * 0.8);
                if (p.lip > 0.3) col = mix(col, [200, 160, 255], (p.lip - 0.3) * 0.6);
                if (glow > 0) col = mix(col, MINT, glow);
                const fill = Math.max(0, nx * 0.55 - ny * 0.15 + nz * 0.6) * 0.22;      // soft fill light from the right
                const alpha = Math.min(1, 0.16 + Math.pow(light, 1.4) * 0.78 + fill + rim + glow * 0.6);
                const size = (0.6 + light * 0.9 + glow * 1.1 + p.iris * 0.8) * (0.85 + z * 0.4);
                proj[pi] = { sx, sy: syy, col, alpha, row: p.row };
                ctx.fillStyle = rgba(col, Math.min(1, alpha + p.iris * 0.4));
                ctx.fillRect(sx - size / 2, syy - size / 2, size, size);
            }
            // contour lines: join neighbours in each row so the surface reads as a 3D scan
            ctx.lineWidth = 0.8;
            for (let pi = 1; pi < pts.length; pi++) {
                const A = proj[pi - 1], B = proj[pi];
                if (!A || !B || A.row !== B.row) continue;
                if (A.row % 2) continue;                                   // every other row keeps it airy
                ctx.strokeStyle = rgba(B.col, Math.min(A.alpha, B.alpha) * 0.55);
                ctx.beginPath(); ctx.moveTo(A.sx, A.sy); ctx.lineTo(B.sx, B.sy); ctx.stroke();
            }

            // feature outlines: almond eyes (close on blink), iris rings, lips
            const strokeCurve = (fn, n, close) => {
                ctx.beginPath();
                for (let i = 0; i <= n; i++) {
                    const [x, y] = fn(i / n);
                    const z = faceZ(x, y);
                    if (z === null) continue;
                    const [sx, syy] = project(x, y, z + 0.012);
                    i ? ctx.lineTo(sx, syy) : ctx.moveTo(sx, syy);
                }
                if (close) ctx.closePath();
                ctx.stroke();
            };
            const fc = phase === 4 ? MINT : mix(tint, [190, 235, 255], 0.5);
            const open = 1 - blinkK;
            ctx.lineWidth = 1.2;
            ctx.strokeStyle = rgba(fc, 0.75);
            for (const ex of [-0.29, 0.29]) {
                strokeCurve(t => { const a = t * Math.PI * 2, s = Math.sin(a); return [ex + 0.125 * Math.cos(a), -0.12 + (s < 0 ? 0.05 : 0.032) * s * Math.max(0.06, open)]; }, 28, true);
            }
            if (open > 0.45) {
                ctx.strokeStyle = rgba([120, 230, 255], 0.85 * Math.min(1, (open - 0.45) * 3));
                for (const ex of [-0.29, 0.29]) strokeCurve(t => [ex + 0.032 * Math.cos(t * Math.PI * 2), -0.122 + 0.032 * Math.sin(t * Math.PI * 2) * open], 18, true);
            }
            ctx.strokeStyle = rgba(mix(fc, [205, 170, 255], 0.35), 0.6);
            strokeCurve(t => { const x = -0.2 + t * 0.4; return [x, 0.483 - 0.028 * (1 - (x / 0.2) ** 2) + 0.012 * Math.exp(-((x / 0.04) ** 2))]; }, 24, false);  // upper lip
            strokeCurve(t => { const x = -0.2 + t * 0.4; return [x, 0.487 + 0.05 * (1 - (x / 0.2) ** 2)]; }, 24, false);                                   // lower lip
            strokeCurve(t => { const x = -0.2 + t * 0.4; return [x, 0.486]; }, 12, false);                                                                 // mouth line

            // scan beam, only as wide as the face at that height
            if (scanning && bandN > 2) {
                const by = bandY / bandN, pad = 18;
                const beam = ctx.createLinearGradient(bandMin - pad, 0, bandMax + pad, 0);
                beam.addColorStop(0, "rgba(79,240,196,0)"); beam.addColorStop(0.5, "rgba(79,240,196,0.9)"); beam.addColorStop(1, "rgba(79,240,196,0)");
                ctx.fillStyle = beam; ctx.fillRect(bandMin - pad, by - 0.75, bandMax - bandMin + pad * 2, 1.5);
                ctx.fillStyle = "rgba(79,240,196,0.10)"; ctx.fillRect(bandMin - pad, by - 0.75, bandMax - bandMin + pad * 2, 1.5);
            }

            // landmarks fade in from the identity phase
            const lmA = phase >= 2 ? Math.min(1, (phase === 2 ? local * 3 : 1)) : 0;
            if (lmA > 0) {
                const P = {};
                for (const k in LM) {
                    let [x, y, z] = LM[k];
                    if (blinkK && k[0] === "e") y = -0.12;
                    P[k] = project(x, y, z);
                }
                const lc = phase === 4 ? MINT : [200, 215, 255];
                ctx.lineWidth = 1;
                ctx.strokeStyle = rgba(lc, 0.28 * lmA);
                ctx.beginPath();
                for (const [a, b] of LM_EDGES) { ctx.moveTo(P[a][0], P[a][1]); ctx.lineTo(P[b][0], P[b][1]); }
                ctx.stroke();
                for (const k in P) {
                    const [x, y] = P[k];
                    const pulse = (phase === 3 && k[0] === "e") ? 1 + blinkK * 0.8 : 1;
                    ctx.fillStyle = rgba(lc, 0.95 * lmA);
                    ctx.fillRect(x - 1.5 * pulse, y - 1.5 * pulse, 3 * pulse, 3 * pulse);
                    ctx.strokeStyle = rgba(lc, 0.45 * lmA);
                    ctx.strokeRect(x - 4 * pulse, y - 4 * pulse, 8 * pulse, 8 * pulse);
                }
            }

            // tracking box hugging the face
            if (isFinite(minX)) {
                const pad = 16;
                const tx0 = minX - pad, ty0 = minY - pad, tx1 = maxX + pad, ty1 = maxY + pad;
                if (!box.init) Object.assign(box, { x0: tx0, y0: ty0, x1: tx1, y1: ty1, init: true });
                box.x0 = lerp(box.x0, tx0, 0.12); box.y0 = lerp(box.y0, ty0, 0.12);
                box.x1 = lerp(box.x1, tx1, 0.12); box.y1 = lerp(box.y1, ty1, 0.12);
                const breathe = phase === 0 ? Math.sin(now / 260) * 3 : 0;
                const x0 = box.x0 - breathe, y0 = box.y0 - breathe, x1 = box.x1 + breathe, y1 = box.y1 + breathe;
                const c = Math.min(22, (x1 - x0) * 0.14), bc = phase === 4 ? MINT : tint;
                ctx.strokeStyle = rgba(bc, 0.95); ctx.lineWidth = 2; ctx.lineCap = "round";
                ctx.beginPath();
                ctx.moveTo(x0, y0 + c); ctx.lineTo(x0, y0); ctx.lineTo(x0 + c, y0);
                ctx.moveTo(x1 - c, y0); ctx.lineTo(x1, y0); ctx.lineTo(x1, y0 + c);
                ctx.moveTo(x1, y1 - c); ctx.lineTo(x1, y1); ctx.lineTo(x1 - c, y1);
                ctx.moveTo(x0 + c, y1); ctx.lineTo(x0, y1); ctx.lineTo(x0, y1 - c);
                ctx.stroke();
                // verified chip
                if (phase === 4) {
                    const k = Math.min(1, local * 4), label = "VERIFIED";
                    ctx.font = "700 10px 'Cascadia Mono', Consolas, monospace";
                    const tw = ctx.measureText(label).width, cw = tw + 34, ch = 22;
                    const cx = Math.min(w - cw - 10, x1 + 12), cyy = y0 + 6 + (1 - k) * 8;
                    ctx.globalAlpha = k;
                    roundRect(cx, cyy, cw, ch, 11);
                    ctx.fillStyle = "rgba(79,240,196,0.14)"; ctx.fill();
                    ctx.strokeStyle = "rgba(79,240,196,0.7)"; ctx.lineWidth = 1; ctx.stroke();
                    ctx.strokeStyle = "rgb(79,240,196)"; ctx.lineWidth = 1.8;
                    ctx.beginPath(); ctx.moveTo(cx + 10, cyy + 11); ctx.lineTo(cx + 13.5, cyy + 14.5); ctx.lineTo(cx + 19, cyy + 7.5); ctx.stroke();
                    ctx.fillStyle = "rgb(214,255,243)"; ctx.textBaseline = "middle";
                    ctx.fillText(label, cx + 25, cyy + ch / 2 + 0.5);
                    ctx.globalAlpha = 1;
                }
            }
            if (!reduced) requestAnimationFrame(frame);
        }
        requestAnimationFrame(frame);
    }

    /* ---------- scanner: re-animate the status text whenever it changes ---------- */
    function mountStatusSwap() {
        const nodes = [document.getElementById("title"), document.getElementById("description")].filter(Boolean);
        if (reduced || !nodes.length || !document.querySelector(".feedback")) return;
        const mo = new MutationObserver(records => {
            const seen = new Set(records.map(r => r.target.nodeType === 1 ? r.target : r.target.parentElement));
            seen.forEach(el => { if (!el) return; el.classList.remove("fx-swap"); void el.offsetWidth; el.classList.add("fx-swap"); });
        });
        nodes.forEach(n => mo.observe(n, { childList: true, characterData: true, subtree: true }));
    }

    function init() {
        mountBackground();
        mountSpotlight();
        mountMagnetic();
        mountTilt();
        mountRipple();
        mountReveal();
        mountCounters();
        mountScrollState();
        mountFaceMesh();
        mountStatusSwap();
    }
    if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", init);
    else init();
})();
