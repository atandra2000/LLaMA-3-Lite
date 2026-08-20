/* LLaMA-3-Lite Documentation Portal Client-Side Behaviors
   ======================================================================
   • GQA 8Q/4KV & RoPE θ=500K Phase-Space Dynamics (FIG · A0)
   • Live 16-Layer Pipeline Pass & Memory Stack Telemetry (FIG · A1)
   • GQA 8Q/4KV KV-Cache Compression & Router Lab (MCH · 01)
   • Chunked Cross-Entropy 256-Token Slicer Lab (MCH · 02)
   • RoPE θ=500K Long-Context Extrapolation Lab (MCH · 03)
   • Expandable Code Blocks (>14 lines) & Copy-to-Clipboard
   • Navigation Filtering & Sidebar Mobile Toggle
   • Table of Contents Scrollspy
   • Highlight.js & KaTeX bootstrap
*/

(function () {
    'use strict';

    var reduced = window.matchMedia &&
                  window.matchMedia('(prefers-reduced-motion: reduce)').matches;

    // ------------------------------------------------------------------
    // 1. GQA 8Q/4KV & RoPE θ=500K Phase-Space Dynamics (FIG · A0)
    //    Interactive high-DPI canvas simulation:
    //    - GQA: 8 Query heads mapping onto 4 Shared KV heads (2:1 ratio)
    //    - RoPE: 64 Rotary frequency phasors across θ=500K base
    //    - Chunked CE: 256-token sliced logit projection in SRAM
    //    - Memory: 8-technique memory stack (78% VRAM reduction: 92GB -> 20GB)
    // ------------------------------------------------------------------
    function initHeroCanvas() {
        var canvas = document.getElementById('heroStateCanvas') || document.getElementById('complexSpiralCanvas');
        if (!canvas) return;

        var ctx = canvas.getContext('2d');
        if (!ctx) return;

        var container = canvas.parentElement;
        var probeEl = document.getElementById('heroProbeHUD') || document.getElementById('hudProbe');
        var probeTag = document.getElementById('probeTag');
        var probeCoords = document.getElementById('probeCoords');
        var probeDecay = document.getElementById('probeDecay');
        var hudModeLabel = document.getElementById('hudModeLabel');
        var hudKVCut = document.getElementById('hudKVCut') || document.getElementById('hudNormVal');
        var pauseBtn = document.getElementById('heroPauseBtn') || document.getElementById('figPauseBtn');
        var speedBtn = document.getElementById('heroSpeedBtn');
        var modeBtns = document.querySelectorAll('.fig-controls .fig-btn[data-mode]');

        var currentMode = 'gqa';
        var isPaused = false;
        var simSpeed = 1.0;
        var mouseX = -9999, mouseY = -9999;
        var pulseWaves = [];
        var animId = null;
        var lastTime = 0;
        var simTime = 0;

        var PALETTE = {
            paperBg: '#0e0c0a',
            paperCenter: '#17130f',
            gridRule: 'rgba(58, 50, 38, 0.45)',
            gridAxis: 'rgba(201, 163, 92, 0.35)',
            unitCircle: 'rgba(201, 163, 92, 0.22)',
            olive: '#9a9440',
            oliveGlow: 'rgba(154, 148, 64, 0.65)',
            oliveTint: 'rgba(154, 148, 64, 0.15)',
            terracotta: '#e07a3f',
            terracottaGlow: 'rgba(224, 122, 63, 0.65)',
            terracottaTint: 'rgba(224, 122, 63, 0.15)',
            gold: '#c9a35c',
            goldGlow: 'rgba(201, 163, 92, 0.75)',
            goldTint: 'rgba(201, 163, 92, 0.18)',
            coreHot: '#fffaf0',
            ink: '#d8ccb4',
            inkSoft: '#b3a68c',
            inkFaint: '#7a7160'
        };

        var Q_HEADS = 8;
        var KV_HEADS = 4;
        var qHeadNodes = [];
        for (var q = 0; q < Q_HEADS; q++) {
            var angle = (q / Q_HEADS) * Math.PI * 2 - Math.PI / 2;
            var kvIdx = Math.floor(q / 2);
            qHeadNodes.push({
                id: q,
                angle: angle,
                kvIdx: kvIdx,
                activity: 0.5 + 0.5 * Math.sin(q * 1.5),
                pulse: 0
            });
        }

        var kvHeadNodes = [];
        for (var k = 0; k < KV_HEADS; k++) {
            var kAngle = (k / KV_HEADS) * Math.PI * 2 - Math.PI / 2 + Math.PI / 8;
            kvHeadNodes.push({
                id: k,
                angle: kAngle,
                weight: 1.0,
                pulse: 0
            });
        }

        var ROPE_DIMS = 64;
        var ropePhasors = [];
        for (var r = 0; r < ROPE_DIMS; r++) {
            var frac = r / (ROPE_DIMS - 1);
            var freq = Math.pow(500000, -2 * r / 128);
            var baseR = 0.2 + 0.75 * Math.pow(frac, 0.75);
            ropePhasors.push({
                id: r,
                dim: r * 2,
                freq: freq,
                r: baseR,
                phase0: (r * 2.39996) % (Math.PI * 2),
                theta: 0
            });
        }

        var PARTICLES_COUNT = 48;
        var particles = [];
        for (var p = 0; p < PARTICLES_COUNT; p++) {
            particles.push({
                qIdx: p % Q_HEADS,
                t: Math.random(),
                speed: 0.3 + 0.3 * Math.random(),
                size: 1.2 + 1.2 * Math.random(),
                alpha: 0.4 + 0.6 * Math.random()
            });
        }

        var width = 0, height = 0, cx = 0, cy = 0, maxR = 0;
        function resize() {
            var rect = container.getBoundingClientRect();
            var dpr = Math.min(window.devicePixelRatio || 1, 2);
            width = rect.width;
            height = rect.height || 380;
            canvas.width = Math.round(width * dpr);
            canvas.height = Math.round(height * dpr);
            ctx.setTransform(1, 0, 0, 1, 0, 0);
            ctx.scale(dpr, dpr);
            cx = width / 2;
            cy = height / 2;
            maxR = Math.min(cx, cy) * 0.84;
        }
        window.addEventListener('resize', resize);
        resize();

        function setMode(mode) {
            currentMode = mode;
            modeBtns.forEach(function (btn) {
                if (btn.getAttribute('data-mode') === mode) {
                    btn.classList.add('active');
                } else {
                    btn.classList.remove('active');
                }
            });

            if (hudModeLabel) {
                if (mode === 'gqa') hudModeLabel.textContent = 'GQA 8Q / 4KV ATTENTION MANIFOLD';
                else if (mode === 'rope') hudModeLabel.textContent = 'RoPE θ=500K ROTARY PHASORS';
                else if (mode === 'chunkce') hudModeLabel.textContent = 'CHUNKED CROSS-ENTROPY 256-TOKEN SLICING';
                else if (mode === 'memory') hudModeLabel.textContent = '8-TECHNIQUE MEMORY STACK (78% VRAM SAVED)';
            }
        }

        modeBtns.forEach(function (btn) {
            btn.addEventListener('click', function () {
                setMode(btn.getAttribute('data-mode'));
            });
        });

        if (pauseBtn) {
            pauseBtn.addEventListener('click', function () {
                isPaused = !isPaused;
                pauseBtn.innerHTML = isPaused ? '&#9658;' : '&#10074;&#10074;';
                pauseBtn.title = isPaused ? 'Resume simulation' : 'Pause simulation';
            });
        }

        if (speedBtn) {
            speedBtn.addEventListener('click', function () {
                if (simSpeed === 1.0) { simSpeed = 2.0; speedBtn.textContent = '2×'; }
                else if (simSpeed === 2.0) { simSpeed = 0.5; speedBtn.textContent = '0.5×'; }
                else { simSpeed = 1.0; speedBtn.textContent = '1×'; }
            });
        }

        canvas.addEventListener('click', function (e) {
            var rect = canvas.getBoundingClientRect();
            var x = e.clientX - rect.left;
            var y = e.clientY - rect.top;
            pulseWaves.push({ x: x, y: y, r: 0, maxR: maxR * 1.2, alpha: 1.0 });
        });

        canvas.addEventListener('mousemove', function (e) {
            var rect = canvas.getBoundingClientRect();
            mouseX = e.clientX - rect.left;
            mouseY = e.clientY - rect.top;
        });

        canvas.addEventListener('mouseleave', function () {
            mouseX = -9999;
            mouseY = -9999;
            if (probeEl) probeEl.style.opacity = '0';
        });

        function drawBackground() {
            ctx.fillStyle = PALETTE.paperBg;
            ctx.fillRect(0, 0, width, height);

            var grad = ctx.createRadialGradient(cx, cy, 0, cx, cy, maxR * 1.3);
            grad.addColorStop(0, PALETTE.paperCenter);
            grad.addColorStop(0.7, '#100e0b');
            grad.addColorStop(1, PALETTE.paperBg);
            ctx.fillStyle = grad;
            ctx.fillRect(0, 0, width, height);

            ctx.strokeStyle = PALETTE.gridRule;
            ctx.lineWidth = 1;
            ctx.setLineDash([2, 4]);

            for (var ring = 1; ring <= 4; ring++) {
                ctx.beginPath();
                ctx.arc(cx, cy, (maxR * ring) / 4, 0, Math.PI * 2);
                ctx.stroke();
            }

            ctx.beginPath();
            ctx.moveTo(cx - maxR * 1.1, cy);
            ctx.lineTo(cx + maxR * 1.1, cy);
            ctx.moveTo(cx, cy - maxR * 1.1);
            ctx.lineTo(cx, cy + maxR * 1.1);
            ctx.stroke();
            ctx.setLineDash([]);
        }

        function drawGQAMode(t) {
            var outerR = maxR * 0.85;
            var innerR = maxR * 0.42;

            ctx.strokeStyle = PALETTE.oliveGlow;
            ctx.lineWidth = 2;
            ctx.beginPath();
            ctx.arc(cx, cy, innerR, 0, Math.PI * 2);
            ctx.stroke();

            ctx.strokeStyle = PALETTE.terracottaGlow;
            ctx.lineWidth = 1.5;
            ctx.beginPath();
            ctx.arc(cx, cy, outerR, 0, Math.PI * 2);
            ctx.stroke();

            var hoveredNode = null;

            for (var q = 0; q < Q_HEADS; q++) {
                var qNode = qHeadNodes[q];
                var qx = cx + Math.cos(qNode.angle) * outerR;
                var qy = cy + Math.sin(qNode.angle) * outerR;

                var kvNode = kvHeadNodes[qNode.kvIdx];
                var kvx = cx + Math.cos(kvNode.angle) * innerR;
                var kvy = cy + Math.sin(kvNode.angle) * innerR;

                var flow = 0.5 + 0.5 * Math.sin(t * 2.5 + q * 0.8);
                ctx.strokeStyle = (q % 2 === 0) ? 'rgba(224, 122, 63, ' + (0.2 + flow * 0.45) + ')' :
                                                  'rgba(154, 148, 64, ' + (0.2 + flow * 0.45) + ')';
                ctx.lineWidth = 1.5 + flow * 1.5;

                ctx.beginPath();
                ctx.moveTo(qx, qy);
                ctx.quadraticCurveTo(cx, cy, kvx, kvy);
                ctx.stroke();

                ctx.fillStyle = PALETTE.terracotta;
                ctx.beginPath();
                ctx.arc(qx, qy, 5 + flow * 2, 0, Math.PI * 2);
                ctx.fill();

                ctx.fillStyle = PALETTE.ink;
                ctx.font = '9px JetBrains Mono, monospace';
                ctx.textAlign = 'center';
                ctx.textBaseline = 'middle';
                var labelX = cx + Math.cos(qNode.angle) * (outerR + 16);
                var labelY = cy + Math.sin(qNode.angle) * (outerR + 16);
                ctx.fillText('Q' + q, labelX, labelY);

                var dQ = Math.hypot(mouseX - qx, mouseY - qy);
                if (dQ < 14) {
                    hoveredNode = { type: 'Query Head', id: q, kvIdx: qNode.kvIdx, flow: flow };
                }
            }

            for (var k = 0; k < KV_HEADS; k++) {
                var kNode = kvHeadNodes[k];
                var kx = cx + Math.cos(kNode.angle) * innerR;
                var ky = cy + Math.sin(kNode.angle) * innerR;

                var kPulse = 0.5 + 0.5 * Math.cos(t * 2.0 + k * 1.2);
                ctx.fillStyle = PALETTE.olive;
                ctx.beginPath();
                ctx.arc(kx, ky, 7 + kPulse * 2, 0, Math.PI * 2);
                ctx.fill();

                ctx.strokeStyle = PALETTE.gold;
                ctx.lineWidth = 1.5;
                ctx.stroke();

                ctx.fillStyle = PALETTE.coreHot;
                ctx.font = 'bold 8px JetBrains Mono, monospace';
                ctx.textAlign = 'center';
                ctx.textBaseline = 'middle';
                ctx.fillText('KV' + k, kx, ky);

                var dKV = Math.hypot(mouseX - kx, mouseY - ky);
                if (dKV < 16) {
                    hoveredNode = { type: 'Shared KV Head', id: k, pairedQ: [k * 2, k * 2 + 1], pulse: kPulse };
                }
            }

            for (var p = 0; p < PARTICLES_COUNT; p++) {
                var pt = particles[p];
                pt.t = (pt.t + 0.015 * pt.speed * simSpeed) % 1.0;

                var qN = qHeadNodes[pt.qIdx];
                var kvN = kvHeadNodes[qN.kvIdx];

                var px0 = cx + Math.cos(qN.angle) * outerR;
                var py0 = cy + Math.sin(qN.angle) * outerR;
                var px1 = cx + Math.cos(kvN.angle) * innerR;
                var py1 = cy + Math.sin(kvN.angle) * innerR;

                var u = pt.t;
                var invU = 1 - u;
                var currX = invU * invU * px0 + 2 * invU * u * cx + u * u * px1;
                var currY = invU * invU * py0 + 2 * invU * u * cy + u * u * py1;

                ctx.fillStyle = (pt.qIdx % 2 === 0) ? PALETTE.terracotta : PALETTE.olive;
                ctx.globalAlpha = pt.alpha * (1 - pt.t);
                ctx.beginPath();
                ctx.arc(currX, currY, pt.size, 0, Math.PI * 2);
                ctx.fill();
                ctx.globalAlpha = 1.0;
            }

            if (hoveredNode && probeEl) {
                probeEl.style.opacity = '1';
                probeEl.style.left = (mouseX + 14) + 'px';
                probeEl.style.top = (mouseY - 14) + 'px';
                if (probeTag) probeTag.textContent = hoveredNode.type + ' #' + hoveredNode.id;
                if (probeCoords) {
                    if (hoveredNode.type === 'Query Head') {
                        probeCoords.textContent = 'Routes to KV Head #' + hoveredNode.kvIdx + ' (128 dims)';
                        if (probeDecay) probeDecay.textContent = 'Attention Activity: ' + (hoveredNode.flow * 100).toFixed(1) + '%';
                    } else {
                        probeCoords.textContent = 'Serves Queries Q' + hoveredNode.pairedQ[0] + ' & Q' + hoveredNode.pairedQ[1];
                        if (probeDecay) probeDecay.textContent = 'KV Cache Footprint: 64 MiB/seq at S=2048 (2× Cut)';
                    }
                }
            } else if (probeEl) {
                probeEl.style.opacity = '0';
            }
        }

        function drawRoPEMode(t) {
            var rMax = maxR * 0.9;

            for (var i = 0; i < ROPE_DIMS; i += 2) {
                var p = ropePhasors[i];
                var ang = p.phase0 + t * p.freq * 15 * simSpeed;
                var px = cx + Math.cos(ang) * (p.r * rMax);
                var py = cy + Math.sin(ang) * (p.r * rMax);

                var isHighFreq = (i < 16);
                var color = isHighFreq ? PALETTE.terracotta : (i < 48 ? PALETTE.olive : PALETTE.gold);

                ctx.strokeStyle = color;
                ctx.globalAlpha = 0.25;
                ctx.lineWidth = 1;
                ctx.beginPath();
                ctx.arc(cx, cy, p.r * rMax, 0, Math.PI * 2);
                ctx.stroke();

                ctx.globalAlpha = 0.9;
                ctx.fillStyle = color;
                ctx.beginPath();
                ctx.arc(px, py, 3.5, 0, Math.PI * 2);
                ctx.fill();

                ctx.lineWidth = 1;
                ctx.beginPath();
                ctx.moveTo(cx, cy);
                ctx.lineTo(px, py);
                ctx.stroke();
            }
            ctx.globalAlpha = 1.0;

            ctx.fillStyle = PALETTE.coreHot;
            ctx.beginPath();
            ctx.arc(cx, cy, 6, 0, Math.PI * 2);
            ctx.fill();
            ctx.strokeStyle = PALETTE.gold;
            ctx.lineWidth = 2;
            ctx.stroke();

            ctx.fillStyle = PALETTE.ink;
            ctx.font = '10px JetBrains Mono, monospace';
            ctx.textAlign = 'center';
            ctx.fillText('θ = 500,000', cx, cy + 22);

            if (mouseX > 0 && probeEl) {
                var dCenter = Math.hypot(mouseX - cx, mouseY - cy);
                var dimEstimate = Math.min(63, Math.max(0, Math.round((dCenter / rMax) * 64)));
                probeEl.style.opacity = '1';
                probeEl.style.left = (mouseX + 14) + 'px';
                probeEl.style.top = (mouseY - 14) + 'px';
                if (probeTag) probeTag.textContent = 'RoPE Dimension Pair #' + (dimEstimate * 2);
                if (probeCoords) {
                    var lambdaEst = (2 * Math.PI / Math.pow(500000, -2 * dimEstimate / 128)).toFixed(0);
                    probeCoords.textContent = 'Wavelength λ ≈ ' + lambdaEst + ' tokens';
                }
                if (probeDecay) probeDecay.textContent = 'Extrapolation Range: 8,192+ Context (No aliasing)';
            }
        }

        function drawChunkCEMode(t) {
            var boxW = width * 0.72;
            var boxH = height * 0.62;
            var bx = cx - boxW / 2;
            var by = cy - boxH / 2;

            ctx.fillStyle = 'rgba(22, 19, 16, 0.85)';
            ctx.fillRect(bx, by, boxW, boxH);
            ctx.strokeStyle = PALETTE.gridRule;
            ctx.lineWidth = 1;
            ctx.strokeRect(bx, by, boxW, boxH);

            ctx.fillStyle = PALETTE.inkFaint;
            ctx.font = '9px JetBrains Mono, monospace';
            ctx.textAlign = 'left';
            ctx.fillText('FULL LOGIT TENSOR: [196,608 × 128,000] · 50.3 GB BF16 (OOM ON 1× GPU)', bx + 8, by - 8);

            var slices = 12;
            var sliceW = boxW / slices;
            var activeSlice = Math.floor((t * 2.5 * simSpeed) % slices);

            for (var s = 0; s < slices; s++) {
                var sx = bx + s * sliceW;
                var isCurrent = (s === activeSlice);

                ctx.fillStyle = isCurrent ? 'rgba(224, 122, 63, 0.45)' : 'rgba(154, 148, 64, 0.12)';
                ctx.fillRect(sx, by, sliceW - 2, boxH);

                ctx.strokeStyle = isCurrent ? PALETTE.terracotta : PALETTE.gridRule;
                ctx.lineWidth = isCurrent ? 2 : 1;
                ctx.strokeRect(sx, by, sliceW - 2, boxH);

                if (isCurrent) {
                    ctx.fillStyle = PALETTE.coreHot;
                    ctx.font = 'bold 9px JetBrains Mono, monospace';
                    ctx.textAlign = 'center';
                    ctx.fillText('SRAM TILE (0.20 GB)', sx + sliceW / 2, by + boxH / 2);
                    ctx.fillText('CHUNK #' + s, sx + sliceW / 2, by + boxH / 2 + 14);

                    ctx.strokeStyle = PALETTE.terracottaGlow;
                    ctx.lineWidth = 2;
                    ctx.beginPath();
                    ctx.moveTo(sx + sliceW / 2, by + boxH);
                    ctx.lineTo(sx + sliceW / 2, by + boxH + 30);
                    ctx.stroke();
                }
            }

            ctx.fillStyle = PALETTE.terracotta;
            ctx.font = 'bold 10px JetBrains Mono, monospace';
            ctx.textAlign = 'center';
            ctx.fillText('99.4% PEAK MEMORY REDUCTION VIA 256-TOKEN STREAMING SLICES + Z-LOSS', cx, by + boxH + 32);
        }

        function drawMemoryMode(t) {
            var barW = width * 0.68;
            var barH = 32;
            var startX = cx - barW / 2;
            var startY = cy - 45;

            ctx.fillStyle = PALETTE.inkFaint;
            ctx.font = '10px JetBrains Mono, monospace';
            ctx.textAlign = 'left';
            ctx.fillText('UNOPTIMIZED DENSE BASELINE: 92.4 GB VRAM (OOM ON 1× A100-80GB)', startX, startY - 8);

            ctx.fillStyle = 'rgba(232, 96, 110, 0.4)';
            ctx.fillRect(startX, startY, barW, barH);
            ctx.strokeStyle = '#e8606e';
            ctx.lineWidth = 1.5;
            ctx.strokeRect(startX, startY, barW, barH);

            var optY = startY + 65;
            var optW = barW * (20.2 / 92.4);

            ctx.fillStyle = PALETTE.olive;
            ctx.fillText('LLaMA-3-Lite 8-TECHNIQUE MEMORY STACK: 20.2 GB VRAM (78% REDUCTION / 4× HEADROOM)', startX, optY - 8);

            var grad = ctx.createLinearGradient(startX, optY, startX + optW, optY);
            grad.addColorStop(0, PALETTE.terracotta);
            grad.addColorStop(0.5, PALETTE.olive);
            grad.addColorStop(1, PALETTE.gold);

            ctx.fillStyle = grad;
            ctx.fillRect(startX, optY, optW, barH);
            ctx.strokeStyle = PALETTE.gold;
            ctx.lineWidth = 1.5;
            ctx.strokeRect(startX, optY, optW, barH);

            ctx.fillStyle = PALETTE.coreHot;
            ctx.font = 'bold 11px JetBrains Mono, monospace';
            ctx.textAlign = 'center';
            ctx.fillText('20.2 GB PEAK (BATCH 96 FIT)', startX + optW / 2, optY + 20);
        }

        function drawShockwaves() {
            for (var i = pulseWaves.length - 1; i >= 0; i--) {
                var w = pulseWaves[i];
                w.r += 3.5 * simSpeed;
                w.alpha *= 0.96;

                ctx.strokeStyle = 'rgba(224, 122, 63, ' + w.alpha + ')';
                ctx.lineWidth = 2;
                ctx.beginPath();
                ctx.arc(w.x, w.y, w.r, 0, Math.PI * 2);
                ctx.stroke();

                if (w.alpha < 0.02 || w.r > w.maxR) {
                    pulseWaves.splice(i, 1);
                }
            }
        }

        function render(timestamp) {
            if (!lastTime) lastTime = timestamp;
            var dt = (timestamp - lastTime) / 1000;
            lastTime = timestamp;

            if (!isPaused && !reduced) {
                simTime += dt * simSpeed;
            }

            drawBackground();

            if (currentMode === 'gqa') drawGQAMode(simTime);
            else if (currentMode === 'rope') drawRoPEMode(simTime);
            else if (currentMode === 'chunkce') drawChunkCEMode(simTime);
            else if (currentMode === 'memory') drawMemoryMode(simTime);

            drawShockwaves();

            animId = requestAnimationFrame(render);
        }

        animId = requestAnimationFrame(render);
    }

    // ------------------------------------------------------------------
    // 2. Full 16-Layer Training Step Pipeline (FIG · A1)
    // ------------------------------------------------------------------
    function initPassDiagram() {
        var canvas = document.getElementById('passDiagramCanvas');
        if (!canvas) return;

        var ctx = canvas.getContext('2d');
        if (!ctx) return;

        var container = canvas.parentElement;
        var tooltipEl = document.getElementById('passStageTooltip');
        var stTag = document.getElementById('stTag');
        var stOp = document.getElementById('stOp');
        var stShape = document.getElementById('stShape');
        var stDesc = document.getElementById('stDesc');
        var phaseEl = document.getElementById('phCurrentPhase');
        var tickerText = document.getElementById('passTickerText');
        var pauseBtn = document.getElementById('passPauseBtn');
        var phaseBtns = document.querySelectorAll('.pass-controls .pass-btn[data-phase]');

        var currentPhaseMode = 'cycle';
        var isPaused = false;
        var isHovered = false;
        var mouseX = -9999, mouseY = -9999;
        var hoveredStation = null;
        var animId = null;
        var lastTime = 0;
        var simTime = 0;

        var PALETTE = {
            paperBg: '#0e0c0a',
            paperStation: '#161310',
            paperStationHover: '#1c1813',
            rule: '#2c261c',
            ruleStrong: '#3a3226',
            terracotta: '#e07a3f',
            terracottaGlow: 'rgba(224, 122, 63, 0.75)',
            olive: '#9a9440',
            oliveGlow: 'rgba(154, 148, 64, 0.75)',
            gold: '#c9a35c',
            goldGlow: 'rgba(201, 163, 92, 0.8)',
            ink: '#d8ccb4',
            inkSoft: '#b3a68c',
            inkFaint: '#7a7160'
        };

        var STAGES = [
            {
                id: 1,
                tag: 'STAGE 01 · UNTIED EMBEDDING',
                badge: '01 · EMB',
                title: 'Embed (x_t)',
                sub: '128K → 1024',
                chip: 'Disk Prefetch',
                op: 'h_0 = Embedding(x_t) · Untied Weights',
                shape: 'Input: [B=96, 2048] uint32 → Output: [B, 2048, 1024] bf16',
                desc: 'Disk-backed uint32 token cache prefetch; separate LM head parameter weight matrix'
            },
            {
                id: 2,
                tag: 'STAGE 02 · RMSNORM PRE-NORM',
                badge: '02 · NORM',
                title: 'Pre-RMSNorm',
                sub: 'RMS(x) · γ',
                chip: 'Bias-Free',
                op: 'x_norm = x / sqrt(mean(x^2) + 1e-5) * weight',
                shape: 'Normalized: [B, 2048, 1024] bf16',
                desc: 'Bias-free RMSNorm pre-norm; optional fused Triton kernel'
            },
            {
                id: 3,
                tag: 'STAGE 03 · GQA 8Q / 4KV ATTN',
                badge: '03 · GQA',
                title: 'GQA + RoPE',
                sub: 'Flash-Attn 2',
                chip: '2× KV Cut',
                op: 'Attention(Q_8, K_4, V_4) · RoPE θ=500K',
                shape: 'Q: [B, 8, 2048, 128], KV: [B, 4, 2048, 128] bf16',
                desc: '2× KV cache compression; RoPE θ=500K long context extrapolation up to 8192'
            },
            {
                id: 4,
                tag: 'STAGE 04 · FUSED SWIGLU FFN',
                badge: '04 · FFN',
                title: 'SwiGLU FFN',
                sub: '1024 → 4096',
                chip: 'Fused GEMM',
                op: 'FFN(x) = (SiLU(x·W_gate) ⊙ (x·W_up)) · W_down',
                shape: 'Intermediate: [B, 2048, 4096] bf16',
                desc: 'Fused gate_up projection in a single GEMM; elementwise SiLU in SRAM'
            },
            {
                id: 5,
                tag: 'STAGE 05 · GRAD CHECKPOINTING',
                badge: '05 · RECOMP',
                title: 'Recompute',
                sub: '16 Blocks',
                chip: '78% Saved',
                op: 'Save layer boundary hidden states; discard intra-layer buffers',
                shape: 'Boundary: [16, B, 2048, 1024] bf16',
                desc: 'Cuts activation memory from 70 GB to 1 layer buffer during forward pass'
            },
            {
                id: 6,
                tag: 'STAGE 06 · CHUNKED CE & ADAMW',
                badge: '06 · LOSS',
                title: 'Chunked CE',
                sub: '256 Slices',
                chip: '99.4% Cut',
                op: 'Loss = Chunked_CE(h_L, W_head) + α·log²(Z) → AdamW',
                shape: 'SRAM Slice: [256, 128000] · Parameter Step: Δθ',
                desc: 'Avoids 50.3 GB logit tensor; directly streams backprop into parameter update'
            }
        ];

        var forwardParticles = [];
        for (var f = 0; f < 24; f++) {
            forwardParticles.push({
                xFrac: f / 24,
                speed: 0.18 + 0.08 * Math.random(),
                size: 1.6 + 1.0 * Math.random(),
                lane: (f % 3) - 1
            });
        }

        var backwardParticles = [];
        for (var b = 0; b < 24; b++) {
            backwardParticles.push({
                xFrac: b / 24,
                speed: 0.20 + 0.08 * Math.random(),
                size: 1.6 + 1.0 * Math.random(),
                lane: (b % 3) - 1
            });
        }

        var adamParticles = [];
        var width = 0, height = 0;
        var stations = [];

        function layoutStations() {
            var rect = container.getBoundingClientRect();
            var dpr = Math.min(window.devicePixelRatio || 1, 2);
            width = rect.width;
            height = rect.height || 260;
            canvas.width = Math.round(width * dpr);
            canvas.height = Math.round(height * dpr);
            ctx.setTransform(1, 0, 0, 1, 0, 0);
            ctx.scale(dpr, dpr);

            stations = [];
            var numStages = STAGES.length;
            var marginX = 16;
            var availW = width - (marginX * 2);
            var gap = Math.max(8, Math.min(16, (availW - (numStages * 82)) / (numStages - 1)));
            var stationW = Math.max(76, (availW - (gap * (numStages - 1))) / numStages);
            var stationH = Math.min(124, height * 0.52);
            var stationY = (height - stationH) / 2 + 2;

            for (var i = 0; i < numStages; i++) {
                var sx = marginX + i * (stationW + gap);
                stations.push({
                    stage: STAGES[i],
                    x: sx,
                    y: stationY,
                    w: stationW,
                    h: stationH,
                    cx: sx + stationW / 2,
                    cy: stationY + stationH / 2,
                    glowForward: 0,
                    glowBackward: 0,
                    glowAdam: 0
                });
            }
        }

        if (window.ResizeObserver) {
            var ro = new ResizeObserver(function () {
                layoutStations();
            });
            ro.observe(container);
        } else {
            window.addEventListener('resize', layoutStations);
        }
        layoutStations();

        container.addEventListener('mousemove', function (e) {
            var rect = canvas.getBoundingClientRect();
            mouseX = e.clientX - rect.left;
            mouseY = e.clientY - rect.top;
            isHovered = true;
        });

        container.addEventListener('mouseleave', function () {
            isHovered = false;
            mouseX = -9999;
            mouseY = -9999;
            hoveredStation = null;
            if (tooltipEl) tooltipEl.style.opacity = '0';
        });

        container.addEventListener('click', function (e) {
            var rect = canvas.getBoundingClientRect();
            var clickX = e.clientX - rect.left;
            var clickY = e.clientY - rect.top;

            stations.forEach(function (st) {
                if (clickX >= st.x && clickX <= st.x + st.w && clickY >= st.y && clickY <= st.y + st.h) {
                    st.glowForward = 1.0;
                    st.glowBackward = 1.0;
                    for (var k = 0; k < 12; k++) {
                        adamParticles.push({
                            x: st.cx,
                            y: st.cy,
                            vx: (Math.random() - 0.5) * 160,
                            vy: (Math.random() - 0.5) * 160,
                            life: 1.0,
                            color: PALETTE.gold
                        });
                    }
                }
            });
        });

        if (pauseBtn) {
            pauseBtn.addEventListener('click', function (e) {
                e.stopPropagation();
                isPaused = !isPaused;
                pauseBtn.innerHTML = isPaused ? '&#9658;' : '&#10074;&#10074;';
                pauseBtn.setAttribute('aria-label', isPaused ? 'Resume pipeline' : 'Pause pipeline');
            });
        }

        phaseBtns.forEach(function (btn) {
            btn.addEventListener('click', function (e) {
                e.stopPropagation();
                phaseBtns.forEach(function (b) { b.classList.remove('active'); });
                btn.classList.add('active');
                currentPhaseMode = btn.getAttribute('data-phase') || 'cycle';
            });
        });

        var CYCLE_DURATION = 8.6;

        function getCycleState(t) {
            if (currentPhaseMode === 'forward') {
                return { phase: 'forward', progress: (t * 0.4) % 1.0, subText: 'FORWARD ACTIVATIONS STREAMING' };
            }
            if (currentPhaseMode === 'backward') {
                return { phase: 'backward', progress: (t * 0.4) % 1.0, subText: 'AUTOGRAD GRADIENT PROPAGATION & CHECKPOINTING' };
            }

            var cycleT = t % CYCLE_DURATION;
            if (cycleT < 3.2) {
                return { phase: 'forward', progress: cycleT / 3.2, subText: 'FORWARD · Activations x_t → 16× Blocks → Hidden States' };
            } else if (cycleT < 4.2) {
                return { phase: 'loss', progress: (cycleT - 3.2) / 1.0, subText: 'LOSS COMPUTATION · Chunked Cross-Entropy (78% VRAM Saved)' };
            } else if (cycleT < 7.0) {
                return { phase: 'backward', progress: (cycleT - 4.2) / 2.8, subText: 'AUTOGRAD BACKWARD · Checkpointing Recomputes Activations' };
            } else if (cycleT < 8.0) {
                return { phase: 'adam', progress: (cycleT - 7.0) / 1.0, subText: 'ADAMW STEP · Weight Updates θ ← θ - η·∇ℓ & RNG Snapshot' };
            } else {
                return { phase: 'rest', progress: (cycleT - 8.0) / 0.6, subText: 'STEP COMMITTED · Next Mini-Batch Ingest' };
            }
        }

        function drawBackground() {
            ctx.fillStyle = PALETTE.paperBg;
            ctx.fillRect(0, 0, width, height);

            ctx.save();
            ctx.strokeStyle = 'rgba(58, 50, 38, 0.22)';
            ctx.lineWidth = 1;
            ctx.setLineDash([2, 8]);
            for (var x = 20; x < width; x += 40) {
                ctx.beginPath();
                ctx.moveTo(x, 0);
                ctx.lineTo(x, height);
                ctx.stroke();
            }
            ctx.setLineDash([]);

            var fwdY = height * 0.18;
            ctx.beginPath();
            ctx.moveTo(14, fwdY);
            ctx.lineTo(width - 14, fwdY);
            ctx.strokeStyle = 'rgba(224, 122, 63, 0.28)';
            ctx.lineWidth = 1.5;
            ctx.setLineDash([4, 6]);
            ctx.stroke();

            var bwdY = height * 0.82;
            ctx.beginPath();
            ctx.moveTo(14, bwdY);
            ctx.lineTo(width - 14, bwdY);
            ctx.strokeStyle = 'rgba(154, 148, 64, 0.28)';
            ctx.lineWidth = 1.5;
            ctx.setLineDash([4, 6]);
            ctx.stroke();
            ctx.setLineDash([]);

            ctx.fillStyle = PALETTE.terracotta;
            ctx.font = 'bold 8.5px "JetBrains Mono", monospace';
            ctx.textAlign = 'left';
            ctx.fillText('FORWARD CONDUIT (ACTIVATIONS) →', 20, fwdY - 6);

            ctx.fillStyle = PALETTE.olive;
            ctx.font = 'bold 8.5px "JetBrains Mono", monospace';
            ctx.textAlign = 'right';
            ctx.fillText('← BACKWARD CONDUIT (GRADIENTS)', width - 20, bwdY + 14);
            ctx.restore();
        }

        function drawParticles(state, dt) {
            ctx.save();
            var fwdY = height * 0.18;
            var bwdY = height * 0.82;

            if (state.phase === 'forward' || state.phase === 'loss' || currentPhaseMode === 'forward') {
                forwardParticles.forEach(function (p) {
                    if (!reduced) p.xFrac += p.speed * dt;
                    if (p.xFrac > 1.0) p.xFrac = 0.0;

                    var px = 16 + p.xFrac * (width - 32);
                    var py = fwdY + p.lane * 3.0;

                    ctx.beginPath();
                    ctx.arc(px, py, p.size, 0, Math.PI * 2);
                    ctx.fillStyle = PALETTE.terracottaGlow;
                    ctx.fill();

                    ctx.beginPath();
                    ctx.moveTo(px, py);
                    ctx.lineTo(px - 14 * p.speed, py);
                    ctx.strokeStyle = 'rgba(224, 122, 63, 0.35)';
                    ctx.lineWidth = p.size * 0.7;
                    ctx.stroke();
                });
            }

            if (state.phase === 'backward' || currentPhaseMode === 'backward') {
                backwardParticles.forEach(function (p) {
                    if (!reduced) p.xFrac += p.speed * dt;
                    if (p.xFrac > 1.0) p.xFrac = 0.0;

                    var px = width - 16 - p.xFrac * (width - 32);
                    var py = bwdY + p.lane * 3.0;

                    ctx.beginPath();
                    ctx.arc(px, py, p.size, 0, Math.PI * 2);
                    ctx.fillStyle = PALETTE.oliveGlow;
                    ctx.fill();

                    ctx.beginPath();
                    ctx.moveTo(px, py);
                    ctx.lineTo(px + 14 * p.speed, py);
                    ctx.strokeStyle = 'rgba(154, 148, 64, 0.35)';
                    ctx.lineWidth = p.size * 0.7;
                    ctx.stroke();
                });
            }

            for (var k = adamParticles.length - 1; k >= 0; k--) {
                var ap = adamParticles[k];
                ap.x += ap.vx * dt;
                ap.y += ap.vy * dt;
                ap.life -= dt * 1.6;

                if (ap.life <= 0) {
                    adamParticles.splice(k, 1);
                    continue;
                }

                ctx.beginPath();
                ctx.arc(ap.x, ap.y, 2.0 * ap.life, 0, Math.PI * 2);
                ctx.fillStyle = 'rgba(201, 163, 92, ' + ap.life.toFixed(2) + ')';
                ctx.fill();
            }

            ctx.restore();
        }

        function drawStations(state, dt) {
            ctx.save();
            var closest = null;
            var numStations = stations.length;

            stations.forEach(function (st, idx) {
                var frac = idx / (numStations - 1);

                var isFwdActive = false;
                var isBwdActive = false;
                var isAdamActive = (state.phase === 'adam');

                if (state.phase === 'forward') {
                    isFwdActive = state.progress >= (frac - 0.12) && state.progress <= (frac + 0.18);
                } else if (state.phase === 'loss') {
                    isFwdActive = (idx === numStations - 1);
                } else if (state.phase === 'backward') {
                    var bwdFrac = 1.0 - frac;
                    isBwdActive = state.progress >= (bwdFrac - 0.12) && state.progress <= (bwdFrac + 0.18);
                }

                if (isFwdActive) st.glowForward = 1.0;
                else st.glowForward = Math.max(0, st.glowForward - dt * 2.2);

                if (isBwdActive) st.glowBackward = 1.0;
                else st.glowBackward = Math.max(0, st.glowBackward - dt * 2.2);

                if (isAdamActive) st.glowAdam = 1.0;
                else st.glowAdam = Math.max(0, st.glowAdam - dt * 2.0);

                var isHover = isHovered && (mouseX >= st.x && mouseX <= st.x + st.w && mouseY >= st.y && mouseY <= st.y + st.h);
                if (isHover) closest = st;

                var cardBg = isHover ? 'rgba(36, 30, 22, 0.98)' : PALETTE.paperStation;
                var borderColor = PALETTE.ruleStrong;

                if (isHover) {
                    borderColor = PALETTE.gold;
                } else if (st.glowAdam > 0.1) {
                    borderColor = PALETTE.gold;
                    cardBg = 'rgba(201, 163, 92, ' + (0.15 * st.glowAdam).toFixed(2) + ')';
                } else if (st.glowForward > 0.1) {
                    borderColor = PALETTE.terracotta;
                    cardBg = 'rgba(224, 122, 63, ' + (0.14 * st.glowForward).toFixed(2) + ')';
                } else if (st.glowBackward > 0.1) {
                    borderColor = PALETTE.olive;
                    cardBg = 'rgba(154, 148, 64, ' + (0.14 * st.glowBackward).toFixed(2) + ')';
                }

                ctx.fillStyle = cardBg;
                ctx.strokeStyle = isHover ? PALETTE.gold : borderColor;
                ctx.lineWidth = isHover ? 2.0 : ((st.glowForward > 0.3 || st.glowBackward > 0.3) ? 1.5 : 1.0);

                ctx.beginPath();
                ctx.rect(st.x, st.y, st.w, st.h);
                ctx.fill();
                ctx.stroke();

                var crLen = isHover ? 6 : 4;
                ctx.strokeStyle = isHover ? PALETTE.gold : PALETTE.ruleStrong;
                ctx.lineWidth = isHover ? 1.5 : 1;
                ctx.beginPath();
                ctx.moveTo(st.x, st.y + crLen); ctx.lineTo(st.x, st.y); ctx.lineTo(st.x + crLen, st.y);
                ctx.moveTo(st.x + st.w - crLen, st.y); ctx.lineTo(st.x + st.w, st.y); ctx.lineTo(st.x + st.w, st.y + crLen);
                ctx.moveTo(st.x, st.y + st.h - crLen); ctx.lineTo(st.x, st.y + st.h); ctx.lineTo(st.x + crLen, st.y + st.h);
                ctx.moveTo(st.x + st.w - crLen, st.y + st.h); ctx.lineTo(st.x + st.w, st.y + st.h); ctx.lineTo(st.x + st.w, st.y + st.h - crLen);
                ctx.stroke();

                var ledColor = PALETTE.inkFaint;
                if (isHover) ledColor = PALETTE.gold;
                else if (st.glowAdam > 0.2) ledColor = PALETTE.gold;
                else if (st.glowForward > 0.2) ledColor = PALETTE.terracotta;
                else if (st.glowBackward > 0.2) ledColor = PALETTE.olive;

                ctx.beginPath();
                ctx.arc(st.x + 8, st.y + 10, isHover ? 3.0 : 2.5, 0, Math.PI * 2);
                ctx.fillStyle = ledColor;
                ctx.fill();

                var isCompact = st.w < 78;
                var displayBadge = isCompact ? '0' + st.stage.id : st.stage.badge;
                var displayTitle = isCompact ? (st.stage.id === 1 ? 'Embed' : st.stage.id === 2 ? 'Norm' : st.stage.id === 3 ? 'GQA' : st.stage.id === 4 ? 'SwiGLU' : st.stage.id === 5 ? 'Recomp' : 'Loss') : st.stage.title;

                function drawFitText(text, fontSize, isBold, color, yPos) {
                    var curSize = fontSize;
                    ctx.font = (isBold ? 'bold ' : '') + curSize + 'px "JetBrains Mono", monospace';
                    var maxTextW = st.w - 10;
                    while (ctx.measureText(text).width > maxTextW && curSize > 6.5) {
                        curSize -= 0.5;
                        ctx.font = (isBold ? 'bold ' : '') + curSize + 'px "JetBrains Mono", monospace';
                    }
                    ctx.fillStyle = color;
                    ctx.textAlign = 'center';
                    ctx.fillText(text, st.cx, yPos);
                }

                // Stage Number Badge
                ctx.font = isHover ? 'bold 8.5px "JetBrains Mono", monospace' : (isCompact ? 'bold 7.5px "JetBrains Mono", monospace' : 'bold 8.5px "JetBrains Mono", monospace');
                ctx.fillStyle = isHover ? PALETTE.terracotta : (st.glowForward > 0.2 ? PALETTE.terracotta : (st.glowBackward > 0.2 ? PALETTE.olive : PALETTE.inkSoft));
                ctx.textAlign = 'left';
                ctx.fillText(displayBadge, st.x + (isCompact ? 13 : 14), st.y + 13);

                // Stage Title
                drawFitText(displayTitle, isHover ? 10.5 : (isCompact ? 9 : 10), true, isHover ? '#ffffff' : PALETTE.ink, st.y + st.h * 0.40);

                // Subtitle
                if (st.h > 80) {
                    drawFitText(st.stage.sub, isHover ? 9 : (isCompact ? 7.5 : 8.5), isHover, isHover ? '#ffd780' : PALETTE.gold, st.y + st.h * 0.60);
                }

                // Chip Tag
                if (st.h > 100 && !isCompact) {
                    drawFitText(st.stage.chip, isHover ? 8.5 : 8, isHover, isHover ? '#fffaf0' : PALETTE.inkSoft, st.y + st.h * 0.78);
                }

                if ((isHover || st.glowBackward > 0.3) && (idx >= 1 && idx <= 4)) {
                    drawFitText('RE-COMPUTE', 8, true, isHover ? '#ffd780' : PALETTE.olive, st.y + st.h - 6);
                } else if (st.glowAdam > 0.3) {
                    drawFitText('θ UPDATE', 8, true, PALETTE.gold, st.y + st.h - 6);
                }
            });

            if (closest && tooltipEl) {
                tooltipEl.style.opacity = '1';
                var st = closest.stage;
                var cardW = Math.min(320, width - 24);
                var leftPx = closest.x + closest.w / 2 - cardW / 2;
                leftPx = Math.max(12, Math.min(width - cardW - 12, leftPx));

                var topPx = closest.y - 110;
                if (topPx < 36) {
                    topPx = closest.y + closest.h + 10;
                }

                tooltipEl.style.left = leftPx + 'px';
                tooltipEl.style.top = topPx + 'px';

                if (stTag) stTag.textContent = st.tag;
                if (stOp) stOp.textContent = st.op;
                if (stShape) stShape.textContent = st.shape;
                if (stDesc) stDesc.textContent = st.desc;
            } else if (tooltipEl) {
                tooltipEl.style.opacity = '0';
            }

            ctx.restore();
        }

        function updateHUD(state) {
            if (phaseEl) phaseEl.textContent = state.subText;
            if (tickerText) tickerText.textContent = state.subText;
        }

        function loop(timestamp) {
            if (!lastTime) lastTime = timestamp;
            var dt = Math.min((timestamp - lastTime) / 1000, 0.1);
            lastTime = timestamp;

            if (!isPaused && !reduced) {
                simTime += dt;
            }

            var state = getCycleState(simTime);
            drawBackground();
            drawParticles(state, dt);
            drawStations(state, dt);
            updateHUD(state);

            animId = requestAnimationFrame(loop);
        }

        animId = requestAnimationFrame(loop);
    }

    // ------------------------------------------------------------------
    // 3. Interactive Mechanism Benchmark Labs
    // ------------------------------------------------------------------

    /* Card 1: GQA 8Q / 4KV Compression & Router Analyzer */
    function initGqaMechanism() {
        var slider = document.getElementById('gqaContextSlider');
        var label = document.getElementById('gqaContextLabel');
        var statMha = document.getElementById('statMhaVram');
        var statGqa = document.getElementById('statGqaVram');
        var statSaved = document.getElementById('statGqaSaved');
        var canvas = document.getElementById('gqaRouterCanvas');
        var routeBtn = document.getElementById('gqaRouteBtn');

        if (!slider && !canvas) return;

        var activeQ = 0;
        var animT = 0;

        function updateMath() {
            var ctxLen = slider ? parseInt(slider.value, 10) : 32768;
            if (label) label.textContent = ctxLen.toLocaleString() + ' tokens';

            // MHA: 8 KV heads * 128 head_dim * 16 layers * 2 (K+V) * 2 bytes (bf16)
            var mhaBytes = ctxLen * 8 * 128 * 16 * 2 * 2;
            var mhaGB = mhaBytes / (1024 * 1024 * 1024);

            // GQA: 4 KV heads * 128 head_dim * 16 layers * 2 (K+V) * 2 bytes (bf16)
            var gqaBytes = ctxLen * 4 * 128 * 16 * 2 * 2;
            var gqaGB = gqaBytes / (1024 * 1024 * 1024);

            var savedGB = mhaGB - gqaGB;

            if (statMha) statMha.textContent = mhaGB >= 1 ? mhaGB.toFixed(2) + ' GB' : (mhaGB * 1024).toFixed(0) + ' MB';
            if (statGqa) statGqa.textContent = gqaGB >= 1 ? gqaGB.toFixed(2) + ' GB' : (gqaGB * 1024).toFixed(0) + ' MB';
            if (statSaved) {
                var savedStr = savedGB >= 1 ? savedGB.toFixed(2) + ' GB' : (savedGB * 1024).toFixed(0) + ' MB';
                statSaved.textContent = savedStr + ' (2.00× cut · 50% savings)';
            }
        }

        function drawRouter() {
            if (!canvas) return;
            var ctx = canvas.getContext('2d');
            if (!ctx) return;

            var dpr = Math.min(window.devicePixelRatio || 1, 2);
            var rect = canvas.getBoundingClientRect();
            if (canvas.width !== Math.round(rect.width * dpr)) {
                canvas.width = Math.round(rect.width * dpr);
                canvas.height = Math.round(rect.height * dpr);
            }
            ctx.setTransform(1, 0, 0, 1, 0, 0);
            ctx.scale(dpr, dpr);

            var w = rect.width, h = rect.height;
            ctx.fillStyle = '#0e0c0a';
            ctx.fillRect(0, 0, w, h);

            // Grid decoration
            ctx.strokeStyle = 'rgba(58, 50, 38, 0.3)';
            ctx.lineWidth = 0.5;
            ctx.setLineDash([2, 4]);
            ctx.beginPath();
            ctx.moveTo(w * 0.45, 10); ctx.lineTo(w * 0.45, h - 10);
            ctx.stroke();
            ctx.setLineDash([]);

            var kvTarget = Math.floor(activeQ / 2);

            // Draw 8 Queries on Left
            for (var q = 0; q < 8; q++) {
                var qy = 12 + q * 12.5;
                var isSel = (q === activeQ);

                ctx.fillStyle = isSel ? '#e07a3f' : 'rgba(122, 113, 96, 0.35)';
                ctx.fillRect(16, qy, 28, 9);

                ctx.strokeStyle = isSel ? '#fffaf0' : 'rgba(44, 38, 28, 0.8)';
                ctx.lineWidth = isSel ? 1.5 : 1;
                ctx.strokeRect(16, qy, 28, 9);

                ctx.fillStyle = isSel ? '#fffaf0' : '#b3a68c';
                ctx.font = 'bold 7.5px "JetBrains Mono", monospace';
                ctx.textAlign = 'center';
                ctx.fillText('Q' + q, 30, qy + 7);
            }

            // Draw 4 KV Heads on Right
            for (var k = 0; k < 4; k++) {
                var ky = 18 + k * 25;
                var isTarget = (k === kvTarget);

                ctx.fillStyle = isTarget ? 'rgba(154, 148, 64, 0.45)' : 'rgba(22, 19, 16, 0.9)';
                ctx.fillRect(w - 52, ky, 36, 16);

                ctx.strokeStyle = isTarget ? '#9a9440' : '#2c261c';
                ctx.lineWidth = isTarget ? 1.5 : 1;
                ctx.strokeRect(w - 52, ky, 36, 16);

                ctx.fillStyle = isTarget ? '#fffaf0' : '#7a7160';
                ctx.font = 'bold 8px "JetBrains Mono", monospace';
                ctx.textAlign = 'center';
                ctx.fillText('KV' + k, w - 34, ky + 11);
            }

            // Draw Projection Stream
            var qY = 12 + activeQ * 12.5 + 4.5;
            var kvY = 18 + kvTarget * 25 + 8;

            ctx.strokeStyle = '#e07a3f';
            ctx.lineWidth = 1.8;
            ctx.beginPath();
            ctx.moveTo(44, qY);
            ctx.bezierCurveTo(w * 0.45, qY, w * 0.45, kvY, w - 52, kvY);
            ctx.stroke();

            // Traveling Energy Pulse
            animT = (animT + 0.03) % 1.0;
            var invT = 1 - animT;
            var pX = invT * invT * 44 + 2 * invT * animT * (w * 0.45) + animT * animT * (w - 52);
            var pY = invT * invT * qY + 2 * invT * animT * ((qY + kvY) / 2) + animT * animT * kvY;

            ctx.fillStyle = '#fffaf0';
            ctx.beginPath();
            ctx.arc(pX, pY, 2.5, 0, Math.PI * 2);
            ctx.fill();

            requestAnimationFrame(drawRouter);
        }

        if (slider) slider.addEventListener('input', updateMath);
        if (routeBtn) {
            routeBtn.addEventListener('click', function () {
                activeQ = (activeQ + 1) % 8;
                updateMath();
            });
        }

        updateMath();
        drawRouter();
    }

    /* Card 2: Chunked Cross-Entropy 256-Token Slicer */
    function initChunkCeMechanism() {
        var slider = document.getElementById('chunkSizeSlider');
        var label = document.getElementById('chunkSizeLabel');
        var memVal = document.getElementById('chunkPeakMem');
        var tileText = document.getElementById('chunkTileText');
        var streamBtn = document.getElementById('chunkStreamBtn');
        var display = document.getElementById('chunkStreamDisplay');

        if (!slider && !streamBtn) return;

        function updateChunk() {
            var chunkSize = slider ? parseInt(slider.value, 10) : 256;
            if (label) label.textContent = chunkSize + ' Tokens';
            var memMB = ((chunkSize * 128000 * 2) / 1e6).toFixed(1);

            if (tileText) {
                tileText.textContent = chunkSize + ' × 128k = ' + memMB + ' MB SRAM Resident Tile';
            }
            if (memVal) {
                memVal.textContent = memMB + ' MB SRAM (vs 50.3 GB Full)';
            }
        }

        if (slider) slider.addEventListener('input', updateChunk);

        if (streamBtn) {
            streamBtn.addEventListener('click', function () {
                var stages = display ? display.querySelectorAll('.chunk-stage') : [];
                for (var s = 0; s < stages.length; s++) {
                    stages[s].style.opacity = '0.3';
                }
                var cur = 0;
                var iv = setInterval(function () {
                    if (cur < stages.length) {
                        stages[cur].style.opacity = '1';
                        cur++;
                    } else {
                        clearInterval(iv);
                    }
                }, 280);
            });
        }

        updateChunk();
    }

    /* Card 3: RoPE θ=500K Long-Context Extrapolation Lab */
    function initRopeMechanism() {
        var slider = document.getElementById('ropePosSlider');
        var label = document.getElementById('ropePosLabel');
        var canvas = document.getElementById('ropeExtrapCanvas');
        var statusEl = document.getElementById('ropePhaseStatus');
        var testBtn = document.getElementById('ropeTestBtn');

        if (!slider && !canvas) return;

        var animOffset = 0;

        function drawWaveforms() {
            if (!canvas) return;
            var ctx = canvas.getContext('2d');
            if (!ctx) return;

            var dpr = Math.min(window.devicePixelRatio || 1, 2);
            var rect = canvas.getBoundingClientRect();
            if (canvas.width !== Math.round(rect.width * dpr)) {
                canvas.width = Math.round(rect.width * dpr);
                canvas.height = Math.round(rect.height * dpr);
            }
            ctx.setTransform(1, 0, 0, 1, 0, 0);
            ctx.scale(dpr, dpr);

            var w = rect.width, h = rect.height;
            ctx.fillStyle = '#0e0c0a';
            ctx.fillRect(0, 0, w, h);

            var pos = slider ? parseInt(slider.value, 10) : 2048;
            if (label) label.textContent = 'Position: ' + pos.toLocaleString() + ' / 8,192';

            if (statusEl) {
                if (pos > 2048) {
                    statusEl.innerHTML = '<span style="color:#e07a3f">θ=500K MONOTONIC (θ=10K ALIASED)</span>';
                } else {
                    statusEl.textContent = 'STABLE (No aliasing)';
                }
            }

            var midY = h / 2;
            ctx.strokeStyle = '#2c261c';
            ctx.lineWidth = 1;
            ctx.beginPath();
            ctx.moveTo(0, midY);
            ctx.lineTo(w, midY);
            ctx.stroke();

            animOffset = (animOffset + 0.02) % (Math.PI * 2);

            // Wave for theta=500,000 (LLaMA-3)
            ctx.strokeStyle = '#e07a3f';
            ctx.lineWidth = 2.0;
            ctx.beginPath();
            for (var x = 0; x < w; x++) {
                var dimFrac = x / w;
                var freq = Math.pow(500000, -2 * dimFrac);
                var y = midY + Math.sin(pos * freq * 0.04 + animOffset) * (h * 0.36);
                if (x === 0) ctx.moveTo(x, y);
                else ctx.lineTo(x, y);
            }
            ctx.stroke();

            // Wave for theta=10,000 (Standard baseline)
            ctx.strokeStyle = '#7a7160';
            ctx.lineWidth = 1.0;
            ctx.setLineDash([3, 3]);
            ctx.beginPath();
            for (var x2 = 0; x2 < w; x2++) {
                var dimFrac2 = x2 / w;
                var freq2 = Math.pow(10000, -2 * dimFrac2);
                var y2 = midY + Math.sin(pos * freq2 * 0.04 + animOffset) * (h * 0.36);
                if (x2 === 0) ctx.moveTo(x2, y2);
                else ctx.lineTo(x2, y2);
            }
            ctx.stroke();
            ctx.setLineDash([]);

            // Labels
            ctx.fillStyle = '#e07a3f';
            ctx.font = 'bold 8px "JetBrains Mono", monospace';
            ctx.textAlign = 'left';
            ctx.fillText('θ = 500,000 (LLaMA-3 Monotonic)', 10, 14);

            ctx.fillStyle = '#7a7160';
            ctx.fillText('θ = 10,000 (Standard Aliased)', 10, 26);

            requestAnimationFrame(drawWaveforms);
        }

        if (slider) slider.addEventListener('input', drawWaveforms);

        if (testBtn) {
            testBtn.addEventListener('click', function () {
                if (slider) {
                    var cur = parseInt(slider.value, 10);
                    slider.value = (cur + 1024) % 8448;
                    if (parseInt(slider.value, 10) === 0) slider.value = 2048;
                    drawWaveforms();
                }
            });
        }

        drawWaveforms();
    }

    // ------------------------------------------------------------------
    // 4. Expandable Code Blocks (>14 lines) & Copy-to-Clipboard
    // ------------------------------------------------------------------
    function fallbackCopy(text) {
        var ta = document.createElement('textarea');
        ta.value = text;
        ta.style.position = 'fixed';
        ta.style.top = '-9999px';
        ta.style.left = '-9999px';
        document.body.appendChild(ta);
        ta.select();
        try { document.execCommand('copy'); } catch (e) {}
        document.body.removeChild(ta);
    }

    window.toggleCode = function (btn) {
        var wrapper = btn.closest('.code-wrapper');
        if (!wrapper) return;
        var collapsed = wrapper.classList.toggle('collapsed');
        var nLines = wrapper.getAttribute('data-lines') || '';
        if (!nLines) {
            var codeEl = wrapper.querySelector('code');
            if (codeEl) {
                var lines = codeEl.innerText.split('\n').length;
                nLines = String(lines);
            }
        }
        btn.textContent = collapsed ? ('expand \u25be \u00b7 ' + nLines + ' lines') : 'collapse \u25b4';
    };

    window.copyCode = function (btn) {
        var wrapper = btn.closest('.code-wrapper');
        if (!wrapper) return;
        var code = wrapper.querySelector('pre code') || wrapper.querySelector('code');
        if (!code) return;
        var text = code.innerText || code.textContent;

        function flash() {
            var orig = btn.textContent;
            btn.textContent = 'Copied!';
            btn.classList.add('copied');
            setTimeout(function () {
                btn.textContent = orig;
                btn.classList.remove('copied');
            }, 1800);
        }

        if (navigator.clipboard && navigator.clipboard.writeText) {
            navigator.clipboard.writeText(text).then(flash).catch(function () {
                fallbackCopy(text);
                flash();
            });
        } else {
            fallbackCopy(text);
            flash();
        }
    };

    // ------------------------------------------------------------------
    // 5. Navigation Filtering & Sidebar Mobile Toggle
    // ------------------------------------------------------------------
    window.filterNav = function () {
        var input = document.getElementById('navSearch') || document.getElementById('sidebarSearch');
        if (!input) return;
        var q = (input.value || '').toLowerCase().trim();
        var links = document.querySelectorAll('.nav-link');
        var groupHeaders = document.querySelectorAll('.nav-group');

        links.forEach(function (l) {
            var txt = l.textContent.toLowerCase();
            var item = l.closest('.nav-item') || l;
            item.style.display = (!q || txt.indexOf(q) !== -1) ? '' : 'none';
        });

        groupHeaders.forEach(function (grp) {
            var visibleItems = grp.querySelectorAll('.nav-item:not([style*="display: none"]), .nav-link:not([style*="display: none"])');
            grp.style.display = (!q || visibleItems.length > 0) ? '' : 'none';
        });
    };

    window.toggleSidebar = function () {
        var sb = document.getElementById('sidebar');
        if (sb) sb.classList.toggle('open');
    };

    // ------------------------------------------------------------------
    // 6. Table of Contents Scrollspy
    // ------------------------------------------------------------------
    function initScrollspy() {
        var tocSidebar = document.querySelector('.toc-sidebar');
        var tocLinks = Array.from(document.querySelectorAll('.toc-link'));
        if (!tocLinks.length) return;

        var headings = [];
        tocLinks.forEach(function (link) {
            var href = link.getAttribute('href');
            if (href && href.startsWith('#')) {
                var targetId = href.substring(1);
                var el = document.getElementById(targetId);
                if (el) headings.push({ el: el, link: link, id: targetId });
            }
        });
        if (!headings.length) return;

        var isTicking = false;
        var activeItem = null;

        function updateSpy() {
            isTicking = false;
            var threshold = 140;
            var current = null;

            for (var i = 0; i < headings.length; i++) {
                var rect = headings[i].el.getBoundingClientRect();
                if (rect.top <= threshold) {
                    current = headings[i];
                } else {
                    break;
                }
            }

            if (!current && headings.length > 0 && window.scrollY < 300) {
                current = headings[0];
            }

            if ((window.innerHeight + window.scrollY) >= (document.body.offsetHeight - 60)) {
                current = headings[headings.length - 1];
            }

            if (current !== activeItem) {
                activeItem = current;
                tocLinks.forEach(function (l) { l.classList.remove('active'); });

                if (current && current.link) {
                    current.link.classList.add('active');

                    if (tocSidebar) {
                        var sidebarRect = tocSidebar.getBoundingClientRect();
                        var linkRect = current.link.getBoundingClientRect();

                        if (linkRect.top < sidebarRect.top + 30 || linkRect.bottom > sidebarRect.bottom - 30) {
                            var linkOffsetInSidebar = current.link.offsetTop;
                            var targetScroll = linkOffsetInSidebar - (tocSidebar.clientHeight / 2) + (current.link.clientHeight / 2);
                            tocSidebar.scrollTo({
                                top: Math.max(0, targetScroll),
                                behavior: 'smooth'
                            });
                        }
                    }
                }
            }
        }

        function onScroll() {
            if (!isTicking) {
                requestAnimationFrame(updateSpy);
                isTicking = true;
            }
        }

        window.addEventListener('scroll', onScroll, { passive: true });
        window.addEventListener('resize', onScroll, { passive: true });

        setTimeout(updateSpy, 100);
        setTimeout(updateSpy, 500);
    }

    // ------------------------------------------------------------------
    // 7. Highlight.js & KaTeX bootstrap
    // ------------------------------------------------------------------
    function initHighlightAndMath() {
        if (window.hljs) hljs.highlightAll();
        if (window.renderMathInElement) {
            renderMathInElement(document.body, {
                delimiters: [
                    {left: '$$', right: '$$', display: true},
                    {left: '\\[', right: '\\]', display: true},
                    {left: '\\(', right: '\\)', display: false},
                    {left: '$', right: '$', display: false}
                ],
                ignoredTags: ["script", "noscript", "style", "textarea", "pre", "code"],
                throwOnError: false
            });
        }
    }

    // ------------------------------------------------------------------
    // Boot Initialization
    // ------------------------------------------------------------------
    document.addEventListener('DOMContentLoaded', function () {
        initHeroCanvas();
        initPassDiagram();
        initGqaMechanism();
        initChunkCeMechanism();
        initRopeMechanism();
        initScrollspy();
        initHighlightAndMath();
    });
})();
