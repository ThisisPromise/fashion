import json
import sys

sys.path.insert(0, "fabric-fill-tool/scratch")
from frontend_assets import ASSETS, FABRICS  # noqa: E402

TEMPLATE = """<title>Swatch Table</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,500;9..144,600&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap">
<style>
:root {
  --bg: #F2EEE4;
  --surface: #FFFFFF;
  --surface-2: #EAE4D6;
  --ink: #1C2C3A;
  --ink-dim: #5B6B78;
  --accent: #C97F1F;
  --accent-ink: #FFFFFF;
  --accent-2: #2E8577;
  --line: #D8D2C2;
  --shadow: 0 1px 2px rgba(28,44,58,0.10);
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    --bg: #131F29;
    --surface: #1B2A37;
    --surface-2: #223447;
    --ink: #ECEDE6;
    --ink-dim: #92A5B3;
    --accent: #E8A33D;
    --accent-ink: #1C2C3A;
    --accent-2: #4FB6A8;
    --line: #2E4356;
    --shadow: 0 1px 3px rgba(0,0,0,0.45);
  }
}
:root[data-theme="dark"] {
  --bg: #131F29;
  --surface: #1B2A37;
  --surface-2: #223447;
  --ink: #ECEDE6;
  --ink-dim: #92A5B3;
  --accent: #E8A33D;
  --accent-ink: #1C2C3A;
  --accent-2: #4FB6A8;
  --line: #2E4356;
  --shadow: 0 1px 3px rgba(0,0,0,0.45);
}

* { box-sizing: border-box; }
body {
  background: var(--bg);
  color: var(--ink);
  font-family: 'IBM Plex Sans', system-ui, -apple-system, sans-serif;
  -webkit-font-smoothing: antialiased;
}
.mono { font-family: 'IBM Plex Mono', ui-monospace, monospace; }

.page { max-width: 1180px; margin: 0 auto; padding: 40px 24px 64px; }

.topbar { margin-bottom: 32px; }
.brand { display: flex; align-items: baseline; gap: 12px; }
.brand-mark {
  font-size: 1.3rem; color: var(--accent);
  transform: translateY(1px);
}
.brand h1 {
  font-family: 'Fraunces', Georgia, serif;
  font-weight: 600;
  font-size: clamp(1.9rem, 3.2vw, 2.5rem);
  letter-spacing: -0.01em;
  margin: 0;
  text-wrap: balance;
}
.tagline {
  margin: 10px 0 0;
  color: var(--ink-dim);
  font-size: 1.02rem;
  max-width: 62ch;
  line-height: 1.55;
}

.workspace {
  display: grid;
  grid-template-columns: minmax(0, 1fr) 300px;
  gap: 28px;
  align-items: start;
}
@media (max-width: 860px) {
  .workspace { grid-template-columns: 1fr; }
}

.stage-panel {
  background: var(--surface);
  border: 1px solid var(--line);
  border-radius: 6px;
  padding: 18px;
  box-shadow: var(--shadow);
}
.stage-toolbar {
  display: flex;
  flex-direction: column;
  gap: 10px;
  margin-bottom: 16px;
}
.image-switch { display: flex; gap: 2px; background: var(--surface-2); border-radius: 5px; padding: 3px; width: fit-content; }
.switch-btn {
  font-family: inherit; font-size: 0.88rem; font-weight: 500;
  padding: 7px 14px; border: none; border-radius: 4px;
  background: transparent; color: var(--ink-dim); cursor: pointer;
  transition: background 0.15s ease, color 0.15s ease;
}
.switch-btn.active { background: var(--surface); color: var(--ink); box-shadow: var(--shadow); }
.switch-btn:hover:not(.active) { color: var(--ink); }
.hint { margin: 0; font-size: 0.86rem; color: var(--ink-dim); line-height: 1.5; }

.canvas-frame {
  background:
    repeating-linear-gradient(0deg, var(--surface-2) 0 1px, transparent 1px 24px),
    repeating-linear-gradient(90deg, var(--surface-2) 0 1px, transparent 1px 24px),
    var(--surface);
  border: 1px solid var(--line);
  border-radius: 4px;
  display: flex;
  justify-content: center;
  padding: 20px;
}
#stage {
  width: min(100%, 460px);
  height: auto;
  cursor: default;
  display: block;
}
#stage.clickable { cursor: pointer; }

.hover-status {
  margin: 12px 2px 0;
  font-size: 0.82rem;
  color: var(--ink-dim);
  min-height: 1.2em;
}

.control-panel { display: flex; flex-direction: column; gap: 14px; }
.panel-block {
  background: var(--surface);
  border: 1px solid var(--line);
  border-radius: 6px;
  padding: 16px 18px;
  box-shadow: var(--shadow);
}
.panel-block h2 {
  font-size: 0.72rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.07em;
  color: var(--ink-dim); margin: 0 0 10px;
}
.stat { margin: 0; display: flex; align-items: baseline; gap: 8px; }
.stat span:first-child { font-family: 'IBM Plex Mono', monospace; font-size: 1.6rem; font-weight: 500; color: var(--ink); font-variant-numeric: tabular-nums; }
.stat-label { font-size: 0.82rem; color: var(--ink-dim); }
#selected-list { margin: 8px 0 0; font-size: 0.86rem; color: var(--ink-dim); word-break: break-word; }

.swatch-row { display: flex; gap: 10px; }
.swatch {
  width: 52px; height: 52px; border-radius: 5px;
  background-size: cover; background-position: center;
  border: 2px solid transparent; cursor: pointer; padding: 0;
  transition: border-color 0.15s ease, transform 0.1s ease;
}
.swatch:hover { transform: translateY(-1px); }
.swatch.active { border-color: var(--accent); }

.actions { display: flex; flex-direction: column; gap: 8px; }
.btn {
  font-family: inherit; font-size: 0.92rem; font-weight: 500;
  padding: 10px 16px; border-radius: 5px; border: 1px solid transparent;
  cursor: pointer; transition: filter 0.15s ease, background 0.15s ease;
}
.btn:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }
.btn-primary { background: var(--accent); color: var(--accent-ink); }
.btn-primary:hover { filter: brightness(1.06); }
.btn-primary:disabled { opacity: 0.45; cursor: not-allowed; filter: none; }
.btn-ghost { background: transparent; color: var(--ink-dim); border-color: var(--line); }
.btn-ghost:hover { color: var(--ink); border-color: var(--ink-dim); }

.footnote { font-size: 0.82rem; color: var(--ink-dim); line-height: 1.55; margin: 0; }

@media (prefers-reduced-motion: reduce) {
  * { transition: none !important; }
}
</style>

<div class="page">
  <header class="topbar">
    <div class="brand">
      <span class="brand-mark">&#9987;</span>
      <h1>Swatch Table</h1>
    </div>
    <p class="tagline">
      Segmentation runs automatically once, when the image loads &mdash; nothing here was marked by hand.
      Click a panel to select it, click several to group them, choose a fabric, and apply it to the whole
      group at once.
    </p>
  </header>

  <main class="workspace">
    <section class="stage-panel">
      <div class="stage-toolbar">
        <div class="image-switch" id="image-switch">
          <button class="switch-btn active" data-image="sketch">Dress sketch</button>
          <button class="switch-btn" data-image="sketch1">Technical flat</button>
        </div>
        <p class="hint">Click inside any panel below to select it &mdash; click again to deselect. Try selecting several pleat strips on the flat before applying.</p>
      </div>
      <div class="canvas-frame">
        <canvas id="stage"></canvas>
      </div>
      <p class="hover-status" id="hover-status">&nbsp;</p>
    </section>

    <aside class="control-panel">
      <div class="panel-block">
        <h2>Regions found</h2>
        <p class="stat"><span id="region-count" class="mono">&ndash;</span><span class="stat-label">automatically, before any click</span></p>
      </div>

      <div class="panel-block">
        <h2>Selection</h2>
        <p class="stat"><span id="selected-count" class="mono">0</span><span class="stat-label">selected</span></p>
        <p id="selected-list" class="mono">&mdash;</p>
      </div>

      <div class="panel-block">
        <h2>Fabric</h2>
        <div class="swatch-row" id="swatch-row"></div>
      </div>

      <div class="panel-block actions">
        <button id="apply-btn" class="btn btn-primary" disabled>Apply to selected</button>
        <button id="reset-btn" class="btn btn-ghost">Reset image</button>
      </div>

      <p class="footnote">Same mechanic either way: one region, or a whole group &mdash; one Apply fills every selected panel with the chosen fabric in a single pass.</p>
    </aside>
  </main>
</div>

<script>
const ASSETS = __ASSETS_JSON__;
const FABRICS = __FABRICS_JSON__;

const state = {
  imageKey: 'sketch',
  selected: new Set(),
  activeFabric: Object.keys(FABRICS)[0],
  labelData: null,
  width: 0,
  height: 0,
  filledCanvas: null,
  filledCtx: null,
};

const fabricData = {};

function loadImage(src) {
  return new Promise((resolve) => {
    const img = new Image();
    img.onload = () => resolve(img);
    img.src = src;
  });
}

async function preloadFabrics() {
  for (const key of Object.keys(FABRICS)) {
    const img = await loadImage(FABRICS[key]);
    const c = document.createElement('canvas');
    c.width = img.width; c.height = img.height;
    const ctx = c.getContext('2d');
    ctx.drawImage(img, 0, 0);
    const id = ctx.getImageData(0, 0, img.width, img.height);
    fabricData[key] = { data: id.data, w: img.width, h: img.height };
  }
}

async function loadImageSet(key) {
  const asset = ASSETS[key];
  state.width = asset.width;
  state.height = asset.height;

  const baseImg = await loadImage(asset.base_data_uri);
  const labelImg = await loadImage(asset.label_data_uri);

  const stage = document.getElementById('stage');
  stage.width = asset.width;
  stage.height = asset.height;

  const filled = document.createElement('canvas');
  filled.width = asset.width;
  filled.height = asset.height;
  const filledCtx = filled.getContext('2d');
  filledCtx.drawImage(baseImg, 0, 0);
  state.filledCanvas = filled;
  state.filledCtx = filledCtx;

  const labelCanvas = document.createElement('canvas');
  labelCanvas.width = asset.width;
  labelCanvas.height = asset.height;
  const labelCtx = labelCanvas.getContext('2d');
  labelCtx.drawImage(labelImg, 0, 0);
  state.labelData = labelCtx.getImageData(0, 0, asset.width, asset.height).data;

  state.selected.clear();
  state.imageKey = key;
  render();
  updateStatus();
}

function render() {
  const stage = document.getElementById('stage');
  const ctx = stage.getContext('2d');
  ctx.drawImage(state.filledCanvas, 0, 0);
  if (state.selected.size > 0) {
    const imgData = ctx.getImageData(0, 0, state.width, state.height);
    const data = imgData.data;
    const label = state.labelData;
    for (let i = 0; i < data.length; i += 4) {
      const rid = label[i];
      if (state.selected.has(rid)) {
        data[i] = data[i] * 0.45 + 232 * 0.55;
        data[i + 1] = data[i + 1] * 0.45 + 163 * 0.55;
        data[i + 2] = data[i + 2] * 0.45 + 61 * 0.55;
      }
    }
    ctx.putImageData(imgData, 0, 0);
  }
}

function regionAt(x, y) {
  if (x < 0 || y < 0 || x >= state.width || y >= state.height) return 0;
  const idx = (y * state.width + x) * 4;
  return state.labelData[idx];
}

function canvasCoords(e) {
  const stage = document.getElementById('stage');
  const rect = stage.getBoundingClientRect();
  const scaleX = stage.width / rect.width;
  const scaleY = stage.height / rect.height;
  return {
    x: Math.floor((e.clientX - rect.left) * scaleX),
    y: Math.floor((e.clientY - rect.top) * scaleY),
  };
}

function applyFabric() {
  if (state.selected.size === 0) return;
  const fab = fabricData[state.activeFabric];
  const ctx = state.filledCtx;
  const imgData = ctx.getImageData(0, 0, state.width, state.height);
  const data = imgData.data;
  const label = state.labelData;
  for (let y = 0; y < state.height; y++) {
    for (let x = 0; x < state.width; x++) {
      const idx = (y * state.width + x) * 4;
      const rid = label[idx];
      if (state.selected.has(rid)) {
        const fx = x % fab.w, fy = y % fab.h;
        const fidx = (fy * fab.w + fx) * 4;
        data[idx] = fab.data[fidx];
        data[idx + 1] = fab.data[fidx + 1];
        data[idx + 2] = fab.data[fidx + 2];
      }
    }
  }
  ctx.putImageData(imgData, 0, 0);
  state.selected.clear();
  render();
  updateStatus();
}

function updateStatus() {
  const asset = ASSETS[state.imageKey];
  document.getElementById('region-count').textContent = asset.region_count;
  document.getElementById('selected-count').textContent = state.selected.size;
  const list = document.getElementById('selected-list');
  list.textContent = state.selected.size
    ? [...state.selected].sort((a, b) => a - b).map((n) => '#' + n).join(', ')
    : String.fromCharCode(8212);
  document.getElementById('apply-btn').disabled = state.selected.size === 0;
}

document.getElementById('stage').addEventListener('click', (e) => {
  const { x, y } = canvasCoords(e);
  const rid = regionAt(x, y);
  if (rid === 0) return;
  if (state.selected.has(rid)) state.selected.delete(rid);
  else state.selected.add(rid);
  render();
  updateStatus();
});

document.getElementById('stage').addEventListener('mousemove', (e) => {
  const { x, y } = canvasCoords(e);
  const rid = regionAt(x, y);
  const stage = document.getElementById('stage');
  const status = document.getElementById('hover-status');
  if (rid === 0) {
    stage.classList.remove('clickable');
    status.textContent = '\\u00A0';
  } else {
    stage.classList.add('clickable');
    status.textContent = 'Region #' + rid + (state.selected.has(rid) ? ' \\u2014 selected' : ' \\u2014 click to select');
  }
});

document.getElementById('image-switch').addEventListener('click', (e) => {
  const btn = e.target.closest('.switch-btn');
  if (!btn) return;
  document.querySelectorAll('.switch-btn').forEach((b) => b.classList.remove('active'));
  btn.classList.add('active');
  loadImageSet(btn.dataset.image);
});

document.getElementById('apply-btn').addEventListener('click', applyFabric);
document.getElementById('reset-btn').addEventListener('click', () => loadImageSet(state.imageKey));

function buildSwatches() {
  const row = document.getElementById('swatch-row');
  Object.keys(FABRICS).forEach((key, i) => {
    const b = document.createElement('button');
    b.className = 'swatch' + (i === 0 ? ' active' : '');
    b.style.backgroundImage = `url(${FABRICS[key]})`;
    b.dataset.fabric = key;
    b.setAttribute('aria-label', 'Fabric ' + (i + 1));
    b.addEventListener('click', () => {
      state.activeFabric = key;
      document.querySelectorAll('.swatch').forEach((s) => s.classList.remove('active'));
      b.classList.add('active');
    });
    row.appendChild(b);
  });
}

(async function init() {
  buildSwatches();
  await preloadFabrics();
  await loadImageSet('sketch');
})();
</script>
"""


def main():
    html = TEMPLATE.replace("__ASSETS_JSON__", json.dumps(ASSETS))
    html = html.replace("__FABRICS_JSON__", json.dumps(FABRICS))
    out_path = "fabric-fill-tool/scratch/swatch_table.html"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"wrote {out_path}, {len(html)} chars")


if __name__ == "__main__":
    main()
