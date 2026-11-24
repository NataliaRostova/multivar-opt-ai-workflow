/**
 * Multivariable Discrete Optimization Engine & 2D Response Surface Simulator
 * Supports exhaustive discrete grid evaluation under linear/nonlinear constraints
 * with real-time Canvas heatmap, feasibility boundary, and local gradient probing.
 */

const SCENARIOS = {
  portfolio: {
    name: "金融资产离散仓位效用最大化",
    xLabel: "权益指数仓位 X (手)",
    yLabel: "贵金属对冲仓位 Y (手)",
    formula: "max U(x,y) = 0.13x + 0.085y - (λ/2)·(0.0022x² + 0.0014y² + 2ρ·0.00175xy) - 0.015·(x+y)  s.t. c₁x + c₂y ≤ B",
    evaluate(x, y, p) {
      const expectedRet = 0.13 * x + 0.085 * y;
      const variance = 0.0022 * x * x + 0.0014 * y * y + 2 * p.rho * 0.00175 * x * y;
      const txCost = 0.015 * (x + y);
      return expectedRet - 0.5 * p.lambda * Math.max(variance, 0) - txCost;
    },
  },
  production: {
    name: "复杂离散条件多要素产出与边际利润最优化",
    xLabel: "要素投入 X (离散档位)",
    yLabel: "要素投入 Y (离散档位)",
    formula: "max Π(x,y) = 4.2·x^0.58·(y+1)^0.42·(1 + 0.18ρ) - 0.04λ·(x - 0.8y)² - 0.35(c₁x + c₂y)  s.t. c₁x + c₂y ≤ B",
    evaluate(x, y, p) {
      const synergy = 1 + 0.18 * p.rho;
      const output = 4.2 * Math.pow(x + 0.5, 0.58) * Math.pow(y + 1.0, 0.42) * synergy;
      const imbalancePenalty = 0.012 * p.lambda * Math.pow(x - 0.8 * y, 2);
      const tierBonus = Math.floor(x / 12) * 1.4 + Math.floor(y / 12) * 1.1;
      const cost = 0.35 * (p.c1 * x + p.c2 * y);
      return output + tierBonus - imbalancePenalty - cost;
    },
  },
  grid: {
    name: "量化网格交易参数二维寻优",
    xLabel: "网格间距 X (×0.1%)",
    yLabel: "单格仓位 Y (千元)",
    formula: "max E[R](x,y) = 28·sin(x/16)·√(y+1)·(1 + 0.25ρ) - 0.008λ·(x·y)^1.15 - 0.12y  s.t. c₁x + c₂y ≤ B",
    evaluate(x, y, p) {
      const capture = 28.0 * Math.sin(Math.min(x, 75) / 24.0) * Math.sqrt(y + 1.0) * (1 + 0.25 * p.rho);
      const inventoryRisk = 0.006 * p.lambda * Math.pow(x * y + 1, 1.12);
      const feeDrag = 0.12 * y + 4.0 / (x + 2.0);
      return capture - inventoryRisk - feeDrag;
    },
  },
};

const dom = {
  scenario: document.getElementById("scenarioSelect"),
  budget: document.getElementById("paramBudget"),
  step: document.getElementById("paramStep"),
  lambda: document.getElementById("paramLambda"),
  rho: document.getElementById("paramRho"),
  c1: document.getElementById("paramC1"),
  c2: document.getElementById("paramC2"),
  valBudget: document.getElementById("valBudget"),
  valStep: document.getElementById("valStep"),
  valLambda: document.getElementById("valLambda"),
  valRho: document.getElementById("valRho"),
  valC1: document.getElementById("valC1"),
  valC2: document.getElementById("valC2"),
  formulaBox: document.getElementById("formulaBox"),
  optVal: document.getElementById("optVal"),
  optXY: document.getElementById("optXY"),
  feasibleCount: document.getElementById("feasibleCount"),
  optMeta: document.getElementById("optMeta"),
  topTableBody: document.getElementById("topTableBody"),
  canvas: document.getElementById("surfaceCanvas"),
  probeBar: document.getElementById("probeBar"),
};

const ctx = dom.canvas.getContext("2d");
const PAD = { left: 56, right: 24, top: 24, bottom: 48 };
let lastSolveState = null;

function readParams() {
  const p = {
    scenarioKey: dom.scenario.value,
    budget: parseFloat(dom.budget.value),
    step: parseInt(dom.step.value, 10),
    lambda: parseFloat(dom.lambda.value),
    rho: parseFloat(dom.rho.value),
    c1: parseFloat(dom.c1.value),
    c2: parseFloat(dom.c2.value),
  };
  dom.valBudget.textContent = p.budget;
  dom.valStep.textContent = p.step;
  dom.valLambda.textContent = p.lambda.toFixed(1);
  dom.valRho.textContent = p.rho.toFixed(2);
  dom.valC1.textContent = p.c1.toFixed(1);
  dom.valC2.textContent = p.c2.toFixed(1);
  return p;
}

function heatColor(t) {
  // Viridis-inspired institutional palette
  const clamped = Math.max(0, Math.min(1, t));
  const stops = [
    [18, 28, 62],
    [33, 86, 138],
    [34, 152, 138],
    [136, 204, 78],
    [248, 224, 52],
  ];
  const scaled = clamped * (stops.length - 1);
  const idx = Math.min(stops.length - 2, Math.floor(scaled));
  const frac = scaled - idx;
  const c0 = stops[idx];
  const c1 = stops[idx + 1];
  const r = Math.round(c0[0] + (c1[0] - c0[0]) * frac);
  const g = Math.round(c0[1] + (c1[1] - c0[1]) * frac);
  const b = Math.round(c0[2] + (c1[2] - c0[2]) * frac);
  return `rgb(${r},${g},${b})`;
}

function solveAndRender() {
  const p = readParams();
  const sc = SCENARIOS[p.scenarioKey];
  dom.formulaBox.textContent = sc.formula;

  const maxX = Math.max(20, Math.ceil(p.budget / p.c1));
  const maxY = Math.max(20, Math.ceil(p.budget / p.c2));

  const candidates = [];
  let minF = Infinity;
  let maxF = -Infinity;

  for (let x = 0; x <= maxX; x += p.step) {
    for (let y = 0; y <= maxY; y += p.step) {
      const cost = p.c1 * x + p.c2 * y;
      if (cost <= p.budget + 1e-9) {
        const val = sc.evaluate(x, y, p);
        candidates.push({ x, y, cost, val });
        if (val < minF) minF = val;
        if (val > maxF) maxF = val;
      }
    }
  }

  candidates.sort((a, b) => b.val - a.val);
  const best = candidates[0] || { x: 0, y: 0, cost: 0, val: 0 };

  // Central finite-difference sensitivity at optimum
  const eps = 0.25;
  const dfdx = (sc.evaluate(best.x + eps, best.y, p) - sc.evaluate(Math.max(0, best.x - eps), best.y, p)) / (2 * eps);
  const dfdy = (sc.evaluate(best.x, best.y + eps, p) - sc.evaluate(best.x, Math.max(0, best.y - eps), p)) / (2 * eps);

  dom.optVal.textContent = best.val.toFixed(3);
  dom.optXY.textContent = `(${best.x}, ${best.y})`;
  dom.feasibleCount.textContent = `${candidates.length.toLocaleString()} 点`;
  dom.optMeta.textContent = `${((best.cost / p.budget) * 100).toFixed(1)}% | ∇=(${dfdx.toFixed(2)}, ${dfdy.toFixed(2)})`;

  dom.topTableBody.innerHTML = candidates
    .slice(0, 5)
    .map(
      (c, i) => `
      <tr>
        <td>#${i + 1}</td>
        <td>${c.x}</td>
        <td>${c.y}</td>
        <td>${c.cost.toFixed(1)}</td>
        <td>${c.val.toFixed(3)}</td>
        <td>${(c.val - best.val).toFixed(3)}</td>
      </tr>`
    )
    .join("");

  lastSolveState = { p, sc, maxX, maxY, minF, maxF, best };
  drawSurface(lastSolveState);
}

function drawSurface(state) {
  const { p, sc, maxX, maxY, minF, maxF, best } = state;
  const W = dom.canvas.width;
  const H = dom.canvas.height;
  const plotW = W - PAD.left - PAD.right;
  const plotH = H - PAD.top - PAD.bottom;

  ctx.clearRect(0, 0, W, H);
  ctx.fillStyle = "#0b1120";
  ctx.fillRect(0, 0, W, H);

  const cols = 72;
  const rows = 58;
  const cellW = plotW / cols;
  const cellH = plotH / rows;
  const denom = Math.max(maxF - minF, 1e-6);

  for (let i = 0; i < cols; i++) {
    for (let j = 0; j < rows; j++) {
      const xVal = ((i + 0.5) / cols) * maxX;
      const yVal = ((rows - 1 - j + 0.5) / rows) * maxY;
      const cost = p.c1 * xVal + p.c2 * yVal;
      const px = PAD.left + i * cellW;
      const py = PAD.top + j * cellH;

      if (cost <= p.budget) {
        const val = sc.evaluate(xVal, yVal, p);
        ctx.fillStyle = heatColor((val - minF) / denom);
      } else {
        ctx.fillStyle = "#111827";
      }
      ctx.fillRect(px, py, Math.ceil(cellW), Math.ceil(cellH));
    }
  }

  // Draw budget constraint boundary: c1*x + c2*y = B
  const xAtY0 = Math.min(maxX, p.budget / p.c1);
  const yAtX0 = Math.min(maxY, p.budget / p.c2);
  ctx.strokeStyle = "#f43f5e";
  ctx.lineWidth = 2.2;
  ctx.setLineDash([6, 4]);
  ctx.beginPath();
  ctx.moveTo(PAD.left, PAD.top + plotH * (1 - yAtX0 / maxY));
  ctx.lineTo(PAD.left + plotW * (xAtY0 / maxX), PAD.top + plotH);
  ctx.stroke();
  ctx.setLineDash([]);

  // Axes
  ctx.strokeStyle = "#475569";
  ctx.lineWidth = 1.2;
  ctx.strokeRect(PAD.left, PAD.top, plotW, plotH);

  ctx.fillStyle = "#94a3b8";
  ctx.font = "12px Segoe UI, sans-serif";
  ctx.textAlign = "center";
  for (let t = 0; t <= 4; t++) {
    const xv = (maxX * t) / 4;
    const xPos = PAD.left + (plotW * t) / 4;
    ctx.fillText(xv.toFixed(0), xPos, PAD.top + plotH + 18);
  }
  ctx.fillText(sc.xLabel, PAD.left + plotW / 2, H - 10);

  ctx.textAlign = "right";
  for (let t = 0; t <= 4; t++) {
    const yv = (maxY * t) / 4;
    const yPos = PAD.top + plotH * (1 - t / 4);
    ctx.fillText(yv.toFixed(0), PAD.left - 8, yPos + 4);
  }

  ctx.save();
  ctx.translate(16, PAD.top + plotH / 2);
  ctx.rotate(-Math.PI / 2);
  ctx.textAlign = "center";
  ctx.fillText(sc.yLabel, 0, 0);
  ctx.restore();

  // Plot global discrete optimum marker
  const optPx = PAD.left + (best.x / maxX) * plotW;
  const optPy = PAD.top + plotH * (1 - best.y / maxY);

  ctx.beginPath();
  ctx.arc(optPx, optPy, 8, 0, Math.PI * 2);
  ctx.fillStyle = "#f59e0b";
  ctx.fill();
  ctx.lineWidth = 2.5;
  ctx.strokeStyle = "#ffffff";
  ctx.stroke();

  ctx.fillStyle = "#ffffff";
  ctx.font = "bold 12px Consolas, monospace";
  ctx.textAlign = "left";
  ctx.fillText(`OPT (${best.x}, ${best.y}) = ${best.val.toFixed(2)}`, Math.min(optPx + 12, W - 180), Math.max(optPy - 10, PAD.top + 16));
}

dom.canvas.addEventListener("mousemove", (evt) => {
  if (!lastSolveState) return;
  const rect = dom.canvas.getBoundingClientRect();
  const scaleX = dom.canvas.width / rect.width;
  const scaleY = dom.canvas.height / rect.height;
  const cx = (evt.clientX - rect.left) * scaleX;
  const cy = (evt.clientY - rect.top) * scaleY;

  const plotW = dom.canvas.width - PAD.left - PAD.right;
  const plotH = dom.canvas.height - PAD.top - PAD.bottom;
  if (cx < PAD.left || cx > PAD.left + plotW || cy < PAD.top || cy > PAD.top + plotH) return;

  const { p, sc, maxX, maxY } = lastSolveState;
  const x = ((cx - PAD.left) / plotW) * maxX;
  const y = (1 - (cy - PAD.top) / plotH) * maxY;
  const cost = p.c1 * x + p.c2 * y;
  const feasible = cost <= p.budget;
  const val = sc.evaluate(x, y, p);
  dom.probeBar.textContent = `Probe (x=${x.toFixed(1)}, y=${y.toFixed(1)}) | Cost=${cost.toFixed(1)}/${p.budget} [${
    feasible ? "FEASIBLE" : "INFEASIBLE"
  }] | f(x,y) = ${val.toFixed(3)}`;
});

["change", "input"].forEach((ev) => {
  [dom.scenario, dom.budget, dom.step, dom.lambda, dom.rho, dom.c1, dom.c2].forEach((el) =>
    el.addEventListener(ev, solveAndRender)
  );
});

solveAndRender();
