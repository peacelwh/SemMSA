// Results under every random missing rate (Tables 12-14 of the paper) as an interactive line chart.
// Data is embedded in the page as JSON (#rate-data): datasets -> metrics, and methods -> one curve per metric.

const COLORS = {
  'MISA': '#1f77b4', 'Self-MM': '#8C9FCA', 'MMIM': '#2CA02C', 'TETFN': '#E6AB02', 'TFR-Net': '#D62728',
  'LNLN': '#8C564B', 'P-RMF': '#B65FCF', 'TF-Mamba': '#17BECF', 'SemMSA': '#111827',
};
const NS = 'http://www.w3.org/2000/svg';

function svgEl(tag, attrs, text) {
  const e = document.createElementNS(NS, tag);
  for (const k in attrs) e.setAttribute(k, attrs[k]);
  if (text !== undefined) e.textContent = text;
  return e;
}

function niceTicks(lo, hi, n) {
  const span = hi - lo || 1;
  const raw = span / n;
  const mag = Math.pow(10, Math.floor(Math.log10(raw)));
  const step = [1, 2, 2.5, 5, 10].map(m => m * mag).find(s => s >= raw);
  const start = Math.floor(lo / step) * step;
  const ticks = [];
  for (let v = start; v <= hi + step * 0.5; v += step) ticks.push(+v.toFixed(6));
  return ticks;
}

function setupRateChart() {
  const root = document.getElementById('chart-rates');
  const dataEl = document.getElementById('rate-data');
  if (!root || !dataEl) return;
  const data = JSON.parse(dataEl.textContent);
  const rates = data.rates;
  const hidden = new Set();
  let ds = Object.keys(data.datasets)[0];
  let mi = 0;

  const controls = root.querySelector('.chart-controls');
  const dsSeg = document.createElement('div');
  dsSeg.className = 'seg';
  dsSeg.setAttribute('role', 'group');
  dsSeg.setAttribute('aria-label', 'Dataset');
  const mSeg = document.createElement('div');
  mSeg.className = 'seg';
  mSeg.setAttribute('role', 'group');
  mSeg.setAttribute('aria-label', 'Metric');
  controls.append(dsSeg, mSeg);
  const legend = root.querySelector('.chart-legend');
  const body = root.querySelector('.chart-body');
  const tip = document.createElement('div');
  tip.className = 'chart-tip';
  document.body.appendChild(tip);

  function button(label, pressed, onClick) {
    const b = document.createElement('button');
    b.type = 'button';
    b.className = 'seg-btn';
    b.textContent = label;
    b.setAttribute('aria-pressed', pressed ? 'true' : 'false');
    b.addEventListener('click', onClick);
    return b;
  }

  function renderControls() {
    dsSeg.replaceChildren(...Object.keys(data.datasets).map(name =>
      button(name, name === ds, () => { ds = name; mi = Math.min(mi, data.datasets[ds].metrics.length - 1); render(); })));
    mSeg.replaceChildren(...data.datasets[ds].metrics.map((m, i) =>
      button(m.name, i === mi, () => { mi = i; render(); })));
    legend.replaceChildren(...Object.keys(data.datasets[ds].methods).map(name => {
      const b = document.createElement('button');
      b.type = 'button';
      b.className = 'legend-item';
      b.setAttribute('aria-pressed', hidden.has(name) ? 'false' : 'true');
      b.title = 'Show or hide ' + name;
      const sw = document.createElement('span');
      sw.className = 'legend-swatch';
      sw.style.background = COLORS[name] || '#64748b';
      if (name === 'SemMSA') sw.style.height = '4px';
      b.append(sw, document.createTextNode(name === 'SemMSA' ? 'SemMSA (ours)' : name));
      b.addEventListener('click', () => { hidden.has(name) ? hidden.delete(name) : hidden.add(name); render(); });
      return b;
    }));
  }

  function renderChart() {
    const metric = data.datasets[ds].metrics[mi];
    const series = Object.entries(data.datasets[ds].methods)
      .filter(([name]) => !hidden.has(name))
      .map(([name, curves]) => ({ name, values: curves[mi] }))
      .sort((a, b) => (a.name === 'SemMSA') - (b.name === 'SemMSA'));  // ours on top
    const W = 760, H = 360, L = 56, R = 16, T = 14, B = 44;
    const svg = svgEl('svg', { viewBox: `0 0 ${W} ${H}`, role: 'img',
      'aria-label': `${metric.name} on ${ds} under missing rates 0.0 to 0.9` });
    const all = series.flatMap(s => s.values);
    let lo = all.length ? Math.min(...all) : 0, hi = all.length ? Math.max(...all) : 1;
    const pad = (hi - lo) * 0.06 || 0.5;
    lo -= pad;
    hi += pad;
    const ticks = niceTicks(lo, hi, 5).filter(t => t >= lo && t <= hi);
    const x = i => L + (W - L - R) * i / (rates.length - 1);
    const y = v => T + (H - T - B) * (1 - (v - lo) / (hi - lo));
    const fmt = v => metric.decimals === 3 ? v.toFixed(3) : v.toFixed(2);

    ticks.forEach(t => {
      svg.appendChild(svgEl('line', { x1: L, x2: W - R, y1: y(t), y2: y(t), stroke: '#EEF1F4' }));
      svg.appendChild(svgEl('text', { x: L - 8, y: y(t) + 4, 'text-anchor': 'end', 'font-size': 12, fill: '#64748b' }, fmt(t)));
    });
    rates.forEach((r, i) => {
      svg.appendChild(svgEl('text', { x: x(i), y: H - B + 18, 'text-anchor': 'middle', 'font-size': 12, fill: '#64748b' }, r.toFixed(1)));
    });
    svg.appendChild(svgEl('line', { x1: L, x2: W - R, y1: H - B, y2: H - B, stroke: '#94a3b8' }));
    svg.appendChild(svgEl('text', { x: (L + W - R) / 2, y: H - 6, 'text-anchor': 'middle', 'font-size': 13, fill: '#334155' },
      'Random missing rate r'));
    svg.appendChild(svgEl('text', { x: 14, y: (T + H - B) / 2, 'text-anchor': 'middle', 'font-size': 13, fill: '#334155',
      transform: `rotate(-90 14 ${(T + H - B) / 2})` }, metric.name + (metric.lower ? ' (lower is better)' : '')));

    series.forEach(s => {
      const ours = s.name === 'SemMSA';
      const color = COLORS[s.name] || '#64748b';
      svg.appendChild(svgEl('polyline', { points: s.values.map((v, i) => `${x(i)},${y(v)}`).join(' '), fill: 'none',
        stroke: color, 'stroke-width': ours ? 3 : 1.8, 'stroke-linejoin': 'round', opacity: ours ? 1 : 0.85 }));
      s.values.forEach((v, i) => svg.appendChild(svgEl('circle', { cx: x(i), cy: y(v), r: ours ? 3.6 : 2.6, fill: color })));
    });

    const guide = svgEl('line', { y1: T, y2: H - B, stroke: '#94a3b8', 'stroke-dasharray': '3 3', visibility: 'hidden' });
    svg.appendChild(guide);
    const hit = svgEl('rect', { x: L, y: T, width: W - L - R, height: H - T - B, fill: 'transparent' });
    svg.appendChild(hit);
    hit.addEventListener('mousemove', e => {
      const box = svg.getBoundingClientRect();
      const px = (e.clientX - box.left) * W / box.width;
      const i = Math.max(0, Math.min(rates.length - 1, Math.round((px - L) / ((W - L - R) / (rates.length - 1)))));
      guide.setAttribute('x1', x(i));
      guide.setAttribute('x2', x(i));
      guide.setAttribute('visibility', 'visible');
      const rows = series.map(s => ({ name: s.name, v: s.values[i] }))
        .sort((a, b) => metric.lower ? a.v - b.v : b.v - a.v);
      tip.replaceChildren();
      const title = document.createElement('div');
      title.className = 'chart-tip-title';
      title.textContent = `${ds} · ${metric.name} · r = ${rates[i].toFixed(1)}`;
      tip.appendChild(title);
      rows.forEach(r => {
        const row = document.createElement('div');
        row.className = 'chart-tip-row';
        const key = document.createElement('span');
        key.className = 'chart-tip-key';
        key.style.background = COLORS[r.name] || '#64748b';
        const name = document.createElement('span');
        name.textContent = r.name;
        const val = document.createElement(r.name === 'SemMSA' ? 'strong' : 'span');
        val.textContent = fmt(r.v);
        row.append(key, name, val);
        tip.appendChild(row);
      });
      tip.style.display = 'block';
      const tw = tip.offsetWidth, th = tip.offsetHeight;
      let left = e.clientX + 14, top = e.clientY + 14;
      if (left + tw > window.innerWidth - 8) left = e.clientX - tw - 14;
      if (top + th > window.innerHeight - 8) top = window.innerHeight - th - 8;
      tip.style.left = left + 'px';
      tip.style.top = top + 'px';
    });
    hit.addEventListener('mouseleave', () => { tip.style.display = 'none'; guide.setAttribute('visibility', 'hidden'); });
    body.replaceChildren(svg);
  }

  function render() {
    renderControls();
    renderChart();
  }
  render();
}

document.addEventListener('DOMContentLoaded', setupRateChart);
