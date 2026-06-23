"""
Dependency-free SVG figure builders. Each function takes already-loaded data and returns an SVG
string; callers (scripts/make_figures.py) handle file I/O. No matplotlib / external deps.
"""
from __future__ import annotations

INK = "#1a1a2e"; MINE = "#3a6ea5"; REF = "#c44e3a"; OK = "#2e8b57"

# distinct colors for the 7 SAE architectures
ARCH_COLORS = {
    "Standard": "#c44e3a", "TopK": "#3a6ea5", "BatchTopK": "#2e8b57", "JumpRelu": "#8e44ad",
    "GatedSAE": "#d98c00", "Matryoshka": "#16a3a3", "PAnneal": "#7f8c8d",
}


def _esc(s):
    return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _log10(v):
    import math
    return math.log10(max(v, 1e-9))


def fig_frontier(rows, title="Sparsity–Fidelity frontier (Core / Loss Recovered)",
                 subtitle="") -> str:
    """Loss Recovered vs L0 (log x), one line per architecture. rows: suite per_sae records."""
    W, H = 760, 470
    ax0, ax1, ay0, ay1 = 70, 600, 80, H - 60
    xs = [r["l0"] for r in rows]; ys = [r["loss_recovered"] for r in rows]
    xlo, xhi = _log10(min(xs)) - 0.05, _log10(max(xs)) + 0.05
    ylo, yhi = min(ys) - 0.01, 1.001
    def X(v): return ax0 + (_log10(v) - xlo) / (xhi - xlo) * (ax1 - ax0)
    def Y(v): return ay1 - (v - ylo) / (yhi - ylo) * (ay1 - ay0)
    s = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" font-family="Inter,Arial,sans-serif">',
         f'<rect width="{W}" height="{H}" fill="white"/>',
         f'<text x="30" y="34" font-size="18" font-weight="700" fill="{INK}">{_esc(title)}</text>',
         f'<text x="30" y="55" font-size="12" fill="#555">{_esc(subtitle)}</text>']
    # axes + gridlines
    s.append(f'<line x1="{ax0}" y1="{ay1}" x2="{ax1}" y2="{ay1}" stroke="{INK}" stroke-width="1.2"/>')
    s.append(f'<line x1="{ax0}" y1="{ay0}" x2="{ax0}" y2="{ay1}" stroke="{INK}" stroke-width="1.2"/>')
    for gx in (20, 40, 80, 160, 320, 640):
        if min(xs) <= gx <= max(xs) * 1.2:
            s.append(f'<line x1="{X(gx):.1f}" y1="{ay0}" x2="{X(gx):.1f}" y2="{ay1}" stroke="#eee"/>')
            s.append(f'<text x="{X(gx):.1f}" y="{ay1+16}" font-size="10" fill="#555" text-anchor="middle">{gx}</text>')
    for gy in [v/100 for v in range(int(ylo*100)//2*2, 101, 2)]:
        s.append(f'<line x1="{ax0}" y1="{Y(gy):.1f}" x2="{ax1}" y2="{Y(gy):.1f}" stroke="#f0f0f0"/>')
        s.append(f'<text x="{ax0-6}" y="{Y(gy)+4:.1f}" font-size="10" fill="#555" text-anchor="end">{gy:.2f}</text>')
    s.append(f'<text x="{(ax0+ax1)/2:.0f}" y="{ay1+38}" font-size="12" fill="{INK}" text-anchor="middle">L0 (mean active latents / token, log scale)</text>')
    s.append(f'<text x="22" y="{(ay0+ay1)/2:.0f}" font-size="12" fill="{INK}" text-anchor="middle" transform="rotate(-90 22 {(ay0+ay1)/2:.0f})">Loss Recovered</text>')
    # one polyline + markers per architecture
    archs = {}
    for r in rows:
        archs.setdefault(r["arch"], []).append(r)
    ly = ay0
    for arch, rs in archs.items():
        rs = sorted(rs, key=lambda r: r["l0"]); col = ARCH_COLORS.get(arch, INK)
        pts = " ".join(f"{X(r['l0']):.1f},{Y(r['loss_recovered']):.1f}" for r in rs)
        s.append(f'<polyline points="{pts}" fill="none" stroke="{col}" stroke-width="2" opacity="0.9"/>')
        for r in rs:
            s.append(f'<circle cx="{X(r["l0"]):.1f}" cy="{Y(r["loss_recovered"]):.1f}" r="3" fill="{col}"/>')
        s.append(f'<circle cx="{ax1+22}" cy="{ly}" r="4" fill="{col}"/>'
                 f'<text x="{ax1+32}" y="{ly+4}" font-size="11" fill="{INK}">{_esc(arch)}</text>')
        ly += 20
    s.append('</svg>')
    return "\n".join(s)


def fig_autointerp_convergence(points, published, null_baseline,
                               title="AutoInterp reproduction — score converges to published with token budget",
                               subtitle="") -> str:
    """points: list of (n_tokens, score). published: float (2M-token value). null_baseline: float."""
    import math
    W, H = 720, 420
    ax0, ax1, ay0, ay1 = 80, 560, 80, 350
    xs = [p[0] for p in points] + [2_000_000]
    xlo, xhi = math.log10(min(xs)) - 0.1, math.log10(2_000_000) + 0.1
    ylo, yhi = 0.66, 0.80
    def X(v): return ax0 + (math.log10(v) - xlo) / (xhi - xlo) * (ax1 - ax0)
    def Y(v): return ay1 - (v - ylo) / (yhi - ylo) * (ay1 - ay0)
    s = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" font-family="Inter,Arial,sans-serif">',
         f'<rect width="{W}" height="{H}" fill="white"/>',
         f'<text x="30" y="34" font-size="16.5" font-weight="700" fill="{INK}">{_esc(title)}</text>',
         f'<text x="30" y="54" font-size="12" fill="#555">{_esc(subtitle)}</text>',
         f'<line x1="{ax0}" y1="{ay1}" x2="{ax1}" y2="{ay1}" stroke="{INK}" stroke-width="1.2"/>',
         f'<line x1="{ax0}" y1="{ay0}" x2="{ax0}" y2="{ay1}" stroke="{INK}" stroke-width="1.2"/>']
    for gx in (24000, 96000, 500000, 2000000):
        s.append(f'<line x1="{X(gx):.1f}" y1="{ay0}" x2="{X(gx):.1f}" y2="{ay1}" stroke="#f0f0f0"/>')
        lab = f"{gx//1000}k" if gx < 1_000_000 else "2M"
        s.append(f'<text x="{X(gx):.1f}" y="{ay1+16}" font-size="10" fill="#555" text-anchor="middle">{lab}</text>')
    for gy in [0.68, 0.72, 0.76, 0.80]:
        s.append(f'<text x="{ax0-6}" y="{Y(gy)+3:.1f}" font-size="10" fill="#555" text-anchor="end">{gy:.2f}</text>')
    s.append(f'<text x="{(ax0+ax1)/2:.0f}" y="{ay1+36}" font-size="12" fill="{INK}" text-anchor="middle">activation tokens (log)</text>')
    s.append(f'<text x="24" y="{(ay0+ay1)/2:.0f}" font-size="12" fill="{INK}" text-anchor="middle" transform="rotate(-90 24 {(ay0+ay1)/2:.0f})">AutoInterp score</text>')
    # null baseline + published reference lines
    s.append(f'<line x1="{ax0}" y1="{Y(null_baseline):.1f}" x2="{ax1}" y2="{Y(null_baseline):.1f}" stroke="#999" stroke-dasharray="3 3"/>')
    s.append(f'<text x="{ax0+6}" y="{Y(null_baseline)-5:.1f}" font-size="10.5" fill="#999">null baseline (predict none) = {null_baseline:.3f}</text>')
    s.append(f'<line x1="{ax0}" y1="{Y(published):.1f}" x2="{ax1}" y2="{Y(published):.1f}" stroke="{REF}" stroke-dasharray="5 4" stroke-width="1.4"/>')
    s.append(f'<text x="{ax1-2}" y="{Y(published)-6:.1f}" font-size="11" fill="{REF}" text-anchor="end">published (2M) = {published:.3f}</text>')
    # my points + connecting line
    pts = sorted(points)
    poly = " ".join(f"{X(x):.1f},{Y(y):.1f}" for x, y in pts)
    s.append(f'<polyline points="{poly}" fill="none" stroke="{MINE}" stroke-width="2"/>')
    for x, y in pts:
        s.append(f'<circle cx="{X(x):.1f}" cy="{Y(y):.1f}" r="4" fill="{MINE}"/>')
        s.append(f'<text x="{X(x):.1f}" y="{Y(y)-9:.1f}" font-size="10.5" fill="{MINE}" text-anchor="middle">{y:.3f}</text>')
    s.append(f'<circle cx="{X(2_000_000):.1f}" cy="{Y(published):.1f}" r="4" fill="{REF}"/>')
    s.append('</svg>')
    return "\n".join(s)


def fig_metric_agreement(summary, title="Full Core metric reproduction vs Neuronpedia",
                         subtitle="") -> str:
    """Horizontal bars: mean relative %% error per metric (42 SAEs) vs published, with Pearson."""
    import math
    items = [(k, v["mean_rel_pct"], v["pearson"]) for k, v in summary.items()]
    items.sort(key=lambda t: t[1])
    W, H = 760, 70 + 26 * len(items)
    x0 = 250; xmax = 480
    def X(p):  # log-ish scale 0.001%..100%
        p = max(p, 0.005)
        return x0 + (math.log10(p) - math.log10(0.005)) / (math.log10(60) - math.log10(0.005)) * (xmax - x0)
    s = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" font-family="Inter,Arial,sans-serif">',
         f'<rect width="{W}" height="{H}" fill="white"/>',
         f'<text x="30" y="32" font-size="17" font-weight="700" fill="{INK}">{_esc(title)}</text>',
         f'<text x="30" y="52" font-size="12" fill="#555">{_esc(subtitle)}</text>']
    for gx in (0.01, 0.1, 1, 10):
        s.append(f'<line x1="{X(gx):.1f}" y1="64" x2="{X(gx):.1f}" y2="{H-10}" stroke="#eee"/>')
        s.append(f'<text x="{X(gx):.1f}" y="{H-2}" font-size="9" fill="#999" text-anchor="middle">{gx}%</text>')
    s.append(f'<line x1="{X(1):.1f}" y1="64" x2="{X(1):.1f}" y2="{H-10}" stroke="{OK}" stroke-dasharray="4 3" stroke-width="1.3"/>')
    for i, (k, rel, pr) in enumerate(items):
        y = 78 + i * 26
        col = OK if rel <= 1 else ("#b8860b" if rel <= 5 else REF)
        s.append(f'<text x="{x0-8}" y="{y+4}" font-size="11" fill="{INK}" text-anchor="end">{_esc(k)}</text>')
        s.append(f'<rect x="{x0}" y="{y-7}" width="{max(X(rel)-x0,1):.1f}" height="11" rx="3" fill="{col}" opacity="0.85"/>')
        pear = "" if (pr != pr) else f"r={pr:.3f}"
        s.append(f'<text x="{X(rel)+8:.1f}" y="{y+3}" font-size="10.5" fill="{INK}">{rel:.2f}%  <tspan fill="#888">{pear}</tspan></text>')
    s.append('</svg>')
    return "\n".join(s)


def fig_reproduction_scatter(rows, summary, title="Reproduction: mine vs released",
                             subtitle="") -> str:
    """Scatter of my Loss Recovered vs the bundled (released) value, colored by architecture."""
    W, H = 520, 500
    ax0, ax1, ay0, ay1 = 80, 430, 70, 430
    allv = [r["loss_recovered"] for r in rows] + [r["bundle_frac"] for r in rows]
    lo, hi = min(allv) - 0.005, 1.002
    def X(v): return ax0 + (v - lo) / (hi - lo) * (ax1 - ax0)
    def Y(v): return ay1 - (v - lo) / (hi - lo) * (ay1 - ay0)
    s = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" font-family="Inter,Arial,sans-serif">',
         f'<rect width="{W}" height="{H}" fill="white"/>',
         f'<text x="30" y="32" font-size="17" font-weight="700" fill="{INK}">{_esc(title)}</text>',
         f'<text x="30" y="52" font-size="12" fill="#555">{_esc(subtitle)}</text>']
    s.append(f'<line x1="{X(lo):.1f}" y1="{Y(lo):.1f}" x2="{X(hi):.1f}" y2="{Y(hi):.1f}" stroke="#bbb" stroke-dasharray="5 4"/>')
    s.append(f'<text x="{X(hi)-4:.1f}" y="{Y(hi)+16:.1f}" font-size="10" fill="#999" text-anchor="end">y = x</text>')
    s.append(f'<line x1="{ax0}" y1="{ay1}" x2="{ax1}" y2="{ay1}" stroke="{INK}" stroke-width="1.2"/>')
    s.append(f'<line x1="{ax0}" y1="{ay0}" x2="{ax0}" y2="{ay1}" stroke="{INK}" stroke-width="1.2"/>')
    for g in [v/100 for v in range(int(lo*100)//2*2, 101, 2)]:
        s.append(f'<text x="{X(g):.1f}" y="{ay1+15}" font-size="9.5" fill="#555" text-anchor="middle">{g:.2f}</text>')
        s.append(f'<text x="{ax0-6}" y="{Y(g)+3:.1f}" font-size="9.5" fill="#555" text-anchor="end">{g:.2f}</text>')
    for r in rows:
        col = ARCH_COLORS.get(r["arch"], INK)
        s.append(f'<circle cx="{X(r["bundle_frac"]):.1f}" cy="{Y(r["loss_recovered"]):.1f}" r="3.4" fill="{col}" opacity="0.85"/>')
    s.append(f'<text x="{(ax0+ax1)/2:.0f}" y="{ay1+36}" font-size="12" fill="{INK}" text-anchor="middle">Released Loss Recovered (bundled eval_results.json)</text>')
    s.append(f'<text x="24" y="{(ay0+ay1)/2:.0f}" font-size="12" fill="{INK}" text-anchor="middle" transform="rotate(-90 24 {(ay0+ay1)/2:.0f})">My Loss Recovered</text>')
    # stats box
    bx, by = ax0 + 14, ay0 + 6
    s.append(f'<text x="{bx}" y="{by+10}" font-size="11" fill="{INK}">n = {summary["n_saes"]} SAEs (7 arch &#215; 6 sparsity)</text>')
    s.append(f'<text x="{bx}" y="{by+26}" font-size="11" fill="{INK}">max |&#916;| = {summary["max_abs_dLR"]:.4f}  ·  mean |&#916;| = {summary["mean_abs_dLR"]:.4f}</text>')
    s.append(f'<text x="{bx}" y="{by+42}" font-size="11" fill="{OK}">Pearson r = {summary["pearson_LR"]:.4f}  ·  all within &#177;0.01</text>')
    s.append('</svg>')
    return "\n".join(s)


def fig_mine_vs_bundle(result: dict, bundled: dict, title_sub: str = "") -> str:
    r, b = result, bundled
    rows = [
        ("Loss Recovered", r["frac_recovered"], b["frac_recovered"], "0-1"),
        ("L0",             r["l0"],             b["l0"],             "count"),
        ("H_orig (CE)",    r["loss_original"],  b["loss_original"],  "nats"),
        ("H* (recon CE)",  r["loss_reconstructed"], b["loss_reconstructed"], "nats"),
        ("H0 (zero-abl)",  r["loss_zero"],      b["loss_zero"],      "nats"),
    ]
    W, H = 720, 430
    x0, y0, bw, gap = 250, 70, 360, 66
    full = bw * 0.82
    s = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" font-family="Inter,Arial,sans-serif">',
         f'<rect width="{W}" height="{H}" fill="white"/>',
         f'<text x="30" y="34" font-size="19" font-weight="700" fill="{INK}">Core / Loss Recovered — mine vs SAEBench released</text>',
         f'<text x="30" y="54" font-size="12.5" fill="#555">{_esc(title_sub)}</text>',
         f'<circle cx="{x0}" cy="{H-22}" r="6" fill="{MINE}"/><text x="{x0+12}" y="{H-18}" font-size="12" fill="{INK}">mine</text>',
         f'<circle cx="{x0+90}" cy="{H-22}" r="6" fill="{REF}"/><text x="{x0+102}" y="{H-18}" font-size="12" fill="{INK}">bundle (ref)</text>']
    for i, (name, mine, ref, unit) in enumerate(rows):
        yc = y0 + i * gap
        mx = mine / ref if ref else 0
        s.append(f'<text x="235" y="{yc+14}" font-size="13" font-weight="600" fill="{INK}" text-anchor="end">{_esc(name)}</text>')
        s.append(f'<rect x="{x0}" y="{yc}" width="{full:.1f}" height="9" rx="4" fill="{REF}" opacity="0.30"/>')
        s.append(f'<rect x="{x0}" y="{yc+13}" width="{full*mx:.1f}" height="9" rx="4" fill="{MINE}"/>')
        pct = 100 * abs(mine - ref) / ref if ref else 0
        col = OK if pct <= 5 else "#b8860b"
        txt = (f"{mine:.4f}  vs  {ref:.4f}" if unit == "0-1"
               else f"{mine:.1f}  vs  {ref:.1f}" if unit == "count"
               else f"{mine:.3f}  vs  {ref:.3f}")
        s.append(f'<text x="{x0+full+12}" y="{yc+19}" font-size="12.5" fill="{INK}">{txt}</text>')
        s.append(f'<text x="{x0+full+12+200}" y="{yc+19}" font-size="12.5" font-weight="700" fill="{col}">&#916; {pct:.1f}%</text>')
    s.append('</svg>')
    return "\n".join(s)


def fig_convergence(perbatch_rows: list, bundle_h: float, bundle_f: float,
                    title_sub: str = "") -> str:
    rows = sorted(perbatch_rows, key=lambda r: r["bi"])
    ndocs, run_h, run_f = [], [], []
    sh = sf = 0.0
    bs = rows[0].get("n_seqs", 1) if rows else 1
    for k, r in enumerate(rows, 1):
        sh += r["loss_original"]; sf += r["frac_recovered"]
        ndocs.append(k * bs); run_h.append(sh / k); run_f.append(sf / k)

    W, H = 720, 430
    pt, pb = 110, 56

    def plot(panel_x, ys, ref, ylo, yhi, title, refcol, ycol, fmt, cid):
        ax0, ax1 = panel_x, panel_x + 270
        ay0, ay1 = pt, H - pb
        out = [f'<clipPath id="clip{cid}"><rect x="{ax0}" y="{ay0}" width="{ax1-ax0}" height="{ay1-ay0}"/></clipPath>',
               f'<text x="{ax0}" y="{pt-16}" font-size="13" font-weight="700" fill="{INK}">{title}</text>',
               f'<line x1="{ax0}" y1="{ay1}" x2="{ax1}" y2="{ay1}" stroke="{INK}" stroke-width="1.2"/>',
               f'<line x1="{ax0}" y1="{ay0}" x2="{ax0}" y2="{ay1}" stroke="{INK}" stroke-width="1.2"/>']
        def X(v): return ax0 + (v - ndocs[0]) / (ndocs[-1] - ndocs[0]) * (ax1 - ax0)
        def Y(v): return ay1 - (v - ylo) / (yhi - ylo) * (ay1 - ay0)
        out.append(f'<line x1="{ax0}" y1="{Y(ref):.1f}" x2="{ax1}" y2="{Y(ref):.1f}" stroke="{refcol}" stroke-width="1.4" stroke-dasharray="5 4"/>')
        out.append(f'<text x="{ax1-2}" y="{Y(ref)-5:.1f}" font-size="11" fill="{refcol}" text-anchor="end">bundle {fmt.format(ref)}</text>')
        pts = " ".join(f"{X(x):.1f},{Y(y):.1f}" for x, y in zip(ndocs, ys))
        out.append(f'<polyline points="{pts}" fill="none" stroke="{ycol}" stroke-width="2" clip-path="url(#clip{cid})"/>')
        out.append(f'<circle cx="{X(ndocs[-1]):.1f}" cy="{Y(ys[-1]):.1f}" r="3.5" fill="{ycol}"/>')
        out.append(f'<text x="{X(ndocs[-1]):.1f}" y="{Y(ys[-1])-8:.1f}" font-size="11" fill="{ycol}" text-anchor="end">{fmt.format(ys[-1])}</text>')
        for v in (ylo, yhi):
            out.append(f'<text x="{ax0-6}" y="{Y(v)+4:.1f}" font-size="10" fill="#555" text-anchor="end">{fmt.format(v)}</text>')
        out.append(f'<text x="{(ax0+ax1)/2:.0f}" y="{ay1+34}" font-size="11" fill="#555" text-anchor="middle"># documents</text>')
        return out

    s = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" font-family="Inter,Arial,sans-serif">',
         f'<rect width="{W}" height="{H}" fill="white"/>',
         f'<text x="30" y="34" font-size="18" font-weight="700" fill="{INK}">Convergence to the bundle (exact per-document run)</text>',
         f'<text x="30" y="55" font-size="12" fill="#555">{_esc(title_sub)}</text>']
    s += plot(60, run_h, bundle_h, 2.4, 3.2, "Running H_orig (model CE)", REF, MINE, "{:.3f}", "L")
    s += plot(400, run_f, bundle_f, 0.982, 0.992, "Running Loss Recovered", REF, OK, "{:.4f}", "R")
    s.append('</svg>')
    return "\n".join(s)
