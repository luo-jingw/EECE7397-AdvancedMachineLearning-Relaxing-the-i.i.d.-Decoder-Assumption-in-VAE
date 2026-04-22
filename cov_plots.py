"""Plots for residual-covariance experiments: mean residual, K row slices, diagonal, recon grid."""
from __future__ import annotations

import base64
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision.utils import make_grid


def save_mean_residual_map(mean_r: torch.Tensor, path: Path) -> None:
    """mean_r: [C,H,W] mean residual map."""
    mr = mean_r.detach().cpu().float().numpy()
    c = mr.shape[0]
    path.parent.mkdir(parents=True, exist_ok=True)
    if c == 1:
        fig, ax = plt.subplots(figsize=(4, 4))
        im = ax.imshow(mr[0], cmap="coolwarm", aspect="equal")
        ax.set_title(r"$\bar{r}$ (mean residual)")
        fig.colorbar(im, ax=ax, fraction=0.046)
        ax.set_axis_off()
    else:
        fig, axes = plt.subplots(1, c, figsize=(3 * c, 3.2))
        axes = np.atleast_1d(axes).ravel().tolist()
        titles = ["R", "G", "B"] if c == 3 else [f"ch{i}" for i in range(c)]
        for i in range(c):
            im = axes[i].imshow(mr[i], cmap="coolwarm", aspect="equal")
            axes[i].set_title(titles[i] + r" $\bar{r}$")
            fig.colorbar(im, ax=axes[i], fraction=0.046)
            axes[i].set_axis_off()
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _nchw_flat_index_to_rc(idx: int, c: int, h: int, w: int) -> tuple[int, int, int]:
    hw = h * w
    ch = idx // hw
    rem = idx % hw
    row = rem // w
    col = rem % w
    return ch, row, col


def save_K_row_slice_heatmaps(
    K: np.ndarray,
    in_channels: int,
    img_h: int,
    img_w: int,
    path: Path,
    row_index: int | None = None,
) -> None:
    """Static PNG: one row of |K[row,:]| heatmaps (NCHW flattening)."""
    d = K.shape[0]
    path.parent.mkdir(parents=True, exist_ok=True)
    if row_index is None:
        row_index = d // 2
    row_index = int(np.clip(row_index, 0, d - 1))
    v = np.asarray(K[row_index], dtype=np.float64)
    hw = img_h * img_w

    if in_channels == 1 or d == hw:
        mat = v.reshape(img_h, img_w)
        fig, ax = plt.subplots(figsize=(4.5, 4))
        im = ax.imshow(np.abs(mat), cmap="viridis", aspect="equal")
        ax.set_title(f"|K[row={row_index}, :]| (reshaped {img_h}x{img_w})")
        fig.colorbar(im, ax=ax, fraction=0.046)
        ax.set_axis_off()
    elif d == in_channels * hw:
        fig, axes = plt.subplots(1, in_channels, figsize=(3.2 * in_channels, 3.5))
        if in_channels == 1:
            axes = [axes]
        ch0, r0, c0 = _nchw_flat_index_to_rc(row_index, in_channels, img_h, img_w)
        for ch in range(in_channels):
            sl = v[ch * hw : (ch + 1) * hw].reshape(img_h, img_w)
            im = axes[ch].imshow(np.abs(sl), cmap="viridis", aspect="equal")
            axes[ch].set_title(f"|K[row, :]| ch={ch}")
            fig.colorbar(im, ax=axes[ch], fraction=0.046)
            axes[ch].set_axis_off()
        fig.suptitle(
            f"row index {row_index} (ch,row,col)=({ch0},{r0},{c0})",
            fontsize=10,
        )
    else:
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.plot(np.abs(v), lw=0.5)
        ax.set_title(f"|K[{row_index}, :]| (length {d})")
        ax.set_xlabel("column index")
        fig.tight_layout()

    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def write_K_row_slice_interactive_html(
    path: Path,
    K: np.ndarray,
    in_channels: int,
    img_h: int,
    img_w: int,
    *,
    picker_bg: np.ndarray | None = None,
    title: str = "K row slice (hover to select)",
) -> None:
    """
    Interactive HTML: hover over the mean (picker) heatmap to show |K[row,:]|.
    Flattening matches residuals: NCHW, index = c*(H*W) + row*W + col.
    Only d == H*W or d == C*H*W; otherwise returns without writing.
    """
    K = np.asarray(K, dtype=np.float64)
    d = int(K.shape[0])
    if K.ndim != 2 or K.shape[1] != d:
        return
    h, w, c = int(img_h), int(img_w), int(in_channels)
    hw = h * w
    if d == hw:
        mode = "single"
        c_eff = 1
    elif d == c * hw:
        mode = "joined"
        c_eff = c
    else:
        return

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    flat = np.asarray(K, dtype=np.float32).ravel(order="C")
    k_b64 = base64.b64encode(flat.tobytes()).decode("ascii")

    pick_b64: str | None = None
    bg_for_range: np.ndarray | None = None
    if picker_bg is not None:
        bg = np.asarray(picker_bg, dtype=np.float32)
        if mode == "single" and bg.ndim == 3 and bg.shape[0] == 1:
            bg = bg[0]
        if mode == "single" and bg.shape == (h, w):
            pick_b64 = base64.b64encode(bg.ravel(order="C").tobytes()).decode("ascii")
            bg_for_range = bg
        elif mode == "joined" and bg.shape == (c_eff, h, w):
            pick_b64 = base64.b64encode(bg.ravel(order="C").tobytes()).decode("ascii")
            bg_for_range = bg

    slice_zmax = float(np.abs(K).max())
    if slice_zmax < 1e-20:
        slice_zmax = 1.0
    if bg_for_range is not None:
        mean_zmin = float(bg_for_range.min())
        mean_zmax = float(bg_for_range.max())
        if mean_zmax <= mean_zmin:
            mean_zmax = mean_zmin + 1e-8
    else:
        mean_zmin = 0.0
        mean_zmax = 1.0

    def _axis_tick_indices(n: int, target: int = 8) -> list[int]:
        if n <= 1:
            return [0]
        step = max(1, int(round((n - 1) / float(max(2, target - 1)))))
        vals = list(range(0, n, step))
        if vals[-1] != n - 1:
            vals.append(n - 1)
        return vals

    tick_x = _axis_tick_indices(w)
    tick_y = _axis_tick_indices(h)

    meta = {
        "H": h,
        "W": w,
        "C": c_eff,
        "d": d,
        "mode": mode,
        "title": title,
        "hasPickerBg": pick_b64 is not None,
        "meanZMin": mean_zmin,
        "meanZMax": mean_zmax,
        "sliceZMin": 0.0,
        "sliceZMax": slice_zmax,
        "tickValsX": tick_x,
        "tickValsY": tick_y,
        "plotW": 480,
        "plotH": 510,
    }
    meta_json = json.dumps(meta, ensure_ascii=False)
    pick_json = json.dumps(pick_b64)
    title_esc = _html_escape(title)
    k_js = json.dumps(k_b64)

    template = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>__TITLE_ESC__</title>
  <script src="https://cdn.plot.ly/plotly-2.27.0.min.js" charset="utf-8"></script>
  <style>
    body { font-family: system-ui, sans-serif; margin: 16px; }
    h1 { font-size: 1.1rem; margin: 0 0 8px 0; }
    #status { margin: 0 0 10px 0; color: #333; font-size: 13px; min-height: 1.2em; }
    #strips { display: flex; flex-direction: column; gap: 14px; }
    .strip {
      display: flex;
      flex-direction: row;
      flex-wrap: wrap;
      align-items: flex-start;
      gap: 20px;
      border-bottom: 1px solid #e8e8e8;
      padding-bottom: 12px;
    }
    .strip:last-child { border-bottom: none; }
    .mean-col, .slice-col {
      display: flex;
      flex-direction: column;
      align-items: center;
    }
    .col-label { font-size: 12px; color: #555; margin-top: 6px; max-width: 520px; text-align: center; }
    .mean-col h2, .slice-col h2 {
      font-size: 11px; font-weight: 600; color: #444; margin: 0 0 6px 0;
      text-transform: uppercase; letter-spacing: 0.04em;
    }
  </style>
</head>
<body>
  <h1>__TITLE_ESC__</h1>
  <div id="status"></div>
  <div id="strips"></div>
  <script type="application/json" id="meta-json">__META_JSON__</script>
  <script>
  (function() {
    var META = JSON.parse(document.getElementById('meta-json').textContent);
    var H = META.H, W = META.W, C = META.C, d = META.d, mode = META.mode;
    var hw = H * W;
    var PW = META.plotW, PH = META.plotH;
    var tickValsX = META.tickValsX, tickValsY = META.tickValsY;

    function decodeB64ToFloat32(b64) {
      var bin = atob(b64);
      var n = bin.length;
      var u = new Uint8Array(n);
      for (var i = 0; i < n; i++) u[i] = bin.charCodeAt(i);
      return new Float32Array(u.buffer);
    }

    var Kflat = decodeB64ToFloat32(__K_JS__);
    var Pflat = null;
    if (__PICK_JSON__ !== null) Pflat = decodeB64ToFloat32(__PICK_JSON__);

    function rowVector(r) {
      var off = r * d;
      return Kflat.subarray(off, off + d);
    }

    function zPickerMatrix(ch) {
      var z = [];
      for (var i = 0; i < H; i++) {
        z.push([]);
        for (var j = 0; j < W; j++) {
          if (Pflat !== null) {
            var t;
            if (mode === 'single') t = Pflat[i * W + j];
            else t = Pflat[ch * hw + i * W + j];
            z[i].push(t);
          } else {
            z[i].push((i * W + j) / Math.max(hw - 1, 1));
          }
        }
      }
      return z;
    }

    function absSliceZ(v, ch) {
      var z = [];
      if (mode === 'single') {
        for (var i = 0; i < H; i++) {
          z.push([]);
          for (var j = 0; j < W; j++)
            z[i].push(Math.abs(v[i * W + j]));
        }
      } else {
        for (var i = 0; i < H; i++) {
          z.push([]);
          for (var j = 0; j < W; j++) {
            var idx = ch * hw + i * W + j;
            z[i].push(Math.abs(v[idx]));
          }
        }
      }
      return z;
    }

    var xArr = Array.from({length: W}, function(_, j) { return j; });
    var yArr = Array.from({length: H}, function(_, i) { return i; });

    var axisCommon = {
      constrain: 'domain',
      zeroline: false,
      showgrid: true,
      gridcolor: '#dddddd',
      gridwidth: 1,
      tickfont: {size: 10},
      showline: true,
      linewidth: 1,
      linecolor: '#888',
      tickmode: 'array'
    };

    function buildLayout() {
      return {
        width: PW,
        height: PH,
        autosize: false,
        title: '',
        margin: {l: 58, r: 84, t: 14, b: 52},
        xaxis: Object.assign({
          range: [-0.5, W - 0.5],
          tickvals: tickValsX,
          ticktext: tickValsX.map(String)
        }, axisCommon),
        yaxis: Object.assign({
          range: [H - 0.5, -0.5],
          tickvals: tickValsY,
          ticktext: tickValsY.map(String),
          scaleanchor: 'x',
          scaleratio: 1
        }, axisCommon)
      };
    }

    var colorbarPick = {
      thickness: 16,
      len: 0.82,
      outlinewidth: 0,
      tickfont: {size: 9}
    };
    var colorbarSlice = {
      thickness: 16,
      len: 0.82,
      outlinewidth: 0,
      tickfont: {size: 9}
    };

    function heatmapPickTrace(z) {
      return {
        type: 'heatmap',
        x: xArr,
        y: yArr,
        z: z,
        zmin: META.meanZMin,
        zmax: META.meanZMax,
        zsmooth: false,
        colorscale: 'Viridis',
        hovertemplate: 'row=%{y} col=%{x}<extra></extra>',
        colorbar: colorbarPick,
        showscale: true
      };
    }

    function heatmapSliceTrace(z) {
      return {
        type: 'heatmap',
        x: xArr,
        y: yArr,
        z: z,
        zmin: META.sliceZMin,
        zmax: META.sliceZMax,
        zsmooth: false,
        colorscale: 'Viridis',
        hovertemplate: 'row=%{y} col=%{x}<extra></extra>',
        colorbar: colorbarSlice,
        showscale: true
      };
    }

    var nStrips = (mode === 'joined' ? C : 1);
    var pickIds = [];
    var sliceIds = [];
    var sliceLayouts = [];

    var stripsEl = document.getElementById('strips');
    for (var si = 0; si < nStrips; si++) {
      var ch = si;
      var strip = document.createElement('div');
      strip.className = 'strip';

      var meanCol = document.createElement('div');
      meanCol.className = 'mean-col';
      var meanH = document.createElement('h2');
      meanH.textContent = mode === 'joined' ? ('Mean map (channel ' + ch + ')') : 'Mean map';
      meanCol.appendChild(meanH);
      var pid = 'pick-' + ch;
      var pdiv = document.createElement('div');
      pdiv.id = pid;
      pdiv.style.width = PW + 'px';
      pdiv.style.height = PH + 'px';
      meanCol.appendChild(pdiv);
      var meanCap = document.createElement('div');
      meanCap.className = 'col-label';
      meanCap.textContent = 'Hover cell';
      meanCol.appendChild(meanCap);

      var sliceCol = document.createElement('div');
      sliceCol.className = 'slice-col';
      var sliceH = document.createElement('h2');
      sliceH.textContent = mode === 'joined' ? ('|K[row,:]| channel ' + ch) : '|K[row,:]|';
      sliceCol.appendChild(sliceH);
      var sid = 'slice-' + ch;
      var sdiv = document.createElement('div');
      sdiv.id = sid;
      sdiv.style.width = PW + 'px';
      sdiv.style.height = PH + 'px';
      sliceCol.appendChild(sdiv);

      strip.appendChild(meanCol);
      strip.appendChild(sliceCol);
      stripsEl.appendChild(strip);

      pickIds.push(pid);
      sliceIds.push(sid);

      var loPick = buildLayout();
      var z0 = zPickerMatrix(mode === 'joined' ? ch : 0);
      Plotly.newPlot(pid, [heatmapPickTrace(z0)], loPick, {displaylogo: false, responsive: false});

      var loSl = buildLayout();
      sliceLayouts.push(loSl);
      Plotly.newPlot(sid, [heatmapSliceTrace(absSliceZ(rowVector(Math.floor(d / 2)), mode === 'single' ? 0 : ch))], loSl, {displaylogo: false, responsive: false});
    }

    function flatIndexFromHover(chPick, row, col) {
      row = Math.max(0, Math.min(H - 1, row));
      col = Math.max(0, Math.min(W - 1, col));
      if (mode === 'single') return row * W + col;
      return chPick * hw + row * W + col;
    }

    function updateSlices(rowIdx) {
      rowIdx = Math.max(0, Math.min(d - 1, rowIdx));
      var v = rowVector(rowIdx);
      document.getElementById('status').textContent = 'row_index=' + rowIdx;
      for (var cidx = 0; cidx < sliceIds.length; cidx++) {
        var zS = absSliceZ(v, mode === 'single' ? 0 : cidx);
        Plotly.react(sliceIds[cidx], [heatmapSliceTrace(zS)], sliceLayouts[cidx]);
      }
    }

    var r0 = Math.floor(d / 2);
    updateSlices(r0);

    function bindPickHover(chPick, pid) {
      var gd = document.getElementById(pid);
      gd.on('plotly_hover', function(ev) {
        if (!ev || !ev.points || !ev.points.length) return;
        var pt = ev.points[0];
        var col = Math.round(pt.x);
        var row = Math.round(pt.y);
        var fi = flatIndexFromHover(chPick, row, col);
        updateSlices(fi);
      });
    }
    for (var i = 0; i < pickIds.length; i++) bindPickHover(i, pickIds[i]);
  })();
  </script>
</body>
</html>
"""
    html = (
        template.replace("__TITLE_ESC__", title_esc)
        .replace("__META_JSON__", meta_json)
        .replace("__K_JS__", k_js)
        .replace("__PICK_JSON__", pick_json)
    )
    path.write_text(html, encoding="utf-8")


def _html_escape(s: str) -> str:
    return (
        s.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def save_K_diagonal_plot(K: np.ndarray, path: Path, title_suffix: str = "") -> None:
    diag = np.diag(np.asarray(K, dtype=np.float64))
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.plot(np.arange(diag.size), diag, lw=0.8)
    ax.set_xlabel("index")
    ax.set_ylabel("diag(K)")
    ax.set_title("Diagonal of " + r"$\hat{K}$ " + title_suffix)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_recon_grid_png(
    originals: torch.Tensor,
    recons: torch.Tensor,
    path: Path,
    nrow: int,
) -> None:
    """originals / recons: [N,C,H,W] in [0,1]."""
    pad = 2
    g1 = make_grid(originals, nrow=nrow, padding=pad, pad_value=1.0)
    g2 = make_grid(recons, nrow=nrow, padding=pad, pad_value=1.0)
    _, h, _ = g1.shape
    gap = torch.ones(g1.size(0), h, 16)
    combined = torch.cat([g1, gap, g2], dim=2)
    arr = combined.permute(1, 2, 0).numpy()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(12, 5))
    if arr.shape[2] == 1:
        plt.imshow(arr[..., 0], cmap="gray", vmin=0, vmax=1)
    else:
        plt.imshow(np.clip(arr, 0, 1))
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.savefig(path, dpi=150, bbox_inches="tight", pad_inches=0.05)
    plt.close()
