# /// script
# description = "Gradio UI: search EgoDex by hand-pose scenario plus SigLIP semantic text, with clip playback"
# requires-python = ">=3.10, <3.13"
# dependencies = ["daft>=0.7.15", "gradio>=5,<6", "torch", "transformers", "numpy"]
# ///
"""Pose-first scenario search over EgoDex, with semantic ranking and per-match playback.

Primary control is the HAND-POSE query (top, always visible) — a Daft DataFrame
`.where(…)` over per-frame pose features. It defines *matching segments*: the
contiguous stretches where the predicate is true (the action's start→end).
The semantic text query (optional, below) only *ranks* the matched episodes.

Flow:
  • Search → master gallery of matched episodes (thumbnail at first match).
  • Click an episode → detail: a LOOPING clip of just the matched segment,
    its exact start/end, a segment stepper (for the N times it matches), and a
    timeline showing where the matches sit in the episode.

  python query_ui.py        # prints a public *.gradio.live URL
"""

from __future__ import annotations

import os
import json
import subprocess
import threading
import time
from collections import OrderedDict, defaultdict
from concurrent.futures import ThreadPoolExecutor
from itertools import count

import daft
import gradio as gr
import numpy as np
import torch
from daft import col, lit
from transformers import AutoModel, AutoProcessor

import lerobot
import pose_features as pf
from clip_features import DEVICE, MODEL_ID, _normalized_embedding

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.environ.get("OUT", os.path.join(HERE, "out", "clip_features"))
POSE_OUT = os.environ.get("POSE_OUT", os.path.join(HERE, "out", "pose_features"))  # precomputed geometry
DATASET = os.environ.get("DATASET", "./egodex_lerobot_full")  # full pose(48+204)+video dataset
CLIPS_DIR, FRAMES_DIR = os.path.join(HERE, "ui_clips"), os.path.join(HERE, "ui_frames")
os.makedirs(CLIPS_DIR, exist_ok=True)
os.makedirs(FRAMES_DIR, exist_ok=True)

KMAX = 24                 # max episodes in the master gallery
SEG_GAP_MERGE = 5         # merge matching runs separated by < this many frames (~0.17s); keeps spans tight
SEG_MIN_FRAMES = 1        # keep even instant (1-frame) events, e.g. the moment of grasping
MAX_SEGS = 12             # cap segments shown per episode (keep the longest)
CLIP_PAD = 0.3            # seconds of lead-in / lead-out around a segment clip
MIN_CLIP_SECS = 1.2       # expand brief segments (e.g. a grasp) to at least this, so the loop is watchable
SEG_CLIP_MAX = 12.0       # cap clip length; a long continuous match shows its middle 12 s
SEMANTIC_WIN = 1.5        # seconds each side of the best frame for semantic-only (no pose) fallback
MAX_THUMBS = 16           # max exact-matched-frame thumbnails shown per segment
TWIST_ROLL_THR = 2.0      # rad/s of TRUE forearm roll (pronation/supination); absolute, not percentile

# ── warm state: embeddings ──────────────────────────────────────────────────
print("loading embeddings…")
_e = daft.read_parquet(OUT).to_pydict()
EP = np.asarray(_e["episode_index"]); FR = np.asarray(_e["frame_index"])
E = np.asarray(_e["clip_emb"], dtype=np.float32)
EP_ROWS = {int(x): np.where(EP == x)[0] for x in np.unique(EP)}
print(f"  {E.shape[0]} embeddings over {len(EP_ROWS)} episodes")

print("loading episode metadata…")
FPS = float(lerobot._read_info(lerobot._normalize_dataset_root(DATASET))["fps"])
_k = "observation.image"
_m = lerobot.read_episodes(DATASET, include_video_metadata=True).select(
    "episode_index", "tasks", f"videos/{_k}/chunk_index", f"videos/{_k}/file_index",
    f"videos/{_k}/from_timestamp", f"videos/{_k}/to_timestamp").to_pydict()
TASK, WINDOW = {}, {}
for e, t, ci, fi, a, b in zip(_m["episode_index"], _m["tasks"], _m[f"videos/{_k}/chunk_index"],
                              _m[f"videos/{_k}/file_index"], _m[f"videos/{_k}/from_timestamp"],
                              _m[f"videos/{_k}/to_timestamp"]):
    e = int(e)
    TASK[e] = (t[0] if isinstance(t, (list, tuple)) and t else (t or ""))
    WINDOW[e] = (os.path.join(DATASET, "videos", _k, f"chunk-{int(ci):03d}", f"file-{int(fi):03d}.mp4"),
                 float(a), float(b))

# ── warm state: raw pose, ONLY for the player's live skeleton overlay (viz) ──
print("loading raw pose for the player overlay…")
_p = lerobot.read(DATASET).where(col("episode_index").is_in(sorted(EP_ROWS))).select(
    "episode_index", "frame_index", "observation.state", "observation.extrinsics").to_pydict()
pep = np.asarray(_p["episode_index"]); pfr = np.asarray(_p["frame_index"])
S = np.asarray(_p["observation.state"], dtype=np.float32)
X = np.asarray(_p["observation.extrinsics"], dtype=np.float32)
# 48-D features for the live inspection table + landmark overlay ONLY (not the query path)
F = pf.compute_raw_features(S)
pf.add_temporal_features(F, pep, pfr, FPS)
pf.add_angular_velocity(F, S, pep, pfr, FPS)

# ── query features: read the PRECOMPUTED geometric parquet (run_pose_features.py) ──
# Decoupled from the UI — scenarios are computed once offline, never recomputed here.
print("loading precomputed pose features…")
_q = daft.read_parquet(POSE_OUT).where(col("episode_index").is_in(sorted(EP_ROWS))).to_pydict()
qep = np.asarray(_q["episode_index"]); qfr = np.asarray(_q["frame_index"])
# nearest embedded frame per query row (bridges 30 fps pose <-> 1 fps embeddings)
_by = defaultdict(list)
for i, (e, f) in enumerate(zip(EP, FR)):
    _by[int(e)].append((int(f), i))
_ep_emb = {e: (np.array([x[0] for x in sorted(v)]), np.array([x[1] for x in sorted(v)])) for e, v in _by.items()}
emb_row = np.empty(len(qep), dtype=np.int64)
for e in np.unique(qep):
    idx = np.where(qep == e)[0]; ef, er = _ep_emb[int(e)]
    pos = np.clip(np.searchsorted(ef, qfr[idx]), 0, len(ef) - 1)
    prev = np.clip(pos - 1, 0, len(ef) - 1)
    emb_row[idx] = er[np.where(np.abs(ef[pos] - qfr[idx]) <= np.abs(ef[prev] - qfr[idx]), pos, prev)]
POSE_DF = daft.from_pydict({**{k: np.asarray(v) for k, v in _q.items()}, "emb_row": emb_row}).collect()
_allclose = np.concatenate([np.asarray(_q["closure_L"]), np.asarray(_q["closure_R"])])
CLOSE_LO, CLOSE_HI = float(np.percentile(_allclose, 2)), float(np.percentile(_allclose, 98))
print(f"  POSE_DF: {len(qep)} frames × precomputed pose features (read, not computed)")

GRASP_RATE, LIFT_VEL = 0.20, 0.20    # 48-D thresholds: curl-closing rate (grasping), wrist up-speed (lifting)

print("loading SigLIP-2 text tower…")
_model = AutoModel.from_pretrained(MODEL_ID).to(DEVICE).eval()
_proc = AutoProcessor.from_pretrained(MODEL_ID)


def encode(text):
    inp = _proc(text=[text], return_tensors="pt", padding="max_length").to(DEVICE)
    with torch.no_grad():
        return _normalized_embedding(_model.get_text_features(**inp))[0].cpu().numpy().astype(np.float32)


# ── media: thumbnails + looping segment clips, LRU-cached ────────────────────
def _lru(maxsize):
    cache, lock = OrderedDict(), threading.Lock()

    def put(key, path):
        with lock:
            cache[key] = path; cache.move_to_end(key)
            while len(cache) > maxsize:
                _, old = cache.popitem(last=False)
                try:
                    os.remove(old)
                except OSError:
                    pass

    def get(key):
        with lock:
            if key in cache:
                cache.move_to_end(key); return cache[key]
        return None
    return get, put


_clip_get, _clip_put = _lru(300)
_frame_get, _frame_put = _lru(2000)


def get_frame(e, fr):
    if (hit := _frame_get((e, fr))):
        return hit
    path = os.path.join(FRAMES_DIR, f"ep{e:04d}_f{fr:05d}.jpg")
    if not os.path.exists(path):
        shard, a, _ = WINDOW[e]
        subprocess.run(["ffmpeg", "-y", "-ss", f"{a + fr / FPS:.4f}", "-i", shard, "-frames:v", "1", "-q:v", "3",
                        path, "-loglevel", "error"], check=True)
    _frame_put((e, fr), path); return path


def get_segment_clip(e, start_frame, end_frame):
    """A short, playable clip covering exactly [start_frame, end_frame] (+pad) — meant to be looped."""
    key = (e, int(start_frame), int(end_frame))
    if (hit := _clip_get(key)):
        return hit
    shard, a, b = WINDOW[e]
    # center the clip on the segment, expanded to >= MIN_CLIP_SECS and capped at SEG_CLIP_MAX
    want = min(SEG_CLIP_MAX, max(MIN_CLIP_SECS, (end_frame - start_frame) / FPS + 2 * CLIP_PAD))
    center = a + (start_frame + end_frame) / 2 / FPS
    start_ts = max(a, center - want / 2)
    dur = max(0.4, min(want, b - start_ts))
    path = os.path.join(CLIPS_DIR, f"ep{e:04d}_seg{int(start_frame):05d}_{int(end_frame):05d}.mp4")
    if not os.path.exists(path):
        # re-encode (not stream-copy) so the cut is FRAME-ACCURATE to [start,end] — we want just the match
        subprocess.run(["ffmpeg", "-y", "-ss", f"{start_ts:.4f}", "-i", shard, "-t", f"{dur:.4f}",
                        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "veryfast", "-crf", "23",
                        "-movflags", "+faststart", path, "-loglevel", "error"], check=True)
    _clip_put(key, path); return path


# ── pose predicate (Daft) + segment extraction ──────────────────────────────
# Scenarios grouped by kind. States read one frame; actions read motion over frames.
STATE_SCENARIOS = ["any", "hand openness", "writing grip", "hammer grip"]
ACTION_SCENARIOS = ["any", "twisting", "lifting", "reaching", "grasping", "in-hand manipulation"]


def scenario_predicate(category, scenario, hand, open_lo, open_hi):
    """Daft boolean for one selected scenario, or None (no pose filter)."""
    if scenario in (None, "any"):
        return None
    # openness in [0,1] (1 = fully open palm) -> closure bounds (low closure = open)
    span = (CLOSE_HI - CLOSE_LO) or 1.0
    clo_lo = CLOSE_HI - open_hi * span       # higher openness -> lower closure
    clo_hi = CLOSE_HI - open_lo * span

    def for_hand(h):
        if scenario == "hand openness":      # openness band, mapped onto the closure column
            return (col(f"closure_{h}") >= clo_lo) & (col(f"closure_{h}") <= clo_hi)
        if scenario == "writing grip":
            return col(f"sc_writing_{h}")
        if scenario == "hammer grip":
            return col(f"sc_hammer_{h}")
        if scenario == "twisting":
            return col(f"sc_twisting_{h}")
        if scenario == "reaching":
            return col(f"sc_reaching_{h}")
        if scenario == "in-hand manipulation":
            return col(f"sc_inhand_{h}")
        if scenario == "grasping":           # action: 48-D curl closing
            return col(f"curl_rate_{h}") <= -GRASP_RATE
        if scenario == "lifting":            # action: 48-D wrist moving up
            return col(f"wrist_vert_vel_{h}") >= LIFT_VEL
        return lit(True)
    return for_hand("L") if hand == "left" else for_hand("R") if hand == "right" else (for_hand("L") | for_hand("R"))


def segments_of(frames):
    """Contiguous matching runs (merging gaps < SEG_GAP_MERGE). Returns [(start, end), …]."""
    frames = sorted(frames)
    if not frames:
        return []
    runs, s, prev = [], frames[0], frames[0]
    for f in frames[1:]:
        if f - prev > SEG_GAP_MERGE:
            runs.append((s, prev)); s = f
        prev = f
    runs.append((s, prev))
    return [(a, b) for a, b in runs if b - a + 1 >= SEG_MIN_FRAMES] or runs


def search(query, k, hand, category, scenario, open_lo, open_hi):
    """POSE_DF [join sim] → where(scenario predicate) → rank episodes → segment each."""
    pred = scenario_predicate(category, scenario, hand, open_lo, open_hi)
    has_text = bool(query.strip())
    if pred is None and not has_text:
        return [], gr.update(value="Pick a State/Action scenario (and/or type a Semantic query)."), [], ""

    df = POSE_DF
    sims = None
    if has_text:
        sims = E @ encode(query.strip())
        df = df.join(daft.from_pydict({"emb_row": np.arange(len(sims)), "sim": sims.astype(np.float32)}), on="emb_row")
    if pred is not None:
        df = df.where(pred)
    score = (col("sim").max() if has_text else col("frame_index").count()).alias("score")
    ranked = df.groupby("episode_index").agg(col("frame_index").count().alias("n"), score).sort("score", desc=True).limit(int(k)).to_pydict()
    top = [int(e) for e in ranked["episode_index"]]
    scores = list(ranked["score"])

    # matching frames for the ranked episodes → per-episode segments
    mf = df.where(col("episode_index").is_in(top)).select("episode_index", "frame_index").to_pydict()
    frames_by_ep = defaultdict(list)
    for e, f in zip(mf["episode_index"], mf["frame_index"]):
        frames_by_ep[int(e)].append(int(f))

    results = []
    for e, sc in zip(top, scores):
        if pred is not None:
            segs = sorted(sorted(segments_of(frames_by_ep[e]), key=lambda p: p[1] - p[0], reverse=True)[:MAX_SEGS])
            framelist = sorted(frames_by_ep[e])                    # exact frames where the predicate is true
        else:  # semantic-only: a window around the episode's best-matching frame
            j = EP_ROWS[e][int(np.argmax(sims[EP_ROWS[e]]))]
            c = int(FR[j]); w = int(SEMANTIC_WIN * FPS)
            segs = [(max(0, c - w), c + w)]
            framelist = list(range(segs[0][0], segs[0][1] + 1, max(1, int(FPS // 3))))
        if segs:
            results.append({"ep": e, "score": float(sc), "has_text": has_text, "segs": segs, "frames": framelist, "task": TASK.get(e, "")})

    items = []
    for r in results:
        s0 = r["segs"][0][0]
        sct = f"sim {r['score']:.3f}" if has_text else f"{int(r['score'])} frames"
        items.append((get_frame(r["ep"], s0), f"ep {r['ep']} · {sct} · {len(r['segs'])} match(es)"))
    status = f"✅ {len(results)} episodes" + (f" · ranked by {query.strip()!r}" if has_text else "") + " · click one (left) to inspect its matches"
    return items, gr.update(value=status), results, ""


# ── client-side player: per-frame pose pushed to the browser, synced to <video> ──
POSE_ROW = {(int(e), int(f)): i for i, (e, f) in enumerate(zip(pep, pfr))}  # (episode,frame) -> pose-feature row
_DER_KEYS = ("curl", "palm_up", "pinch", "wrist_speed", "curl_rate", "wrist_angvel")
FX = FY = 736.6339            # EgoDex constant camera intrinsics (apple/ml-egodex), 1920x1080
CX, CY = 960.0, 540.0


def _project(state48, extr16):
    """Project a frame's wrist+5 fingertips (both hands) to image pixels (visualize_predictions math).

    Returns 24 flat numbers: [L wrist u,v, L tip0..4 u,v, R wrist u,v, R tip0..4 u,v];
    None for points behind the camera.
    """
    cfw = np.linalg.inv(np.asarray(extr16, np.float64).reshape(4, 4))
    flat = []
    for side in (0, 1):
        base = side * 24
        world = np.vstack([np.asarray(state48[base:base + 3]),
                           np.asarray(state48[base + 9:base + 24]).reshape(5, 3)])
        cam = (cfw @ np.hstack([world, np.ones((6, 1))]).T).T[:, :3]
        z = cam[:, 2]
        with np.errstate(divide="ignore", invalid="ignore"):
            u, v = FX * cam[:, 0] / z + CX, FY * cam[:, 1] / z + CY
        for k in range(6):
            if z[k] <= 0 or not (np.isfinite(u[k]) and np.isfinite(v[k])):
                flat += [None, None]
            else:
                flat += [round(float(u[k]), 1), round(float(v[k]), 1)]
    return flat


def _t(frame):
    s = frame / FPS
    return f"{int(s // 60)}:{s % 60:04.1f}"


def file_url(path):
    return "/gradio_api/file=" + path           # Gradio serves allowed_paths files here


def _clip_bounds(e, start, end):
    """Mirror get_segment_clip's trim math so the embedded data aligns with the video."""
    _, a, b = WINDOW[e]
    want = min(SEG_CLIP_MAX, max(MIN_CLIP_SECS, (end - start) / FPS + 2 * CLIP_PAD))
    center = a + (start + end) / 2 / FPS
    start_ts = max(a, center - want / 2)
    dur = max(0.4, min(want, b - start_ts))
    return int(round((start_ts - a) * FPS)), max(1, int(round(dur * FPS)))


def _seg_json(e, start, end, matched):
    """One segment's clip URL + per-clip-frame pose values, projected landmarks, and a
    per-frame `mt` flag (1 = this exact frame satisfied the pose predicate)."""
    path = get_segment_clip(e, start, end)
    f0, n = _clip_bounds(e, start, end)
    fing, der, pts, mt, last = [], [], [], [], None
    for k in range(n):
        i = POSE_ROW.get((e, f0 + k), last)
        if i is None:
            continue
        last = i
        dL, dR = F["fingerdist_L"][i], F["fingerdist_R"][i]
        fing.append([round(float(x), 3) for x in dL] + [round(float(x), 3) for x in dR])
        der.append([round(float(F[f"{kk}_L"][i]), 3) for kk in _DER_KEYS]
                   + [round(float(F[f"{kk}_R"][i]), 3) for kk in _DER_KEYS])
        pts.append(_project(S[i], X[i]))                       # 2D landmark overlay for this frame
        mt.append(1 if (f0 + k) in matched else 0)             # did THIS frame satisfy the predicate?
    return {"url": file_url(path), "fps": FPS, "f0": f0,
            "label": f"{_t(start)}–{_t(end)} ({(end-start)/FPS:.1f}s)", "fing": fing, "der": der, "pts": pts, "mt": mt}


_render_seq = count(1)   # unique token per player render, so the JS re-inits on every new click


def build_player(result):
    """Prefetch all of an episode's segment clips + embed per-frame pose → a self-contained JS player."""
    e, segs = result["ep"], result["segs"]
    token = next(_render_seq)
    with ThreadPoolExecutor(max_workers=8) as ex:          # prefetch every segment clip in parallel
        list(ex.map(lambda seg: get_segment_clip(e, seg[0], seg[1]), segs))
    matched = set(result["frames"])
    payload = json.dumps([_seg_json(e, s, en, matched) for s, en in segs])
    _, a, b = WINDOW[e]; total = max(1.0, (b - a) * FPS)
    bars = "".join(f'<div class="seg-bar" style="position:absolute;left:{100*s/total:.1f}%;'
                   f'width:{max(0.6, 100*(en-s)/total):.1f}%;top:0;bottom:0;border-radius:2px;background:#93c5fd"></div>'
                   for s, en in segs)
    bs = "padding:4px 12px;border:1px solid #ccc;border-radius:6px;background:#f7f7f7;cursor:pointer;font-size:14px"
    return f'''<div style="font-family:system-ui,sans-serif">
  <div style="font-weight:600;margin-bottom:8px">ep {e} — {result["task"][:90]}</div>
  <div style="display:grid;grid-template-columns:1.6fr 1fr;gap:18px;align-items:start">
    <div>
      <div style="position:relative;width:100%;line-height:0">
        <video id="poseVideo" controls autoplay loop muted playsinline style="width:100%;border-radius:8px;background:#000"></video>
        <canvas id="poseCanvas" style="position:absolute;top:0;left:0;pointer-events:none"></canvas>
      </div>
      <div style="margin:8px 0;display:flex;align-items:center;gap:12px;line-height:1.4">
        <button onclick="window.poseNav(-1)" style="{bs}">◀ prev</button>
        <span id="poseSeg" style="font-weight:600"></span>
        <button onclick="window.poseNav(1)" style="{bs}">next ▶</button>
        <label style="margin-left:auto;font-size:13px;cursor:pointer"><input type="checkbox" checked onchange="window.POVERLAY=this.checked;window._pIdx=-1;window.poseTick()"> hand-tracking overlay</label>
      </div>
      <div style="position:relative;height:18px;background:#e5e7eb;border-radius:3px;width:100%">{bars}</div>
      <div style="font-size:11px;color:#888;margin-top:4px">episode timeline — matches highlighted</div>
    </div>
    <div>
      <div id="poseFrame" style="margin-bottom:8px;font-size:14px">—</div>
      <table id="poseTable" style="width:100%;border-collapse:collapse;font-size:13px"></table>
      <div style="font-size:11px;color:#888;margin-top:10px">⟳ table updates live with the video — scrub or pause to inspect any frame</div>
    </div>
  </div>
  <script type="application/json" id="poseData" data-token="{token}">{payload}</script>
</div>'''


# JS defined once on load; the per-episode HTML embeds data + an <img onerror> that calls poseInit().
BOOTSTRAP_JS = """() => {
  const $ = (id) => document.getElementById(id);
  const cell = (v) => `<td style="padding:2px 8px;border-bottom:1px solid #eee;text-align:right">${v.toFixed(3)}</td>`;
  window.poseTick = () => {
    const v = $('poseVideo'); if (!v || !window.PSEGS) return;
    const S = window.PSEGS[window.PCUR]; if (!S || !S.fing.length) return;
    let i = Math.round(v.currentTime * S.fps);
    i = Math.max(0, Math.min(i, S.fing.length - 1));
    if (i === window._pIdx) return;            // only redraw when the frame actually changes
    window._pIdx = i;
    const fn = ['thumb','index','middle','ring','pinky'];
    const dn = ['curl (open↑)','palm ↑y','pinch','wrist speed','curl rate','wrist ang.vel'];
    let h = '<tr><th style="text-align:left;padding:2px 8px">fingers / pose</th><th style="padding:2px 8px">L</th><th style="padding:2px 8px">R</th></tr>';
    for (let k=0;k<5;k++) h += `<tr><td style="padding:2px 8px;border-bottom:1px solid #eee">${fn[k]}</td>${cell(S.fing[i][k])}${cell(S.fing[i][k+5])}</tr>`;
    for (let k=0;k<6;k++) h += `<tr><td style="padding:2px 8px;border-bottom:1px solid #eee">${dn[k]}</td>${cell(S.der[i][k])}${cell(S.der[i][k+6])}</tr>`;
    $('poseTable').innerHTML = h;
    const ep = S.f0 + i, inMatch = !!(S.mt && S.mt[i]);
    $('poseFrame').innerHTML = `frame <b>${ep}</b> · ${(ep/S.fps).toFixed(2)}s` + (inMatch ? ' · <span style="color:#16a34a">● MATCH</span>' : ' · <span style="color:#999">(context)</span>');
    window.poseDraw(S, i);
  };
  window.poseDraw = (S, i) => {
    const v = $('poseVideo'), c = $('poseCanvas'); if (!v || !c) return;
    const W = v.clientWidth, H = v.clientHeight; c.width = W; c.height = H;
    const ctx = c.getContext('2d'); ctx.clearRect(0, 0, W, H);
    if (!window.POVERLAY || !S.pts || !S.pts[i]) return;
    const sx = W / (v.videoWidth || 1920), sy = H / (v.videoHeight || 1080), P = S.pts[i];
    const hand = (off, color) => {
      const wx = P[off], wy = P[off + 1]; if (wx == null) return;
      ctx.strokeStyle = color; ctx.fillStyle = color; ctx.lineWidth = 2;
      for (let t = 0; t < 5; t++) { const tx = P[off + 2 + t*2], ty = P[off + 3 + t*2]; if (tx == null) continue;
        ctx.beginPath(); ctx.moveTo(wx*sx, wy*sy); ctx.lineTo(tx*sx, ty*sy); ctx.stroke();
        ctx.beginPath(); ctx.arc(tx*sx, ty*sy, 4, 0, 6.283); ctx.fill(); }
      ctx.beginPath(); ctx.arc(wx*sx, wy*sy, 6, 0, 6.283); ctx.fill();
    };
    hand(0, '#22d3ee'); hand(12, '#f59e0b');   // left = cyan, right = amber
  };
  window.poseLoad = (i) => {
    if (!window.PSEGS) return;
    const n = window.PSEGS.length; window.PCUR = ((i % n) + n) % n;
    const S = window.PSEGS[window.PCUR], v = $('poseVideo');
    v.src = S.url; v.load(); v.play().catch(()=>{});
    $('poseSeg').textContent = `match ${window.PCUR+1} / ${n} · ${S.label}`;
    document.querySelectorAll('.seg-bar').forEach((b,j)=> b.style.background = (j===window.PCUR) ? '#2563eb' : '#93c5fd');
    window._pIdx = -1; window.poseTick();
  };
  window.poseNav = (d) => window.poseLoad((window.PCUR||0) + d);
  window.poseInit = () => {
    const el = $('poseData'); if (!el) return false;
    const v = $('poseVideo'); if (!v) return false;
    let segs; try { segs = JSON.parse(el.textContent); } catch(e) { return false; }
    window.PSEGS = segs;
    if (window.POVERLAY === undefined) window.POVERLAY = true;
    window.PCUR = 0; window.poseLoad(0);
    cancelAnimationFrame(window._praf || 0);                 // drive the table per-frame, not per timeupdate
    const loop = () => { window.poseTick(); window._praf = requestAnimationFrame(loop); };
    loop();
    return true;
  };
  // Re-init whenever the rendered player changes (token differs), so clicking a new
  // episode always swaps the video — a sticky one-shot flag could get stuck (stale clip)
  // or fire against mismatched state (wrong clip).
  setInterval(() => {
    const el = document.getElementById('poseData');
    if (el && el.dataset.token !== window._poseToken) {
      if (window.poseInit()) window._poseToken = el.dataset.token;
    }
  }, 120);
}"""


def on_select(results, evt: gr.SelectData):
    return build_player(results[evt.index])


# ── layout: compact query bar on top; left = episodes, right = live player ───
with gr.Blocks(title="EgoDex pose dashboard", css=".gradio-container{max-width:100% !important}", js=BOOTSTRAP_JS) as demo:
    gr.Markdown("## EgoDex — pose query dashboard &nbsp;<span style='font-weight:400;color:#888'>pose filter (Daft) defines matches · semantic ranks them · click an episode; the table tracks the video live</span>")

    with gr.Row():     # compact query bar: Type (State/Action) → Scenario
        hand = gr.Dropdown(["either", "left", "right"], value="either", label="Hand", scale=1)
        category = gr.Dropdown(["State", "Action"], value="State", label="Type", scale=1)
        scenario = gr.Dropdown(STATE_SCENARIOS, value="any", label="Scenario", scale=2)
        query = gr.Textbox(label="Semantic query (optional)", scale=3)
        k_in = gr.Number(label="Max eps", value=12, precision=0, scale=1)
        btn = gr.Button("Search", variant="primary", scale=1)
    with gr.Row(visible=False) as closure_row:   # shown only for the 'hand openness' state
        open_lo = gr.Slider(0.0, 1.0, value=0.6, step=0.02,
                            label="Openness ≥  (0 = fist · 1 = fully open palm)", scale=1)
        open_hi = gr.Slider(0.0, 1.0, value=1.0, step=0.02,
                            label="Openness ≤  (keep at 1 to include fully-open hands)", scale=1)
    status = gr.Markdown()

    with gr.Row():
        with gr.Column(scale=2):   # left: episode selection
            gr.Markdown("**Episodes** — click one to load the player")
            ep_gallery = gr.Gallery(columns=2, height=560, object_fit="contain", show_label=False)
        with gr.Column(scale=8):   # right: self-contained client-side player (video + live table + match nav)
            player = gr.HTML()

    results_state = gr.State([])

    def _update_scenarios(category):
        ch = STATE_SCENARIOS if category == "State" else ACTION_SCENARIOS
        return gr.update(choices=ch, value="any"), gr.update(visible=False)

    def _toggle_closure(scenario):
        return gr.update(visible=(scenario == "hand openness"))

    category.change(_update_scenarios, [category], [scenario, closure_row])
    scenario.change(_toggle_closure, [scenario], [closure_row])

    pose_inputs = [hand, category, scenario, open_lo, open_hi]
    btn.click(search, [query, k_in, *pose_inputs], [ep_gallery, status, results_state, player])
    query.submit(search, [query, k_in, *pose_inputs], [ep_gallery, status, results_state, player])
    ep_gallery.select(on_select, [results_state], [player])

if __name__ == "__main__":
    demo.launch(share=True, allowed_paths=[CLIPS_DIR, FRAMES_DIR])
