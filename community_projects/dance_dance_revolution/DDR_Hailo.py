#!/usr/bin/env python3
"""
Dance Dance Revolution — Hailo Pose Estimation Edition
Single-player DDR: move your wrists and legs into the correct column
when a matching-coloured node reaches the hit line.

Limb → Colour:
  Left Wrist  → Red    (COCO-17 keypoint 9)
  Right Wrist → Blue   (COCO-17 keypoint 10)
  Left Leg    → Green  (avg of knee 13 + ankle 15)
  Right Leg   → Yellow (avg of knee 14 + ankle 16)

Nodes of any colour fall in any of the 4 columns at random.
When a node reaches the hit line, place the matching limb in that column to score.

Usage:
  source ~/hailo-rpi5-examples/setup_env.sh
  python DDR_Hailo.py --input usb       # USB webcam
  python DDR_Hailo.py --input rpi       # Raspberry Pi Camera

Keyboard fallback (no Hailo hardware):
  Left wrist:  Q / W / E / R  → columns 1 / 2 / 3 / 4
  Right wrist: A / S / D / F  → columns 1 / 2 / 3 / 4
  Left knee:   Z / X / C / V  → columns 1 / 2 / 3 / 4
  Right knee:  1 / 2 / 3 / 4  → columns 1 / 2 / 3 / 4
"""

import threading
import queue
import argparse
import sys
import random
import os
import json
import numpy as np
import time

import pygame
import pygame.sndarray

# =============================================================================
# ARGUMENT PARSING
# Normalise --input before hailo_apps_infra reads sys.argv
# =============================================================================

def _parse_and_normalise_input():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--input', '-i', default='usb')
    args, _ = parser.parse_known_args()
    raw = args.input.strip()
    device_path = None
    if raw.startswith('/dev/video'):
        device_path = raw
        for idx, val in enumerate(sys.argv):
            if val == raw:
                sys.argv[idx] = 'usb'
                break
        return 'usb', device_path
    if '--input' not in sys.argv and '-i' not in sys.argv:
        sys.argv += ['--input', raw]
    return raw, device_path

INPUT_SOURCE, CAMERA_DEVICE = _parse_and_normalise_input()

# =============================================================================
# HAILO / GSTREAMER — optional import
# Falls back to keyboard control if Hailo env is not active.
# =============================================================================

HAILO_AVAILABLE = False
_glib_loop = None

try:
    import gi
    gi.require_version('Gst', '1.0')
    from gi.repository import Gst, GLib
    Gst.init(None)
    import hailo
    from hailo_apps.hailo_app_python.core.common.buffer_utils import get_caps_from_pad, get_numpy_from_buffer
    from hailo_apps.hailo_app_python.core.gstreamer.gstreamer_app import app_callback_class
    from hailo_apps.hailo_app_python.apps.pose_estimation.pose_estimation_pipeline import GStreamerPoseEstimationApp
    HAILO_AVAILABLE = True
    print("[INFO] Hailo AI HAT+ detected — pose estimation enabled.")
    print(f"[INFO] Input source: {INPUT_SOURCE}" + (f"  ({CAMERA_DEVICE})" if CAMERA_DEVICE else ""))
except Exception as _hailo_err:
    print(f"[WARN] Hailo / hailo_apps_infra not found: {_hailo_err}")
    print("[INFO] Running in KEYBOARD fallback mode.")

    class app_callback_class:          # noqa: N801
        pass

    class _FakeApp:
        def run(self): pass

    def GStreamerPoseEstimationApp(cb, ud):
        return _FakeApp()

    class Gst:
        class PadProbeReturn:
            OK = 0

    class GLib:
        class MainLoop:
            def run(self): pass


def _start_glib_loop():
    """GStreamer requires a running GLib main loop to dispatch pipeline bus messages."""
    global _glib_loop
    if not HAILO_AVAILABLE:
        return
    _glib_loop = GLib.MainLoop()
    try:
        _glib_loop.run()
    except Exception as exc:
        print(f"[WARN] GLib main loop exited: {exc}")

# =============================================================================
# CONSTANTS
# =============================================================================

WINDOW_WIDTH  = 1280
WINDOW_HEIGHT = 720
FPS           = 60
NUM_COLUMNS   = 4

# COCO-17 keypoint indices
LEFT_ELBOW  = 7
RIGHT_ELBOW = 8
LEFT_WRIST  = 9
RIGHT_WRIST = 10
LEFT_KNEE   = 13
RIGHT_KNEE  = 14
LEFT_ANKLE  = 15
RIGHT_ANKLE = 16

# Limbs: (display_name, short_name, primary_keypoint_index, rgb_colour)
# Arms average elbow + wrist; legs average knee + ankle.
LIMBS = [
    ("Left Arm",  "LA", LEFT_WRIST,  (220,  60,  60)),   # Red
    ("Right Arm", "RA", RIGHT_WRIST, ( 60, 100, 230)),   # Blue
    ("Left Leg",  "LL", LEFT_KNEE,   ( 50, 200,  70)),   # Green
    ("Right Leg", "RL", RIGHT_KNEE,  (230, 200,   0)),   # Yellow
]

# Limb index → (kp_a, kp_b) — x position averaged across both keypoints
_KP_PAIRS = {
    0: (LEFT_WRIST,  LEFT_ELBOW),
    1: (RIGHT_WRIST, RIGHT_ELBOW),
    2: (LEFT_KNEE,   LEFT_ANKLE),
    3: (RIGHT_KNEE,  RIGHT_ANKLE),
}
LIMB_COLORS = [l[3] for l in LIMBS]
LIMB_SHORTS = [l[1] for l in LIMBS]

# Set True if left/right limbs appear swapped (non-mirrored camera)
MIRROR_X = False

# Gameplay
NODE_BASE_SPEED      = 4.0    # px / frame at start
NODE_SPEED_INCR      = 0.3    # extra px / frame per score-level
TIME_LEVEL_INTERVAL  = 20     # seconds between time-based speed bumps
TIME_SPEED_INCR      = 0.6    # extra px / frame per time-level
NODE_W           = 90     # node width in pixels
NODE_H           = 44     # node height in pixels
SPAWN_INTERVAL   = 80     # frames between spawns at level 1
SPAWN_MIN        = 28     # minimum frames between spawns
LEVEL_SCORE_STEP = 400    # score needed per level-up

PERFECT_WINDOW   = 28     # px either side of hit line → PERFECT
GOOD_WINDOW      = 60     # px either side → GOOD

POINTS_PERFECT   = 100
POINTS_GOOD      = 50
COMBO_BONUS      = 10     # extra points per combo step beyond 1

ROUND_SECONDS  = 60     # length of a round

# Time bonus: score TIME_BONUS_POINTS within TIME_BONUS_WINDOW seconds → +TIME_BONUS_ADD seconds
TIME_BONUS_POINTS = 300
TIME_BONUS_WINDOW = 15.0
TIME_BONUS_ADD    = 8

HIT_LINE_FRAC    = 0.80   # hit line sits at 80 % of screen height

# COCO-17 skeleton connections for PiP overlay
COCO_SKELETON = [
    (0, 1), (0, 2), (1, 3), (2, 4),
    (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 6), (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16),
]
# Keypoint index → highlight colour in skeleton overlay
_TRACKED_KP_COLOR = {
    LEFT_ELBOW:  (220,  60,  60),   # same red as left arm
    RIGHT_ELBOW: ( 60, 100, 230),   # same blue as right arm
    LEFT_WRIST:  (220,  60,  60),
    RIGHT_WRIST: ( 60, 100, 230),
    LEFT_KNEE:   ( 50, 200,  70),
    RIGHT_KNEE:  (230, 200,   0),
    LEFT_ANKLE:  ( 50, 200,  70),
    RIGHT_ANKLE: (230, 200,   0),
}

# Picture-in-picture camera overlay
PIP_W      = 280
PIP_H      = 160
PIP_MARGIN = 12

# Audio
SAMPLE_RATE = 44100
SFX_VOL     = 0.70
BGM_VOL     = 0.25

# Set paths to real audio files, or leave None to use generated tones
AUDIO_FILES = {'perfect': None, 'good': None, 'miss': None, 'bgm': None}

HIGH_SCORE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.ddr_hs.json')

# Palette
BLACK     = (  0,   0,   0)
WHITE     = (255, 255, 255)
DARK_BG   = ( 12,  10,  30)
COL_BG    = [( 18,  16,  42), ( 22,  20,  50), ( 18,  16,  42), ( 22,  20,  50)]
HIT_COL   = (255, 255, 160)
GREY      = ( 90,  90, 110)
HEART_RED = (220,  55,  55)

# Game states
GS_MENU      = 0
GS_PLAYING   = 1
GS_GAME_OVER = 2

# =============================================================================
# HIGH SCORE
# =============================================================================

def load_high_score():
    try:
        with open(HIGH_SCORE_FILE) as f:
            return json.load(f).get('hs', 0)
    except Exception:
        return 0

def save_high_score(score):
    try:
        with open(HIGH_SCORE_FILE, 'w') as f:
            json.dump({'hs': score}, f)
    except Exception:
        pass

# =============================================================================
# POSE ESTIMATION INTEGRATION
# =============================================================================

class DDRUserData(app_callback_class):
    """Thread-safe store for limb X positions shared between callback and game loop."""

    def __init__(self):
        super().__init__()
        self._lock      = threading.Lock()
        self._limb_x    = [None] * len(LIMBS)   # normalised 0-1, or None if undetected
        self._keypoints = []   # [(x_norm, y_norm), ...] for all COCO-17 kps; empty if undetected
        self.pip_queue  = queue.Queue(maxsize=1)

    def set_limb_x(self, limb_idx, x_norm):
        with self._lock:
            self._limb_x[limb_idx] = x_norm

    def set_keypoints(self, kps):
        with self._lock:
            self._keypoints = kps

    def get_limb_x(self):
        with self._lock:
            return list(self._limb_x)

    def get_keypoints(self):
        with self._lock:
            return list(self._keypoints)

    def clear_limbs(self):
        with self._lock:
            self._limb_x    = [None] * len(LIMBS)
            self._keypoints = []


class DDRCallback(app_callback_class):
    """GStreamer pad probe callback — extracts wrist and knee positions."""

    def __init__(self, user_data):
        super().__init__()
        self.user_data = user_data

    def __call__(self, pad, info, _):
        if not HAILO_AVAILABLE:
            return Gst.PadProbeReturn.OK

        buf = info.get_buffer()
        if buf is None:
            return Gst.PadProbeReturn.OK

        fmt, cam_w, cam_h = get_caps_from_pad(pad)
        if cam_w is None:
            return Gst.PadProbeReturn.OK

        roi  = hailo.get_roi_from_buffer(buf)
        dets = roi.get_objects_typed(hailo.HAILO_DETECTION)

        self.user_data.clear_limbs()

        for det in dets:
            if det.get_label() != "person":
                continue
            lms = det.get_objects_typed(hailo.HAILO_LANDMARKS)
            if not lms:
                continue
            pts  = lms[0].get_points()
            bbox = det.get_bbox()

            for li in range(len(LIMBS)):
                xs = []
                for kp in _KP_PAIRS[li]:
                    if kp < len(pts):
                        x = pts[kp].x() * bbox.width() + bbox.xmin()
                        xs.append(x)
                if not xs:
                    continue
                x_abs = sum(xs) / len(xs)
                if MIRROR_X:
                    x_abs = 1.0 - x_abs
                self.user_data.set_limb_x(li, x_abs)

            # Store all 17 keypoints (normalised frame coords) for skeleton overlay
            all_kps = []
            for pt in pts:
                x_abs = pt.x() * bbox.width() + bbox.xmin()
                y_abs = pt.y() * bbox.height() + bbox.ymin()
                if MIRROR_X:
                    x_abs = 1.0 - x_abs
                all_kps.append((x_abs, y_abs))
            self.user_data.set_keypoints(all_kps)

            # Capture PiP frame
            try:
                frame = get_numpy_from_buffer(buf, fmt, cam_w, cam_h)
                if frame is not None:
                    self.user_data.pip_queue.put_nowait(frame[:, :, :3].copy())
            except Exception:
                pass

            break   # single player — only process first detected person

        return Gst.PadProbeReturn.OK

# =============================================================================
# AUDIO HELPERS
# =============================================================================

def _tone(freq, ms, vol=0.5, wave='sine'):
    n = int(SAMPLE_RATE * ms / 1000)
    t = np.linspace(0, ms / 1000, n, False)
    if wave == 'sine':
        s = np.sin(2 * np.pi * freq * t)
    else:
        s = np.sign(np.sin(2 * np.pi * freq * t))
    fade = max(1, n // 5)
    s[-fade:] *= np.linspace(1, 0, fade)
    s = (s * vol * 32767).astype(np.int16)
    return pygame.sndarray.make_sound(np.column_stack([s, s]))


def _sweep(f0, f1, ms, vol=0.5):
    n    = int(SAMPLE_RATE * ms / 1000)
    freq = np.linspace(f0, f1, n)
    ph   = np.cumsum(2 * np.pi * freq / SAMPLE_RATE)
    s    = np.sin(ph)
    fade = max(1, n // 6)
    s[-fade:] *= np.linspace(1, 0, fade)
    s = (s * vol * 32767).astype(np.int16)
    return pygame.sndarray.make_sound(np.column_stack([s, s]))


def _load_or_gen(key, gen_fn):
    path = AUDIO_FILES.get(key)
    if path and os.path.isfile(path):
        snd = pygame.mixer.Sound(path)
        snd.set_volume(SFX_VOL)
        return snd
    return gen_fn()


def build_sounds():
    return {
        'perfect': _load_or_gen('perfect', lambda: _sweep(440, 880, 110, SFX_VOL * 0.70)),
        'good':    _load_or_gen('good',    lambda: _tone(500,  90, SFX_VOL * 0.50, 'sine')),
        'miss':    _load_or_gen('miss',    lambda: _sweep(300, 120, 200, SFX_VOL * 0.40)),
    }


def start_bgm():
    path = AUDIO_FILES.get('bgm')
    if path and os.path.isfile(path):
        pygame.mixer.music.load(path)
        pygame.mixer.music.set_volume(BGM_VOL)
        pygame.mixer.music.play(-1)
    else:
        # Procedural 4/4 beat drone
        n   = int(SAMPLE_RATE * 1.5)
        t   = np.linspace(0, 1.5, n, False)
        b   = np.sign(np.sin(2 * np.pi * 130 * t)) * 0.18
        k   = np.sign(np.sin(2 * np.pi * 260 * t)) * 0.10
        pls = (np.arange(n) % (SAMPLE_RATE // 2) < SAMPLE_RATE // 8).astype(float)
        sig = ((b + k) * (0.55 + 0.45 * pls) * BGM_VOL * 32767).astype(np.int16)
        pygame.sndarray.make_sound(np.column_stack([sig, sig])).play(loops=-1)

# =============================================================================
# NODE
# =============================================================================

class Node:
    def __init__(self, col, limb_idx, speed):
        self.col      = col
        self.limb_idx = limb_idx
        self.color    = LIMB_COLORS[limb_idx]
        self.speed    = speed
        self.y        = float(-NODE_H)
        self.state    = 'falling'   # 'falling' | 'hit' | 'missed'
        self.label    = LIMB_SHORTS[limb_idx]
        self.alpha    = 255

    def update(self):
        if self.state == 'falling':
            self.y += self.speed
        else:
            self.alpha = max(0, self.alpha - 14)

    @property
    def center_y(self):
        return self.y + NODE_H / 2

    def is_dead(self):
        return self.state != 'falling' and self.alpha == 0

# =============================================================================
# DRAWING HELPERS
# =============================================================================

def draw_heart(surf, cx, cy, size, color):
    """Filled heart using two circles + downward triangle."""
    r = size // 4
    pygame.draw.circle(surf, color, (cx - r, cy - r // 2), r)
    pygame.draw.circle(surf, color, (cx + r, cy - r // 2), r)
    pts = [
        (cx - size // 2, cy - r // 2),
        (cx + size // 2, cy - r // 2),
        (cx,             cy + size // 2),
    ]
    pygame.draw.polygon(surf, color, pts)


def draw_node(surf, nd, cx, node_font):
    x = cx + (surf.get_width() // NUM_COLUMNS - NODE_W) // 2

    # We draw onto a temporary surface so we can set alpha per-node
    ns = pygame.Surface((NODE_W, NODE_H), pygame.SRCALPHA)
    body_color = (*nd.color, nd.alpha)
    pygame.draw.rect(ns, body_color, (0, 0, NODE_W, NODE_H), border_radius=10)
    hi = tuple(min(255, c + 70) for c in nd.color)
    pygame.draw.rect(ns, (*hi, nd.alpha), (5, 4, NODE_W - 10, 9), border_radius=4)
    lbl = node_font.render(nd.label, True, (255, 255, 255))
    lbl.set_alpha(nd.alpha)
    ns.blit(lbl, lbl.get_rect(center=(NODE_W // 2, NODE_H // 2)))
    surf.blit(ns, (x, int(nd.y)))


# =============================================================================
# SKELETON / TRACKING DRAWING
# =============================================================================

def draw_pip_skeleton(surf, keypoints):
    """Overlay COCO-17 skeleton lines and keypoint dots on the PiP surface."""
    if not keypoints:
        return
    w, h = surf.get_size()

    def to_px(kp_idx):
        x, y = keypoints[kp_idx]
        return (int(x * w), int(y * h))

    # Skeleton lines
    for a, b in COCO_SKELETON:
        if a < len(keypoints) and b < len(keypoints):
            pygame.draw.line(surf, (200, 200, 200), to_px(a), to_px(b), 1)

    # Keypoint dots — tracked limbs are larger and coloured
    for idx, (x_norm, y_norm) in enumerate(keypoints):
        px, py = int(x_norm * w), int(y_norm * h)
        color  = _TRACKED_KP_COLOR.get(idx, (160, 160, 160))
        r      = 5 if idx in _TRACKED_KP_COLOR else 2
        pygame.draw.circle(surf, color, (px, py), r)
        if idx in _TRACKED_KP_COLOR:
            pygame.draw.circle(surf, (255, 255, 255), (px, py), r, 1)


def draw_game_skeleton(surf, keypoints, window_w, window_h):
    """Draw a semi-transparent full-size skeleton over the game frame."""
    if not keypoints:
        return
    skel = pygame.Surface((window_w, window_h), pygame.SRCALPHA)

    def to_px(idx):
        x, y = keypoints[idx]
        return (int(x * window_w), int(y * window_h))

    for a, b in COCO_SKELETON:
        if a < len(keypoints) and b < len(keypoints):
            pygame.draw.line(skel, (220, 220, 220, 90), to_px(a), to_px(b), 3)

    for idx, (x_norm, y_norm) in enumerate(keypoints):
        px, py = int(x_norm * window_w), int(y_norm * window_h)
        color  = _TRACKED_KP_COLOR.get(idx, (160, 160, 160))
        r      = 12 if idx in _TRACKED_KP_COLOR else 5
        pygame.draw.circle(skel, (*color, 200), (px, py), r)
        if idx in _TRACKED_KP_COLOR:
            pygame.draw.circle(skel, (255, 255, 255, 220), (px, py), r, 2)

    surf.blit(skel, (0, 0))


def draw_limb_guides(surf, raw_x, window_w, window_h, hit_line_y):
    """Draw a semi-transparent vertical guide and hit-line arrow for each detected limb."""
    guide_surf = pygame.Surface((window_w, window_h), pygame.SRCALPHA)
    for li, x_norm in enumerate(raw_x):
        if x_norm is None:
            continue
        lx    = int(x_norm * window_w)
        color = LIMB_COLORS[li]

        # Thin vertical guide spanning the full game area
        pygame.draw.line(guide_surf, (*color, 55), (lx, 0), (lx, window_h), 2)

        # Small downward-pointing arrow just above the hit line
        tip  = (lx, hit_line_y - 6)
        left = (lx - 7, hit_line_y - 20)
        right= (lx + 7, hit_line_y - 20)
        pygame.draw.polygon(guide_surf, (*color, 210), [tip, left, right])
        pygame.draw.polygon(guide_surf, (255, 255, 255, 180), [tip, left, right], 1)

    surf.blit(guide_surf, (0, 0))


# =============================================================================
# MAIN
# =============================================================================

def main():
    global WINDOW_WIDTH, WINDOW_HEIGHT

    pygame.init()
    pygame.mixer.init(frequency=SAMPLE_RATE, size=-16, channels=2, buffer=512)

    info = pygame.display.Info()
    WINDOW_WIDTH  = info.current_w
    WINDOW_HEIGHT = info.current_h

    HIT_LINE_Y   = int(WINDOW_HEIGHT * HIT_LINE_FRAC)
    ZONE_H       = WINDOW_HEIGHT - HIT_LINE_Y   # height of detection zone below hit line

    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT), pygame.FULLSCREEN)
    pygame.display.set_caption("Dance Dance Revolution — Hailo Edition")
    clock = pygame.time.Clock()

    title_font = pygame.font.Font(None, 96)
    big_font   = pygame.font.Font(None, 72)
    hud_font   = pygame.font.Font(None, 40)
    pop_font   = pygame.font.Font(None, 58)
    small_font = pygame.font.Font(None, 30)
    node_font  = pygame.font.Font(None, 28)
    tiny_font  = pygame.font.Font(None, 24)

    sounds     = build_sounds()
    high_score = load_high_score()
    start_bgm()

    col_w  = WINDOW_WIDTH // NUM_COLUMNS
    col_xs = [i * col_w for i in range(NUM_COLUMNS)]

    # ── Pose pipeline ────────────────────────────────────────────────────────
    user_data = DDRUserData()

    if HAILO_AVAILABLE:
        if CAMERA_DEVICE and not os.path.exists(CAMERA_DEVICE):
            print(f"[WARN] Camera device not found: {CAMERA_DEVICE}. Keyboard fallback.")
        else:
            glib_t = threading.Thread(target=_start_glib_loop, daemon=True, name='GLib-Loop')
            glib_t.start()
            time.sleep(0.1)

            def _run_pose():
                import signal as _s
                _s.signal = lambda *a, **kw: None

                class _PoseApp(GStreamerPoseEstimationApp):
                    def get_pipeline_string(self):
                        self.video_sink = 'fakesink'
                        return super().get_pipeline_string()

                cb = DDRCallback(user_data)
                _PoseApp(cb, user_data).run()

            threading.Thread(target=_run_pose, daemon=True, name='Pose-Pipeline').start()
    else:
        print("[INFO] Keyboard fallback active.")
        print("  Left wrist:  Q / W / E / R  → columns 1-4")
        print("  Right wrist: A / S / D / F  → columns 1-4")
        print("  Left knee:   Z / X / C / V  → columns 1-4")
        print("  Right knee:  1 / 2 / 3 / 4  → columns 1-4")

    # Keyboard fallback: limb_idx → [(key, col_idx), ...]
    KB = [
        [(pygame.K_q, 0), (pygame.K_w, 1), (pygame.K_e, 2), (pygame.K_r, 3)],   # LW
        [(pygame.K_a, 0), (pygame.K_s, 1), (pygame.K_d, 2), (pygame.K_f, 3)],   # RW
        [(pygame.K_z, 0), (pygame.K_x, 1), (pygame.K_c, 2), (pygame.K_v, 3)],   # LK
        [(pygame.K_1, 0), (pygame.K_2, 1), (pygame.K_3, 2), (pygame.K_4, 3)],   # RK
    ]

    # ── Game state ───────────────────────────────────────────────────────────
    game_state        = GS_MENU
    score             = 0
    combo             = 0
    time_left         = float(ROUND_SECONDS)
    time_level        = 0
    bonus_score_at    = 0     # score when current bonus window started
    bonus_elapsed_at  = 0.0   # elapsed seconds when current bonus window started
    nodes             = []
    popups            = []   # [text, color, cx, y, alpha, vy]
    spawn_tmr         = 0
    level             = 1
    miss_flash        = 0    # frames of red screen flash remaining
    _pip_surf         = None

    def reset():
        nonlocal score, combo, time_left, time_level, bonus_score_at, bonus_elapsed_at, \
                 nodes, popups, spawn_tmr, level, miss_flash
        score = combo = spawn_tmr = miss_flash = time_level = 0
        bonus_score_at = 0
        bonus_elapsed_at = 0.0
        time_left = float(ROUND_SECONDS)
        level = 1
        nodes.clear()
        popups.clear()

    def x_to_col(x_norm):
        if x_norm is None:
            return None
        return int(np.clip(int(x_norm * NUM_COLUMNS), 0, NUM_COLUMNS - 1))

    def add_popup(text, color, col_idx):
        cx = col_xs[col_idx] + col_w // 2
        popups.append([text, color, cx, HIT_LINE_Y - 70, 255, -2.5])

    running = True
    while running:
        dt = clock.tick(FPS) / 1000.0   # real seconds elapsed this frame

        # ── Events ───────────────────────────────────────────────────────────
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                running = False
            elif ev.type == pygame.KEYDOWN:
                k = ev.key
                if k == pygame.K_ESCAPE:
                    running = False
                elif k == pygame.K_SPACE:
                    if game_state == GS_MENU:
                        reset()
                        game_state = GS_PLAYING
                    elif game_state == GS_GAME_OVER:
                        game_state = GS_MENU
                elif k == pygame.K_r and game_state == GS_GAME_OVER:
                    reset()
                    game_state = GS_PLAYING

        # ── Keyboard fallback — resolve limb columns ──────────────────────────
        keys      = pygame.key.get_pressed()
        kb_cols   = [None] * len(LIMBS)
        for li, mappings in enumerate(KB):
            for key, col in mappings:
                if keys[key]:
                    kb_cols[li] = col
                    break

        # ── Pose data — resolve limb columns ─────────────────────────────────
        raw_x   = user_data.get_limb_x()
        raw_kps = user_data.get_keypoints() if HAILO_AVAILABLE else []
        if HAILO_AVAILABLE:
            limb_cols = [x_to_col(x) for x in raw_x]
            # Keyboard overrides pose for any pressed keys
            for li, kc in enumerate(kb_cols):
                if kc is not None:
                    limb_cols[li] = kc
        else:
            limb_cols = kb_cols

        # ── PiP camera frame ──────────────────────────────────────────────────
        try:
            frame = user_data.pip_queue.get_nowait()
            pip_raw = pygame.surfarray.make_surface(np.transpose(frame, (1, 0, 2)))
            _pip_surf = pygame.transform.scale(pip_raw, (PIP_W, PIP_H))
            draw_pip_skeleton(_pip_surf, raw_kps)
        except queue.Empty:
            pass

        # ── Game logic ────────────────────────────────────────────────────────
        if game_state == GS_PLAYING:
            level = 1 + score // LEVEL_SCORE_STEP

            # Countdown timer — use real delta time so slow frames don't stretch the clock
            time_left = max(0.0, time_left - dt)
            if time_left <= 0:
                game_state = GS_GAME_OVER
                if score > high_score:
                    high_score = score
                    save_high_score(high_score)

            # Time bonus: score TIME_BONUS_POINTS within TIME_BONUS_WINDOW seconds → +TIME_BONUS_ADD s
            elapsed          = ROUND_SECONDS - time_left
            points_in_window = score - bonus_score_at
            time_in_window   = elapsed - bonus_elapsed_at
            if points_in_window >= TIME_BONUS_POINTS:
                if time_in_window <= TIME_BONUS_WINDOW:
                    time_left += TIME_BONUS_ADD
                    add_popup(f"+{TIME_BONUS_ADD}s!", (80, 255, 180), random.randint(0, NUM_COLUMNS - 1))
                bonus_score_at   = score
                bonus_elapsed_at = elapsed

            # Spawn nodes
            elapsed    = ROUND_SECONDS - time_left
            time_level = int(elapsed / TIME_LEVEL_INTERVAL)
            spawn_tmr += 1
            interval   = max(SPAWN_MIN, SPAWN_INTERVAL - (level - 1) * 5)
            if spawn_tmr >= interval:
                spawn_tmr = 0
                col   = random.randint(0, NUM_COLUMNS - 1)
                limb  = random.randint(0, len(LIMBS) - 1)
                speed = (NODE_BASE_SPEED
                         + NODE_SPEED_INCR * (level - 1)
                         + TIME_SPEED_INCR * time_level)
                nodes.append(Node(col, limb, speed))

            # Update nodes
            for nd in nodes:
                nd.update()

            # Hit detection
            for nd in nodes:
                if nd.state != 'falling':
                    continue
                dist   = abs(nd.center_y - HIT_LINE_Y)
                in_col = (limb_cols[nd.limb_idx] == nd.col)

                if dist <= PERFECT_WINDOW and in_col:
                    nd.state = 'hit'
                    pts      = POINTS_PERFECT + combo * COMBO_BONUS
                    score   += pts
                    combo   += 1
                    sounds['perfect'].play()
                    add_popup('PERFECT!', (255, 255, 80), nd.col)

                elif dist <= GOOD_WINDOW and in_col:
                    nd.state = 'hit'
                    pts      = POINTS_GOOD + combo * COMBO_BONUS
                    score   += pts
                    combo   += 1
                    sounds['good'].play()
                    add_popup('GOOD', (100, 230, 100), nd.col)

                elif nd.center_y > HIT_LINE_Y + GOOD_WINDOW:
                    nd.state   = 'missed'
                    combo      = 0
                    miss_flash = 12
                    sounds['miss'].play()
                    add_popup('MISS', (230, 70, 70), nd.col)

            # Remove dead nodes (faded out or off-screen)
            nodes = [nd for nd in nodes
                     if not nd.is_dead() and nd.y < WINDOW_HEIGHT + 80]

            # Update popups
            for p in popups:
                p[4] -= 5
                p[3] += p[5]
            popups = [p for p in popups if p[4] > 0]

            if miss_flash > 0:
                miss_flash -= 1

            # Track high score live
            if score > high_score:
                high_score = score
                save_high_score(high_score)

        # ── DRAW ──────────────────────────────────────────────────────────────
        screen.fill(DARK_BG)

        # Column backgrounds
        for ci, cx in enumerate(col_xs):
            pygame.draw.rect(screen, COL_BG[ci], (cx, 0, col_w, WINDOW_HEIGHT))

        # Column dividers
        for ci in range(1, NUM_COLUMNS):
            pygame.draw.line(screen, (45, 42, 72), (col_xs[ci], 0), (col_xs[ci], WINDOW_HEIGHT), 2)

        # Limb position guides (pose mode only)
        if HAILO_AVAILABLE:
            draw_limb_guides(screen, raw_x, WINDOW_WIDTH, WINDOW_HEIGHT, HIT_LINE_Y)

        # Miss flash overlay
        if miss_flash > 0:
            fl = pygame.Surface((WINDOW_WIDTH, WINDOW_HEIGHT), pygame.SRCALPHA)
            fl.fill((220, 30, 30, miss_flash * 8))
            screen.blit(fl, (0, 0))

        # Nodes
        for nd in nodes:
            draw_node(screen, nd, col_xs[nd.col], node_font)

        # Hit line
        total_w = col_w * NUM_COLUMNS
        glow = pygame.Surface((total_w, 14), pygame.SRCALPHA)
        glow.fill((255, 255, 160, 35))
        screen.blit(glow, (0, HIT_LINE_Y - 6))
        pygame.draw.line(screen, HIT_COL, (0, HIT_LINE_Y), (total_w, HIT_LINE_Y), 3)

        # Target rings on the hit line — one per column
        for ci, cx in enumerate(col_xs):
            ring_cx = cx + col_w // 2
            pygame.draw.circle(screen, (55, 50, 85), (ring_cx, HIT_LINE_Y), 34)
            pygame.draw.circle(screen, (90, 85, 130), (ring_cx, HIT_LINE_Y), 34, 2)

        # Detection zone (below hit line) — show where each limb currently is
        zone_cy = HIT_LINE_Y + ZONE_H // 2
        for ci, cx in enumerate(col_xs):
            # Collect which limbs are in this column
            in_here = [li for li in range(len(LIMBS)) if limb_cols[li] == ci]
            if in_here:
                # Spread multiple limbs horizontally within the column
                spacing = col_w // (len(in_here) + 1)
                for slot, li in enumerate(in_here):
                    ix = cx + spacing * (slot + 1)
                    pygame.draw.circle(screen, LIMB_COLORS[li], (ix, zone_cy), 18)
                    pygame.draw.circle(screen, WHITE, (ix, zone_cy), 18, 2)
                    lbl = tiny_font.render(LIMB_SHORTS[li], True, WHITE)
                    screen.blit(lbl, lbl.get_rect(center=(ix, zone_cy)))
            else:
                # Empty column indicator
                ring_cx = cx + col_w // 2
                pygame.draw.circle(screen, (40, 38, 65), (ring_cx, zone_cy), 18)
                pygame.draw.circle(screen, (60, 58, 90), (ring_cx, zone_cy), 18, 2)

        # Full skeleton overlay on game frame
        if HAILO_AVAILABLE and raw_kps:
            draw_game_skeleton(screen, raw_kps, WINDOW_WIDTH, WINDOW_HEIGHT)

        # Score popups
        for text, color, cx, y, alpha, _ in popups:
            s = pop_font.render(text, True, color)
            s.set_alpha(alpha)
            screen.blit(s, s.get_rect(centerx=cx, centery=int(y)))

        # ── HUD ───────────────────────────────────────────────────────────────
        if game_state in (GS_PLAYING, GS_GAME_OVER):
            screen.blit(hud_font.render(f"Score: {score}", True, WHITE), (12, 10))
            screen.blit(hud_font.render(f"Best:  {high_score}", True, GREY), (12, 50))
            screen.blit(hud_font.render(f"Combo ×{combo}", True, (255, 215, 50)), (12, 90))
            screen.blit(small_font.render(f"Score Lv {level}  |  Speed Lv {time_level}", True, GREY), (12, 132))

            # Countdown timer
            secs    = int(time_left)
            t_color = (255, 80, 80) if time_left < 10 else WHITE
            t_surf  = big_font.render(f"{secs:02d}", True, t_color)
            t_rect  = t_surf.get_rect(topright=(WINDOW_WIDTH - 14, 8))
            screen.blit(t_surf, t_rect)
            lbl = tiny_font.render("SEC", True, GREY)
            screen.blit(lbl, lbl.get_rect(topright=(t_rect.left - 4, t_rect.centery - 6)))

        # Limb colour legend (top-right, below timer)
        lx = WINDOW_WIDTH - 180
        ly = 80
        for li, (name, short, _, color) in enumerate(LIMBS):
            pygame.draw.circle(screen, color, (lx, ly + 10), 10)
            pygame.draw.circle(screen, WHITE,  (lx, ly + 10), 10, 1)
            screen.blit(small_font.render(f"{short}: {name}", True, color), (lx + 16, ly + 2))
            ly += 28

        # PiP camera overlay
        if _pip_surf and HAILO_AVAILABLE:
            px = WINDOW_WIDTH - PIP_W - PIP_MARGIN
            py = WINDOW_HEIGHT - PIP_H - PIP_MARGIN
            screen.blit(_pip_surf, (px, py))
            pygame.draw.rect(screen, GREY, (px, py, PIP_W, PIP_H), 2)

        # ── MENU OVERLAY ──────────────────────────────────────────────────────
        if game_state == GS_MENU:
            ov = pygame.Surface((WINDOW_WIDTH, WINDOW_HEIGHT), pygame.SRCALPHA)
            ov.fill((0, 0, 0, 158))
            screen.blit(ov, (0, 0))

            cy = WINDOW_HEIGHT // 2
            screen.blit(
                title_font.render("DANCE DANCE", True, (255, 100, 210)),
                title_font.render("DANCE DANCE", True, (255, 100, 210)).get_rect(
                    centerx=WINDOW_WIDTH // 2, centery=cy - 120))
            screen.blit(
                title_font.render("REVOLUTION", True, (100, 200, 255)),
                title_font.render("REVOLUTION", True, (100, 200, 255)).get_rect(
                    centerx=WINDOW_WIDTH // 2, centery=cy - 30))

            screen.blit(
                hud_font.render("Press SPACE to start", True, WHITE),
                hud_font.render("Press SPACE to start", True, WHITE).get_rect(
                    centerx=WINDOW_WIDTH // 2, centery=cy + 60))

            if high_score > 0:
                screen.blit(
                    hud_font.render(f"High Score: {high_score}", True, (255, 220, 50)),
                    hud_font.render(f"High Score: {high_score}", True, (255, 220, 50)).get_rect(
                        centerx=WINDOW_WIDTH // 2, centery=cy + 108))

            # Limb colour key
            key_y = cy + 165
            screen.blit(small_font.render("Limbs:", True, GREY), (WINDOW_WIDTH // 2 - 230, key_y))
            for li, (name, short, _, color) in enumerate(LIMBS):
                kx = WINDOW_WIDTH // 2 - 150 + li * 110
                pygame.draw.circle(screen, color, (kx, key_y + 8), 13)
                screen.blit(tiny_font.render(short, True, color),
                            tiny_font.render(short, True, color).get_rect(centerx=kx, centery=key_y + 30))

        # ── GAME OVER OVERLAY ─────────────────────────────────────────────────
        elif game_state == GS_GAME_OVER:
            ov = pygame.Surface((WINDOW_WIDTH, WINDOW_HEIGHT), pygame.SRCALPHA)
            ov.fill((0, 0, 0, 178))
            screen.blit(ov, (0, 0))

            cy = WINDOW_HEIGHT // 2
            go = big_font.render("GAME OVER", True, (255, 70, 70))
            sc = hud_font.render(f"Score: {score}", True, WHITE)
            hs = hud_font.render(f"High Score: {high_score}", True, (255, 220, 50))
            rt = small_font.render("R = Play Again     SPACE = Menu", True, GREY)

            screen.blit(go, go.get_rect(centerx=WINDOW_WIDTH // 2, centery=cy - 90))
            screen.blit(sc, sc.get_rect(centerx=WINDOW_WIDTH // 2, centery=cy - 15))
            screen.blit(hs, hs.get_rect(centerx=WINDOW_WIDTH // 2, centery=cy + 40))
            screen.blit(rt, rt.get_rect(centerx=WINDOW_WIDTH // 2, centery=cy + 100))

        pygame.display.flip()

    pygame.quit()
    sys.exit(0)


if __name__ == '__main__':
    main()
