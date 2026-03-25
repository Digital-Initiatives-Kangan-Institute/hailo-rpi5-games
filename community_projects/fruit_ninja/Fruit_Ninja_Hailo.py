#!/usr/bin/env python3
"""
Fruit Ninja — Hailo Pose Estimation Edition
Controls: Wave your wrists/hands in front of the camera to slice fruits.
Runs on Raspberry Pi 5 + Hailo AI HAT+.

Usage:
  source ~/hailo-rpi5-examples/setup_env.sh
  python Fruit_Ninja_Hailo.py --input usb       # USB webcam
  python Fruit_Ninja_Hailo.py --input rpi       # Raspberry Pi Camera Module

Mouse fallback (no Hailo hardware):
  Hold left mouse button and drag to slice.
"""

import threading
import queue
import argparse
import sys
from collections import deque
import pygame
import pygame.sndarray
import random
import math
import os
import numpy as np
import time

# ─────────────────────────────────────────────────────────────────────────────
# ARGUMENT PARSING — normalise --input before hailo_apps_infra reads sys.argv
# ─────────────────────────────────────────────────────────────────────────────
def _parse_and_normalise_input():
    """
    Normalise --input so that raw device paths like /dev/video0
    are rewritten to the 'usb' keyword that hailo_apps_infra expects.
    Returns (normalised_source, raw_device_or_None).
    """
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--input', '-i', default='usb')
    args, _ = parser.parse_known_args()
    raw = args.input.strip()

    if raw.startswith('/dev/video'):
        for idx, val in enumerate(sys.argv):
            if val == raw:
                sys.argv[idx] = 'usb'
                break
        return 'usb', raw

    # Ensure --input is explicitly in sys.argv so GStreamerPoseEstimationApp picks it up
    if '--input' not in sys.argv and '-i' not in sys.argv:
        sys.argv += ['--input', raw]

    return raw, None

INPUT_SOURCE, CAMERA_DEVICE = _parse_and_normalise_input()

# ─────────────────────────────────────────────────────────────────────────────
# HAILO / GSTREAMER — optional import
# Falls back to mouse control if Hailo env is not active.
# ─────────────────────────────────────────────────────────────────────────────
HAILO_AVAILABLE = False
_glib_loop       = None

try:
    import gi
    gi.require_version('Gst', '1.0')
    from gi.repository import Gst, GLib

    Gst.init(None)   # initialise GStreamer once here

    import hailo
    from hailo_apps.hailo_app_python.core.common.buffer_utils import get_caps_from_pad, get_numpy_from_buffer
    from hailo_apps.hailo_app_python.core.gstreamer.gstreamer_app import app_callback_class
    from hailo_apps.hailo_app_python.apps.pose_estimation.pose_estimation_pipeline import GStreamerPoseEstimationApp

    HAILO_AVAILABLE = True
    print("[INFO] Hailo AI HAT+ detected — pose estimation enabled.")
    print(f"[INFO] Input source : {INPUT_SOURCE}"
          + (f"  ({CAMERA_DEVICE})" if CAMERA_DEVICE else ""))

except Exception as _hailo_err:
    print(f"[WARN] Hailo / hailo_apps_infra not found: {_hailo_err}")
    print("[INFO] Running in MOUSE fallback mode.")
    print("[INFO] To enable Hailo:  source ~/hailo-rpi5-examples/setup_env.sh")

    # Minimal stubs so the rest of the file loads without error
    class app_callback_class:          # noqa: N801
        pass

    class _FakeGStreamerApp:
        def run(self): pass

    def GStreamerPoseEstimationApp(cb, ud):
        return _FakeGStreamerApp()

    class Gst:
        class PadProbeReturn:
            OK = 0

    class GLib:
        class MainLoop:
            def run(self): pass


def _start_glib_loop():
    """
    GStreamer requires a running GLib main loop to dispatch pipeline
    bus messages.  Without it the pipeline builds but never delivers frames.
    """
    global _glib_loop
    if not HAILO_AVAILABLE:
        return
    _glib_loop = GLib.MainLoop()
    try:
        _glib_loop.run()
    except Exception as exc:
        print(f"[WARN] GLib main loop exited: {exc}")

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

WINDOW_WIDTH  = 1280
WINDOW_HEIGHT = 720
FPS           = 60

# Picture-in-picture camera overlay
PIP_W      = 320
PIP_H      = 180
PIP_MARGIN = 12

# Player legend face thumbnails
LEGEND_FACE = 80   # px — square thumbnail size

MAX_PLAYERS = 2
LEFT_WRIST  = 9     # YOLOv8 pose landmark index
RIGHT_WRIST = 10
MIRROR_X    = False  # GStreamer pipeline already applies videoflip horiz

# Gameplay
LIVES              = 3
FRUIT_SPAWN_CHANCE = 0.025   # per frame
BOMB_SPAWN_CHANCE  = 0.005
GRAVITY            = 0.35
LAUNCH_VY_MIN      = -20.0
LAUNCH_VY_MAX      = -14.0
LAUNCH_VX_RANGE    = 4.0
TRAIL_LEN          = 20
SWIPE_MIN_DIST     = 5       # px between consecutive points to count as swipe
SLICE_RADIUS_MULT  = 2.0     # multiplier on fruit radius for hit detection
BOMB_TIMEOUT_FRAMES = 3 * 60 # frames a player is frozen after hitting a bomb
COMBO_WINDOW       = 35      # frames in which consecutive slices = combo
COMBO_BONUS        = 5       # extra pts per additional fruit in a combo
PARTICLE_COUNT     = 14

HIGH_SCORE_FILE = os.path.expanduser("~/.fruit_ninja_hailo_hs")

# ── Colours ───────────────────────────────────────────────────────────────────
BLACK    = (  0,   0,   0)
WHITE    = (255, 255, 255)
DARK_BG  = (  8,   5,  20)
DIMWHITE = (140, 140, 160)

# Per-player blade trail colours (Player 1 = cyan, Player 2 = orange)
BLADE_COLORS = [(80, 210, 255), (255, 155, 55)]

# Fruit definitions: (name, body_color, highlight_color, radius, base_points)
FRUIT_TYPES = [
    ("apple",      (215,  40,  40), (255, 140, 140), 34, 10),
    ("watermelon", ( 35, 168,  35), (130, 230, 130), 46, 15),
    ("orange",     (255, 138,   0), (255, 205, 110), 34, 10),
    ("lemon",      (230, 218,   0), (255, 255, 160), 28, 10),
    ("peach",      (255, 150,  95), (255, 210, 175), 32, 20),
    ("plum",       (128,  35, 148), (200, 120, 225), 28, 20),
]

# ── Audio ─────────────────────────────────────────────────────────────────────
SAMPLE_RATE = 44100
SFX_VOLUME  = 0.7
BGM_VOLUME  = 0.25
# To use real audio files, set the paths here (None = use generated tones):
AUDIO_FILES = {
    'slice':     None,   # e.g. 'sounds/slice.wav'
    'splat':     None,
    'bomb':      None,
    'miss':      None,
    'game_over': None,
    'bgm':       None,   # e.g. 'sounds/bgm.ogg'
}

# ── Game states ───────────────────────────────────────────────────────────────
STATE_WAITING   = 0
STATE_PLAYING   = 1
STATE_GAME_OVER = 2
STATE_WIN       = 3


# ─────────────────────────────────────────────────────────────────────────────
# Persistence
# ─────────────────────────────────────────────────────────────────────────────

def load_high_score():
    try:
        with open(HIGH_SCORE_FILE) as f:
            return int(f.read().strip())
    except Exception:
        return 0

def save_high_score(score):
    try:
        with open(HIGH_SCORE_FILE, 'w') as f:
            f.write(str(score))
    except Exception:
        pass


# ─────────────────────────────────────────────────────────────────────────────
# Thread-safe pose data
# ─────────────────────────────────────────────────────────────────────────────

class FruitNinjaUserData(app_callback_class):
    """Shared state between GStreamer callback thread and main game loop."""
    def __init__(self):
        super().__init__()   # sets running, frame_count, use_frame, frame_queue
        # [player_index][wrist: 0=left, 1=right]
        self.wrist_queues = [
            [queue.Queue(maxsize=2) for _ in range(2)]
            for _ in range(MAX_PLAYERS)
        ]
        self._lock    = threading.Lock()
        self._active  = [False] * MAX_PLAYERS
        self.pip_queue        = queue.Queue(maxsize=1)   # latest camera frame for PiP
        self.head_bboxes      = [None] * MAX_PLAYERS    # (nx1,ny1,nx2,ny2) normalised
        self.head_crop_queues = [queue.Queue(maxsize=1) for _ in range(MAX_PLAYERS)]

    @property
    def active(self):
        with self._lock:
            return list(self._active)

    @active.setter
    def active(self, value):
        with self._lock:
            self._active = list(value)

    def set_active(self, i, state):
        with self._lock:
            self._active[i] = state


# ─────────────────────────────────────────────────────────────────────────────
# Pose callback — extracts wrist positions from Hailo detections
# ─────────────────────────────────────────────────────────────────────────────

class FruitNinjaCallback(app_callback_class):
    def __init__(self, user_data):
        super().__init__()
        self.user_data = user_data

    def __call__(self, pad, info, _):
        if not HAILO_AVAILABLE:
            return Gst.PadProbeReturn.OK

        buffer = info.get_buffer()
        if buffer is None:
            return Gst.PadProbeReturn.OK

        self.user_data.active = [False] * MAX_PLAYERS
        roi        = hailo.get_roi_from_buffer(buffer)
        detections = roi.get_objects_typed(hailo.HAILO_DETECTION)
        _, cam_w, cam_h = get_caps_from_pad(pad)

        count = 0
        for det in detections:
            if det.get_label() != "person" or count >= MAX_PLAYERS:
                continue
            landmarks = det.get_objects_typed(hailo.HAILO_LANDMARKS)
            if not landmarks:
                continue
            pts  = landmarks[0].get_points()
            bbox = det.get_bbox()

            for wrist_slot, pt_idx in enumerate([LEFT_WRIST, RIGHT_WRIST]):
                pt  = pts[pt_idx]
                xn  = pt.x() * bbox.width()  + bbox.xmin()
                yn  = pt.y() * bbox.height() + bbox.ymin()
                xpx = xn * cam_w
                ypx = yn * cam_h
                if MIRROR_X:
                    xpx = cam_w - xpx
                wx = int(np.clip(xpx * WINDOW_WIDTH  / cam_w, 0, WINDOW_WIDTH  - 1))
                wy = int(np.clip(ypx * WINDOW_HEIGHT / cam_h, 0, WINDOW_HEIGHT - 1))
                q  = self.user_data.wrist_queues[count][wrist_slot]
                try:
                    while not q.empty():
                        q.get_nowait()
                    q.put_nowait((wx, wy))
                except queue.Full:
                    pass

            # Head crop bbox — centred on nose landmark (index 0)
            NOSE = 0
            if len(pts) > NOSE and cam_w and cam_h:
                nose    = pts[NOSE]
                nx = nose.x() * bbox.width()  + bbox.xmin()
                ny = nose.y() * bbox.height() + bbox.ymin()
                hr_y = 0.12                          # head half-height in normalised coords
                hr_x = hr_y * (cam_h / cam_w)       # keep square in pixel space
                x1 = max(0.0, nx - hr_x)
                y1 = max(0.0, ny - hr_y * 1.3)      # forehead above nose
                x2 = min(1.0, nx + hr_x)
                y2 = min(1.0, ny + hr_y * 0.7)      # chin below nose
                with self.user_data._lock:
                    self.user_data.head_bboxes[count] = (x1, y1, x2, y2)

            self.user_data.set_active(count, True)
            count += 1

        # Clear bboxes for players no longer detected
        with self.user_data._lock:
            for i in range(count, MAX_PLAYERS):
                self.user_data.head_bboxes[i] = None

        return Gst.PadProbeReturn.OK


# ─────────────────────────────────────────────────────────────────────────────
# Audio helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_tone(freq, dur_ms, vol=0.5, wave='square'):
    n   = int(SAMPLE_RATE * dur_ms / 1000)
    t   = np.linspace(0, dur_ms / 1000, n, False)
    if wave == 'square':
        sig = np.sign(np.sin(2 * np.pi * freq * t))
    elif wave == 'sine':
        sig = np.sin(2 * np.pi * freq * t)
    else:  # noise
        sig = np.random.uniform(-1, 1, n)
    fade = max(1, int(n * 0.2))
    sig[-fade:] *= np.linspace(1, 0, fade)
    sig = (sig * vol * 32767).astype(np.int16)
    return pygame.sndarray.make_sound(np.column_stack([sig, sig]))


def _make_sweep(f0, f1, dur_ms, vol=0.5):
    n     = int(SAMPLE_RATE * dur_ms / 1000)
    freqs = np.linspace(f0, f1, n)
    phase = np.cumsum(2 * np.pi * freqs / SAMPLE_RATE)
    sig   = np.sin(phase)
    fade  = max(1, int(n * 0.15))
    sig[-fade:] *= np.linspace(1, 0, fade)
    sig = (sig * vol * 32767).astype(np.int16)
    return pygame.sndarray.make_sound(np.column_stack([sig, sig]))


def _load_or_gen(key, fn):
    path = AUDIO_FILES.get(key)
    if path and os.path.isfile(path):
        s = pygame.mixer.Sound(path)
        s.set_volume(SFX_VOLUME)
        return s
    return fn()


def build_sounds():
    return {
        'slice':     _load_or_gen('slice',     lambda: _make_sweep(800, 200, 80,  vol=SFX_VOLUME * 0.55)),
        'splat':     _load_or_gen('splat',     lambda: _make_tone(150, 150,  vol=SFX_VOLUME * 0.75, wave='noise')),
        'bomb':      _load_or_gen('bomb',      lambda: _make_tone(80,  400,  vol=SFX_VOLUME,        wave='noise')),
        'miss':      _load_or_gen('miss',      lambda: _make_sweep(420, 140, 220, vol=SFX_VOLUME * 0.5)),
        'game_over': _load_or_gen('game_over', lambda: _make_sweep(440, 100, 900, vol=SFX_VOLUME * 0.9)),
    }


def start_bgm():
    path = AUDIO_FILES.get('bgm')
    if path and os.path.isfile(path):
        pygame.mixer.music.load(path)
        pygame.mixer.music.set_volume(BGM_VOLUME)
        pygame.mixer.music.play(-1)
    else:
        n     = int(SAMPLE_RATE * 2.0)
        t     = np.linspace(0, 2.0, n, False)
        drone = np.sign(np.sin(2 * np.pi * 65   * t)) * 0.3
        drone += np.sign(np.sin(2 * np.pi * 65.8 * t)) * 0.3
        pulse = 0.5 + 0.5 * np.sin(2 * np.pi * 0.4 * t)
        sig   = (drone * pulse * BGM_VOLUME * 32767).astype(np.int16)
        pygame.sndarray.make_sound(np.column_stack([sig, sig])).play(loops=-1)


# ─────────────────────────────────────────────────────────────────────────────
# Game object factories
# ─────────────────────────────────────────────────────────────────────────────

def make_fruit():
    ftype = random.choice(FRUIT_TYPES)
    return {
        'x':      float(random.randint(80, WINDOW_WIDTH - 80)),
        'y':      float(WINDOW_HEIGHT + ftype[3] + 10),
        'vx':     random.uniform(-LAUNCH_VX_RANGE, LAUNCH_VX_RANGE),
        'vy':     random.uniform(LAUNCH_VY_MIN, LAUNCH_VY_MAX),
        'type':   ftype,
        'angle':  random.uniform(0, math.pi * 2),
        'spin':   random.uniform(-0.07, 0.07),
        'sliced': False,
    }


def make_bomb():
    return {
        'x':      float(random.randint(100, WINDOW_WIDTH - 100)),
        'y':      float(WINDOW_HEIGHT + 35),
        'vx':     random.uniform(-2.0, 2.0),
        'vy':     random.uniform(LAUNCH_VY_MIN + 2, LAUNCH_VY_MAX + 2),
        'angle':  0.0,
        'spin':   random.uniform(-0.04, 0.04),
        'fuse_t': 0,
        'sliced': False,
    }


def make_halves(fruit):
    _, color, highlight, radius, _ = fruit['type']
    cut = random.uniform(0, math.pi)
    halves = []
    for side in (0, 1):
        sign = -1 if side == 0 else 1
        halves.append({
            'x':        float(fruit['x']),
            'y':        float(fruit['y']),
            'vx':       sign * random.uniform(2, 6),
            'vy':       random.uniform(-5, -1),
            'color':    color,
            'highlight': highlight,
            'radius':   radius,
            'cut':      cut,
            'angle':    fruit['angle'],
            'spin':     sign * random.uniform(0.06, 0.15),
            't':        55,
            'max_t':    55,
            'side':     side,
        })
    return halves


def make_particles(x, y, color):
    parts = []
    for _ in range(PARTICLE_COUNT):
        a = random.uniform(0, math.pi * 2)
        s = random.uniform(3, 13)
        parts.append({
            'x':     float(x),
            'y':     float(y),
            'vx':    math.cos(a) * s,
            'vy':    math.sin(a) * s - 2,
            'color': color,
            't':     random.randint(18, 42),
            'max_t': 42,
            'r':     random.randint(3, 9),
        })
    return parts


def make_popup(x, y, text, color):
    return {'x': float(x), 'y': float(y), 'text': text,
            'color': color, 't': 52, 'max_t': 52}


# ─────────────────────────────────────────────────────────────────────────────
# Geometry
# ─────────────────────────────────────────────────────────────────────────────

def segment_hits_circle(p1, p2, cx, cy, r):
    """True if line segment p1→p2 intersects circle centred at (cx, cy) radius r."""
    dx, dy = p2[0] - p1[0], p2[1] - p1[1]
    fx, fy = p1[0] - cx,    p1[1] - cy
    a = dx*dx + dy*dy
    if a < 1e-9:
        return math.hypot(fx, fy) <= r
    b    = 2 * (fx*dx + fy*dy)
    c    = fx*fx + fy*fy - r*r
    disc = b*b - 4*a*c
    if disc < 0:
        return False
    sq = math.sqrt(disc)
    t1 = (-b - sq) / (2*a)
    t2 = (-b + sq) / (2*a)
    return (0 <= t1 <= 1) or (0 <= t2 <= 1)


# ─────────────────────────────────────────────────────────────────────────────
# Drawing
# ─────────────────────────────────────────────────────────────────────────────

def draw_background(surf, frame):
    surf.fill((8, 5, 20))


def draw_fruit(surf, fruit):
    name, color, highlight, radius, _ = fruit['type']
    x, y = int(fruit['x']), int(fruit['y'])
    a    = fruit['angle']

    pygame.draw.circle(surf, color, (x, y), radius)
    # Shine
    hx = x + int(math.cos(a - 0.5) * radius * 0.38)
    hy = y + int(math.sin(a - 0.5) * radius * 0.38)
    pygame.draw.circle(surf, highlight, (hx, hy), max(4, radius // 3))
    # Type-specific markings
    if name == "watermelon":
        for i in range(4):
            ang = a + i * math.pi / 2
            pygame.draw.line(surf, (20, 110, 20),
                             (x + int(math.cos(ang)*radius*0.25), y + int(math.sin(ang)*radius*0.25)),
                             (x + int(math.cos(ang)*radius*0.88), y + int(math.sin(ang)*radius*0.88)), 2)
    elif name == "orange":
        for i in range(6):
            ang = a + i * math.pi / 3
            pygame.draw.line(surf, (190, 95, 0), (x, y),
                             (x + int(math.cos(ang)*radius*0.88), y + int(math.sin(ang)*radius*0.88)), 1)
    elif name == "lemon":
        for sign in (-1, 1):
            tip = (x + int(math.cos(a) * (radius+5) * sign),
                   y + int(math.sin(a) * (radius+5) * sign))
            pygame.draw.circle(surf, color, tip, 5)


def draw_half(surf, half):
    ratio     = half['t'] / half['max_t']
    color     = tuple(int(c * ratio) for c in half['color'])
    highlight = tuple(int(c * ratio) for c in half['highlight'])
    radius    = half['radius']
    x, y      = int(half['x']), int(half['y'])
    cut       = half['cut']
    sign      = -1 if half['side'] == 0 else 1
    ox = int(math.cos(cut + math.pi/2) * sign * radius * 0.2)
    oy = int(math.sin(cut + math.pi/2) * sign * radius * 0.2)
    pygame.draw.circle(surf, color, (x + ox, y + oy), radius)
    # Flat cut line
    pygame.draw.line(surf, highlight,
                     (x + int(math.cos(cut)*radius), y + int(math.sin(cut)*radius)),
                     (x - int(math.cos(cut)*radius), y - int(math.sin(cut)*radius)), 3)


def draw_bomb(surf, bomb):
    x, y = int(bomb['x']), int(bomb['y'])
    pygame.draw.circle(surf, (30, 30, 30),  (x, y), 28)
    pygame.draw.circle(surf, (70, 70, 70),  (x, y), 28, 2)
    pygame.draw.circle(surf, (80, 80, 80),  (x - 8, y - 8), 7)
    # Fuse
    ft  = bomb['fuse_t']
    flen = 12 + int(math.sin(ft * 0.3) * 4)
    fa   = bomb['angle'] + 1.2
    fx   = x + int(math.cos(fa) * 28)
    fy   = y + int(math.sin(fa) * 28)
    pygame.draw.line(surf, (160, 120, 40), (fx, fy), (fx, fy - flen), 3)
    if (ft // 4) % 2 == 0:
        pygame.draw.circle(surf, (255, 200, 50), (fx, fy - flen), 4)


def draw_blade(surf, trail, color):
    n = len(trail)
    if n < 2:
        return
    for i in range(n - 1):
        ratio = (i + 1) / n
        r = min(255, int(color[0]*ratio + 255*(1-ratio)*0.25))
        g = min(255, int(color[1]*ratio + 200*(1-ratio)*0.25))
        b = min(255, int(color[2]*ratio + 255*(1-ratio)*0.25))
        w = max(2, int(16 * ratio))
        pygame.draw.line(surf, (r, g, b),
                         (int(trail[i][0]),   int(trail[i][1])),
                         (int(trail[i+1][0]), int(trail[i+1][1])), w)
    tip = trail[-1]
    pygame.draw.circle(surf, color, (int(tip[0]), int(tip[1])), 9)


def draw_particles(surf, particles):
    for p in particles:
        ratio  = p['t'] / p['max_t']
        color  = tuple(int(c * ratio) for c in p['color'])
        radius = max(1, int(p['r'] * ratio))
        pygame.draw.circle(surf, color, (int(p['x']), int(p['y'])), radius)


def draw_popups(surf, popups, font):
    for p in popups:
        ratio = p['t'] / p['max_t']
        color = tuple(min(255, int(c * ratio)) for c in p['color'])
        txt   = font.render(p['text'], True, color)
        surf.blit(txt, (int(p['x']) - txt.get_width()//2, int(p['y'])))


def draw_lives(surf, lives_left, font):
    for i in range(LIVES):
        color = (215, 40, 40) if i < lives_left else (55, 55, 75)
        pygame.draw.circle(surf, color, (WINDOW_WIDTH - 38 - i*44, 34), 15)
        if i < lives_left:
            pygame.draw.circle(surf, (255, 140, 140), (WINDOW_WIDTH - 44 - i*44, 27), 6)


def draw_hud(surf, score, high_score, lives_left, combo_count, t_left, hud_font, big_font):
    _text(surf, f"SCORE  {score}",      20, 14, WHITE,             hud_font)
    _text(surf, f"BEST   {high_score}", 20, 44, (180, 180, 200),   hud_font)
    if combo_count >= 2:
        label = f"COMBO  x{combo_count}!"
        col   = (255, 220, 50) if combo_count < 4 else (255, 100, 50)
        cx    = WINDOW_WIDTH // 2
        _text(surf, label, cx - big_font.size(label)[0]//2, 16, col, big_font)
    # Timer (top-right)
    if t_left <= 10 and (pygame.time.get_ticks() // 500) % 2 == 0:
        timer_col = (255, 50, 50)
    else:
        timer_col = WHITE
    _text(surf, f"TIME  {t_left:02d}", WINDOW_WIDTH - 160, 14, timer_col, hud_font)


def draw_game_over(surf, score, high_score, new_record, big_font, hud_font):
    overlay = pygame.Surface((WINDOW_WIDTH, WINDOW_HEIGHT), pygame.SRCALPHA)
    overlay.fill((0, 0, 0, 165))
    surf.blit(overlay, (0, 0))
    cx = WINDOW_WIDTH // 2
    cy = WINDOW_HEIGHT // 2
    _centered(surf, "GAME  OVER",         cx, cy - 130, (220,  50,  50), big_font)
    _centered(surf, f"SCORE :  {score}",  cx, cy -  40, WHITE,           hud_font)
    if new_record:
        _centered(surf, "NEW  RECORD!",   cx, cy +  10, (255, 215,  50), hud_font)
    else:
        _centered(surf, f"BEST :  {high_score}", cx, cy + 10, (140, 140, 160), hud_font)
    _centered(surf, "Wave  to  play  again", cx, cy + 72, (80, 210, 255), hud_font)


def draw_waiting(surf, big_font, hud_font, frame):
    cx, cy = WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2
    pulse  = 0.5 + 0.5 * math.sin(frame * 0.06)
    col    = (int(80 + 175*pulse), int(180 + 60*pulse), 255)
    _centered(surf, "FRUIT  NINJA",                          cx, cy - 80, (255, 210, 50), big_font)
    _centered(surf, "Stand  in  front  of  the  camera",     cx, cy +  8, col,            hud_font)
    _centered(surf, "Wave  your  hands  to  slice  fruits!", cx, cy + 44, DIMWHITE,       hud_font)


def draw_win_screen(surf, score, high_score, face_surfs, frame, big_font, hud_font):
    """Full-screen win overlay shown when the 60-second timer runs out."""
    overlay = pygame.Surface((WINDOW_WIDTH, WINDOW_HEIGHT), pygame.SRCALPHA)
    overlay.fill((0, 0, 0, 210))
    surf.blit(overlay, (0, 0))
    cx = WINDOW_WIDTH // 2
    cy = WINDOW_HEIGHT // 2
    _centered(surf, "TIME'S  UP!",              cx, cy - 220, (255, 210,  50), big_font)
    _centered(surf, f"FINAL  SCORE  :  {score}", cx, cy - 130, WHITE,           hud_font)
    if score > high_score:
        _centered(surf, "NEW  RECORD!",          cx, cy -  90, (255, 215,   0), hud_font)
    else:
        _centered(surf, f"BEST  :  {high_score}", cx, cy - 90, (140, 140, 160), hud_font)
    # Face snapshots captured at win moment (no tracking lines — raw crop)
    face_size   = 140
    active_faces = [(pi, fs) for pi, fs in enumerate(face_surfs) if fs is not None]
    if active_faces:
        total    = len(active_faces)
        spacing  = 190
        start_x  = cx - spacing * (total - 1) // 2
        for idx, (pi, fs) in enumerate(active_faces):
            fx     = start_x + idx * spacing
            scaled = pygame.transform.scale(fs, (face_size, face_size))
            col    = BLADE_COLORS[pi % len(BLADE_COLORS)]
            pygame.draw.rect(surf, col,
                             (fx - face_size // 2 - 4, cy - 50, face_size + 8, face_size + 8), 3)
            surf.blit(scaled, (fx - face_size // 2, cy - 46))
    _centered(surf, "Press  R  to  play  again", cx, cy + 130, (80, 210, 255), hud_font)


def _text(surf, text, x, y, color, font):
    surf.blit(font.render(text, True, color), (x, y))


def _centered(surf, text, cx, y, color, font):
    rendered = font.render(text, True, color)
    surf.blit(rendered, (cx - rendered.get_width()//2, y))


# ─────────────────────────────────────────────────────────────────────────────
# Game state
# ─────────────────────────────────────────────────────────────────────────────

def new_game():
    return {
        'state':       STATE_WAITING,
        'score':       0,
        'lives':       LIVES,
        'fruits':      [],
        'bombs':       [],
        'halves':      [],
        'particles':   [],
        'popups':      [],
        'combo_count': 0,
        'combo_timer': 0,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main loop
# ─────────────────────────────────────────────────────────────────────────────

def main():
    global WINDOW_WIDTH, WINDOW_HEIGHT

    pygame.init()
    pygame.mixer.init(frequency=SAMPLE_RATE, size=-16, channels=2, buffer=512)

    # Auto-detect display resolution and go fullscreen
    info = pygame.display.Info()
    WINDOW_WIDTH  = info.current_w
    WINDOW_HEIGHT = info.current_h

    screen    = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT), pygame.FULLSCREEN)
    pygame.display.set_caption("Fruit Ninja — Hailo Edition")
    clock     = pygame.time.Clock()
    game_surf = pygame.Surface((WINDOW_WIDTH, WINDOW_HEIGHT))

    big_font = pygame.font.Font(None, 82)
    hud_font = pygame.font.Font(None, 36)
    pop_font = pygame.font.Font(None, 44)

    sounds     = build_sounds()
    start_bgm()
    high_score = load_high_score()

    # Wrist blade trails — [player][wrist: 0=left, 1=right]
    trails = [[deque(maxlen=TRAIL_LEN) for _ in range(2)]
              for _ in range(MAX_PLAYERS)]

    # Bomb timeout counters per player (frames remaining)
    bomb_timeouts = [0] * MAX_PLAYERS

    # ── Camera + pose pipeline startup ────────────────────────────────────────
    user_data = FruitNinjaUserData()

    if HAILO_AVAILABLE:
        # Check device node exists before trying to build the pipeline
        if CAMERA_DEVICE and not os.path.exists(CAMERA_DEVICE):
            print(f"[ERROR] Camera device not found: {CAMERA_DEVICE}")
            print( "        Check: ls /dev/video*")
            print( "[INFO]  Continuing in mouse fallback mode.")
        else:
            # 1. GLib main loop MUST be running before app.run() so
            #    GStreamer bus messages are dispatched properly
            glib_thread = threading.Thread(
                target=_start_glib_loop, daemon=True, name='GLib-Loop'
            )
            glib_thread.start()
            time.sleep(0.1)  # let the loop spin up

            # 2. Build and run the pose pipeline in its own thread
            def _run_pose():
                # Suppress signal.signal() calls (require main thread)
                import signal as _sig
                _sig.signal = lambda *a, **kw: None

                # Subclass that routes video to fakesink so no separate window opens
                _fq = user_data.pip_queue  # closure ref for probe

                class _PoseApp(GStreamerPoseEstimationApp):
                    def get_pipeline_string(self):
                        self.video_sink = 'fakesink'
                        return super().get_pipeline_string()

                    def run(self):
                        _hcq = user_data.head_crop_queues
                        _hbl = user_data._lock
                        _hbb = user_data.head_bboxes

                        # Identity probe — raw frame (no skeleton) → face crops only
                        identity = self.pipeline.get_by_name("identity_callback")
                        if identity:
                            id_pad = identity.get_static_pad("src")
                            if id_pad:
                                def _identity_face_probe(pad, info, _):
                                    buf = info.get_buffer()
                                    if buf is None:
                                        return Gst.PadProbeReturn.OK
                                    fmt, w, h = get_caps_from_pad(pad)
                                    if fmt is None:
                                        return Gst.PadProbeReturn.OK
                                    f = get_numpy_from_buffer(buf, fmt, w, h)
                                    if f is None:
                                        return Gst.PadProbeReturn.OK
                                    with _hbl:
                                        bboxes = list(_hbb)
                                    for i, bb in enumerate(bboxes):
                                        if bb is None:
                                            continue
                                        x1, y1, x2, y2 = bb
                                        px1 = int(x1 * w); py1 = int(y1 * h)
                                        px2 = int(x2 * w); py2 = int(y2 * h)
                                        if px2 > px1 and py2 > py1:
                                            crop = f[py1:py2, px1:px2].copy()
                                            try:
                                                while not _hcq[i].empty():
                                                    _hcq[i].get_nowait()
                                                _hcq[i].put_nowait(crop)
                                            except queue.Full:
                                                pass
                                    return Gst.PadProbeReturn.OK
                                id_pad.add_probe(Gst.PadProbeType.BUFFER, _identity_face_probe, None)

                        # Overlay probe — post-skeleton frame → PiP only
                        overlay = self.pipeline.get_by_name("hailo_display_overlay")
                        if overlay:
                            ov_pad = overlay.get_static_pad("src")
                            if ov_pad:
                                def _overlay_pip_probe(pad, info, _):
                                    buf = info.get_buffer()
                                    if buf is None:
                                        return Gst.PadProbeReturn.OK
                                    fmt, w, h = get_caps_from_pad(pad)
                                    if fmt is None:
                                        return Gst.PadProbeReturn.OK
                                    f = get_numpy_from_buffer(buf, fmt, w, h)
                                    if f is None:
                                        return Gst.PadProbeReturn.OK
                                    try:
                                        while not _fq.empty():
                                            _fq.get_nowait()
                                        _fq.put_nowait(f.copy())
                                    except queue.Full:
                                        pass
                                    return Gst.PadProbeReturn.OK
                                ov_pad.add_probe(Gst.PadProbeType.BUFFER, _overlay_pip_probe, None)
                        super().run()

                try:
                    cb  = FruitNinjaCallback(user_data)
                    app = _PoseApp(cb, user_data)
                    app.run()
                except Exception as exc:
                    print(f"[ERROR] Pose pipeline crashed: {exc}")
                    print( "[INFO]  Continuing in mouse fallback mode.")

            pose_thread = threading.Thread(
                target=_run_pose, daemon=True, name='Pose-Pipeline'
            )
            pose_thread.start()
            print("[INFO] Camera pipeline starting…  "
                  "(first frame may take 2–3 s)")
    else:
        print("[INFO] Mouse fallback active — hold left mouse button and drag to slice.")

    g                = new_game()
    frame            = 0
    GAME_DURATION    = 60                # 1-minute game
    game_start_time  = None             # set when gameplay begins
    winner_face_surfs = [None] * MAX_PLAYERS  # face snapshot at win moment
    _pip_surf  = None                    # cached PiP surface
    _face_surfs = [None] * MAX_PLAYERS   # cached per-player face thumbnails
    _player_labels = [
        hud_font.render(f"P{pi + 1}", True, BLADE_COLORS[pi % len(BLADE_COLORS)])
        for pi in range(MAX_PLAYERS)
    ]

    while True:
        # ── Events ────────────────────────────────────────────────────────────
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                save_high_score(high_score)
                pygame.mixer.stop()
                pygame.quit()
                return
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                save_high_score(high_score)
                pygame.mixer.stop()
                pygame.quit()
                return
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                g                 = new_game()
                high_score        = load_high_score()
                game_start_time   = None
                winner_face_surfs = [None] * MAX_PLAYERS

        frame += 1

        # ── Timer ─────────────────────────────────────────────────────────────
        t_left = (max(0, int(GAME_DURATION - (time.time() - game_start_time)))
                  if game_start_time else GAME_DURATION)

        # ── Mouse fallback — left-click and drag to slice ─────────────────────
        if pygame.mouse.get_pressed()[0]:
            mx, my = pygame.mouse.get_pos()
            q = user_data.wrist_queues[0][0]
            try:
                while not q.empty():
                    q.get_nowait()
                q.put_nowait((mx, my))
                user_data.set_active(0, True)
            except queue.Full:
                pass

        # ── Read wrist positions ──────────────────────────────────────────────
        active = user_data.active
        for pi in range(MAX_PLAYERS):
            for wi in range(2):
                try:
                    pos = user_data.wrist_queues[pi][wi].get_nowait()
                    trails[pi][wi].append(pos)
                except queue.Empty:
                    pass

        # ── State machine ─────────────────────────────────────────────────────

        if g['state'] == STATE_WAITING:
            if any(active):
                g['state']      = STATE_PLAYING
                game_start_time = time.time()

        elif g['state'] == STATE_PLAYING:

            # ── Timer expiry → win ────────────────────────────────────────────
            if t_left == 0:
                g['state'] = STATE_WIN
                winner_face_surfs = list(_face_surfs)
                sounds['game_over'].play()
                if g['score'] > high_score:
                    high_score = g['score']
                    save_high_score(high_score)

            # ── Spawn ─────────────────────────────────────────────────────────
            if random.random() < FRUIT_SPAWN_CHANCE:
                g['fruits'].append(make_fruit())
            if random.random() < BOMB_SPAWN_CHANCE:
                g['bombs'].append(make_bomb())

            # ── Physics ───────────────────────────────────────────────────────
            for fruit in g['fruits'][:]:
                fruit['x']     += fruit['vx']
                fruit['y']     += fruit['vy']
                fruit['vy']    += GRAVITY
                fruit['angle'] += fruit['spin']
                too_low = fruit['y'] > WINDOW_HEIGHT + fruit['type'][3] + 20
                if too_low:
                    g['fruits'].remove(fruit)
                    if not fruit['sliced']:
                        sounds['miss'].play()

            for bomb in g['bombs'][:]:
                bomb['x']      += bomb['vx']
                bomb['y']      += bomb['vy']
                bomb['vy']     += GRAVITY
                bomb['angle']  += bomb['spin']
                bomb['fuse_t'] += 1
                if bomb['y'] > WINDOW_HEIGHT + 40:
                    g['bombs'].remove(bomb)

            for h in g['halves'][:]:
                h['x'] += h['vx'];  h['y'] += h['vy']
                h['vy'] += GRAVITY * 0.4
                h['angle'] += h['spin']
                h['t'] -= 1
                if h['t'] <= 0 or h['y'] > WINDOW_HEIGHT + 60:
                    g['halves'].remove(h)

            for p in g['particles'][:]:
                p['x'] += p['vx'];  p['y'] += p['vy']
                p['vy'] += GRAVITY * 0.25
                p['t']  -= 1
                if p['t'] <= 0:
                    g['particles'].remove(p)

            for pop in g['popups'][:]:
                pop['y'] -= 1.6
                pop['t'] -= 1
                if pop['t'] <= 0:
                    g['popups'].remove(pop)

            if g['combo_timer'] > 0:
                g['combo_timer'] -= 1
            else:
                g['combo_count'] = 0

            # Decrement bomb timeouts
            for pi in range(MAX_PLAYERS):
                if bomb_timeouts[pi] > 0:
                    bomb_timeouts[pi] -= 1

            # ── Slice detection ───────────────────────────────────────────────
            if g['state'] == STATE_PLAYING:   # re-check — bomb may have ended game
                for pi in range(MAX_PLAYERS):
                    if not active[pi]:
                        continue
                    if bomb_timeouts[pi] > 0:
                        continue   # player is frozen after hitting a bomb
                    blade_col = BLADE_COLORS[pi % len(BLADE_COLORS)]
                    for wi in range(2):
                        trail = trails[pi][wi]
                        if len(trail) < 2:
                            continue
                        p1, p2 = trail[-2], trail[-1]
                        if math.hypot(p2[0]-p1[0], p2[1]-p1[1]) < SWIPE_MIN_DIST:
                            continue

                        # Fruits
                        for fruit in g['fruits'][:]:
                            if fruit['sliced']:
                                continue
                            r = fruit['type'][3] * SLICE_RADIUS_MULT
                            if segment_hits_circle(p1, p2, fruit['x'], fruit['y'], r):
                                fruit['sliced']    = True
                                g['combo_timer']   = COMBO_WINDOW
                                g['combo_count']  += 1
                                pts  = fruit['type'][4]
                                pts += COMBO_BONUS * max(0, g['combo_count'] - 1)
                                g['score']        += pts
                                g['halves']       += make_halves(fruit)
                                g['particles']    += make_particles(
                                    fruit['x'], fruit['y'], fruit['type'][1])
                                label = (f"+{pts}" if g['combo_count'] < 2
                                         else f"COMBO x{g['combo_count']}!  +{pts}")
                                g['popups'].append(
                                    make_popup(fruit['x'], fruit['y']-22, label, blade_col))
                                sounds['slice'].play()
                                sounds['splat'].play()
                                if g['score'] > high_score:
                                    high_score = g['score']
                                    save_high_score(high_score)

                        # Bombs — flash and vanish; player frozen for 3s
                        for bomb in g['bombs'][:]:
                            if bomb['sliced']:
                                continue
                            if segment_hits_circle(p1, p2, bomb['x'], bomb['y'],
                                                   28 * SLICE_RADIUS_MULT):
                                bomb['sliced'] = True
                                g['bombs'].remove(bomb)
                                g['particles'] += make_particles(
                                    bomb['x'], bomb['y'], (200, 100, 30))
                                bomb_timeouts[pi] = BOMB_TIMEOUT_FRAMES
                                g['popups'].append(
                                    make_popup(bomb['x'], bomb['y']-22,
                                               f"P{pi+1} TIMEOUT!", (255, 80, 30)))
                                sounds['bomb'].play()
                                break

        # ─────────────────────────────────────────────────────────────────────
        # DRAW
        # ─────────────────────────────────────────────────────────────────────
        draw_background(game_surf, frame)

        if g['state'] == STATE_WAITING:
            draw_waiting(game_surf, big_font, hud_font, frame)

        else:
            for fruit in g['fruits']:
                if not fruit['sliced']:
                    draw_fruit(game_surf, fruit)

            for bomb in g['bombs']:
                if not bomb['sliced']:
                    draw_bomb(game_surf, bomb)

            for h in g['halves']:
                draw_half(game_surf, h)

            draw_particles(game_surf, g['particles'])

            for pi in range(MAX_PLAYERS):
                if active[pi]:
                    bc = BLADE_COLORS[pi % len(BLADE_COLORS)]
                    for wi in range(2):
                        draw_blade(game_surf, trails[pi][wi], bc)

            draw_popups(game_surf, g['popups'], pop_font)
            draw_hud(game_surf, g['score'], high_score,
                     0, g['combo_count'], t_left, hud_font, big_font)

        # ── Player legend — face thumbnails with colour border ────────────────
        for pi in range(MAX_PLAYERS):
            try:
                crop = user_data.head_crop_queues[pi].get_nowait()
                h_c, w_c = crop.shape[:2]
                raw = pygame.image.frombuffer(
                    np.ascontiguousarray(crop).tobytes(), (w_c, h_c), 'RGB')
                _face_surfs[pi] = pygame.transform.scale(raw, (LEGEND_FACE, LEGEND_FACE))
            except queue.Empty:
                pass
            if _face_surfs[pi] is not None:
                col  = BLADE_COLORS[pi % len(BLADE_COLORS)]
                lx   = PIP_MARGIN
                ly   = WINDOW_HEIGHT - (MAX_PLAYERS - pi) * (LEGEND_FACE + 30) - PIP_MARGIN
                pygame.draw.rect(game_surf, col,
                                 (lx - 3, ly - 3, LEGEND_FACE + 6, LEGEND_FACE + 6), 3)
                game_surf.blit(_face_surfs[pi], (lx, ly))
                lbl = _player_labels[pi]
                game_surf.blit(lbl, (lx + LEGEND_FACE // 2 - lbl.get_width() // 2,
                                     ly + LEGEND_FACE + 4))

        # ── Picture-in-picture: pose estimation feed (bottom-right corner) ───
        try:
            cam_frame = user_data.pip_queue.get_nowait()
            h_p, w_p = cam_frame.shape[:2]
            raw = pygame.image.frombuffer(
                np.ascontiguousarray(cam_frame).tobytes(), (w_p, h_p), 'RGB')
            _pip_surf = pygame.transform.scale(raw, (PIP_W, PIP_H))
        except queue.Empty:
            pass
        if _pip_surf is not None:
            pip_x = WINDOW_WIDTH  - PIP_W - PIP_MARGIN
            pip_y = WINDOW_HEIGHT - PIP_H - PIP_MARGIN
            pygame.draw.rect(game_surf, (255, 255, 255),
                             (pip_x - 2, pip_y - 2, PIP_W + 4, PIP_H + 4), 2)
            game_surf.blit(_pip_surf, (pip_x, pip_y))

        # ── Win screen overlay ─────────────────────────────────────────────
        if g['state'] == STATE_WIN:
            draw_win_screen(game_surf, g['score'], high_score,
                            winner_face_surfs, frame, big_font, hud_font)

        screen.fill(BLACK)
        screen.blit(game_surf, (0, 0))
        pygame.display.flip()
        clock.tick(FPS)


if __name__ == "__main__":
    from pathlib import Path
    import os
    project_root = Path(__file__).resolve().parent.parent
    env_file     = project_root / ".env"
    os.environ["HAILO_ENV_FILE"] = str(env_file)
    main()
