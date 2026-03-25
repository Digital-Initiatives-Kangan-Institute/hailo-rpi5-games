# =============================================================================
#  Hailo 3D Space Invaders — Complete Version
#  Raspberry Pi 5 + Hailo AI HAT+ | Pose Estimation Multiplayer
#  Kangan Digital Initiative
# =============================================================================

import threading
import queue
import argparse
from collections import deque
import pygame
import pygame.sndarray
import random
import math
import os
import json
import numpy as np
import time
import sys

# =============================================================================
# ARGUMENT PARSING
# Normalise --input before hailo_apps_infra reads sys.argv
# =============================================================================
def _parse_and_normalise_input():
    """
    Parse --input from the command line and normalise it so that:
      /dev/video0  →  usb
      /dev/video1  →  usb  (with device override stored separately)
      usb          →  usb
      rpi          →  rpi
    Returns the normalised input string and the raw device path (or None).
    """
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--input', '-i', default='usb')
    args, _ = parser.parse_known_args()

    raw = args.input.strip()
    device_path = None

    if raw.startswith('/dev/video'):
        device_path = raw
        # Rewrite sys.argv so hailo_apps_infra sees the expected keyword
        for idx, val in enumerate(sys.argv):
            if val == raw:
                sys.argv[idx] = 'usb'
                break
        return 'usb', device_path

    # Ensure --input is explicitly in sys.argv so GStreamerPoseEstimationApp picks it up
    if '--input' not in sys.argv and '-i' not in sys.argv:
        sys.argv += ['--input', raw]

    return raw, device_path

INPUT_SOURCE, CAMERA_DEVICE = _parse_and_normalise_input()

# =============================================================================
# HAILO / GSTREAMER — optional import
# If the Hailo environment is not active the game falls back to
# keyboard + mouse control automatically.
# =============================================================================
HAILO_AVAILABLE = False
_glib_loop       = None   # kept alive so GStreamer events are processed

try:
    import gi
    gi.require_version('Gst', '1.0')
    from gi.repository import Gst, GLib

    # Initialise GStreamer once, here, before anything else uses it
    Gst.init(None)

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
    print("[INFO] Running in KEYBOARD / MOUSE fallback mode.")
    print("[INFO] To enable Hailo:  source ~/hailo-rpi5-examples/setup_env.sh")

    # ------------------------------------------------------------------
    # Minimal stub classes so the rest of the file parses without error
    # ------------------------------------------------------------------
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
    GStreamer needs a running GLib main loop to dispatch pipeline bus
    messages (EOS, errors, state changes).  Without it the camera
    pipeline builds successfully but never delivers frames.
    """
    global _glib_loop
    if not HAILO_AVAILABLE:
        return
    _glib_loop = GLib.MainLoop()
    try:
        _glib_loop.run()
    except Exception as exc:
        print(f"[WARN] GLib main loop exited: {exc}")


def _check_camera_device(device: str) -> bool:
    """Return True if the V4L2 device node exists and is accessible."""
    if device and not os.path.exists(device):
        print(f"[ERROR] Camera device not found: {device}")
        print( "        Check: ls /dev/video*")
        return False
    return True

# =============================================================================
# CONSTANTS
# =============================================================================

# --- Window ---
WINDOW_WIDTH  = 1280
WINDOW_HEIGHT = 720
FPS           = 60

# Picture-in-picture camera overlay
PIP_W      = 320
PIP_H      = 180
PIP_MARGIN = 12

# Player legend face thumbnails
LEGEND_FACE = 80

# --- Players ---
MAX_PLAYERS      = 2
SMOOTHING_WINDOW = 10       # frames of head-position smoothing

# --- Ship ---
SHIP_WIDTH  = 60
SHIP_HEIGHT = 40

# --- Bullets ---
BULLET_SPEED         = 12
BULLET_RADIUS        = 4
SHOOT_INTERVAL_FRAMES = 20
BULLET_TRAIL_LEN     = 6

# --- Enemies ---
ENEMY_WIDTH  = 48
ENEMY_HEIGHT = 40
ENEMY_SPEED  = 1.5
ENEMY_SPAWN_CHANCE = 0.02
ENEMY_DRIFT_SPEED  = 1.5

# Enemy types: (speed_mult, points, ring_color, spawn_weight)
ENEMY_TYPES = [
    (1.0,  10, None,            60),   # normal
    (2.0,  20, (255,  80,  80), 25),   # fast  — red ring
    (0.5,  30, ( 80,  80, 255), 15),   # tank  — blue ring
]
_ENEMY_TYPE_WEIGHTS = [t[3] for t in ENEMY_TYPES]

# --- Round ---
ROUND_TIME           = 60   # seconds per round

# --- Banner / Fireworks ---
FIREWORK_CHANCE      = 0.15
BANNER_SPEED         = 5
BANNER_WOBBLE_AMP    = 20
BANNER_WAVE_AMP      = 5
INITIAL_BANNER_TIMER = 120
FIREWORK_DURATION    = 20

# --- Visuals ---
SHAKE_FRAMES       = 18
SHAKE_AMPLITUDE    = 8
LOW_TIME_THRESHOLD = 10
GHOST_ALPHA        = 60

# --- Stars (3D background) ---
NUM_STARS    = 180
STAR_SPEED_Z = 0.015   # depth movement per frame
FOV          = 300.0

# --- Colours ---
WHITE  = (255, 255, 255)
BLACK  = (  0,   0,   0)
GREEN  = (  0, 220,   0)
BLUE   = (  0, 120, 255)
YELLOW = (255, 230,   0)
PURPLE = (180,   0, 255)
ORANGE = (255, 140,   0)
RED    = (220,  30,  30)
CYAN   = (  0, 220, 220)

PLAYER_COLORS = [GREEN, BLUE, YELLOW, PURPLE]

# --- Audio ---
SAMPLE_RATE = 44100
BGM_VOLUME  = 0.35
SFX_VOLUME  = 0.70

# Set paths to real audio files, or leave None to use generated tones
AUDIO_FILES = {
    'shoot':     None,   # e.g. 'sounds/shoot.wav'
    'explode_0': None,
    'explode_1': None,
    'explode_2': None,
    'round_end': None,
    'tick':      None,
    'bgm':       None,   # .ogg recommended for looping
}

# --- High score file ---
HIGH_SCORE_FILE = os.path.join(os.path.dirname(__file__), '.highscore.json')


# =============================================================================
# HIGH SCORE
# =============================================================================

def load_high_score():
    try:
        with open(HIGH_SCORE_FILE, 'r') as f:
            return json.load(f).get('high_score', 0)
    except Exception:
        return 0


def save_high_score(score):
    try:
        with open(HIGH_SCORE_FILE, 'w') as f:
            json.dump({'high_score': score}, f)
    except Exception:
        pass


# =============================================================================
# POSE ESTIMATION INTEGRATION
# =============================================================================

class PoseInvadersUserData(app_callback_class):
    def __init__(self):
        super().__init__()
        self.position_queues  = [queue.Queue(maxsize=5) for _ in range(MAX_PLAYERS)]
        self.active_players   = [False] * MAX_PLAYERS
        self.pip_queue        = queue.Queue(maxsize=1)
        self._bbox_lock       = threading.Lock()
        self.head_bboxes      = [None] * MAX_PLAYERS
        self.head_crop_queues = [queue.Queue(maxsize=1) for _ in range(MAX_PLAYERS)]

    def update_position(self, player_idx, x_norm):
        try:
            self.position_queues[player_idx].put_nowait(x_norm)
        except queue.Full:
            pass


class PoseInvadersCallback(app_callback_class):
    # COCO17 keypoint index — nose tracks ship position
    NOSE_IDX = 0

    def __init__(self, user_data):
        super().__init__()
        self.user_data = user_data

    def __call__(self, pad, info, u_data):
        if not HAILO_AVAILABLE:
            return Gst.PadProbeReturn.OK

        buffer = info.get_buffer()
        if buffer is None:
            return Gst.PadProbeReturn.OK
        _fmt, cam_w, cam_h = get_caps_from_pad(pad)
        if cam_w is None:
            return Gst.PadProbeReturn.OK

        roi  = hailo.get_roi_from_buffer(buffer)
        dets = roi.get_objects_typed(hailo.HAILO_DETECTION)

        active = [False] * MAX_PLAYERS
        # (cx_norm, head_bbox) tuples — sorted left-to-right for consistent player IDs
        persons = []

        for det in dets:
            if det.get_label() != "person":
                continue
            lms = det.get_objects_typed(hailo.HAILO_LANDMARKS)
            if not lms:
                continue
            pts  = lms[0].get_points()
            bbox = det.get_bbox()
            if len(pts) <= self.NOSE_IDX:
                continue
            nose    = pts[self.NOSE_IDX]
            nx      = nose.x() * bbox.width()  + bbox.xmin()
            ny      = nose.y() * bbox.height() + bbox.ymin()
            cx_norm = nx   # ship follows the nose horizontally

            # Head bbox — centred on nose
            head_bb = None
            if cam_w and cam_h:
                hr_y = 0.12
                hr_x = hr_y * (cam_h / cam_w)
                head_bb = (max(0.0, nx - hr_x), max(0.0, ny - hr_y * 1.3),
                           min(1.0, nx + hr_x), min(1.0, ny + hr_y * 0.7))
            persons.append((cx_norm, head_bb))

        # Assign left-to-right so P1 is always the left player
        persons.sort(key=lambda p: p[0])
        with self.user_data._bbox_lock:
            for idx in range(MAX_PLAYERS):
                if idx < len(persons):
                    cx_norm, head_bb = persons[idx]
                    active[idx] = True
                    self.user_data.update_position(idx, cx_norm)
                    self.user_data.head_bboxes[idx] = head_bb
                else:
                    self.user_data.head_bboxes[idx] = None

        self.user_data.active_players = active
        return Gst.PadProbeReturn.OK


# =============================================================================
# AUDIO HELPERS
# =============================================================================

def _make_tone(freq, duration_ms, volume=0.5, wave='square'):
    n   = int(SAMPLE_RATE * duration_ms / 1000)
    t   = np.linspace(0, duration_ms / 1000, n, False)
    if wave == 'square':
        sig = np.sign(np.sin(2 * np.pi * freq * t))
    elif wave == 'sine':
        sig = np.sin(2 * np.pi * freq * t)
    elif wave == 'sawtooth':
        sig = 2 * (t * freq - np.floor(0.5 + t * freq))
    elif wave == 'noise':
        sig = np.random.uniform(-1, 1, n)
    else:
        sig = np.sin(2 * np.pi * freq * t)
    fade = max(1, int(n * 0.2))
    sig[-fade:] *= np.linspace(1, 0, fade)
    sig = (sig * volume * 32767).astype(np.int16)
    return pygame.sndarray.make_sound(np.column_stack([sig, sig]))


def _make_sweep(f0, f1, duration_ms, volume=0.5, wave='sine'):
    n     = int(SAMPLE_RATE * duration_ms / 1000)
    t     = np.linspace(0, duration_ms / 1000, n, False)
    freq  = np.linspace(f0, f1, n)
    phase = np.cumsum(2 * np.pi * freq / SAMPLE_RATE)
    sig   = np.sin(phase) if wave == 'sine' else np.sign(np.sin(phase))
    fade  = max(1, int(n * 0.15))
    sig[-fade:] *= np.linspace(1, 0, fade)
    sig = (sig * volume * 32767).astype(np.int16)
    return pygame.sndarray.make_sound(np.column_stack([sig, sig]))


def _load_or_gen(key, gen_fn):
    path = AUDIO_FILES.get(key)
    if path and os.path.isfile(path):
        snd = pygame.mixer.Sound(path)
        snd.set_volume(SFX_VOLUME)
        return snd
    return gen_fn()


def build_sounds():
    return {
        'shoot':     _load_or_gen('shoot',     lambda: _make_tone(880, 55,  SFX_VOLUME * 0.6, 'square')),
        'explode_0': _load_or_gen('explode_0', lambda: _make_tone(180, 220, SFX_VOLUME * 0.8, 'noise')),
        'explode_1': _load_or_gen('explode_1', lambda: _make_tone(320, 130, SFX_VOLUME * 0.7, 'noise')),
        'explode_2': _load_or_gen('explode_2', lambda: _make_tone(90,  380, SFX_VOLUME,       'noise')),
        'round_end': _load_or_gen('round_end', lambda: _make_sweep(220, 880, 700, SFX_VOLUME * 0.9, 'sine')),
        'tick':      _load_or_gen('tick',      lambda: _make_tone(520, 80,  SFX_VOLUME * 0.5, 'square')),
    }


def start_bgm():
    path = AUDIO_FILES.get('bgm')
    if path and os.path.isfile(path):
        pygame.mixer.music.load(path)
        pygame.mixer.music.set_volume(BGM_VOLUME)
        pygame.mixer.music.play(-1)
        return
    # Synthesised drone — two detuned square waves with slow pulse
    n      = int(SAMPLE_RATE * 2.0)
    t      = np.linspace(0, 2.0, n, False)
    wave_a = np.sign(np.sin(2 * np.pi * 55.0 * t))
    wave_b = np.sign(np.sin(2 * np.pi * 55.7 * t))
    pulse  = 0.5 + 0.5 * np.sin(2 * np.pi * 0.5 * t)
    sig    = ((wave_a + wave_b) * 0.5 * pulse * BGM_VOLUME * 32767).astype(np.int16)
    bgm    = pygame.sndarray.make_sound(np.column_stack([sig, sig]))
    bgm.play(loops=-1)


# =============================================================================
# STARFIELD (3D depth scroll)
# =============================================================================

def generate_stars_3d():
    cx, cy = WINDOW_WIDTH / 2, WINDOW_HEIGHT / 2
    stars  = []
    for _ in range(NUM_STARS):
        x = random.uniform(-cx, cx)
        y = random.uniform(-cy, cy)
        z = random.uniform(0.1, 1.0)
        stars.append([x, y, z])
    return stars


def draw_background_3d(surf, stars):
    surf.fill((5, 5, 18))
    cx, cy = WINDOW_WIDTH / 2, WINDOW_HEIGHT / 2
    for star in stars:
        # Move toward viewer
        star[2] -= STAR_SPEED_Z
        if star[2] <= 0:
            star[0] = random.uniform(-cx, cx)
            star[1] = random.uniform(-cy, cy)
            star[2] = 1.0

        sx = int(cx + star[0] / star[2])
        sy = int(cy + star[1] / star[2])
        if 0 <= sx < WINDOW_WIDTH and 0 <= sy < WINDOW_HEIGHT:
            brightness = int(220 * (1 - star[2]))
            size       = max(1, int(3 * (1 - star[2])))
            color      = (brightness, brightness, min(255, brightness + 40))
            if size == 1:
                surf.set_at((sx, sy), color)
            else:
                pygame.draw.circle(surf, color, (sx, sy), size)


# =============================================================================
# DRAW HELPERS
# =============================================================================

_font_cache = {}
def draw_text(surf, text, x, y, color=WHITE, size=28):
    if size not in _font_cache:
        _font_cache[size] = pygame.font.Font(None, size)
    s = _font_cache[size].render(text, True, color)
    surf.blit(s, (x, y))


def draw_ship(surf, x, y, frame, color):
    """Draw player spaceship with animated engine glow."""
    cx   = x + SHIP_WIDTH  // 2
    by_  = y + SHIP_HEIGHT

    # Engine exhaust glow (animated)
    glow_size = 6 + int(4 * math.sin(frame * 0.3))
    glow_surf = pygame.Surface((glow_size * 2, glow_size * 2), pygame.SRCALPHA)
    pygame.draw.circle(glow_surf, (*ORANGE, 120), (glow_size, glow_size), glow_size)
    surf.blit(glow_surf, (cx - glow_size, by_ - glow_size // 2))

    # Exhaust flame
    flame_h = 10 + int(5 * math.sin(frame * 0.4))
    pygame.draw.polygon(surf, ORANGE,
                        [(cx, by_ + flame_h), (cx - 6, by_), (cx + 6, by_)])
    pygame.draw.polygon(surf, YELLOW,
                        [(cx, by_ + flame_h - 3), (cx - 3, by_), (cx + 3, by_)])

    # Main hull — triangle
    tip = (cx, y)
    bl  = (x, by_)
    br  = (x + SHIP_WIDTH, by_)
    pygame.draw.polygon(surf, color, [tip, bl, br])
    pygame.draw.polygon(surf, WHITE,  [tip, bl, br], 2)

    # Cockpit dome
    dome_r = SHIP_WIDTH // 6
    pygame.draw.circle(surf, WHITE, (cx, y + SHIP_HEIGHT // 2), dome_r)
    pygame.draw.circle(surf, color, (cx, y + SHIP_HEIGHT // 2), dome_r - 2)

    # Wing accents
    wing_col = tuple(min(255, c + 60) for c in color)
    pygame.draw.line(surf, wing_col, (cx, y + SHIP_HEIGHT // 2), bl, 2)
    pygame.draw.line(surf, wing_col, (cx, y + SHIP_HEIGHT // 2), br, 2)


def draw_enemy(surf, x, y):
    """Draw a classic alien invader sprite."""
    cx = int(x + ENEMY_WIDTH  // 2)
    cy = int(y + ENEMY_HEIGHT // 2)

    # Body (hexagon-ish)
    body_pts = [
        (cx - 10, y),
        (cx + 10, y),
        (cx + ENEMY_WIDTH // 2, cy),
        (cx + 10, y + ENEMY_HEIGHT),
        (cx - 10, y + ENEMY_HEIGHT),
        (cx - ENEMY_WIDTH // 2, cy),
    ]
    pygame.draw.polygon(surf, (0, 200, 80), body_pts)
    pygame.draw.polygon(surf, (0, 255, 120), body_pts, 2)

    # Eyes
    pygame.draw.circle(surf, RED, (cx - 8, cy - 4), 5)
    pygame.draw.circle(surf, RED, (cx + 8, cy - 4), 5)
    pygame.draw.circle(surf, (255, 100, 100), (cx - 8, cy - 4), 2)
    pygame.draw.circle(surf, (255, 100, 100), (cx + 8, cy - 4), 2)

    # Antennae
    pygame.draw.line(surf, (0, 200, 80), (cx - 8, y),  (cx - 14, y - 10), 2)
    pygame.draw.line(surf, (0, 200, 80), (cx + 8, y),  (cx + 14, y - 10), 2)
    pygame.draw.circle(surf, YELLOW, (cx - 14, y - 10), 3)
    pygame.draw.circle(surf, YELLOW, (cx + 14, y - 10), 3)

    # Mouth / grill lines
    for k in range(3):
        lx = cx - 8 + k * 8
        pygame.draw.line(surf, (0, 150, 50), (lx, cy + 4), (lx, cy + 10), 2)


def draw_explosion(surf, cx, cy, t):
    """Animated particle burst explosion."""
    num_particles = 14
    for k in range(num_particles):
        angle  = 2 * math.pi * k / num_particles + t * 0.1
        dist   = (20 - t) * 1.8 + random.uniform(0, 6)
        px     = int(cx + math.cos(angle) * dist)
        py     = int(cy + math.sin(angle) * dist)
        radius = max(1, t // 3)
        heat   = min(255, t * 18)
        color  = (heat, max(0, heat - 80), 0)
        if 0 <= px < WINDOW_WIDTH and 0 <= py < WINDOW_HEIGHT:
            pygame.draw.circle(surf, color, (px, py), radius)

    # Core flash
    if t > 10:
        core_r = (t - 10) * 2
        flash  = pygame.Surface((core_r * 2, core_r * 2), pygame.SRCALPHA)
        pygame.draw.circle(flash, (255, 255, 200, min(200, t * 15)), (core_r, core_r), core_r)
        surf.blit(flash, (int(cx) - core_r, int(cy) - core_r))


def _draw_bullet_trail(surf, bullet, player_color):
    trail = bullet.get('trail', [])
    n     = len(trail)
    if n < 2:
        return
    for k in range(n - 1):
        ratio   = k / n
        r       = min(255, int(player_color[0] * ratio + 255 * (1 - ratio)))
        g       = min(255, int(player_color[1] * ratio))
        b       = min(255, int(player_color[2] * ratio))
        w_px    = max(1, int(BULLET_RADIUS * ratio))
        pygame.draw.line(surf, (r, g, b),
                         (int(trail[k][0]),   int(trail[k][1])),
                         (int(trail[k+1][0]), int(trail[k+1][1])),
                         w_px)


def _draw_enemy_overlay(surf, en):
    """Draw coloured ring around fast/tank enemies."""
    _, _, ring_color, _ = ENEMY_TYPES[en['type']]
    if ring_color is None:
        return
    cx     = int(en['x'] + ENEMY_WIDTH  // 2)
    cy     = int(en['y'] + ENEMY_HEIGHT // 2)
    radius = ENEMY_WIDTH // 2 + 4
    pulse  = int(3 * math.sin(en['y'] * 0.05 + en['phase']))
    pygame.draw.circle(surf, ring_color, (cx, cy), radius + pulse, 2)


def _draw_score_popups(surf, popups, popup_font):
    for p in popups:
        alpha     = int(255 * p['t'] / p['max_t'])
        col       = tuple(min(255, int(c * alpha / 255)) for c in p['color'])
        text_surf = popup_font.render(p['text'], True, col)
        surf.blit(text_surf, (int(p['x']) - text_surf.get_width() // 2, int(p['y'])))


def _draw_ghost_ship(surf, x, color):
    """Draw a semi-transparent ship outline for inactive players."""
    ghost = pygame.Surface((SHIP_WIDTH, SHIP_HEIGHT), pygame.SRCALPHA)
    cx    = SHIP_WIDTH // 2
    pts   = [(cx, 0), (0, SHIP_HEIGHT), (SHIP_WIDTH, SHIP_HEIGHT)]
    pygame.draw.polygon(ghost, (*color, GHOST_ALPHA),      pts)
    pygame.draw.polygon(ghost, (*color, GHOST_ALPHA + 40), pts, 2)
    surf.blit(ghost, (x, WINDOW_HEIGHT - SHIP_HEIGHT))


def _draw_hud(surf, scores, high_score, t_left, frame, active_players):
    """Draw all heads-up display elements."""
    # Player scores
    for i, score in enumerate(scores):
        status = "" if active_players[i] else " (away)"
        draw_text(surf, f"P{i+1}: {score}{status}", 10, 10 + i * 32, PLAYER_COLORS[i])

    # High score
    draw_text(surf, f"BEST: {high_score}", 10, 10 + MAX_PLAYERS * 32 + 8, WHITE)

    # Timer — flash red when low
    if t_left <= LOW_TIME_THRESHOLD and frame % 20 < 10:
        timer_col = (255, 50, 50)
    else:
        timer_col = WHITE
    draw_text(surf, f"TIME: {t_left:02d}", WINDOW_WIDTH - 160, 10, timer_col)

    # Active player indicator dots
    for i in range(MAX_PLAYERS):
        dot_col = PLAYER_COLORS[i] if active_players[i] else (60, 60, 60)
        pygame.draw.circle(surf, dot_col, (WINDOW_WIDTH - 20, 20 + i * 18), 6)


def _draw_win_screen(surf, scores, winner_idx, face_surfs, high_score, frame,
                     banner_font, popup_font):
    """Full-screen win overlay shown after the 60-second round ends."""
    overlay = pygame.Surface((WINDOW_WIDTH, WINDOW_HEIGHT), pygame.SRCALPHA)
    overlay.fill((0, 0, 20, 215))
    surf.blit(overlay, (0, 0))
    cx = WINDOW_WIDTH  // 2
    cy = WINDOW_HEIGHT // 2

    # Title
    pulse  = 0.5 + 0.5 * math.sin(frame * 0.05)
    title_col = (int(200 + 55*pulse), int(200 + 55*pulse), int(50 + 100*pulse))
    title = banner_font.render("TIME'S  UP!", True, title_col)
    surf.blit(title, (cx - title.get_width() // 2, cy - 240))

    # Winner announcement
    if all(s == 0 for s in scores):
        winner_text = "DRAW!"
    elif len(scores) > 1 and scores[0] == scores[1]:
        winner_text = "TIE GAME!"
    else:
        winner_text = f"P{winner_idx + 1}  WINS!"
    win_surf = banner_font.render(winner_text, True, PLAYER_COLORS[winner_idx % len(PLAYER_COLORS)])
    surf.blit(win_surf, (cx - win_surf.get_width() // 2, cy - 160))

    # Scores
    for i, sc in enumerate(scores):
        label = popup_font.render(f"P{i+1}: {sc}", True, PLAYER_COLORS[i % len(PLAYER_COLORS)])
        surf.blit(label, (cx - label.get_width() // 2, cy - 100 + i * 36))

    # High score
    hs_surf = popup_font.render(f"BEST: {high_score}", True, WHITE)
    surf.blit(hs_surf, (cx - hs_surf.get_width() // 2, cy - 100 + len(scores) * 36 + 8))

    # Winner face snapshot (no tracking lines — raw crop from identity probe)
    face_size = 150
    winner_face = face_surfs[winner_idx] if winner_idx < len(face_surfs) else None
    if winner_face is not None:
        scaled = pygame.transform.scale(winner_face, (face_size, face_size))
        col    = PLAYER_COLORS[winner_idx % len(PLAYER_COLORS)]
        pygame.draw.rect(surf, col,
                         (cx - face_size // 2 - 4, cy + 20, face_size + 8, face_size + 8), 3)
        surf.blit(scaled, (cx - face_size // 2, cy + 24))

    prompt = popup_font.render("Press  R  to  play  again", True, CYAN)
    surf.blit(prompt, (cx - prompt.get_width() // 2, cy + 200))


# =============================================================================
# MAIN GAME LOOP
# =============================================================================

def main():
    global WINDOW_WIDTH, WINDOW_HEIGHT

    pygame.init()
    pygame.mixer.init(frequency=SAMPLE_RATE, size=-16, channels=2, buffer=512)

    # Auto-detect display resolution and go fullscreen
    info = pygame.display.Info()
    WINDOW_WIDTH  = info.current_w
    WINDOW_HEIGHT = info.current_h

    screen    = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT), pygame.FULLSCREEN)
    pygame.display.set_caption("Hailo 3D Space Invaders — Kangan Digital Initiative")
    clock     = pygame.time.Clock()
    stars3d   = generate_stars_3d()
    game_surf = pygame.Surface((WINDOW_WIDTH, WINDOW_HEIGHT))

    banner_font = pygame.font.Font(None, 72)
    popup_font  = pygame.font.Font(None, 36)

    sounds = build_sounds()
    start_bgm()

    bullet_outer_colors = [tuple(min(255, c + 80) for c in col) for col in PLAYER_COLORS]

    # Game state
    ship_positions     = [WINDOW_WIDTH // (MAX_PLAYERS + 1) * (i + 1) for i in range(MAX_PLAYERS)]
    position_histories = [
        deque([ship_positions[i]] * SMOOTHING_WINDOW, maxlen=SMOOTHING_WINDOW)
        for i in range(MAX_PLAYERS)
    ]
    bullets        = [[] for _ in range(MAX_PLAYERS)]
    enemies        = []
    explosions     = []
    fireworks      = []
    score_popups   = []
    scores         = [0] * MAX_PLAYERS
    high_score     = load_high_score()
    shoot_cooldowns= [0] * MAX_PLAYERS

    start_time       = time.time()
    frame            = 0
    banner_timer     = 0
    banner_y         = -100
    shake_timer      = 0
    last_tick_second = -1

    # Win-screen state
    GS_PLAYING     = 0
    GS_WIN_SCREEN  = 1
    game_state       = GS_PLAYING
    winner_face_surfs = [None] * MAX_PLAYERS
    winner_scores    = list(scores)
    winner_idx       = 0

    # ------------------------------------------------------------------
    # CAMERA + POSE ESTIMATION STARTUP
    # ------------------------------------------------------------------
    user_data = PoseInvadersUserData()

    if HAILO_AVAILABLE:
        # 1. Validate the camera device node exists
        if CAMERA_DEVICE and not _check_camera_device(CAMERA_DEVICE):
            print("[WARN] Camera device missing — switching to keyboard fallback.")
            _hailo_ok = False
        else:
            _hailo_ok = True

        if _hailo_ok:
            # 2. GLib main loop — MUST start before app.run() so the
            #    GStreamer pipeline bus has an event dispatcher ready
            glib_thread = threading.Thread(
                target=_start_glib_loop, daemon=True, name='GLib-Loop'
            )
            glib_thread.start()
            time.sleep(0.1)   # give GLib loop time to spin up

            # 3. Build and start the pose pipeline
            def _run_pose_app():
                # GStreamerApp.__init__ calls signal.signal() which requires the
                # main thread.  Temporarily suppress it so we can run in a daemon thread.
                # GStreamerApp calls signal.signal() in both __init__ and shutdown(),
                # which requires the main thread.  Suppress for the full thread lifetime.
                import signal as _sig
                _sig.signal = lambda *a, **kw: None

                # Route video to fakesink so no separate window opens
                _fq  = user_data.pip_queue   # closure ref for PiP probe
                _hcq = user_data.head_crop_queues
                _hbl = user_data._bbox_lock
                _hbb = user_data.head_bboxes

                class _PoseApp(GStreamerPoseEstimationApp):
                    def get_pipeline_string(self):
                        self.video_sink = 'fakesink'
                        return super().get_pipeline_string()

                    def run(self):
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
                                        px1 = int(x1*w); py1 = int(y1*h)
                                        px2 = int(x2*w); py2 = int(y2*h)
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
                    callback = PoseInvadersCallback(user_data)
                    app      = _PoseApp(callback, user_data)
                    app.run()
                except Exception as exc:
                    print(f"[ERROR] Pose pipeline crashed: {exc}")
                    print("[INFO]  Continuing in keyboard fallback mode.")

            pose_thread = threading.Thread(
                target=_run_pose_app, daemon=True, name='Pose-Pipeline'
            )
            pose_thread.start()
            print("[INFO] Camera pipeline starting…  "
                  "(first frame may take 2–3 s)")
    else:
        print("[INFO] Keyboard / mouse fallback active.")
        print("       P1: A / D keys    P2: LEFT / RIGHT keys")

    BGM_RESTART_EVENT = pygame.USEREVENT + 1
    _pip_surf   = None
    _face_surfs = [None] * MAX_PLAYERS
    _player_labels = [
        popup_font.render(f"P{pi + 1}", True, PLAYER_COLORS[pi % len(PLAYER_COLORS)])
        for pi in range(MAX_PLAYERS)
    ]

    running = True
    while running:
        # ----------------------------------------------------------------
        # EVENTS
        # ----------------------------------------------------------------
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                running = False
            elif ev.type == pygame.KEYDOWN and ev.key == pygame.K_ESCAPE:
                running = False
            elif ev.type == BGM_RESTART_EVENT:
                start_bgm()
            elif ev.type == pygame.KEYDOWN and ev.key == pygame.K_r:
                if game_state == GS_WIN_SCREEN:
                    game_state       = GS_PLAYING
                    scores           = [0] * MAX_PLAYERS
                    enemies.clear()
                    [bl.clear() for bl in bullets]
                    score_popups.clear()
                    explosions.clear()
                    start_time       = time.time()
                    last_tick_second = -1
                    start_bgm()

        # ----------------------------------------------------------------
        # KEYBOARD FALLBACK (A/D — P1,  ←/→ — P2)
        # ----------------------------------------------------------------
        keys = pygame.key.get_pressed()
        kb   = [(pygame.K_a, pygame.K_d), (pygame.K_LEFT, pygame.K_RIGHT)]
        for i, (kl, kr) in enumerate(kb):
            if i >= MAX_PLAYERS:
                break
            if keys[kl]:
                position_histories[i].append(max(0, position_histories[i][-1] - 8))
            elif keys[kr]:
                position_histories[i].append(min(WINDOW_WIDTH - SHIP_WIDTH, position_histories[i][-1] + 8))

        # ----------------------------------------------------------------
        # TIMING
        # ----------------------------------------------------------------
        elapsed = time.time() - start_time
        t_left  = max(0, int(ROUND_TIME - elapsed))

        # ----------------------------------------------------------------
        # ROUND END → WIN SCREEN
        # ----------------------------------------------------------------
        if t_left == 0 and game_state == GS_PLAYING:
            game_state = GS_WIN_SCREEN
            shake_timer = SHAKE_FRAMES
            sounds['round_end'].play()
            pygame.mixer.stop()
            if max(scores) > high_score:
                high_score = max(scores)
                save_high_score(high_score)
            winner_scores     = list(scores)
            winner_idx        = scores.index(max(scores)) if any(s > 0 for s in scores) else 0
            winner_face_surfs = list(_face_surfs)

        # ----------------------------------------------------------------
        # LOW-TIME TICK  (once per second, only while playing)
        # ----------------------------------------------------------------
        if game_state == GS_PLAYING and 0 < t_left <= LOW_TIME_THRESHOLD and t_left != last_tick_second:
            sounds['tick'].play()
            last_tick_second = t_left

        # ----------------------------------------------------------------
        # READ POSE POSITIONS  (callback sends normalised 0-1 values)
        # ----------------------------------------------------------------
        for i in range(MAX_PLAYERS):
            try:
                cx_norm = user_data.position_queues[i].get_nowait()
                hx_px   = cx_norm * WINDOW_WIDTH
                position_histories[i].append(hx_px)
            except queue.Empty:
                pass

        ship_positions = [
            int(np.clip(
                sum(hist) / len(hist) - SHIP_WIDTH // 2,
                0, WINDOW_WIDTH - SHIP_WIDTH
            ))
            for hist in position_histories
        ]

        # ----------------------------------------------------------------
        # SHOOTING  (only while game is running)
        # ----------------------------------------------------------------
        # In fallback mode all players are always considered active
        active_players = (
            user_data.active_players if HAILO_AVAILABLE
            else [True] * MAX_PLAYERS
        )
        for i in range(MAX_PLAYERS) if game_state == GS_PLAYING else []:
            if active_players[i]:
                shoot_cooldowns[i] -= 1
                if shoot_cooldowns[i] <= 0:
                    lx = ship_positions[i] + SHIP_WIDTH  * 0.2
                    rx = ship_positions[i] + SHIP_WIDTH  * 0.8
                    yb = WINDOW_HEIGHT - SHIP_HEIGHT * 0.5
                    bullets[i].append({'x': lx, 'y': yb, 'trail': [], 'color': PLAYER_COLORS[i]})
                    bullets[i].append({'x': rx, 'y': yb, 'trail': [], 'color': PLAYER_COLORS[i]})
                    shoot_cooldowns[i] = SHOOT_INTERVAL_FRAMES
                    sounds['shoot'].play()

        # ----------------------------------------------------------------
        # MOVE BULLETS
        # ----------------------------------------------------------------
        for blist in bullets:
            for b in blist[:]:
                b['trail'].append((b['x'], b['y']))
                if len(b['trail']) > BULLET_TRAIL_LEN:
                    b['trail'].pop(0)
                b['y'] -= BULLET_SPEED
                if b['y'] < 0:
                    blist.remove(b)

        # ----------------------------------------------------------------
        # SPAWN ENEMIES  (only while playing)
        # ----------------------------------------------------------------
        EDGE_MARGIN = max(60, WINDOW_WIDTH // 20)
        if game_state == GS_PLAYING and random.random() < ENEMY_SPAWN_CHANCE:
            etype = random.choices(range(len(ENEMY_TYPES)), weights=_ENEMY_TYPE_WEIGHTS)[0]
            enemies.append({
                'x':     float(random.randint(EDGE_MARGIN, WINDOW_WIDTH - ENEMY_WIDTH - EDGE_MARGIN)),
                'y':     float(-ENEMY_HEIGHT),   # slide in from above the screen
                'type':  etype,
                'phase': random.uniform(0, math.pi * 2),
            })

        # ----------------------------------------------------------------
        # MOVE ENEMIES + COLLISION DETECTION
        # ----------------------------------------------------------------
        for en in enemies[:]:
            speed_mult, points, _, _ = ENEMY_TYPES[en['type']]
            en['y'] += ENEMY_SPEED * speed_mult
            drift    = math.sin(en['y'] * 0.02 + en['phase']) * ENEMY_DRIFT_SPEED
            en['x']  = float(np.clip(en['x'] + drift, EDGE_MARGIN, WINDOW_WIDTH - ENEMY_WIDTH - EDGE_MARGIN))

            if en['y'] > WINDOW_HEIGHT:
                enemies.remove(en)
                continue

            ecx = int(en['x'] + ENEMY_WIDTH  // 2)
            ecy = int(en['y'] + ENEMY_HEIGHT // 2)
            hit = False
            for i, blist in enumerate(bullets):
                for b in blist[:]:
                    if math.hypot(b['x'] - ecx, b['y'] - ecy) < ENEMY_WIDTH / 2:
                        blist.remove(b)
                        enemies.remove(en)
                        explosions.append({'x': ecx, 'y': ecy, 't': 15})
                        scores[i] += points
                        score_popups.append({
                            'x': ecx, 'y': ecy,
                            'text': f'+{points}',
                            'color': PLAYER_COLORS[i],
                            't': 40, 'max_t': 40,
                        })
                        sounds[f'explode_{en["type"]}'].play()
                        hit = True
                        break
                if hit:
                    break

        # ----------------------------------------------------------------
        # AGE EXPLOSIONS & POPUPS
        # ----------------------------------------------------------------
        for ex in explosions[:]:
            ex['t'] -= 1
            if ex['t'] <= 0:
                explosions.remove(ex)

        for p in score_popups[:]:
            p['y'] -= 1.2
            p['t'] -= 1
            if p['t'] <= 0:
                score_popups.remove(p)

        # ----------------------------------------------------------------
        # SCREEN SHAKE COUNTDOWN
        # ----------------------------------------------------------------
        if shake_timer > 0:
            shake_timer -= 1

        # ================================================================
        # DRAW
        # ================================================================
        draw_background_3d(game_surf, stars3d)

        # Ghost ships (inactive players)
        for i in range(MAX_PLAYERS):
            if not active_players[i]:
                _draw_ghost_ship(game_surf, ship_positions[i], PLAYER_COLORS[i])

        # Active ships + bullet trails
        for i in range(MAX_PLAYERS):
            if active_players[i]:
                sy = WINDOW_HEIGHT - SHIP_HEIGHT
                draw_ship(game_surf, ship_positions[i], sy, frame, PLAYER_COLORS[i])
                outer = bullet_outer_colors[i]
                inner = PLAYER_COLORS[i]
                for b in bullets[i]:
                    _draw_bullet_trail(game_surf, b, PLAYER_COLORS[i])
                    fh = BULLET_RADIUS * 6
                    bx, by_ = int(b['x']), int(b['y'])
                    br = BULLET_RADIUS
                    pygame.draw.polygon(game_surf, outer,
                                        [(bx, by_), (bx - br, by_ + fh), (bx + br, by_ + fh)])
                    pygame.draw.polygon(game_surf, inner,
                                        [(bx, by_ + int(fh * 0.4)),
                                         (bx - br // 2, by_ + fh),
                                         (bx + br // 2, by_ + fh)])

        # Enemies
        for en in enemies:
            draw_enemy(game_surf, en['x'], en['y'])
            _draw_enemy_overlay(game_surf, en)

        # Explosions
        for ex in explosions:
            draw_explosion(game_surf, ex['x'], ex['y'], ex['t'])

        # Score popups
        _draw_score_popups(game_surf, score_popups, popup_font)

        # End-of-round banner
        if banner_timer > 0:
            banner_y = min(200, banner_y + BANNER_SPEED)
            progress = INITIAL_BANNER_TIMER - banner_timer
            wobble   = math.sin(progress / 10) * BANNER_WOBBLE_AMP
            text     = "KANGAN DIGITAL INITIATIVE"
            total_w  = banner_font.size(text)[0]
            start_x  = WINDOW_WIDTH / 2 - total_w / 2 + wobble
            for j, ch in enumerate(text):
                ch_surf = banner_font.render(ch, True, YELLOW)
                ch_x    = start_x + banner_font.size(text[:j])[0]
                ch_y    = banner_y + 30 + math.sin((progress + j * 5) / 5) * BANNER_WAVE_AMP
                game_surf.blit(ch_surf, (ch_x, ch_y))
            if random.random() < FIREWORK_CHANCE:
                fireworks.append({
                    'x': random.uniform(0, WINDOW_WIDTH),
                    'y': banner_y + random.uniform(10, 90),
                    't': FIREWORK_DURATION,
                })
            for fw in fireworks[:]:
                draw_explosion(game_surf, fw['x'], fw['y'], fw['t'])
                fw['t'] -= 1
                if fw['t'] <= 0:
                    fireworks.remove(fw)
            banner_timer -= 1

        # HUD
        _draw_hud(game_surf, scores, high_score, t_left, frame, active_players)

        # Win screen overlay (drawn on top of everything)
        if game_state == GS_WIN_SCREEN:
            _draw_win_screen(game_surf, winner_scores, winner_idx, winner_face_surfs,
                             high_score, frame, banner_font, popup_font)

        # Apply screen shake
        if shake_timer > 0:
            amp = int(SHAKE_AMPLITUDE * shake_timer / SHAKE_FRAMES)
            sx  = random.randint(-amp, amp)
            sy  = random.randint(-amp, amp)
        else:
            sx, sy = 0, 0

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
                col = PLAYER_COLORS[pi % len(PLAYER_COLORS)]
                lx  = PIP_MARGIN
                ly  = WINDOW_HEIGHT - (MAX_PLAYERS - pi) * (LEGEND_FACE + 30) - PIP_MARGIN
                pygame.draw.rect(game_surf, col,
                                 (lx - 3, ly - 3, LEGEND_FACE + 6, LEGEND_FACE + 6), 3)
                game_surf.blit(_face_surfs[pi], (lx, ly))
                lbl = _player_labels[pi]
                game_surf.blit(lbl, (lx + LEGEND_FACE // 2 - lbl.get_width() // 2,
                                     ly + LEGEND_FACE + 4))

        # ── Picture-in-picture camera feed (bottom-right corner) ─────────────
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

        screen.fill(BLACK)
        screen.blit(game_surf, (sx, sy))

        pygame.display.flip()
        clock.tick(FPS)
        frame += 1

    # ----------------------------------------------------------------
    # CLEANUP
    # ----------------------------------------------------------------
    save_high_score(high_score)
    pygame.mixer.stop()
    pygame.quit()


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    from pathlib import Path
    import os
    project_root = Path(__file__).resolve().parent.parent
    env_file     = project_root / ".env"
    os.environ["HAILO_ENV_FILE"] = str(env_file)
    main()
