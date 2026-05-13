# =============================================================================
#  Hailo Soccer — 3D Perspective Penalty Kick
#  Raspberry Pi 5 + Hailo AI HAT+
#  Kick the ball FORWARD into the goal using your leg.
#  Perspective view: camera behind player, goal in the distance.
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
import numpy as np
import time
import sys

# =============================================================================
# ARGUMENT PARSING
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
# HAILO / GSTREAMER
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
    print(f"[INFO] Input source : {INPUT_SOURCE}"
          + (f"  ({CAMERA_DEVICE})" if CAMERA_DEVICE else ""))
except Exception as _hailo_err:
    print(f"[WARN] Hailo not found: {_hailo_err}")
    print("[INFO] Running in KEYBOARD fallback mode.")

    class app_callback_class:
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
    global _glib_loop
    if not HAILO_AVAILABLE:
        return
    _glib_loop = GLib.MainLoop()
    try:
        _glib_loop.run()
    except Exception as exc:
        print(f"[WARN] GLib main loop exited: {exc}")


def _check_camera_device(device):
    if device and not os.path.exists(device):
        print(f"[ERROR] Camera device not found: {device}")
        return False
    return True


# =============================================================================
# CONSTANTS — Window
# =============================================================================
WINDOW_WIDTH  = 1280
WINDOW_HEIGHT = 720
FPS           = 60

PIP_W      = 320
PIP_H      = 180
PIP_MARGIN = 12

# =============================================================================
# CONSTANTS — 3D Perspective Projection
# =============================================================================
# Camera sits at world (0, CAM_H, 0) looking along +z.
# project(wx, wy, wz) maps world coords to screen (sx, sy, scale).
#   wx: left/right  (0 = centre of field)
#   wy: height      (0 = ground)
#   wz: depth       (0 = camera, increases toward goal)

CAM_H  = 4.0     # camera height above ground (world units)
FOCAL  = 860.0   # focal length (pixels · world-unit⁻¹)
NEAR_Z = 0.5     # clip anything closer than this

# Convenience: derived screen-space horizon position
#   At wz→∞, ground projects to SCREEN_CY.
#   At wz=wz0: sy_ground = SCREEN_CY + CAM_H * FOCAL / wz0

# We want the penalty-spot ball (wz=PENALTY_Z) to sit at ~85 % of screen height.
# SCREEN_CY + CAM_H * FOCAL / PENALTY_Z = 0.85 * H
# → SCREEN_CY = 0.85*H - CAM_H*FOCAL/PENALTY_Z   (computed at runtime after resize)
# We set it after display init; placeholder here:
SCREEN_CX = 640
SCREEN_CY = 180   # recalculated in main()

# =============================================================================
# CONSTANTS — World / Field
# =============================================================================
FIELD_HALF_W = 26.0    # half field width
FIELD_Z_FAR  = 80.0    # how far to draw the field (past the goal)
PENALTY_Z    = 10.0    # ball starts here (penalty spot)

GOAL_Z      = 64.0    # depth of goal line
GOAL_HW     = 5.6     # goal half-width
GOAL_H      = 5.5     # goal height
POST_R      = 0.25    # post half-width (world)

GK_HW       = 0.65    # goalkeeper half-width (world)
GK_H        = 4.2     # goalkeeper height (world)
GK_SPEED    = 0.13    # world units per frame (lateral)
GK_REACT_Z  = 30.0    # start tracking when ball z > GOAL_Z - GK_REACT_Z

BALL_R_W    = 0.75    # ball radius in world units
GRAVITY_W   = 0.0055  # gravity per frame² (world)
BOUNCE_DAMP = 0.48    # vertical velocity kept on ground bounce
FRICTION_W  = 0.995   # air friction per frame

# Kick physics
KICK_VZ_BASE  = 1.35   # minimum forward speed (world/frame)
KICK_VZ_BONUS = 0.04   # extra per px/frame of ankle speed
KICK_LAT      = 0.013  # screen-px offset → lateral world velocity
KICK_LIFT     = 0.016  # screen-px offset → lift world velocity

# Kick detection (screen space)
KICK_SCREEN_R  = 85    # px proximity
KICK_VEL_THR   = 4.0   # ankle speed threshold (px/frame) — pose mode
KICK_COOLDOWN  = 65    # frames between kicks

# Leg keypoint indices (COCO17)
KP_L_HIP    = 11; KP_R_HIP   = 12
KP_L_KNEE   = 13; KP_R_KNEE  = 14
KP_L_ANKLE  = 15; KP_R_ANKLE = 16
LEG_KP      = [KP_L_HIP, KP_R_HIP, KP_L_KNEE, KP_R_KNEE, KP_L_ANKLE, KP_R_ANKLE]
LEG_SKEL    = [(KP_L_HIP, KP_L_KNEE), (KP_L_KNEE, KP_L_ANKLE),
               (KP_R_HIP, KP_R_KNEE), (KP_R_KNEE, KP_R_ANKLE)]
ANKLE_IDX   = [KP_L_ANKLE, KP_R_ANKLE]

# Colors
WHITE       = (255, 255, 255)
BLACK       = (  0,   0,   0)
F_GREEN     = ( 46, 125,  50)
F_STRIPE    = ( 56, 142,  60)
SKY_A       = ( 65, 125, 200)
SKY_B       = (130, 190, 235)
CROWD_COL   = ( 72,  95, 135)
GOAL_COL    = (235, 235, 235)
NET_COL     = (180, 180, 180)
GK_COL      = (255, 140,   0)
YELLOW      = (255, 220,   0)
CYAN        = (  0, 210, 220)
RED         = (220,  40,  40)

# States
S_WAITING = 'waiting'
S_FLIGHT  = 'flight'
S_GOAL    = 'goal'
S_BLOCKED = 'blocked'
S_MISS    = 'miss'
STATE_HOLD = 140

# Audio
SAMPLE_RATE = 44100
SFX_VOL     = 0.70
BGM_VOL     = 0.28


# =============================================================================
# PROJECTION
# =============================================================================

def project(wx, wy, wz):
    """
    Project world (wx, wy, wz) to screen.
    Returns (sx, sy, scale) where scale = FOCAL/wz.
    Returns None if behind near clip.
    """
    if wz < NEAR_Z:
        return None
    s = FOCAL / wz
    sx = int(SCREEN_CX + wx * s)
    sy = int(SCREEN_CY + (CAM_H - wy) * s)
    return sx, sy, s


def proj_ground(wx, wz):
    """Project a point on the ground (wy=0)."""
    return project(wx, 0.0, wz)


# =============================================================================
# AUDIO
# =============================================================================

def _tone(freq, ms, vol=0.5, wave='square'):
    n = int(SAMPLE_RATE * ms / 1000)
    t = np.linspace(0, ms / 1000, n, False)
    sig = np.sign(np.sin(2 * np.pi * freq * t)) if wave == 'square' \
          else np.random.uniform(-1, 1, n) if wave == 'noise' \
          else np.sin(2 * np.pi * freq * t)
    fade = max(1, int(n * 0.15))
    sig[-fade:] *= np.linspace(1, 0, fade)
    sig = (sig * vol * 32767).astype(np.int16)
    return pygame.sndarray.make_sound(np.column_stack([sig, sig]))


def _sweep(f0, f1, ms, vol=0.5):
    n = int(SAMPLE_RATE * ms / 1000)
    t = np.linspace(0, ms / 1000, n, False)
    ph = np.cumsum(2 * np.pi * np.linspace(f0, f1, n) / SAMPLE_RATE)
    sig = np.sin(ph)
    fade = max(1, int(n * 0.12))
    sig[-fade:] *= np.linspace(1, 0, fade)
    sig = (sig * vol * 32767).astype(np.int16)
    return pygame.sndarray.make_sound(np.column_stack([sig, sig]))


def build_sounds():
    return {
        'kick':    _tone(110, 90,  SFX_VOL, 'noise'),
        'goal':    _sweep(440, 1000, 550, SFX_VOL),
        'blocked': _sweep(320, 160, 380, SFX_VOL * 0.8),
        'miss':    _tone(180, 260,  SFX_VOL * 0.55, 'square'),
        'bounce':  _tone(260,  35,  SFX_VOL * 0.35, 'noise'),
    }


def start_bgm():
    n = int(SAMPLE_RATE * 2.5)
    t = np.linspace(0, 2.5, n, False)
    w = (np.sign(np.sin(2 * np.pi * 64.0 * t)) +
         np.sign(np.sin(2 * np.pi * 64.4 * t))) * 0.5
    pulse = 0.5 + 0.5 * np.sin(2 * np.pi * 0.7 * t)
    sig = (w * pulse * BGM_VOL * 32767).astype(np.int16)
    pygame.sndarray.make_sound(np.column_stack([sig, sig])).play(loops=-1)


# =============================================================================
# POSE INTEGRATION
# =============================================================================

class SoccerUserData(app_callback_class):
    def __init__(self):
        super().__init__()
        self.leg_queue = queue.Queue(maxsize=4)
        self.pip_queue = queue.Queue(maxsize=1)


class SoccerCallback(app_callback_class):
    """Extracts hip/knee/ankle keypoints (COCO17 indices 11-16) only."""

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

        persons = []
        for det in dets:
            if det.get_label() != 'person':
                continue
            lms = det.get_objects_typed(hailo.HAILO_LANDMARKS)
            if not lms:
                continue
            pts  = lms[0].get_points()
            bbox = det.get_bbox()

            nose_x = (pts[0].x() * bbox.width() + bbox.xmin()) if pts else 0.5
            kps = {}
            for idx in LEG_KP:
                if idx < len(pts):
                    pt = pts[idx]
                    kps[idx] = (
                        float(pt.x() * bbox.width()  + bbox.xmin()),
                        float(pt.y() * bbox.height() + bbox.ymin()),
                    )
            persons.append((nose_x, kps))

        if persons:
            persons.sort(key=lambda p: p[0])
            try:
                self.user_data.leg_queue.put_nowait(persons[0][1])
            except queue.Full:
                pass

        return Gst.PadProbeReturn.OK


# =============================================================================
# DRAW — FIELD
# =============================================================================

def _build_crowd_surf(rng):
    """Pre-render the crowd stand as a static surface."""
    surf = pygame.Surface((WINDOW_WIDTH, 120), pygame.SRCALPHA)
    for cx in range(0, WINDOW_WIDTH, 20):
        h = rng.randint(18, 38)
        shade = max(50, CROWD_COL[0] + rng.randint(-25, 25))
        pygame.draw.rect(surf, (shade, shade - 12, min(255, shade + 28)),
                         (cx, 120 - h, 16, h))
        # Head
        pygame.draw.circle(surf, (min(255, shade + 40), shade - 5, shade + 20),
                           (cx + 8, 120 - h - 6), 5)
    return surf


def draw_sky(surf):
    for row in range(SCREEN_CY + 20):
        t   = row / max(1, SCREEN_CY + 20)
        col = tuple(int(SKY_A[k] + (SKY_B[k] - SKY_A[k]) * t) for k in range(3))
        pygame.draw.line(surf, col, (0, row), (WINDOW_WIDTH, row))


def draw_field_stripes(surf):
    """Draw alternating perspective stripes on the ground."""
    stripe_wz = 6.0   # stripe depth in world units
    z = NEAR_Z + 0.1
    i = 0
    while z < FIELD_Z_FAR:
        z2   = min(z + stripe_wz, FIELD_Z_FAR)
        col  = F_GREEN if i % 2 == 0 else F_STRIPE
        # Four corners of this stripe
        pts  = []
        for wz in (z, z2):
            for wx in (-FIELD_HALF_W, FIELD_HALF_W):
                p = proj_ground(wx, wz)
                if p:
                    pts.append(p[:2])
        # Reorder: near-left, near-right, far-right, far-left
        if len(pts) == 4:
            poly = [pts[0], pts[1], pts[3], pts[2]]
            pygame.draw.polygon(surf, col, poly)
        z  += stripe_wz
        i  += 1


def draw_field_lines(surf):
    """Project field markings onto the ground."""
    lw = 2

    def gline(x1, z1, x2, z2):
        a = proj_ground(x1, z1)
        b = proj_ground(x2, z2)
        if a and b:
            pygame.draw.line(surf, WHITE, a[:2], b[:2], lw)

    # Touch lines (sides)
    gline(-FIELD_HALF_W, NEAR_Z + 0.2, -FIELD_HALF_W, FIELD_Z_FAR)
    gline( FIELD_HALF_W, NEAR_Z + 0.2,  FIELD_HALF_W, FIELD_Z_FAR)
    # Goal line (back)
    gline(-FIELD_HALF_W, GOAL_Z, FIELD_HALF_W, GOAL_Z)
    # Penalty area box
    pa_hw = GOAL_HW + 8.0
    gline(-pa_hw, GOAL_Z,      -pa_hw, GOAL_Z - 12.0)
    gline( pa_hw, GOAL_Z,       pa_hw, GOAL_Z - 12.0)
    gline(-pa_hw, GOAL_Z - 12.0, pa_hw, GOAL_Z - 12.0)
    # Goal area box
    ga_hw = GOAL_HW + 3.5
    gline(-ga_hw, GOAL_Z,      -ga_hw, GOAL_Z - 5.0)
    gline( ga_hw, GOAL_Z,       ga_hw, GOAL_Z - 5.0)
    gline(-ga_hw, GOAL_Z - 5.0, ga_hw, GOAL_Z - 5.0)
    # Penalty spot
    ps = proj_ground(0, PENALTY_Z)
    if ps:
        pygame.draw.circle(surf, WHITE, ps[:2], 4)
    # Penalty arc (semicircle in world space, approximated with line segments)
    ARC_R = 7.0
    prev = None
    for deg in range(-90, 91, 8):
        rad = math.radians(deg)
        wx  = math.sin(rad) * ARC_R
        wz  = PENALTY_Z - math.cos(rad) * ARC_R
        if wz < GOAL_Z - 12.0:   # only draw the part outside penalty area
            p = proj_ground(wx, wz)
            if p and prev:
                pygame.draw.line(surf, WHITE, prev[:2], p[:2], lw)
            prev = p
        else:
            prev = None


def draw_net(surf):
    """Draw goal net as a projected grid on back/side faces."""
    STEPS_X = 10
    STEPS_Y = 8
    GOAL_DEPTH = 3.0   # how deep the net goes behind goal line

    for i in range(STEPS_X + 1):
        wx = -GOAL_HW + i * (GOAL_HW * 2 / STEPS_X)
        a  = project(wx, 0.0,     GOAL_Z)
        b  = project(wx, GOAL_H,  GOAL_Z)
        c  = project(wx, GOAL_H,  GOAL_Z + GOAL_DEPTH)
        d  = project(wx, 0.0,     GOAL_Z + GOAL_DEPTH)
        for p, q in ((a, b), (b, c)):
            if p and q:
                pygame.draw.line(surf, (*NET_COL, 130), p[:2], q[:2], 1)

    for j in range(STEPS_Y + 1):
        wy = j * (GOAL_H / STEPS_Y)
        a  = project(-GOAL_HW, wy, GOAL_Z)
        b  = project( GOAL_HW, wy, GOAL_Z)
        c  = project( GOAL_HW, wy, GOAL_Z + GOAL_DEPTH)
        d  = project(-GOAL_HW, wy, GOAL_Z + GOAL_DEPTH)
        for p, q in ((a, b), (b, c)):
            if p and q:
                pygame.draw.line(surf, (*NET_COL, 130), p[:2], q[:2], 1)


def draw_goal_posts(surf):
    """Draw left post, right post, and crossbar as thick projected lines."""
    pw = POST_R   # half-width

    def post_quad(wx_l, wy_b, wx_r, wy_t, wz):
        """Draw a rectangular post face at depth wz."""
        corners = [
            project(wx_l, wy_b, wz),
            project(wx_r, wy_b, wz),
            project(wx_r, wy_t, wz),
            project(wx_l, wy_t, wz),
        ]
        pts = [c[:2] for c in corners if c]
        if len(pts) == 4:
            pygame.draw.polygon(surf, GOAL_COL, pts)
            pygame.draw.polygon(surf, WHITE, pts, 1)

    # Left post
    post_quad(-GOAL_HW - pw, 0, -GOAL_HW + pw, GOAL_H, GOAL_Z)
    # Right post
    post_quad( GOAL_HW - pw, 0,  GOAL_HW + pw, GOAL_H, GOAL_Z)
    # Crossbar
    post_quad(-GOAL_HW - pw, GOAL_H - pw, GOAL_HW + pw, GOAL_H + pw, GOAL_Z)


def draw_goalkeeper(surf, gk_x, frame):
    """Draw goalkeeper as a projected quad inside the goal."""
    gk_wx_l = gk_x - GK_HW
    gk_wx_r = gk_x + GK_HW
    gk_wz   = GOAL_Z - 0.3   # slightly in front of goal line

    bl = project(gk_wx_l, 0.0,  gk_wz)
    br = project(gk_wx_r, 0.0,  gk_wz)
    tl = project(gk_wx_l, GK_H, gk_wz)
    tr = project(gk_wx_r, GK_H, gk_wz)

    if not all([bl, br, tl, tr]):
        return

    pts = [bl[:2], br[:2], tr[:2], tl[:2]]
    pygame.draw.polygon(surf, GK_COL, pts)

    # Jersey stripe
    ml = project(gk_x - GK_HW * 0.1, 0.0,  gk_wz)
    mr = project(gk_x + GK_HW * 0.1, 0.0,  gk_wz)
    tl2= project(gk_x - GK_HW * 0.1, GK_H, gk_wz)
    tr2= project(gk_x + GK_HW * 0.1, GK_H, gk_wz)
    if all([ml, mr, tl2, tr2]):
        pygame.draw.polygon(surf, (200, 100, 0), [ml[:2], mr[:2], tr2[:2], tl2[:2]])

    # Outline
    pygame.draw.polygon(surf, (180, 80, 0), pts, 2)

    # Head (above body)
    head_c = project(gk_x, GK_H + 0.7, gk_wz)
    if head_c:
        hr = max(3, int(0.65 * FOCAL / gk_wz))
        pygame.draw.circle(surf, (200, 165, 120), head_c[:2], hr)


# =============================================================================
# STATIC BACKGROUND CACHE
# =============================================================================

def build_static_bg(crowd_surf):
    """
    Pre-render sky, crowd, field stripes, lines, net and goal posts
    into a single surface.  Call once after SCREEN_CX/SCREEN_CY are set.
    """
    bg = pygame.Surface((WINDOW_WIDTH, WINDOW_HEIGHT))
    draw_sky(bg)
    crowd_y = SCREEN_CY - crowd_surf.get_height() + 10
    bg.blit(crowd_surf, (0, crowd_y))
    draw_field_stripes(bg)
    draw_field_lines(bg)
    draw_net(bg)
    draw_goal_posts(bg)
    return bg


# =============================================================================
# DRAW — BALL
# =============================================================================

def draw_ball_3d(surf, ball, ball_rot):
    p = project(ball['x'], ball['y'], ball['z'])
    if p is None:
        return
    bsx, bsy, scale = p
    br = max(3, int(BALL_R_W * scale))

    # Ground shadow — simple dark ellipse, no alpha surface
    shadow_p = proj_ground(ball['x'], ball['z'])
    if shadow_p:
        h_ratio = max(0.0, 1.0 - ball['y'] / (GOAL_H * 1.5))
        sh_rx = max(2, int(br * 1.6 * h_ratio))
        sh_ry = max(1, int(br * 0.55 * h_ratio))
        if sh_rx > 0 and sh_ry > 0:
            pygame.draw.ellipse(surf, (30, 90, 30),
                                (shadow_p[0] - sh_rx, shadow_p[1] - sh_ry,
                                 sh_rx * 2, sh_ry * 2))

    # Ball body
    pygame.draw.circle(surf, WHITE, (bsx, bsy), br)

    # Rotating pentagon dots
    for k in range(5):
        angle = math.radians(ball_rot + k * 72)
        px = int(bsx + math.cos(angle) * br * 0.55)
        py = int(bsy + math.sin(angle) * br * 0.55)
        dot_r = max(1, br // 4)
        pygame.draw.circle(surf, (30, 30, 30), (px, py), dot_r)

    # Seam
    for k in range(3):
        a = math.radians(ball_rot * 0.7 + k * 60)
        x1 = int(bsx + math.cos(a) * (br - 1))
        y1 = int(bsy + math.sin(a) * (br - 1))
        x2 = int(bsx - math.cos(a) * (br - 1))
        y2 = int(bsy - math.sin(a) * (br - 1))
        pygame.draw.line(surf, (100, 100, 100), (x1, y1), (x2, y2), max(1, br // 7))

    # Outline + shine
    pygame.draw.circle(surf, (60, 60, 60), (bsx, bsy), br, max(1, br // 8))
    pygame.draw.circle(surf, WHITE, (bsx - br // 3, bsy - br // 3), max(1, br // 4))


# =============================================================================
# DRAW — LEG SKELETON
# =============================================================================

def draw_leg_skeleton(surf, leg_pos, leg_hist, kick_cd, ball_sx, ball_sy, frame):
    if not leg_pos:
        return

    # Skeleton lines — draw directly, no SRCALPHA surface
    for a_idx, b_idx in LEG_SKEL:
        if a_idx in leg_pos and b_idx in leg_pos:
            ax = int(leg_pos[a_idx][0] * WINDOW_WIDTH)
            ay = int(leg_pos[a_idx][1] * WINDOW_HEIGHT)
            bx = int(leg_pos[b_idx][0] * WINDOW_WIDTH)
            by = int(leg_pos[b_idx][1] * WINDOW_HEIGHT)
            pygame.draw.line(surf, (180, 180, 180), (ax, ay), (bx, by), 4)

    # Joint dots
    for kp_idx in LEG_KP:
        if kp_idx not in leg_pos:
            continue
        kx = int(leg_pos[kp_idx][0] * WINDOW_WIDTH)
        ky = int(leg_pos[kp_idx][1] * WINDOW_HEIGHT)
        if kp_idx in (KP_L_ANKLE, KP_R_ANKLE):
            col = (255, 180, 0)
        elif kp_idx in (KP_L_KNEE, KP_R_KNEE):
            col = (100, 220, 100)
        else:
            col = (0, 200, 255)
        pygame.draw.circle(surf, col, (kx, ky), 7)

    # Ankle kick ring
    for ankle_idx in ANKLE_IDX:
        if ankle_idx not in leg_pos:
            continue
        fx = int(leg_pos[ankle_idx][0] * WINDOW_WIDTH)
        fy = int(leg_pos[ankle_idx][1] * WINDOW_HEIGHT)
        dist = math.hypot(fx - ball_sx, fy - ball_sy)

        hist = leg_hist.get(ankle_idx)
        vel  = 0.0
        if hist and len(hist) >= 2:
            vel = math.hypot(hist[-1][0] - hist[-2][0], hist[-1][1] - hist[-2][1])

        in_range = dist < KICK_SCREEN_R
        ready    = in_range and vel > KICK_VEL_THR * 0.6 and kick_cd <= 0

        if in_range:
            col = (255, 220, 0) if ready else (80, 200, 255)
            pygame.draw.circle(surf, col, (fx, fy), KICK_SCREEN_R, 3)

        if kick_cd > 40:
            fc = (255, 60, 60)
        elif ready:
            fc = (255, 240, 0)
        else:
            fc = (255, 165, 0)
        pygame.draw.circle(surf, fc, (fx, fy), 11)
        pygame.draw.circle(surf, WHITE, (fx, fy), 11, 2)


def draw_virtual_foot(surf, vfx, vfy, kick_cd, ball_sx, ball_sy, frame):
    """Keyboard fallback: draw a shoe-shaped cursor — no alpha surfaces."""
    dist  = math.hypot(vfx - ball_sx, vfy - ball_sy)
    ready = dist < KICK_SCREEN_R
    col   = (255, 220, 0) if ready else (80, 200, 255)

    pygame.draw.ellipse(surf, col, (int(vfx) - 20, int(vfy) - 4, 40, 16))
    pygame.draw.ellipse(surf, col, (int(vfx) + 4,  int(vfy) - 14, 16, 12))
    if ready:
        pygame.draw.circle(surf, col, (int(vfx), int(vfy)), KICK_SCREEN_R, 2)


# =============================================================================
# DRAW — HUD
# =============================================================================

_font_cache = {}
def font(size):
    if size not in _font_cache:
        _font_cache[size] = pygame.font.Font(None, size)
    return _font_cache[size]


def draw_text(surf, text, x, y, color=WHITE, size=28):
    surf.blit(font(size).render(text, True, color), (x, y))


def draw_hud(surf, score, attempts, state, state_timer, frame):
    # Score panel
    panel = pygame.Surface((210, 62), pygame.SRCALPHA)
    panel.fill((0, 0, 0, 110))
    surf.blit(panel, (10, 10))
    draw_text(surf, f"GOALS: {score}", 18, 16, YELLOW, 32)
    if attempts > 0:
        pct = int(score / attempts * 100)
        draw_text(surf, f"Shots: {attempts}   {pct}%", 18, 44, (180, 180, 180), 22)

    # Kick prompt when waiting
    if state == S_WAITING:
        pulse = 0.5 + 0.5 * math.sin(frame * 0.12)
        col   = tuple(int(c * pulse) for c in (255, 230, 60))
        draw_text(surf, "KICK!", WINDOW_WIDTH // 2 - 50, WINDOW_HEIGHT - 60, col, 52)

    # State banners
    if state == S_GOAL:
        pulse = 0.5 + 0.5 * math.sin(frame * 0.20)
        col   = (int(255 * pulse), int(220 * pulse), 0)
        scale = 1.0 + 0.25 * abs(math.sin(frame * 0.22))
        txt   = font(110).render("GOAL!", True, col)
        scaled = pygame.transform.scale(txt, (int(txt.get_width() * scale),
                                              int(txt.get_height() * scale)))
        surf.blit(scaled, (WINDOW_WIDTH // 2 - scaled.get_width() // 2,
                           WINDOW_HEIGHT // 2 - scaled.get_height() // 2 - 80))

    elif state == S_BLOCKED:
        ratio = min(1.0, state_timer / STATE_HOLD)
        col   = (255, int(80 * ratio), 30)
        txt   = font(80).render("SAVED!", True, col)
        surf.blit(txt, (WINDOW_WIDTH // 2 - txt.get_width() // 2,
                        WINDOW_HEIGHT // 2 - 60))

    elif state == S_MISS:
        txt = font(72).render("MISS!", True, (200, 90, 90))
        surf.blit(txt, (WINDOW_WIDTH // 2 - txt.get_width() // 2,
                        WINDOW_HEIGHT // 2 - 50))

    if not HAILO_AVAILABLE:
        draw_text(surf, "WASD / Arrows: move foot   Auto-kicks on contact",
                  WINDOW_WIDTH // 2 - 240, WINDOW_HEIGHT - 28, (150, 150, 150), 21)


# =============================================================================
# MAIN
# =============================================================================

def main():
    global WINDOW_WIDTH, WINDOW_HEIGHT, SCREEN_CX, SCREEN_CY

    pygame.init()
    pygame.mixer.init(frequency=SAMPLE_RATE, size=-16, channels=2, buffer=512)

    info = pygame.display.Info()
    WINDOW_WIDTH  = info.current_w
    WINDOW_HEIGHT = info.current_h

    # Recalculate horizon so that the ball (at PENALTY_Z) sits at 85% screen height.
    # SCREEN_CY + CAM_H * FOCAL / PENALTY_Z = 0.85 * WINDOW_HEIGHT
    SCREEN_CX = WINDOW_WIDTH  // 2
    SCREEN_CY = int(0.85 * WINDOW_HEIGHT - CAM_H * FOCAL / PENALTY_Z)

    screen    = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT), pygame.FULLSCREEN)
    pygame.display.set_caption("Hailo Soccer")
    clock     = pygame.time.Clock()
    game_surf = pygame.Surface((WINDOW_WIDTH, WINDOW_HEIGHT))

    sounds = build_sounds()
    start_bgm()

    # Pre-render static crowd stand
    rng        = random.Random(42)
    crowd_surf = _build_crowd_surf(rng)

    # Pre-render all static scene elements (sky, field, net, posts) into one surface
    static_bg  = build_static_bg(crowd_surf)

    # ------------------------------------------------------------------
    # GAME STATE
    # ------------------------------------------------------------------
    def _new_ball():
        return {'x': 0.0, 'y': 0.0, 'z': float(PENALTY_Z),
                'vx': 0.0, 'vy': 0.0, 'vz': 0.0}

    ball       = _new_ball()
    ball_rot   = 0.0
    gk_x       = 0.0       # goalkeeper world x
    score      = 0
    attempts   = 0
    state      = S_WAITING
    state_timer= 0
    kick_cd    = 0
    frame      = 0
    shake_timer= 0
    particles  = []

    # Leg tracking
    leg_pos  = {}
    leg_hist = {idx: deque(maxlen=6) for idx in LEG_KP}

    # Keyboard fallback virtual foot
    # Start foot near the ball's projected position
    vf_init   = project(0.0, 0.0, PENALTY_Z)
    vfoot_x   = float(vf_init[0]) if vf_init else float(SCREEN_CX)
    vfoot_y   = float(vf_init[1]) if vf_init else float(WINDOW_HEIGHT * 0.85)
    vfoot_hist= deque(maxlen=6)
    vfoot_hist.append((vfoot_x, vfoot_y))

    # ------------------------------------------------------------------
    # CAMERA + POSE PIPELINE  (identical pattern to Space Invaders)
    # ------------------------------------------------------------------
    user_data = SoccerUserData()

    if HAILO_AVAILABLE:
        if CAMERA_DEVICE and not _check_camera_device(CAMERA_DEVICE):
            print("[WARN] Camera missing — keyboard fallback.")
        else:
            glib_thread = threading.Thread(
                target=_start_glib_loop, daemon=True, name='GLib-Loop')
            glib_thread.start()
            time.sleep(0.1)

            def _run_pose_app():
                import signal as _sig
                _sig.signal = lambda *a, **kw: None
                _fq = user_data.pip_queue

                class _PoseApp(GStreamerPoseEstimationApp):
                    def get_pipeline_string(self):
                        self.video_sink = 'fakesink'
                        return super().get_pipeline_string()

                    def run(self):
                        overlay = self.pipeline.get_by_name("hailo_display_overlay")
                        if overlay:
                            ov_pad = overlay.get_static_pad("src")
                            if ov_pad:
                                def _pip_probe(pad, info, _):
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
                                ov_pad.add_probe(Gst.PadProbeType.BUFFER, _pip_probe, None)
                        super().run()

                try:
                    cb  = SoccerCallback(user_data)
                    app = _PoseApp(cb, user_data)
                    app.run()
                except Exception as exc:
                    print(f"[ERROR] Pose pipeline: {exc}")

            pose_thread = threading.Thread(
                target=_run_pose_app, daemon=True, name='Pose-Pipeline')
            pose_thread.start()
            print("[INFO] Camera pipeline starting…")
    else:
        print("[INFO] Keyboard fallback.  WASD/Arrows to move foot.")

    _pip_surf  = None
    _pip_alpha = 220.0

    running = True
    while running:
        # ----------------------------------------------------------------
        # EVENTS
        # ----------------------------------------------------------------
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                running = False
            elif ev.type == pygame.KEYDOWN:
                if ev.key == pygame.K_ESCAPE:
                    running = False
                elif ev.key == pygame.K_r:
                    score = 0; attempts = 0
                    ball  = _new_ball(); state = S_WAITING

        # ----------------------------------------------------------------
        # KEYBOARD FALLBACK
        # ----------------------------------------------------------------
        if not HAILO_AVAILABLE:
            keys  = pygame.key.get_pressed()
            spd   = 9
            dx, dy = 0, 0
            if keys[pygame.K_LEFT]  or keys[pygame.K_a]: dx = -spd
            if keys[pygame.K_RIGHT] or keys[pygame.K_d]: dx =  spd
            if keys[pygame.K_UP]    or keys[pygame.K_w]: dy = -spd
            if keys[pygame.K_DOWN]  or keys[pygame.K_s]: dy =  spd
            vfoot_x = float(np.clip(vfoot_x + dx, 0, WINDOW_WIDTH))
            vfoot_y = float(np.clip(vfoot_y + dy, SCREEN_CY, WINDOW_HEIGHT))
            vfoot_hist.append((vfoot_x, vfoot_y))

        # ----------------------------------------------------------------
        # READ POSE POSITIONS
        # ----------------------------------------------------------------
        try:
            kps = user_data.leg_queue.get_nowait()
            for kp_idx, (xn, yn) in kps.items():
                leg_pos[kp_idx] = (xn, yn)
                leg_hist[kp_idx].append((xn * WINDOW_WIDTH, yn * WINDOW_HEIGHT))
        except queue.Empty:
            pass

        # ----------------------------------------------------------------
        # GOALKEEPER AI
        # ----------------------------------------------------------------
        if state == S_FLIGHT and ball['z'] > GOAL_Z - GK_REACT_Z:
            gk_target = float(np.clip(ball['x'], -(GOAL_HW - GK_HW), GOAL_HW - GK_HW))
            gk_dx     = gk_target - gk_x
            gk_x     += float(np.clip(gk_dx, -GK_SPEED, GK_SPEED))
        elif state != S_FLIGHT:
            # Drift back to center slowly
            gk_x += (0.0 - gk_x) * 0.04

        # ----------------------------------------------------------------
        # BALL PHYSICS  (S_WAITING: gentle bob; S_FLIGHT: full physics)
        # ----------------------------------------------------------------
        ball_p = project(ball['x'], ball['y'], ball['z'])
        ball_sx = ball_p[0] if ball_p else SCREEN_CX
        ball_sy = ball_p[1] if ball_p else int(WINDOW_HEIGHT * 0.85)

        if state == S_WAITING:
            ball['y'] = 0.12 * math.sin(frame * 0.07)   # gentle hover
            ball_rot += 0.8

        elif state == S_FLIGHT:
            ball['x'] += ball['vx']
            ball['y'] += ball['vy']
            ball['z'] += ball['vz']
            ball['vy'] -= GRAVITY_W
            ball['vx'] *= FRICTION_W
            ball_rot   += ball['vz'] * 6

            # Ground bounce
            if ball['y'] < 0.0:
                ball['y']  = 0.0
                if abs(ball['vy']) > 0.02:
                    ball['vy'] *= -BOUNCE_DAMP
                    sounds['bounce'].play()
                else:
                    ball['vy'] = 0.0

            # ---- Goal line reached ----
            if ball['z'] >= GOAL_Z:
                bx, by = ball['x'], ball['y']
                in_goal = (-GOAL_HW < bx < GOAL_HW) and (0.0 < by < GOAL_H)
                hits_gk = (abs(bx - gk_x) < GK_HW * 1.1) and (0.0 < by < GK_H * 1.05)

                if in_goal and not hits_gk:
                    state = S_GOAL;    sounds['goal'].play()
                    score += 1;        attempts += 1
                    shake_timer = 22
                    for _ in range(45):
                        particles.append({
                            'x': float(ball_sx), 'y': float(ball_sy),
                            'vx': rng.uniform(-9, 9), 'vy': rng.uniform(-13, -2),
                            'life': rng.randint(28, 65),
                            'col': (rng.randint(180, 255), rng.randint(140, 255), rng.randint(0, 80)),
                            'r': rng.randint(3, 8),
                        })
                elif hits_gk:
                    state = S_BLOCKED; sounds['blocked'].play()
                    attempts += 1;     shake_timer = 10
                else:
                    state = S_MISS;    sounds['miss'].play()
                    attempts += 1
                state_timer = STATE_HOLD

            # ---- Off screen (flew wide, backward, or too high) ----
            elif (ball['z'] < 0 or
                  abs(ball['x']) > FIELD_HALF_W * 1.5 or
                  ball['y'] > GOAL_H * 3):
                state = S_MISS; sounds['miss'].play()
                attempts += 1;  state_timer = STATE_HOLD // 2

        # State timer countdown
        if state in (S_GOAL, S_BLOCKED, S_MISS):
            state_timer -= 1
            if state_timer <= 0:
                ball  = _new_ball()
                state = S_WAITING
                kick_cd = max(0, kick_cd - STATE_HOLD)

        # ----------------------------------------------------------------
        # KICK DETECTION
        # ----------------------------------------------------------------
        kick_cd = max(0, kick_cd - 1)

        if state == S_WAITING and kick_cd == 0:
            # Ball projected screen position
            bp = project(ball['x'], ball['y'], ball['z'])
            if bp:
                bsx, bsy = bp[0], bp[1]

                feet = []
                if HAILO_AVAILABLE:
                    for ankle_idx in ANKLE_IDX:
                        if ankle_idx in leg_pos:
                            fx = int(leg_pos[ankle_idx][0] * WINDOW_WIDTH)
                            fy = int(leg_pos[ankle_idx][1] * WINDOW_HEIGHT)
                            feet.append((fx, fy, leg_hist[ankle_idx]))
                else:
                    feet.append((int(vfoot_x), int(vfoot_y), vfoot_hist))

                for fx, fy, hist in feet:
                    if math.hypot(fx - bsx, fy - bsy) >= KICK_SCREEN_R:
                        continue

                    foot_speed = 0.0
                    if len(hist) >= 2:
                        foot_speed = math.hypot(hist[-1][0] - hist[-2][0],
                                                hist[-1][1] - hist[-2][1])

                    if foot_speed < KICK_VEL_THR and HAILO_AVAILABLE:
                        continue

                    # Kick direction from foot-to-ball offset on screen
                    dx_s = bsx - fx   # positive = foot left of ball → ball goes right
                    dy_s = bsy - fy   # positive = foot above ball → ball goes down
                    #   foot BELOW ball (fy > bsy → dy_s < 0) → ball goes up (lofted)

                    kick_vz = float(np.clip(
                        KICK_VZ_BASE + foot_speed * KICK_VZ_BONUS, KICK_VZ_BASE, 2.2))
                    kick_vx = dx_s * KICK_LAT
                    kick_vy = -dy_s * KICK_LIFT   # inverted: foot below (dy_s<0) → vy > 0

                    ball['vx'] = kick_vx
                    ball['vy'] = max(0.0, kick_vy)   # can't kick downward from ground
                    ball['vz'] = kick_vz
                    state      = S_FLIGHT
                    kick_cd    = KICK_COOLDOWN
                    shake_timer = 8
                    sounds['kick'].play()
                    break

        # ----------------------------------------------------------------
        # PARTICLES
        # ----------------------------------------------------------------
        for p in particles[:]:
            p['x'] += p['vx']; p['y'] += p['vy']; p['vy'] += 0.45
            p['life'] -= 1
            if p['life'] <= 0:
                particles.remove(p)

        shake_timer = max(0, shake_timer - 1)

        # ================================================================
        # DRAW
        # ================================================================
        game_surf.blit(static_bg, (0, 0))   # single blit for all static elements
        draw_goalkeeper(game_surf, gk_x, frame)

        # Celebration particles
        for p in particles:
            pygame.draw.circle(game_surf, p['col'], (int(p['x']), int(p['y'])), p['r'])

        draw_ball_3d(game_surf, ball, ball_rot)

        # Leg skeleton or virtual foot
        bp2 = project(ball['x'], ball['y'], ball['z'])
        bsx2, bsy2 = (bp2[0], bp2[1]) if bp2 else (SCREEN_CX, int(WINDOW_HEIGHT * 0.85))
        if HAILO_AVAILABLE:
            draw_leg_skeleton(game_surf, leg_pos, leg_hist,
                              kick_cd, bsx2, bsy2, frame)
        else:
            draw_virtual_foot(game_surf, vfoot_x, vfoot_y,
                              kick_cd, bsx2, bsy2, frame)

        draw_hud(game_surf, score, attempts, state, state_timer, frame)

        # ── PiP camera feed ─────────────────────────────────────────────
        try:
            cam_frame = user_data.pip_queue.get_nowait()
            h_p, w_p  = cam_frame.shape[:2]
            raw        = pygame.image.frombuffer(
                np.ascontiguousarray(cam_frame).tobytes(), (w_p, h_p), 'RGB')
            _pip_surf  = pygame.transform.scale(raw, (PIP_W, PIP_H))
        except queue.Empty:
            pass

        if _pip_surf is not None:
            pip_x     = WINDOW_WIDTH  - PIP_W - PIP_MARGIN
            pip_y     = WINDOW_HEIGHT - PIP_H - PIP_MARGIN
            fade_rect = pygame.Rect(pip_x - 30, pip_y - 30, PIP_W + 60, PIP_H + 60)

            obscured = False
            if bp2 and fade_rect.collidepoint(bsx2, bsy2):
                obscured = True
            if not obscured and not HAILO_AVAILABLE:
                if fade_rect.collidepoint(int(vfoot_x), int(vfoot_y)):
                    obscured = True

            target    = 40.0 if obscured else 220.0
            _pip_alpha += (target - _pip_alpha) * 0.12
            ai        = int(_pip_alpha)

            pygame.draw.rect(game_surf, (ai, ai, ai),
                             (pip_x - 2, pip_y - 2, PIP_W + 4, PIP_H + 4), 2)
            _pip_surf.set_alpha(ai)
            game_surf.blit(_pip_surf, (pip_x, pip_y))
            draw_text(game_surf, "CAM", pip_x + 4, pip_y + 4, (200, 200, 200), 20)

        # Screen shake
        sx, sy = 0, 0
        if shake_timer > 0:
            amp = int(9 * shake_timer / 22)
            sx  = rng.randint(-amp, amp)
            sy  = rng.randint(-amp, amp)

        screen.fill(BLACK)
        screen.blit(game_surf, (sx, sy))
        pygame.display.flip()
        clock.tick(FPS)
        frame += 1

    pygame.mixer.stop()
    pygame.quit()


# =============================================================================
# ENTRY POINT
# =============================================================================
if __name__ == "__main__":
    from pathlib import Path
    project_root = Path(__file__).resolve().parent.parent
    os.environ["HAILO_ENV_FILE"] = str(project_root / ".env")
    main()
