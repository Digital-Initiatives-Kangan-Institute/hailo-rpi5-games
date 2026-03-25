# Hailo 3D Space Invaders

A 3D Space Invaders game controlled by body pose estimation using the Hailo AI HAT+.
Supports up to 2 players simultaneously, each controlling their own ship by moving their head/body in front of the camera.

Falls back to keyboard + mouse control automatically when no Hailo hardware is detected.

## Overview

- **Pose Estimation Control**: Head position drives the player's ship left/right
- **Multiplayer**: Up to 2 players detected simultaneously via pose estimation
- **3D Starfield**: Parallax star background
- **Enemy Variety**: Normal, fast (red ring), and tank (blue ring) enemy types
- **Picture-in-Picture**: Live camera feed shown in corner of the game window
- **High Score**: Persistent high score saved between sessions

## Controls

### With Hailo (pose estimation)
- **Move your body left/right** to steer your ship
- Your ship shoots automatically

### Without Hailo (keyboard fallback)
- **Player 1**: Arrow keys to move, auto-fire
- **Player 2**: A/D keys to move, auto-fire
- **ESC**: Quit

## Requirements

- Raspberry Pi 5 + Hailo AI HAT+
- USB camera or Raspberry Pi Camera Module
- Python packages: `pygame`, `numpy`

## Usage

```bash
# From the repo root (recommended)
./play_space_invaders.sh

# Or run directly
python community_projects/space_invaders/Space_Invaders.py --input usb
python community_projects/space_invaders/Space_Invaders.py --input rpi
```

## Game Parameters

Key constants at the top of `Space_Invaders.py`:

| Constant | Default | Description |
|---|---|---|
| `MAX_PLAYERS` | 2 | Maximum simultaneous players |
| `ROUND_TIME` | 60 | Seconds per round |
| `FPS` | 60 | Target frame rate |
| `ENEMY_SPAWN_CHANCE` | 0.02 | Per-frame enemy spawn probability |
| `BULLET_SPEED` | 12 | Bullet velocity in pixels/frame |

## License

Part of the Hailo RPi5 Examples — MIT License.
