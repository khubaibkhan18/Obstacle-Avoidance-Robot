"""
Webots Robot Controller — Fuzzy Logic
- Blue box  → turn left  (scaled by membership)
- Red box   → turn right (scaled by membership)
- Duck      → turn left  (scaled by membership)
- Ball      → approach / stop (scaled by membership)
- Default   → go forward

All rules fire simultaneously; outputs blended via weighted centroid.
"""

from controller import Robot
import cv2
import numpy as np

# ── Configuration ────────────────────────────────────────────────────
SPEED     = 7.0
TIME_STEP = 64

# ── HSV colour ranges ────────────────────────────────────────────────
BLUE_LOWER  = np.array([100, 150,  50]);  BLUE_UPPER  = np.array([130, 255, 255])
RED_LOWER1  = np.array([  0, 150,  80]);  RED_UPPER1  = np.array([ 10, 255, 255])
RED_LOWER2  = np.array([170, 150,  80]);  RED_UPPER2  = np.array([180, 255, 255])
DUCK_LOWER  = np.array([ 20, 100, 100]);  DUCK_UPPER  = np.array([ 35, 255, 255])
BALL_W_LO   = np.array([  0,   0, 180]);  BALL_W_HI   = np.array([180,  40, 255])
BALL_B_LO   = np.array([  0,   0,   0]);  BALL_B_HI   = np.array([180, 255,  50])


# ── Fuzzy membership functions ───────────────────────────────────────
def trapezoid(x, a, b, c, d):
    if x <= a or x >= d:
        return 0.0
    if b <= x <= c:
        return 1.0
    if x < b:
        return (x - a) / (b - a)
    return (d - x) / (d - c)

def triangle(x, a, b, c):
    return trapezoid(x, a, b, b, c)


# ── Membership sets ───────────────────────────────────────────────────
def blue_membership(cov):
    return {
        "far":  triangle(cov,  0.10, 0.30, 0.55),
        "near": trapezoid(cov, 0.40, 0.60, 1.00, 1.00),
    }

def red_membership(cov):
    return {
        "far":  triangle(cov,  0.10, 0.30, 0.55),
        "near": trapezoid(cov, 0.40, 0.60, 1.00, 1.00),
    }

def duck_membership(cov):
    return {
        "far":  triangle(cov,  0.03, 0.10, 0.22),
        "near": trapezoid(cov, 0.15, 0.28, 1.00, 1.00),
    }

def ball_membership(cov):
    return {
        "far":   triangle(cov,  0.03, 0.08, 0.18),
        "near":  triangle(cov,  0.12, 0.20, 0.30),
        "close": trapezoid(cov, 0.22, 0.30, 1.00, 1.00),
    }


# ── Vision helpers ────────────────────────────────────────────────────
def get_frame(camera, w, h):
    raw = camera.getImage()
    if raw is None:
        return None
    img = np.frombuffer(raw, dtype=np.uint8).reshape((h, w, 4))
    return img[:, :, :3]

def coverage(hsv, lo, hi, w, h):
    mask = cv2.inRange(hsv, lo, hi)
    return cv2.countNonZero(mask) / (w * h)

def sense(camera, w, h):
    bgr = get_frame(camera, w, h)
    if bgr is None:
        return 0.0, 0.0, 0.0, 0.0
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

    blue = coverage(hsv, BLUE_LOWER, BLUE_UPPER, w, h)
    red  = (coverage(hsv, RED_LOWER1, RED_UPPER1, w, h) +
            coverage(hsv, RED_LOWER2, RED_UPPER2, w, h))
    duck = coverage(hsv, DUCK_LOWER, DUCK_UPPER, w, h)

    wh   = coverage(hsv, BALL_W_LO, BALL_W_HI, w, h)
    bk   = coverage(hsv, BALL_B_LO, BALL_B_HI, w, h)
    ball = (wh + bk) / 2.0 if (wh > 0.05 and bk > 0.03) else 0.0

    return blue, red, duck, ball


# ── Fuzzy rule engine ─────────────────────────────────────────────────
def fuzzy_inference(blue_cov, red_cov, duck_cov, ball_cov):
    bm  = blue_membership(blue_cov)
    rm  = red_membership(red_cov)
    dm  = duck_membership(duck_cov)
    bam = ball_membership(ball_cov)

    rules = []

    # Blue box → turn left
    w = bm["far"]
    if w > 0:
        rules.append((w, -SPEED * 0.4,  SPEED * 0.4, "blue-far→gentle-left"))
    w = bm["near"]
    if w > 0:
        rules.append((w, -SPEED,  SPEED, "blue-near→hard-left"))

    # Red box → turn right
    w = rm["far"]
    if w > 0:
        rules.append((w,  SPEED * 0.4, -SPEED * 0.4, "red-far→gentle-right"))
    w = rm["near"]
    if w > 0:
        rules.append((w,  SPEED, -SPEED, "red-near→hard-right"))

    # Duck → turn left
    w = dm["far"]
    if w > 0:
        rules.append((w, -SPEED * 0.5,  SPEED * 0.5, "duck-far→gentle-left"))
    w = dm["near"]
    if w > 0:
        rules.append((w, -SPEED,  SPEED, "duck-near→hard-left"))

    # Soccer ball → approach / stop
    w = bam["far"]
    if w > 0:
        rules.append((w,  SPEED,        SPEED,        "ball-far→approach-fast"))
    w = bam["near"]
    if w > 0:
        rules.append((w,  SPEED * 0.6,  SPEED * 0.6,  "ball-near→approach-slow"))
    w = bam["close"]
    if w > 0:
        rules.append((w * 3,  0.0,  0.0, "ball-close→stop"))

    # Default → go forward
    all_w = (bm["far"] + bm["near"] +
             rm["far"] + rm["near"] +
             dm["far"] + dm["near"] +
             bam["far"] + bam["near"] + bam["close"] * 3)
    forward_w = max(0.0, 1.0 - min(1.0, all_w))
    if forward_w > 0:
        rules.append((forward_w, SPEED, SPEED, "nothing→forward"))

    if not rules:
        return SPEED, SPEED
    total_w = sum(r[0] for r in rules)
    if total_w < 1e-6:
        return SPEED, SPEED
    left_out  = sum(r[0] * r[1] for r in rules) / total_w
    right_out = sum(r[0] * r[2] for r in rules) / total_w
    return left_out, right_out


# ── Main ──────────────────────────────────────────────────────────────
def main():
    robot = Robot()

    left_motor  = robot.getDevice("left_motor")
    right_motor = robot.getDevice("right_motor")
    left_motor.setPosition(float("inf"))
    right_motor.setPosition(float("inf"))
    left_motor.setVelocity(0)
    right_motor.setVelocity(0)

    left_sensor  = robot.getDevice("left wheel sensor")
    right_sensor = robot.getDevice("right wheel sensor")
    left_sensor.enable(TIME_STEP)
    right_sensor.enable(TIME_STEP)

    camera = robot.getDevice("camera")
    camera.enable(TIME_STEP)
    cam_w = camera.getWidth()
    cam_h = camera.getHeight()

    print("[INFO] Fuzzy Controller → Made by Khubaib Daad Khan student id: 24179559, starting...")

    ball_stopped = False

    while robot.step(TIME_STEP) != -1:
        blue, red, duck, ball = sense(camera, cam_w, cam_h)
        l_spd, r_spd = fuzzy_inference(blue, red, duck, ball)
        left_motor.setVelocity(l_spd)
        right_motor.setVelocity(r_spd)

        if l_spd == 0.0 and r_spd == 0.0 and ball > 0.0 and not ball_stopped:
            print("Ball found - yay!")
            ball_stopped = True


if __name__ == "__main__":
    main()
