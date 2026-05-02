from controller import Robot
import cv2
import numpy as np

SPEED = 7.0
TIME_STEP = 64

BLUE_LOWER  = np.array([100, 150,  50]);  BLUE_UPPER  = np.array([130, 255, 255])
RED_LOWER1  = np.array([  0, 150,  80]);  RED_UPPER1  = np.array([ 10, 255, 255])
RED_LOWER2  = np.array([170, 150,  80]);  RED_UPPER2  = np.array([180, 255, 255])
DUCK_LOWER  = np.array([ 20, 100, 100]);  DUCK_UPPER  = np.array([ 35, 255, 255])
BALL_W_LO   = np.array([  0,   0, 180]);  BALL_W_HI   = np.array([180,  40, 255])
BALL_B_LO   = np.array([  0,   0,   0]);  BALL_B_HI   = np.array([180, 255,  50])

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

    print("[INFO] Colour sensing added, starting...")

    while robot.step(TIME_STEP) != -1:
        blue, red, duck, ball = sense(camera, cam_w, cam_h)
        left_motor.setVelocity(SPEED)
        right_motor.setVelocity(SPEED)

if __name__ == "__main__":
    main()
