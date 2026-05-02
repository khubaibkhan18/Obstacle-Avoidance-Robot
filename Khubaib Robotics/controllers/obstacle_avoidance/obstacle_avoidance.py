from controller import Robot
import numpy as np

SPEED = 7.0
TIME_STEP = 64

def get_frame(camera, w, h):
    raw = camera.getImage()
    if raw is None:
        return None
    img = np.frombuffer(raw, dtype=np.uint8).reshape((h, w, 4))
    return img[:, :, :3]

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

    print("[INFO] Camera enabled, starting...")

    while robot.step(TIME_STEP) != -1:
        frame = get_frame(camera, cam_w, cam_h)
        left_motor.setVelocity(SPEED)
        right_motor.setVelocity(SPEED)

if __name__ == "__main__":
    main()
