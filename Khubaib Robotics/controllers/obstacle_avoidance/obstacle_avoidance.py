from controller import Robot

SPEED = 7.0
TIME_STEP = 64

def main():
    robot = Robot()

    left_motor  = robot.getDevice("left_motor")
    right_motor = robot.getDevice("right_motor")
    left_motor.setPosition(float("inf"))
    right_motor.setPosition(float("inf"))
    left_motor.setVelocity(0)
    right_motor.setVelocity(0)

    print("[INFO] Robot starting...")

    while robot.step(TIME_STEP) != -1:
        left_motor.setVelocity(SPEED)
        right_motor.setVelocity(SPEED)

if __name__ == "__main__":
    main()