from Agent import Agent
import numpy as np
import os


class PID_Agent(Agent):
    def __init__(self, kp=1.0, ki=0.0, kd=1.0):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.previous_error = 0
        self.integral = 0
        self.target = 0

    def take_action(self, observations):
        # observations are position, velocity, pole angle, pole angular velocity
        angle = observations[2]
        error = self.target - angle
        self.integral += error
        derivative = error - self.previous_error
        self.previous_error = error
        action = self.kp * error + self.ki * self.integral + self.kd * derivative
        return 0 if action > 0 else 1

    def save(self, checkpoint_path):
        if not os.path.exists(checkpoint_path):
            os.makedirs(checkpoint_path)
        np.save(
            open(checkpoint_path + "kpkikd", "wb"),
            np.array([self.kp, self.ki, self.kd]),
        )

    def load(self, checkpoint_path):
        if os.path.exists(checkpoint_path + "kpkikd"):
            kpkikd = np.load(open(checkpoint_path + "kpkikd", "rb"))
            self.kp = kpkikd[0]
            self.ki = kpkikd[1]
            self.kd = kpkikd[2]
        else:
            print("No PID parameters found")
