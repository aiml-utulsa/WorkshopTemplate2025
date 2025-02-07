from Agent import Agent
import random


class Random_Agent(Agent):
    def __init__(self):
        print("initializing random agent")

    def take_action(self, observations):
        return random.randint(0, 1)

    def save(self, checkpoint_path):
        print("Save not implemented")

    def load(self, checkpoint_path):
        print("Load not implemented")
