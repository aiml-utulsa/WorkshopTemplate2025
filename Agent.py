from abc import ABC, abstractmethod


class Agent(ABC):
    # These methods will be used by the environment runner to interact with the agent.
    @abstractmethod
    def take_action(self, observations, id=0):
        """Takes in a single observation (np.array) and returns
        a discrete action."""
        return 0  # Returns a single integer action

    @abstractmethod
    def save(self, checkpoint_path):
        """Given a path such as './competition_models/timAgent/'
        we want to create a folder with that name in that path
        and save our model"""

        print("Save not implemeted")

    @abstractmethod
    def load(self, checkpoint_path):
        """Given a path such as './competition_models/timAgent/'
        we want to load our model from that folder"""

        print("Load not implemented")
