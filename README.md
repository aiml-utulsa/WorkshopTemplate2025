# WorkshopTemplate2025
The 2025 TU AI / ML Spring workshop is coming soon! This repository provides an example of how to submit a model to the competition as well as the abstract class that all agents will be implementing.

## How to format your agent

### Python structure
All agents must implement the `Agent` abstract base class. Your `__init__` function MUST INCLUDE DEFAULT ARGUMENTS FOR ALL ENTRIES!!! This is because your arguments will be filled in by the `load()` function at load time instead of at `init`. For example: Legal: `def __init__(self, x1=2, x2=5, fun=7)`, and illegal: `def __init__(self, x1, x2=5, fun=8)`. Keeping the competition organizers sane by following common rules will never hurt your odds. 

```python
class Agent(ABC):
    # These methods will be used by the environment runner to interact with the agent.
    @abstractmethod
    def take_action(self, observations):
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
```

Your agents will be put into a list so that they can be matched against eachother. The `checkpoint_path` passed to an agent to save or load itself from the drive will be the name of the zip folder you provide, explained further in the "File Structure" subsection. Your agent may implement any other functions that you want with any return types, but these will not be called in the competition. Every agent will be called by the same environment runner with the API provided above.

### File Structure

Your agents will be submitted via zip file such as "timAgent.zip". Please do not include your `__pycache__` folder for the workshop organizer's sanity and hard drive space. Inside your zip folder, you must include a python file `blah.py` which includes a class that implements `Agent` of the form: 

```python
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

```

Your python file will be loaded and run in the same directory as `Agent.py` like it does with the competition runner example provided. This means that you do not need to include `Agent.py` in your submission folder, but you may want to include it in your local file so that you don't have to change import functions upon submission. You may have as many sub folders as you like (no zip bombs, thank you, if you make me reinstall the vm you will get no prize and a dishonorable mention). Your own load function will be passed the path to your folder, what happens in your folder is your business so your model does not have to be a single `.npy` file as with the `kpkikd.npy` example.

## Multi-Agent Changes

For the Multi-Agent competition using MLAgents, another example environment runner file will be provided when the competition opens to allow for testing. You will submit a single agent which will be cloned to control each of the individual units in the multi-agent challenge. This paradigm is called parameter sharing. Parameter sharing is required for this competition because there are 100+ agents at a time, so if you each submit 100 neural networks with 100 million parameters, the competition platform's gpu and harddrives will cry a little and then catch on fire. 

## Q&A

1. If my submission is ill-formatted will I be disqualified? 
    Everybody makes mistakes, Tim will reach out if there is a code problem. 

2. When will the multi agent competition runner be released?
    On the competition start date, March 15, the official single agent and multi_agent environment runners will be released along with an example reinforcement learner for each environment. 

3. What if I can't make it to a workshop?
    The workshop lectures will be posted online after each if you cannot attend in person. 

4. Who can join the competition?
    We have a rules document that is currently under revision and will be sent out via discord. Submissions are not anonymous but elligability requirements are based on the honor system until there is a reason to look further. 

5. Can I work in groups?
    TBD Depending on the number if competitors