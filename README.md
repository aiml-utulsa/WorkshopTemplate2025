# WELCOME!
The **Spring 2025 AIML@TU Reinforcement Learning Zero to Hero Workshop and Competition** is here! This page provides information about how to view the recorded workshop sessions, access workshop materials, and submit a model to the competition as well as the abstract class that all agents will be implementing.

Submission Link: https://forms.office.com/Pages/ResponsePage.aspx?id=PAH_1LdiZ0GST1vZPoIC0z85eZBVSDlNrvRCaZtRpNFUQ0QzNDVOQTZTUUtLRTJERjQzTkxKS1FQSi4u

## [Competition Rules Here](https://github.com/aiml-utulsa/WorkshopTemplate2025/blob/main/RULES.md)

## Setting up the environment!

1. Install Python (preferably 3.11 because pip in 3.12+ removed some depricated libraries for installers and it can cause problems for libraries that are not actively maintained). So go here: https://www.python.org/downloads/ and scroll to 3.11.8 or 3.11.9 to get an installer link and install it like at https://www.python.org/downloads/release/python-3119/ then MAKE SURE TO CHECK THE BOX THAT ADDS PYTHON TO YOUR PATH!!!!! if not you have to do that yourself so look up editing system environment variables on windows or something like etc bashrc in linux. 

2. install git

3. `git clone https://github.com/aiml-utulsa/WorkshopTemplate2025`

4. navigate into the folder in terminal like "cd desktop" "cd WorkshopTemplate2025" that just cloned and create a virtual environment with your python. `python -m venv "som_env_name_you_like"` or if you want a particular version of python `py -3.11 -m venv "blah"`. Then on linux do `source ./blah/Scripts/activate` or on windows `./blah/Scripts/activate` to enter the virtual environment. This is basically a fresh python env with nothing installed

5. Either look at the `pip_commands.txt` file and just do each command in order (might need different torch command based on your gpu/operating system so go here: https://pytorch.org/get-started/locally/ ) and if you do those commands in that order you should be good! Or you can use the requirements.txt method instead. 

## View Session Recordings:

[Session #1 2/22/25 (Introduction to Reinforcement Learning)](https://app.bluedothq.com/preview/67ba0d4c6c00e2181fba914d?timestamp=1064.500395): Multi-armed bandits, contextual bandits, Markov Decision Processes, reinforcement learning objective, frozen lake environment example, policy iteration, Monte Carlo vs. Dynamic Programming, temporal difference learning, neural networks in RL.

[Session #2 3/1/25 (Intro to Q-Learning)](https://app.bluedothq.com/preview/67c34acbb429ce84e208ba11?timestamp=390.18527): RL Goal, environment dynamics, Q-learning concept, Python implementation, Q-learning implementation, policy improvement, intro to Google Colab.

[Session #3 3/8/25 (Implementing Q-Learning)](https://app.bluedothq.com/preview/67cc82d7e070d428efa67aa9): Q-Learning implementation, epsilon-greedy vs. optimistic intitialization, cart pole environment, descretization of continuous states, limitations of discretization.

[Session #4 3/15/25 (Intro to Q-networks)](https://app.bluedothq.com/preview/67d5b31088f84dca4f17eb24): Transition from Q-tables to Q-networks, neural network basics and implementation, Q-network implementation, memory buffer implementation, debugging and final implementation.

[Session #5 3/29/25 (Intro to Neural Networks)](https://app.bluedothq.com/preview/67e82b2785bc4e7b7b98782c): Intro to Neural Networks, Non-linearity in Neural Networks, Training Nerual Networks, Implementing Q-Learning with Neural Networks, Challenges and Improvements, Competition Information.

## Code for all sessions
[Copy/paste this code from Google Colab to get started!](https://colab.research.google.com/drive/1jJOLOlI28JuhT1sGx6J-bsXNhL0oDOC7?usp=sharing)

## How to format your agent

### Python structure
All agents must implement the `Agent` abstract base class. Your `__init__` function MUST INCLUDE DEFAULT ARGUMENTS FOR ALL ENTRIES!!! This is because your arguments will be filled in by the `load()` function at load time instead of at `init`. For example: Legal: `def __init__(self, x1=2, x2=5, fun=7)`, and illegal: `def __init__(self, x1, x2=5, fun=8)`. Keeping the competition organizers sane by following common rules will never hurt your odds. 

```python
class Agent(ABC):
    # These methods will be used by the environment runner to interact with the agent.
    @abstractmethod
    def take_action(self, observations, id=0):
        """Takes in a single observation (np.array) and returns
        a discrete action. the id is this agent's number in case 
        you train multiple policies so that your agent class can
        identify which player is taking an action"""
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

Your agents will be submitted via zip files such as "timAgent1.zip". Please do not include your `__pycache__` folder for the workshop organizer's sanity and hard drive space. Inside your zip folder, you must include a python file `blah.py` which includes a class that implements `Agent` of the form: 

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

For the Multi-Agent competition using MLAgents, another example environment runner `competition_runner_MARL` has been provided. You will submit a single agent class which will control all agents in the environment. If you do not use id to maintain separate agents (which we recommend) this paradigm is called parameter sharing. We recommend parameter sharing because it makes agent coordinatione easier and the poor gpu doesn't want to keep track of 10 neural networks. We are doing the 6v6 version of the environment pending feedback from club members.

## Example Submission

If I tim were to compete in this arena, I would submit a single .zip file called `tim_flavin_single_agent.zip` which contains everything needed for the single agent challenge (besides `__pycache__` do not include your pychache) and one folder for the multi agent challenge. For example, I could submit the `./timAgent/` folder included for this project and the `./marlAgent/Q_net_example/` folder both contained within this github repository. Also include a README.txt or a description in your project report file which specifies which folder is for which challenge and the name of the main python file. For example the `/Q_net_example` folder is for the multi-agent challenge, and the `Q_agent.py` file is the main file. 

In order to test your submission, use the `competitionrunner_Single.py` and `competition_runner_MARL.py` but add your model and model paths to the list of models like 

```python
from CompetitionAgents.marlAgent.rand_agent import Random_Agent
from CompetitionAgents.marlAgent.Q_net_example.Q_agent import Q_Agent

#Your addition: 
from CompetitionAgents.marlAgent.my_new_awesome_submission import coolAgent


    agents = [Random_Agent(), Q_Agent()]
    comp_agent_folders = [
        "./CompetitionAgents/marlAgent/",
        "./CompetitionAgents/marlAgent/Q_net_example/",
        "./CompetitionAgents/marlAgent/", # your path which should be this,
        # but if you have a sub folder like the Q example, you can add more path
        # If you try to ../../../ your way our of my path to look for a passwords
        # file you should know that this will run on a collab instance and you
        # will be disqualified so don't do that.
    ]

```


## Q&A

1. If my submission is ill-formatted will I be disqualified? 
    Everybody makes mistakes, Tim will reach out if there is a code problem. 

2. When will the multi agent competition runner be released?
    It is out! 

3. What if I can't make it to a workshop?
    The workshop lectures will be posted online after each if you cannot attend in person. 

4. Who can join the competition?
    We have a rules document that is currently under revision and will be sent out via discord. Submissions are not anonymous but elligability requirements are based on the honor system until there is a reason to look further. 

5. Can I work in groups?
    TBD Depending on the number if competitors

6. Where do I submit sumbissions? 
    Submit your agents via the "2025-rl-workshop-submissions" channel microsoft form!
