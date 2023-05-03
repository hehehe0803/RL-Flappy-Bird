# Reinforcement Learning Project - Flappy Bird
This project aims to teach an agent to play the popular game Flappy Bird using reinforcement learning with different algorithms and compare their results. The project is implemented in Python 3.9 and requires the following dependencies:
## Prerequisites
- Python 3.9 or later
- flappy-bird-gym
## Setup
1. Clone the repository to your local machine.
2. Create a virtual environment `venv` inside the repo directory to contain all the dependencies with:
```
$ python -m venv venv
```
3. Activate `venv`
    1. in Windows with:
   ```$ venv/Scripts/activate```
   2. in MacOS/Ubuntu with:
   ```$ source venv/bin/activate```
4. Install the required dependencies by running:
```
$ pip install -r requirements.txt
```
## Training DDQN for features state space
Choose one of the configurations in the `config/` folder, then run:
```python main.py --config_file <config>```

## Setup - cnn_dqn.py

To run cnn_dqn, please install the following dependencies:
```
numpy
tensorflow
opencv-python-headless
flappy-bird-gymnasium
```

Following that, go to Lib/site-packages/flappy_bird_gymnasium/evs/renderer.py and change the value of `[43] FILL_BACKGROUND_COLOR` (line 43) to `(0, 0, 0)`

## Setup - Tabular methods
Requirements:
```
flappy-bird-gym
gym
numpy
pygame
matplotlib
```
flappy-bird-gym installation:
```$ pip install --no-dependencies flappy-bird-gym```

To test an agent's Q_table performance run "validate_agent_test.py" and select the appropriate Q_table from the comments:

```
# give Q_table file name
#q_table_name = "Q_table_Q_lambda.json"
#q_table_name = "Q_table_SARSA_lambda.json"
```
The agents are in the files "Q_lambda_agent.py" and "SARSA_lambda_agent.py"

## Acknowledgments
This project is inspired by the original Flappy Bird game created by Dong Nguyen.
