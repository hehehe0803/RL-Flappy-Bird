# Reinforcement Learning Project - Flappy Bird
This project aims to teach an agent to play the popular game Flappy Bird using reinforcement learning with different algorithms and compare their results. The project is implemented in Python 3.9 and requires the following dependencies:
## Prerequisites
- Python 3.9 or later
- flappy-bird-gymnasium==0.2.1
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
## Setup - cnn_dqn.py

To run cnn_dqn, please install the following dependencies:
```
numpy
tensorflow
opencv-python-headless
flappy-bird-gymnasium
```

Following that, go to Lib/site-packages/flappy_bird_gymnasium/evs/renderer.py and change the value of `[43] FILL_BACKGROUND_COLOR` (line 43) to `(0, 0, 0)`

## Acknowledgments
This project is inspired by the original Flappy Bird game created by Dong Nguyen.
