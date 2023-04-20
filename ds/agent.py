import random

class Agent:
    """A Q Learning Agent designed to learn how to play flappy bird."""

    def __init__(self, alpha, epsilon, gamma, init_state):
        """Initialise the agent with an empty q table"""
        self.alpha = alpha
        self.epsilon = epsilon
        self.gamma = gamma

        # initialise q with starting state
        self.q = {
            self.state_to_key(init_state) : [0.0, 0.0]
        }

        # initialise history
        self.history = []

    @classmethod
    def fromJson(self, alpha, epsilon, gamma, init_state, q_to_load):
        """Initialise the agent, passing in a pre-trained q table"""
        self.alpha = alpha
        self.epsilon = epsilon
        self.gamma = gamma

        # load q from json
        self.q = q_to_load
        if not self.state_to_key(init_state) in self.q:
            # add starting state if not present
            self.q[self.state_to_key(init_state)] = [0.0, 0.0]

        # initialise history
        self.history = []

    def choose_action(self, state) -> int:
        """Choose an action using epsilon greedy algorithm."""
        # get the q values for this state
        state_action_qs = self.q[self.state_to_key(state)]

        action = 0 # default to action 0 (do nothing)
        if random.uniform(0, 1) < self.epsilon:
            # select a random action
            action = random.randint(0, 1)
        else:  
            # select action with biggest q value
            max_q = max(state_action_qs)
            action = state_action_qs.index(max_q)
        return action
    
    def learn(self):
        """A function that will update the q values based off the states in history"""

        self.history = [] # clear history
        pass

    def add_to_history(self, state, action, reward):
        """A function which adds a state, action and reward to history"""
        self.history.append((state, action, reward))

    def state_to_key(self, state) -> str:  
        """Convert agent state into a key which can be used to index q table"""      
        # Calculate unique index from row and column indices
        key = "{}_{}_{}".format(state[0], state[1], state[2])
        return key
