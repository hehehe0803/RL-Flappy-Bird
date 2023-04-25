import random

class Agent:
    """A Q Learning Agent designed to learn how to play flappy bird."""

    def __init__(self, epsilon, epsilon_decay, epsilon_min, gamma, init_state, q):
        """Initialise the agent with q table"""
        self.epsilon = epsilon
        self.gamma = gamma
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        self.q = q

        # initialise q with starting state
        if not self.state_to_key(init_state) in self.q:
            # add starting state if not present
            self.q[self.state_to_key(init_state)] = [0.0, 0.0, 0]


    def choose_action(self, state) -> int:
        """Choose an action using epsilon greedy algorithm."""
        # get the q values for this state
        state_action_qs = self.q[self.state_to_key(state)][:2]

        action = 0 # default to action 0 (do nothing)
        if random.uniform(0, 1) < self.epsilon:
            # select a random action
            action = random.randint(0, 1)
        else:  
            # select action with biggest q value
            max_q = max(state_action_qs)
            action = state_action_qs.index(max_q)
        return action
    
    def learn(self, state, action, reward, new_state):
        """A function that will update the q table based off the current state"""
        # update rule based on how many times state has been visited
        alpha = 1 / (1 + self.q[self.state_to_key(state)][2])

        # Get the current Q-value for the state and action
        current_q = self.q[self.state_to_key(state)][action]

        # Find the maximum Q-value for the next state
        max_q = max(self.q[self.state_to_key(new_state)][:2])

        # Update the Q-value for the current state and action
        new_q = current_q + alpha * (reward + self.gamma * max_q - current_q)
        self.q[self.state_to_key(state)][action] = new_q

        self.q[self.state_to_key(state)][2] += 1 # update how many times state has been visited


    def discover_state(self, state):
        """A function to add state to q table if not present. Returns whether new state found"""
        if not self.state_to_key(state) in self.q:
            # add state if not present
            self.q[self.state_to_key(state)] = [0.0, 0.0, 0]
            return True
        return False

    def state_to_key(self, state) -> str:  
        """Convert agent state into a key which can be used to index q table"""      
        # Calculate unique index from row and column indices
        key = "{}_{}".format(state[0], state[1])
        return key
    
    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon - self.epsilon_decay)

