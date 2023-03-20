import random
import numpy as np
import tensorflow as tf
import pickle

# Define the state space
all_possible_boards = [' ' * 9]
for i in range(1, 10):
    new_boards = []
    for board in all_possible_boards:
        for player in ['X', 'O']:
            new_board = board[:i-1] + player + board[i:]
            if new_board not in new_boards:
                new_boards.append(new_board)
    all_possible_boards += new_boards
all_possible_boards.sort()

# Define the action space
all_possible_actions = [i for i in range(9)]

def convert_tuple_board_to_numeric(state_tuple):
    MY_DICT = {'X':0, 'O':1, ' ':2}
    return [MY_DICT[i] for i in list(state_tuple)]

def print_board(board):
    print(" "+board[0]+" | "+board[1]+" | "+board[2]+" ")
    print("-----------")
    print(" "+board[3]+" | "+board[4]+" | "+board[5]+" ")
    print("-----------")
    print(" "+board[6]+" | "+board[7]+" | "+board[8]+" ")

def is_winner(board, player):
    # Check rows
    for i in range(3):
        row = board[i*3:i*3+3]
        if row == [player] * 3:
            return True
    # Check columns
    for i in range(3):
        col = [board[i+j*3] for j in range(3)]
        if col == [player] * 3:
            return True
    # Check diagonals
    diagonal1 = [board[i] for i in [0, 4, 8]]
    diagonal2 = [board[i] for i in [2, 4, 6]]
    if diagonal1 == [player] * 3 or diagonal2 == [player] * 3:
        return True
    # If no winner is found, return False
    return False

def is_game_over(board):
    return ' ' not in board or is_winner(board, 'X') or is_winner(board, 'O')

def get_other_player(player):
    return 'O' if player == 'X' else 'X'

def get_all_possible_actions(state):
    board, player, turn = state
    return [i for i in range(len(board)) if board[i] == ' ']

def select_random(actions):
    try:
        return random.choice(actions)
    except IndexError:
        raise IndexError;

# Define the reward function
def reward(state):
    board, player, turn = state
    if is_winner(board, player):
        return 1.0
    elif is_winner(board, get_other_player(player)):
        return -1.0
    else:
        return 0.0

# Define the Q-function using a neural network
class QFunction:
    def __init__(self, state_shape, num_actions, hidden_units=64, learning_rate=0.001):
        self.state_shape = state_shape
        self.num_actions = num_actions
        self.hidden_units = hidden_units
        self.learning_rate = learning_rate

        self.model = self._build_model()
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)

    def _build_model(self):
        inputs = tf.keras.layers.Input(shape=self.state_shape)
        x = tf.keras.layers.Dense(self.hidden_units, activation='relu')(inputs)
        x = tf.keras.layers.Dense(self.hidden_units, activation='relu')(x)
        outputs = tf.keras.layers.Dense(self.num_actions)(x)
        model = tf.keras.models.Model(inputs=inputs, outputs=outputs)
        return model

    def __call__(self, state):
        state = np.array(convert_tuple_board_to_numeric(state[0]))
        state = state.reshape(1, *state.shape)
        q_values = self.model.predict(state)[0]
        return q_values

    def update(self, state, action, target):
        state = np.array(convert_tuple_board_to_numeric(state[0]))
        state = state.reshape(1, *state.shape)
        action_one_hot = np.zeros(self.num_actions)
        action_one_hot[action] = 1
        with tf.GradientTape() as tape:
            q_values = self.model(state)
            q_value = tf.reduce_sum(tf.multiply(q_values, action_one_hot))
            loss = tf.square(target - q_value)
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

    def getModel(self):
        return self.model

    def setModel(self, model):
        self.model = model

# Define the agent class
class Agent:
    def __init__(self, state_shape, num_actions, hidden_units=64, learning_rate=0.001, gamma=0.99, epsilon=0.1):
        self.q_function = QFunction(state_shape, num_actions, hidden_units, learning_rate)
        self.gamma = gamma
        self.epsilon = epsilon

    def act(self, state):
        if random.random() < self.epsilon:
            all_actions = get_all_possible_actions(state)
            action = select_random(all_actions)
        else:
            q_values = self.q_function(state)
            action = np.argmax(q_values)
        return action

    def model_act(self, state):
        q_values = self.q_function(state)
        actions = get_all_possible_actions(state)
        while True:
            action = np.argmax(q_values)
            if action not in actions:
                q_values[action] = -1_000_000
            else:
                break
        return action

    def learn(self, state, action, reward, next_state):
        next_q_values = self.q_function(next_state)
        target = reward + self.gamma * np.max(next_q_values)
        self.q_function.update(state, action, target)

    def getModel(self):
        return self.q_function.getModel()

    def setModel(self, model):
        self.q_function.setModel(model)

def perform_tic_tac_toe_training():
    # Train the agent
    # num_episodes = 10000
    num_episodes = 100
    # agent = Agent(state_shape=(9,), num_actions=9, hidden_units=64, learning_rate=0.001, gamma=0.99, epsilon=0.1)
    agent = Agent(state_shape=(9,), num_actions=9, hidden_units=64, learning_rate=0.01, gamma=0.99, epsilon=0.1)

    for episode in range(num_episodes):
        board = [' ' for _ in range(9)]
        player = 'X'
        turn = 1
        while not is_game_over(board):
            state = (tuple(board), player, turn)
            action = agent.act(state)
            board[action] = player
            turn += 1
            next_state = (tuple(board), get_other_player(player), turn)
            reward_value = reward(next_state)
            agent.learn(state, action, reward_value, next_state)
            player = get_other_player(player)

        print("Episode", episode)

    # Save the agent object to disk
    with open('./model/tic_tac_toe_agent.pkl', 'wb') as f:
        pickle.dump(agent.getModel(), f)


def play_simulation():
    agent = Agent(state_shape=(9,), num_actions=9, hidden_units=64, learning_rate=0.01, gamma=0.99, epsilon=0.1)
    with open('./model/tic_tac_toe_agent.pkl', 'rb') as f:
        model = pickle.load(f)
        agent.setModel(model)

    # Play the game
    board = [' ' for _ in range(9)]
    player = 'X'
    turn = 1
    while not is_game_over(board):
        state = (tuple(board), player, turn)
        action = agent.model_act(state)
        board[action] = player
        turn += 1
        player = get_other_player(player)
        print_board(board)

