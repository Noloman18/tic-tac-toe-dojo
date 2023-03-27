import math
import os
import pickle
import random
import time

import numpy as np
import tensorflow as tf

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

def convert_to_numeric(state_tuple):
    MY_DICT = {'X':0, 'O':1, ' ':2}
    return [MY_DICT[i] for i in state_tuple]

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

def get_all_possible_actions(board):
    return [i for i in range(len(board)) if board[i] == ' ']

def select_random(actions):
    try:
        return random.choice(actions)
    except IndexError:
        raise IndexError;

# Define the reward function
def reward(state, next_state):
    if len(get_all_possible_actions(state[0])) == len(get_all_possible_actions(next_state[0])):
        return -6.0

    if is_game_over(next_state[0]):
        if is_winner(next_state[0], state[1]):
            return 2.0

        if is_winner(next_state[0], next_state[1]):
            return -2.0

        return 1

    return 0.0

# Define the Q-function using a neural network
class QFunction:
    def __init__(self, state_shape, num_actions, hidden_units=128, learning_rate=0.001):
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
        state = np.array(convert_to_numeric(state[0]))
        state = state.reshape(1, *state.shape)
        q_values = self.model.predict(state)[0]
        return q_values

    def update(self, state, action, q_value_new):
        state = np.array(convert_to_numeric(state[0]))
        state = state.reshape(1, *state.shape)
        action_one_hot = np.zeros(self.num_actions)
        action_one_hot[action] = 1
        with tf.GradientTape() as tape:
            q_value_old = self.model(state)
            q_value = tf.reduce_sum(tf.multiply(q_value_old, action_one_hot))
            loss = tf.square(q_value_new - q_value)
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
        self.actions_performed = {'X': [], 'O': []}

    def act(self, state):
        notRandomSelection = True
        actions = get_all_possible_actions(state[0])
        if random.random() < self.epsilon:
            action = select_random(actions)
            notRandomSelection = False
        else:
            q_values = self.q_function(state)
            maximum = -1_000_000
            for possible_action in actions:
                if q_values[possible_action] > maximum:
                    maximum = q_values[possible_action]
                    action = possible_action

        self.actions_performed.get(state[1]).append(f"{'NR' if notRandomSelection else 'R'} {action}")
        return action

    def learn(self, state, action, reward, next_state):
        target = reward
        if not is_game_over(next_state[0]):
            next_q_values = self.q_function(next_state)
            target += self.gamma * np.max(next_q_values)
        self.q_function.update(state, action, target)

    def getModel(self):
        return self.q_function.getModel()

    def setModel(self, model):
        self.q_function.setModel(model)

    def new_game(self):
        self.actions_performed = {'X':[], 'O':[]}

def reload_memento(agent, initial_epsilon):
    training_context = {'elapsed_time_in_seconds': 0, 'num_iterations' : 0, 'epsilon':initial_epsilon, 'training_logs':[]}
    if os.path.exists('./model/tic_tac_toe_agent.pkl'):
        with open('./model/tic_tac_toe_agent.pkl', 'rb') as f:
            model = pickle.load(f)
            agent.setModel(model)
    if os.path.exists('./training_context.pkl'):
        with open('./training_context.pkl', 'rb') as f:
            training_context = pickle.load(f)
    return training_context

def create_memento(agent, training_context):
    with open('./model/tic_tac_toe_agent.pkl', 'wb') as f:
        pickle.dump(agent.getModel(), f)
    with open('./training_context.pkl', 'wb') as f:
        pickle.dump(training_context, f)

def calculate_epsilon(epsilon,number_of_iterations,min,max):
    if epsilon <= min:
        return min

    decay =math.pow(min/max,2/number_of_iterations)
    return epsilon*decay


def perform_tic_tac_toe_training():
    # Train the agent
    training_instance_iterations = 3_000
    min_epsilon = 0.1
    max_epsilon = 0.6

    agent = Agent(state_shape=(9,), num_actions=9, hidden_units=64, learning_rate=0.01, gamma=0.99, epsilon=max_epsilon)

    training_context = reload_memento(agent, max_epsilon)
    agent.epsilon = training_context['epsilon']

    original_elapsed_time = training_context['elapsed_time_in_seconds']
    st = time.time()

    def calculate_average_reward(rewards):
        total_reward = 0
        for reward in rewards:
            total_reward += reward
        return total_reward / len(rewards)

    game_rewards = []
    for episode in range(training_context['num_iterations'],training_instance_iterations+1):
        board = [' ' for _ in range(9)]
        player = 'X'
        agent.new_game()
        training_context['num_iterations'] += 1
        turns = 0
        reward_value = 0
        agent.epsilon = calculate_epsilon(agent.epsilon, training_instance_iterations, min_epsilon, max_epsilon)
        while not is_game_over(board):
            state = (board[:], player)
            training_context['epsilon'] = agent.epsilon
            action = agent.act(state)
            board[action] = player
            turns +=1
            next_state = (board[:], get_other_player(player))
            reward_value = reward(state, next_state)
            agent.learn(state, action, reward_value, next_state)
            player = get_other_player(player)
            elapsed_time = time.time() - st
            training_context['elapsed_time_in_seconds'] = original_elapsed_time + elapsed_time

        game_rewards.append(reward_value)
        if episode > 0 and episode %100 ==0:
            terse_training_context = {'elapsed_time_in_seconds': training_context['elapsed_time_in_seconds'], 'num_iterations' :training_context['num_iterations']}
            training_log = f"Episode {episode}. Training context {terse_training_context} Epsilon {agent.epsilon:.2f} Average Reward {calculate_average_reward(game_rewards):.2f} Actions performed in last game {agent.actions_performed}. Turns {turns}"
            logs = training_context['training_logs']
            logs.append(training_log)
            print(f"{episode}th Episode logs")
            print(' '*5, '-' * 20, ' '*5)
            for item in logs:
                print(item)
            print(' '*5, '-' * 20, ' '*5)
            game_rewards = []
            create_memento(agent, training_context)

def play_simulation():
    agent = Agent(state_shape=(9,), num_actions=9, hidden_units=64, learning_rate=0.01, gamma=0.99, epsilon=0.0)
    with open('./model/tic_tac_toe_agent.pkl', 'rb') as f:
        model = pickle.load(f)
        agent.setModel(model)

    # Play the game
    board = [' ' for _ in range(9)]
    player = 'X'
    while not is_game_over(board):
        state = (board[:], player)
        action = agent.act(state,True)
        board[action] = player
        player = get_other_player(player)
        print_board(board)

