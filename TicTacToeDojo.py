import itertools
import os
import pickle
import random
import time

import numpy as np

import tensorflow as tf
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.layers import Input
from tensorflow.python.keras.models import clone_model
from tensorflow.keras.optimizers import Adam

TRAINING_ACCURACY_LOG = './training_progress/invalid_move_training_accuracy.log'

VALID_INCONSEQUENTIAL_MOVE_REWARD = 0

INVALID_MOVE_PENALTY = -5

WINNING_REWARD = 2

DISCOUNT_FACTOR = 0.95

LEARNING_RATE = 0.001

SECOND_HIDDEN_LAYER_NODE_NUMBER = 32

FIRST_HIDDEN_LAYER_NODE_NUMBER = 32

NUMBER_OF_EPOCHS = 2_000_000
INVALID_MOVE_TRAINING_EPOCHS = 100_000
BATCH_SIZE = 4096

CONTEXT_PKL = './training_progress/training_context.pkl'
BASE_AGENT = './model/base_agent.pkl'
TAC_TOE_SECOND_AGENT_PKL = './model/tic_tac_toe_second_agent.pkl'
TAC_TOE_FIRST_AGENT_PKL = './model/tic_tac_toe_first_agent.pkl'

def clear_console():
    os.system('cls' if os.name=='nt' else 'clear')
class Environment:
    def __init__(self):
        self.reset_board()
        self.network_mapper = EnvironmentNetworkMapper(self)

    def update_with_action(self,action, player):
        is_valid = self.is_valid_move(action)
        self.board[action] = player
        return is_valid

    def get_all_possible_combinations_of_moves(self):
        board = ['_', '_', '_', '_', '_', '_', '_', '_', '_']
        possible_boards = []

        # Generate all possible combinations of X and O on the board
        combinations = itertools.product(['X', 'O',' '], repeat=9)
        for combo in combinations:
            for i, val in enumerate(combo):
                board[i] = val
            if not self.is_game_over(board):
                x_count = board.count('X')
                o_count = board.count('O')
                if (x_count >0 or o_count < 0) and abs(x_count - o_count) <= 1 :
                    possible_boards.append(board[:])
        return possible_boards

    def undo_action(self,action):
        self.board[action] = ' '
        return self.board

    def get_possible_actions(self,board = None):
        if board is None:
            board = self.board
        return [i for i in range(len(board)) if board[i] ==' ']

    def is_valid_move(self, action):
        actions = self.get_possible_actions()
        return action in actions

    def reset_board(self):
        self.board = [' '] * 9

    def get_board(self):
        return self.board[:]

    def first_player_symbol(self):
        return 'X'

    def second_player_symbol(self):
        return 'O'
    def is_winner(self,player, board = None):
        if board is None:
            board = self.board
        # Check rows
        for i in range(3):
            row = self.board[i*3:i*3+3]
            if row == [player] * 3:
                return True
        # Check columns
        for i in range(3):
            col = [self.board[i+j*3] for j in range(3)]
            if col == [player] * 3:
                return True
        # Check diagonals
        diagonal1 = [self.board[i] for i in [0, 4, 8]]
        diagonal2 = [self.board[i] for i in [2, 4, 6]]
        if diagonal1 == [player] * 3 or diagonal2 == [player] * 3:
            return True
        # If no winner is found, return False
        return False

    def get_winner(self):
        if self.is_winner(self.first_player_symbol()):
            return self.first_player_symbol()
        elif self.is_winner(self.second_player_symbol()):
            return self.second_player_symbol()
        else:
            return None
    def print_board(self):
        def printRow(row):
            offset = (row-1)*3
            print(f"{self.board[0+offset]}|{self.board[1+offset]}|{self.board[2+offset]}")
        printRow(1)
        printRow(2)
        printRow(3)

    def is_game_over(self, board = None):
        if board is None:
            board = self.board
        return ' ' not in board or self.is_winner( self.first_player_symbol()) or self.is_winner( self.second_player_symbol())

    def get_other_player(self, player):
        return self.second_player_symbol() if player == self.first_player_symbol() else self.first_player_symbol()


class EnvironmentNetworkMapper:

    def __init__(self, env:Environment):
        self.env = env

    def perform_prediction(self,model:Sequential):
        numpy_input = np.array(self.env_state_to_network_input())
        return model.predict(numpy_input.reshape(1,-1), verbose=0)

    def teach_model(self, model:Sequential, input, target):
        numpy_input = np.array(input)
        model.fit(numpy_input.reshape(1,-1), target.reshape(1,-1), epochs=1, verbose=0)

    def env_state_to_network_input(self, board = None):
        board = self.env.get_board() if board is None else board
        input = [0] * 2*len(board)
        for i in range(len(board)):
            if board[i] == 'X':
                input[i] = 1
            elif board[i] == 'O':
                input[i+9] = 1
        return input

    def env_input_length(self):
        return (2*len(self.env.get_board()),)

    def env_output_length(self):
        return len(self.env.get_board())

# Define the agent class
class Agent:
    def __init__(self, model:Sequential, agent_symbol,epsilon=0.0):
        self.games_played = 0
        self.games_won = 0
        self.games_drawn = 0
        self.games_lost = 0
        self.epsilon = epsilon
        self.agent_symbol = agent_symbol
        self.actions_performed_in_last_game = []
        self.current_chosen_action = None
        self.current_q_values = None
        self.current_input = None
        self.model = model
        self.was_random_selection = False
        self.cloned_model = self.backup_brain()

    def new_game(self):
        self.actions_performed_in_last_game = []
        self.games_played += 1
        self.cloned_model = self.backup_brain()


    def backup_brain(self):
        clone = clone_model(self.model)
        clone.set_weights(self.model.get_weights())
        return clone

    def act(self, env:Environment):
        if np.random.random() < self.epsilon:
            actions = env.get_possible_actions()
            chosen_action = actions[np.random.randint(len(actions))]
            self.was_random_selection = True
            self.current_input = None
            self.current_q_values = None
        else:
            self.current_input = env.network_mapper.env_state_to_network_input()
            self.current_q_values = env.network_mapper.perform_prediction(self.get_brain())[0]
            chosen_action = np.argmax(self.current_q_values)
            self.was_random_selection = False

        self.current_chosen_action = chosen_action
        return chosen_action

    def get_brain(self):
        return self.model

    def get_backup_brain(self):
        return self.cloned_model

    def to_string(self):
        return f"[ Symbol:{self.agent_symbol} P:{self.games_played} W:{100*self.games_won/self.games_played:.2f} D:{100*self.games_drawn/self.games_played:.2f} L:{100*self.games_lost/self.games_played:.2f} ]"

class TrainingContext:
    def __init__(self):
        self.elapsed_time_in_seconds = 0
        self.num_iterations = 0
        self.epsilon = 0.1
        self.training_logs = []

        if os.path.exists('./training_context.pkl'):
            with open('./training_context.pkl', 'rb') as f:
                training_context = pickle.load(f)
            self.elapsed_time_in_seconds = training_context.elapsed_time_in_seconds
            self.num_iterations = training_context.num_iterations
            self.epsilon = training_context.epsilon
            self.training_logs = training_context.training_logs

def create_mementos(first_agent: Agent,second_agent:Agent, training_context):
    first_agent.get_brain().save(TAC_TOE_FIRST_AGENT_PKL)
    second_agent.get_brain().save(TAC_TOE_SECOND_AGENT_PKL)
    with open(CONTEXT_PKL, 'wb') as f:
        pickle.dump(training_context, f)

def create_new_model(env:Environment):
    model = Sequential()
    model.add(Input(shape=env.network_mapper.env_input_length()))
    model.add(Dense(FIRST_HIDDEN_LAYER_NODE_NUMBER, activation='relu'))
    if SECOND_HIDDEN_LAYER_NODE_NUMBER>0:
        model.add(Dense(SECOND_HIDDEN_LAYER_NODE_NUMBER, activation='relu'))
    model.add(Dense(env.network_mapper.env_output_length()))
    optimizer = Adam(learning_rate=LEARNING_RATE)
    model.compile(loss='mse',  metrics='accuracy', optimizer=optimizer)
    return model

def create_or_get_base_model(env:Environment):
    if os.path.exists(BASE_AGENT):
        print('Loading model....')
        return tf.keras.models.load_model(BASE_AGENT)
    return create_new_model(env)
def create_or_get_agent_brain(file_name,symbol,env:Environment):
    if os.path.exists(file_name):
        model = tf.keras.models.load_model(file_name)
    else:
        if os.path.exists(BASE_AGENT):
            model = tf.keras.models.load_model(BASE_AGENT)
        else:
            model = create_new_model(env)

    return Agent(model, symbol,0.1)

def create_first_player(env:Environment):
    return create_or_get_agent_brain(TAC_TOE_FIRST_AGENT_PKL, env.first_player_symbol(),env)

def create_second_player(env:Environment):
    return create_or_get_agent_brain(TAC_TOE_SECOND_AGENT_PKL, env.second_player_symbol(),env)

def calculate_epsilon(number_of_iterations):
    if 0 <= number_of_iterations <= 5000:
        return 0.5
    elif 5000 < number_of_iterations <= 10_000:
        return 0.3
    elif 10_000 < number_of_iterations <= 15_000:
        return 0.2
    elif 15_000 < number_of_iterations < 20_000:
        return 0.3
    return 0.1

def teach_current_agent(current_agent:Agent, next_agent:Agent, current_action,is_action_valid,env:Environment, debug=False):
    if current_agent.was_random_selection:
        return

    discount_factor = DISCOUNT_FACTOR
    reward = VALID_INCONSEQUENTIAL_MOVE_REWARD
    is_done = env.is_game_over()
    if is_done:
        if env.is_winner(current_agent.agent_symbol):
            reward = WINNING_REWARD
    elif not is_action_valid:
        reward = INVALID_MOVE_PENALTY

    network_mapper = env.network_mapper
    target = reward
    if not is_done:
        q_next_values = network_mapper.perform_prediction(next_agent.get_backup_brain())[0]
        target += discount_factor* np.amax(q_next_values)

    q_value_current = current_agent.current_q_values[:]
    q_value_current[current_action] = target
    network_mapper.teach_model(current_agent.model, current_agent.current_input, q_value_current)
    if debug:
        board = env.get_board()
        old_q_values = current_agent.current_q_values
        new_q_values = network_mapper.perform_prediction(current_agent.get_brain())[0]
        old_action = current_action
        new_action = np.argmax(new_q_values)
        break_point_here = True

def perform_tic_tac_toe_training():
    print('Going to the mountains to train at 50 times the gravity')
    env = Environment()
    training_context = TrainingContext()

    first_agent = create_first_player(env)
    second_agent = create_second_player(env)
    original_elapsed_time = training_context.elapsed_time_in_seconds

    st = time.time()

    def new_game():
        env.reset_board()
        first_agent.new_game()
        second_agent.new_game()
        training_context.num_iterations += 1
        training_context.epsilon = first_agent.epsilon = second_agent.epsilon = calculate_epsilon(episode)
        return first_agent

    total_move_count = 0
    invalid_move_count = 0
    games_played = 0

    def get_other_agent(current_agent):
        return first_agent if current_agent == second_agent else second_agent

    for episode in range(training_context.num_iterations,NUMBER_OF_EPOCHS+1):
        current_agent = new_game()
        games_played += 1
        is_not_finished = not env.is_game_over()
        while is_not_finished:
            total_move_count+=1
            action = current_agent.act(env)
            next_agent = get_other_agent(current_agent)
            valid_selection = env.update_with_action(action, current_agent.agent_symbol)
            teach_current_agent(current_agent, next_agent, action,valid_selection,env,True)
            if valid_selection:
                current_agent = next_agent
            else:
                env.undo_action(action)
                invalid_move_count += 1

            is_not_finished = not env.is_game_over()
            if not is_not_finished:
                winner = env.get_winner()
                if winner == env.first_player_symbol():
                    first_agent.games_won += 1
                    second_agent.games_lost += 1
                elif winner == env.second_player_symbol():
                    second_agent.games_won += 1
                    first_agent.games_lost += 1
                else:
                    first_agent.games_drawn += 1
                    second_agent.games_drawn += 1

        elapsed_time = time.time() - st
        training_context.elapsed_time_in_seconds = original_elapsed_time + elapsed_time

        if episode > 0 and episode%100 ==0:
            # create_mementos(first_agent,second_agent,training_context) TODO:Uncomment....
            clear_console()
            training_log = f"Episode {episode}. Elapsed time {training_context.elapsed_time_in_seconds:.0f} Epsilon {training_context.epsilon} Avg. Invalid Moves: {invalid_move_count}/{total_move_count} = {100*invalid_move_count/total_move_count:.2f} Avg. Turns = {total_move_count}/ {games_played} = {total_move_count/games_played:.2f} Agent Stats [{first_agent.to_string()}] "
            logs = training_context.training_logs
            logs.append(training_log)
            print(f"{episode}th Episode logs")
            print(' '*5, '-' * 20, ' '*5)
            for item in logs:
                print(item)
            print(' '*5, '-' * 20, ' '*5)
            invalid_move_count = 0
            total_move_count = 0
            games_played = 0

def load_or_init_invalid_move_model_and_context(env:Environment):
    base_model = create_or_get_base_model(env)
    return base_model
def pre_train_base_model_to_exclude_invalid_moves():
    env = Environment()
    network_mapper = env.network_mapper
    model = load_or_init_invalid_move_model_and_context(env)
    possible_combinations = env.get_all_possible_combinations_of_moves()
    random.shuffle(possible_combinations)
    X = []
    y = []
    def calculate_target(action, actions):
        if action in actions:
            return VALID_INCONSEQUENTIAL_MOVE_REWARD
        else:
            return -1

    for combination in possible_combinations:
        actions = env.get_possible_actions(combination)
        inputs = network_mapper.env_state_to_network_input(combination)
        outputs = [calculate_target(action, actions) for action in range(9)]
        X.append(inputs)
        y.append(outputs)

    num_train_and_validation = int(len(X) * 0.7)  # Use 70% of the data for training
    num_test = int(len(X) * 0.9)
    X_train, y_train = X[:num_train_and_validation], y[:num_train_and_validation]
    X_val, y_val = X[num_train_and_validation:num_test], y[num_train_and_validation:num_test]
    X_test, y_test = X[num_test:], y[num_test:]

    # Split the data into training and validation sets
    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train)).shuffle(len(X_train)).batch(BATCH_SIZE)
    val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(BATCH_SIZE)
    test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(BATCH_SIZE)

    st = time.time()

    model.fit(train_dataset, validation_data=val_dataset, epochs=INVALID_MOVE_TRAINING_EPOCHS, verbose=2)
    test_loss, test_acc = model.evaluate(test_dataset, verbose=1)
    elapsed_time = time.time() - st
    print('Test accuracy:', test_acc, 'Test loss:', test_loss, 'Training took', f"{elapsed_time/60:.2f} minutes")

    model.save(BASE_AGENT)


if __name__ == "__main__":
    # perform_tic_tac_toe_training()
    pre_train_base_model_to_exclude_invalid_moves()