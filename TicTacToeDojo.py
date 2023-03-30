import os
import pickle
import time

import numpy as np
from keras import Sequential
from keras.layers import Dense
from keras.layers import Input

from tensorflow import keras

CONTEXT_PKL = './training_context.pkl'

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

    def undo_action(self,action):
        self.board[action] = ' '
        return self.board

    def get_possible_actions(self):
        return [i for i in range(len(self.board)) if self.board[i] == ' ']

    def is_valid_move(self, action):
        return action in self.get_possible_actions()

    def reset_board(self):
        self.board = [' '] * 9

    def get_board(self):
        return self.board[:]

    def first_player_symbol(self):
        return 'X'

    def second_player_symbol(self):
        return 'O'
    def is_winner(self,player):
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

    def is_game_over(self):
        return ' ' not in self.board or self.is_winner( self.first_player_symbol()) or self.is_winner( self.second_player_symbol())

    def get_other_player(self, player):
        return self.second_player_symbol() if player == self.first_player_symbol() else self.first_player_symbol()


class EnvironmentNetworkMapper:

    def __init__(self, env:Environment):
        self.env = env

    def perform_prediction(self,model:Sequential):
        return model.predict([self.env_state_to_network_input()], verbose=0)

    def teach_model(self, model:Sequential, input, target):
        numpy_input = np.array(input)
        model.fit(numpy_input.reshape(1,-1), target.reshape(1,-1), epochs=1, verbose=0)

    def env_state_to_network_input(self):
        board = self.env.get_board()
        input = [0] * 3*len(board)
        for i in range(len(board)):
            if board[i] == ' ':
                input[i] = 1
            elif board[i] == 'X':
                input[i+9] = 1
            else:
                input[i+18] = 1
        return input

    def env_input_length(self):
        return (3*len(self.env.get_board()),)

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

    def new_game(self):
        self.actions_performed_in_last_game = []
        self.games_played += 1

    def act(self, env:Environment):
        self.current_input = env.network_mapper.env_state_to_network_input()
        self.current_q_values = env.network_mapper.perform_prediction(self.model)[0]
        chosen_action = np.argmax(self.current_q_values)
        if np.random.random() < self.epsilon:
            actions = env.get_possible_actions()
            chosen_action = actions[np.random.randint(len(actions))]

        self.current_chosen_action = chosen_action
        return chosen_action

    def get_brain(self):
        return self.model

    def to_string(self):
        return f"[ Symbol:{self.agent_symbol} P:{self.games_played} W:{self.games_won} D:{self.games_drawn} L:{self.games_lost} ]"

class TrainingContext:
    def __init__(self):
        self.elapsed_time_in_seconds = 0
        self.num_iterations = 0
        self.epsilon = 0.1
        self.training_logs = []

        if os.path.exists('./training_context.pkl'):
            with open('./training_context.pkl', 'rb') as f:
                training_context = pickle.load(f)
            self.elapsed_time_in_seconds = training_context['elapsed_time_in_seconds']
            self.num_iterations = training_context['num_iterations']
            self.epsilon = training_context['epsilon']
            self.training_logs = training_context['training_logs']

def create_mementos(first_agent: Agent,second_agent:Agent, training_context):
    with open(TAC_TOE_FIRST_AGENT_PKL, 'wb') as f:
        pickle.dump(first_agent.get_brain(), f)
    with open(TAC_TOE_SECOND_AGENT_PKL, 'wb') as f:
        pickle.dump(second_agent.get_brain(), f)
    with open(CONTEXT_PKL, 'wb') as f:
        pickle.dump(training_context, f)

def create_or_get_agent_brain(file_name,symbol,env:Environment):
    if os.path.exists(file_name):
        with open(file_name, 'rb') as f:
            model = pickle.load(f)
            model
    else:
        model = Sequential()
        model.add(Input(shape=env.network_mapper.env_input_length()))
        model.add(Dense(512, activation='relu'))
        model.add(Dense(128, activation='relu'))
        model.add(Dense(env.network_mapper.env_output_length()))
        model.compile(loss='mse', optimizer=keras.optimizers.Adam(learning_rate=0.01))

    return Agent(model, symbol,0.1)

def create_first_player(env:Environment):
    return create_or_get_agent_brain(TAC_TOE_FIRST_AGENT_PKL, env.first_player_symbol(),env)

def create_second_player(env:Environment):
    return create_or_get_agent_brain(TAC_TOE_SECOND_AGENT_PKL, env.second_player_symbol(),env)

def calculate_epsilon(number_of_iterations):
    if 0 <= number_of_iterations <= 1000:
        return 0.5
    elif 1000 < number_of_iterations <= 2000:
        return 0.3
    elif 2000 < number_of_iterations <= 3000:
        return 0.2
    elif 6000 < number_of_iterations < 7000:
        return 0.3
    return 0.1

def teach_current_agent(current_agent:Agent, next_agent:Agent, current_action,is_action_valid,env:Environment):
    discount_factor = 0.95
    reward = 0
    is_done = env.is_game_over()
    if is_done:
        if env.is_winner(current_agent.agent_symbol):
            reward = 2
        else:
            reward = 1
    elif not is_action_valid:
        reward = -5

    network_mapper = env.network_mapper
    target = reward
    if not is_done:
        q_next_values = network_mapper.perform_prediction(next_agent.model)[0]
        target += discount_factor* np.amax(q_next_values)

    q_value_current = current_agent.current_q_values
    q_value_current[current_action] = target
    network_mapper.teach_model(current_agent.model, current_agent.current_input, q_value_current)

def perform_tic_tac_toe_training():
    training_instance_iterations = 100_000

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

    for episode in range(training_context.num_iterations,training_instance_iterations+1):
        current_agent = new_game()
        games_played += 1
        is_not_finished = not env.is_game_over()
        while is_not_finished:
            total_move_count+=1
            action = current_agent.act(env)
            next_agent = get_other_agent(current_agent)
            valid_selection = env.update_with_action(action, current_agent.agent_symbol)
            teach_current_agent(current_agent, next_agent, action,valid_selection,env)
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

        if episode > 0 and episode%10 ==0:
            clear_console()
            training_log = f"Episode {episode}. Elapsed time {training_context.elapsed_time_in_seconds:.0f} Epsilon {training_context.epsilon} Avg. Invalid Moves: {invalid_move_count}/{total_move_count} = {100*invalid_move_count/total_move_count:.2f} Avg. Turns = {total_move_count}/ {games_played} = {total_move_count/games_played:.2f} Agent Stats [{first_agent.to_string()} & {second_agent.to_string()}] "
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
        # if episode > 0 and episode %100 ==0:
        #     create_memento(agent, training_context)

if __name__ == "__main__":
    perform_tic_tac_toe_training()