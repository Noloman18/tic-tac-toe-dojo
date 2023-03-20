# Tic Tac Toe game in Python

board = [' ' for x in range(10)]

def initialize_board():
    for x in range(10):
        board[x] =' '
#Initialize it below.....
def insert_letter(letter, pos):
    board[pos] = letter

def space_is_free(pos):
    return board[pos] == ' '

def print_board(board):
    print("   |   |   ")
    print(" "+board[1]+" | "+board[2]+" | "+board[3]+" ")
    print("   |   |   ")
    print("-----------")
    print("   |   |   ")
    print(" "+board[4]+" | "+board[5]+" | "+board[6]+" ")
    print("   |   |   ")
    print("-----------")
    print("   |   |   ")
    print(" "+board[7]+" | "+board[8]+" | "+board[9]+" ")
    print("   |   |   ")

def is_winner(board, letter):
    return (board[1] == letter and board[2] == letter and board[3] == letter) or \
           (board[4] == letter and board[5] == letter and board[6] == letter) or \
           (board[7] == letter and board[8] == letter and board[9] == letter) or \
           (board[1] == letter and board[4] == letter and board[7] == letter) or \
           (board[2] == letter and board[5] == letter and board[8] == letter) or \
           (board[3] == letter and board[6] == letter and board[9] == letter) or \
           (board[1] == letter and board[5] == letter and board[9] == letter) or \
           (board[3] == letter and board[5] == letter and board[7] == letter)

def player_move():
    run = True
    while run:
        move = input("Please select a position to place an 'X' (1-9): ")
        try:
            move = int(move)
            if move > 0 and move < 10:
                if space_is_free(move):
                    run = False
                    insert_letter('X', move)
                else:
                    print("Sorry, this space is occupied!")
            else:
                print("Please type a number within the range!")
        except:
            print("Please type a number!")

def computer_move():
    possible_moves = [x for x, letter in enumerate(board) if letter == ' ' and x != 0]
    move = 0

    for letter in ['O', 'X']:
        for i in possible_moves:
            board_copy = board[:]
            board_copy[i] = letter
            if is_winner(board_copy, letter):
                move = i
                return move

    corners_open = []
    for i in possible_moves:
        if i in [1, 3, 7, 9]:
            corners_open.append(i)
    if len(corners_open) > 0:
        move = select_random(corners_open)
        return move

    if 5 in possible_moves:
        move = 5
        return move

    edges_open = []
    for i in possible_moves:
        if i in [2, 4, 6, 8]:
            edges_open.append(i)
    if len(edges_open) > 0:
        move = select_random(edges_open)
        return move

    return move

def select_random(lst):
    import random
    ln = len(lst)
    r = random.randrange(0, ln)
    return lst[r]

def play_tic_tac_toe():
    initialize_board()
    print("Welcome to Tic Tac Toe!")
    print_board(board)

    while True:
        # Player's move
        player_move()
        print_board(board)
        if is_winner(board, 'X'):
            print("Congratulations! You have won the game!")
            break
        if ' ' not in board:
            print("The game is a tie!")
            break

        # Computer's move
        move = computer_move()
        if move == 0:
            print("Tie game!")
        else:
            insert_letter('O', move)
            print("Computer placed an 'O' in position", move)
            print_board(board)
            if is_winner(board, 'O'):
                print("Sorry, the computer has won the game.")
                break
            if ' ' not in board:
                print("The game is a tie!")
                break


