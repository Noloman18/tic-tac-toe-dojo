import math

import numpy as np

import tensorflow as tf
from tensorflow.python.client import device_lib

from TicTacToe import play_tic_tac_toe

if __name__ == '__main__':
    # Start the game
    # play_tic_tac_toe()
    # perform_tic_tac_toe_training()
    print(tf.version.VERSION)
    print(device_lib.list_local_devices())
    # play_simulation()


