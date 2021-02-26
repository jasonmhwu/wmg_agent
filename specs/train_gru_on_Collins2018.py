# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

###  CONTROLS  (non-tunable)  ###

# general
TYPE_OF_RUN = train  # train, test, test_episodes, render
LOAD_MODEL_FROM = None
SAVE_MODELS_TO = models/new_gru_Collins2018.pth

# worker.py
ENV = Collins2018_Env
ENV_RANDOM_SEED = randint  # Use an integer for deterministic training.
AGENT_RANDOM_SEED = 1
REPORTING_INTERVAL = 10_000
TOTAL_STEPS = 20_000_000
ANNEAL_LR = False

# A3cAgent
AGENT_NET = GRU_Network

# Collins2018_Env
MIN_NUM_OBJECTS = 3
MAX_NUM_OBJECTS = 4
NUM_ACTIONS = 3
NUM_REPEATS = 13
MAX_OBSERVATIONS = 4
ONE_HOT_PATTERNS = False
ALLOW_DELIBERATION = False
USE_SUCCESS_RATE = False
HELDOUT_TESTING = False

###  HYPERPARAMETERS  (tunable)  ###

# A3cAgent
A3C_T_MAX = 16
LEARNING_RATE = 0.0001
DISCOUNT_FACTOR = 0.5
GRADIENT_CLIP = 4.0
ENTROPY_TERM_STRENGTH = 0.02
ADAM_EPS = 1e-08
REWARD_SCALE = 0.5
WEIGHT_DECAY = 0.


# RNNs
NUM_RNN_UNITS = 64
OBS_EMBED_SIZE = 16
AC_HIDDEN_LAYER_SIZE = 64
