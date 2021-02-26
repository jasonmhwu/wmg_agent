# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

###  CONTROLS  (non-tunable)  ###

# general
TYPE_OF_RUN = train  # train, test, test_episodes, render
LOAD_MODEL_FROM = None
SAVE_MODELS_TO = models/new_wmg_Collins2018.pth

# worker.py
ENV = Collins2018_Env
ENV_RANDOM_SEED = randint  # Use an integer for deterministic training.
AGENT_RANDOM_SEED = 1
REPORTING_INTERVAL = 1_000
TOTAL_STEPS = 2_000_000
ANNEAL_LR = False

# A3cAgent
AGENT_NET = WMG_Network

# WMG
V2 = False

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
LEARNING_RATE = 0.00016
DISCOUNT_FACTOR = 0.5
GRADIENT_CLIP = 16.0
ENTROPY_TERM_STRENGTH = 0.01
ADAM_EPS = 1e-06
REWARD_SCALE = 2.0
WEIGHT_DECAY = 0.

# WMG
WMG_MAX_OBS = 0
WMG_MAX_MEMOS = 16
WMG_MEMO_SIZE = 50
WMG_NUM_LAYERS = 4
WMG_NUM_ATTENTION_HEADS = 6
WMG_ATTENTION_HEAD_SIZE = 12
WMG_HIDDEN_SIZE = 12
AC_HIDDEN_LAYER_SIZE = 50
