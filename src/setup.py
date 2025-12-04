import random
import os
import time
import warnings
from pathlib import Path
from collections import deque, namedtuple

import numpy as np
import seaborn as sns
import pylatexenc
import matplotlib.pyplot as plt

import qiskit
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Statevector, state_fidelity, Pauli, Operator
from qiskit.circuit.library import QFT, QFTGate

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

warnings.filterwarnings("ignore")

ROOT_DIR = str(Path.cwd().parents[0])

MODEL_DIR = os.path.join(ROOT_DIR, "model")
os.makedirs(MODEL_DIR, exist_ok=True)

VECTOR_MODEL_DIR = f'{MODEL_DIR}/ghz_model'
os.makedirs(VECTOR_MODEL_DIR, exist_ok=True)

UNITARY_MODEL_DIR = f'{MODEL_DIR}/qft_model'
os.makedirs(UNITARY_MODEL_DIR, exist_ok=True)

BEST_GHZ_PATH = f'{VECTOR_MODEL_DIR}/ghz_agent_best.pth'
INITIAL_GHZ_PATH = f'{VECTOR_MODEL_DIR}/ghz_agent_initial.pth'
HISTORY_GHZ_PATH = f'{VECTOR_MODEL_DIR}/ghz_train_history.pth'

BEST_QFT_PATH = f'{UNITARY_MODEL_DIR}/qft_agent_best.pth'
INITIAL_QFT_PATH = f'{UNITARY_MODEL_DIR}/qft_agent_initial.pth'
HISTORY_QFT_PATH = f'{UNITARY_MODEL_DIR}/qft_train_history.pth'

GATE_SET = [
    ('h', 1),
    ('rx', 1),
    ('rz', 1),
    ('x', 1),
    ('cx', 2)
]