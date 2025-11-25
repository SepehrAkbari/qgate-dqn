import random
import os
import time
import warnings
from collections import deque, namedtuple

import numpy as np
import seaborn as sns
import pylatexenc
import matplotlib.pyplot as plt

import qiskit
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Statevector, state_fidelity, Pauli

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

warnings.filterwarnings("ignore")

N_QUBITS = 3

TARGET_STATE_VECTOR = Statevector(
    [1/np.sqrt(2), 0, 0, 0, 0, 0, 0, 1/np.sqrt(2)]
) 

GATE_SET = [
    ('h', 1),
    ('rx', 1),
    ('rz', 1),
    ('x', 1),
    ('cx', 2)
]

POSSIBLE_ACTIONS = []
for name, num_q in GATE_SET:
    if num_q == 1:
        for q in range(N_QUBITS):
            POSSIBLE_ACTIONS.append((name, q, None))
    elif num_q == 2:
        for control in range(N_QUBITS):
            for target in range(N_QUBITS):
                if control != target:
                    POSSIBLE_ACTIONS.append((name, target, control))

N_ACTIONS = len(POSSIBLE_ACTIONS)