from ..setup import *

N_QUBITS_QFT = 4

TARGET_UNITARY_OPERATOR = Operator(QFT(N_QUBITS_QFT, do_swaps=False))
TARGET_OBJECT = TARGET_UNITARY_OPERATOR

POSSIBLE_ACTIONS_QFT = []
for name, num_q in GATE_SET:
    if num_q == 1:
        for q in range(N_QUBITS_QFT):
            POSSIBLE_ACTIONS_QFT.append((name, q, None))
    elif num_q == 2:
        for control in range(N_QUBITS_QFT):
            for target in range(N_QUBITS_QFT):
                if control != target:
                    POSSIBLE_ACTIONS_QFT.append((name, target, control))
                    
N_ACTIONS_QFT = len(POSSIBLE_ACTIONS_QFT)