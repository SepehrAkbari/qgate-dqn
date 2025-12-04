from ..setup import *

N_QUBITS_GHZ = 3

TARGET_STATE_VECTOR = Statevector(
    [1/np.sqrt(2), 0, 0, 0, 0, 0, 0, 1/np.sqrt(2)]
) 

POSSIBLE_ACTIONS_GHZ = []

for name, num_q in GATE_SET:
    if num_q == 1:
        for q in range(N_QUBITS_GHZ):
            POSSIBLE_ACTIONS_GHZ.append((name, q, None))
    elif num_q == 2:
        for control in range(N_QUBITS_GHZ):
            for target in range(N_QUBITS_GHZ):
                if control != target:
                    POSSIBLE_ACTIONS_GHZ.append((name, target, control))

N_ACTIONS_GHZ = len(POSSIBLE_ACTIONS_GHZ)