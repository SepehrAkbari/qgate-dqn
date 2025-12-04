from ..setup import *

class QuantumStateCircuitEnv:
    def __init__(self, n_qubits, target_state, possible_actions, REWARD_GATE_PENALTY, REWARD_SUCCESS, FIDELITY_THRESHOLD, REWARD_FIDELITY_WEIGHT, max_gates=20):
        '''
        Initializes the Quantum Circuit MDP environment.
        '''
        self.N_QUBITS = n_qubits
        self.TARGET_STATE_VECTOR = target_state
        self.POSSIBLE_ACTIONS = possible_actions
        self.N_ACTIONS = len(possible_actions)
        self.MAX_GATES = max_gates
        
        self.STATE_DIM = n_qubits * 3 
        self.PAULI_OBSERVABLES = self._get_pauli_observables()
        
        self.last_fidelity = 0.0
        
        self.REWARD_GATE_PENALTY = REWARD_GATE_PENALTY
        self.REWARD_SUCCESS = REWARD_SUCCESS
        self.FIDELITY_THRESHOLD = FIDELITY_THRESHOLD
        self.REWARD_FIDELITY_WEIGHT = REWARD_FIDELITY_WEIGHT
        
        self.reset()
    
    def _get_pauli_observables(self):
        '''
        Generates the list of single-qubit Pauli observables for state representation.
        '''
        operators = []
        pauli_bases = ['X', 'Y', 'Z']
        
        for i in range(self.N_QUBITS):
            for basis in pauli_bases:
                pauli_string = ['I'] * self.N_QUBITS
                pauli_string[i] = basis
                pauli_string_qiskit = "".join(pauli_string[::-1])
                operators.append(Pauli(pauli_string_qiskit))
        return operators
    
    def reset(self):
        '''
        Resets the environment to the initial state.
        '''
        self.current_circuit = QuantumCircuit(self.N_QUBITS)
        
        self.gate_count = 0
        self.max_fidelity = 0.0
        self.last_fidelity = 0.0
        
        initial_state_vector = Statevector(self.current_circuit)
        return self._get_state_representation(initial_state_vector)
    
    def _get_state_representation(self, state_vector):
        '''
        Generates the state representation based on expectation values of Pauli observables.
        '''
        state_features = []
        for pauli_op in self.PAULI_OBSERVABLES:
            exp_val = state_vector.expectation_value(pauli_op)
            state_features.append(np.real(exp_val))
            
        return torch.tensor(state_features, dtype=torch.float32)
    

    def step(self, action_index):
        '''
        Represents one step in the MDP environment, including action execution, fidelity computation, and reward determination.
        '''
        gate_name, target_q, control_q = self.POSSIBLE_ACTIONS[action_index]
        
        if control_q is None:
            if gate_name == 'h':
                self.current_circuit.h(target_q)
            elif gate_name == 'x':
                self.current_circuit.x(target_q)
            elif gate_name == 'rx':
                self.current_circuit.rx(np.pi/2, target_q) 
            elif gate_name == 'rz':
                self.current_circuit.rz(np.pi/2, target_q)
        else:
            if gate_name == 'cx':
                self.current_circuit.cx(control_q, target_q)
        
        self.gate_count += 1
                
        current_state_vector = Statevector(self.current_circuit)
        fidelity = state_fidelity(self.TARGET_STATE_VECTOR, current_state_vector)
        
        self.max_fidelity = max(self.max_fidelity, fidelity)
        
        next_state = self._get_state_representation(current_state_vector)
        
        done = False
        reward = self.REWARD_GATE_PENALTY
        
        if fidelity >= self.FIDELITY_THRESHOLD:
            reward += self.REWARD_SUCCESS 
            done = True
        elif self.gate_count >= self.MAX_GATES:
            done = True
        
        reward += self.REWARD_FIDELITY_WEIGHT * (fidelity - self.last_fidelity)
        self.last_fidelity = fidelity

        return next_state, reward, done