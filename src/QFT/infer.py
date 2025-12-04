from ..setup import *
from config import *
from learn import *
from mdp import QuantumOperatorCircuitEnv
from agent import DQNAgent

env = QuantumOperatorCircuitEnv(
    n_qubits=N_QUBITS_QFT, 
    target_unitary=TARGET_UNITARY_OPERATOR, 
    possible_actions=POSSIBLE_ACTIONS_QFT,
    REWARD_GATE_PENALTY=REWARD_GATE_PENALTY,
    REWARD_SUCCESS=REWARD_SUCCESS,
    FIDELITY_THRESHOLD=FIDELITY_THRESHOLD,
    REWARD_FIDELITY_WEIGHT=REWARD_FIDELITY_WEIGHT,
    max_gates=20)

STATE_DIM = env.STATE_DIM
N_ACTIONS = env.N_ACTIONS

agent = DQNAgent(
    state_dim=STATE_DIM, 
    action_dim=N_ACTIONS, 
    device=DEVICE, 
    gamma=GAMMA, 
    lr=LEARNING_RATE, 
    capacity=REPLAY_CAPACITY)

best_is_loaded = load_agent(agent, BEST_QFT_PATH, DEVICE)
initial_is_loaded = load_agent(agent, INITIAL_QFT_PATH, DEVICE)
history_is_loaded = os.path.exists(HISTORY_QFT_PATH)

if not best_is_loaded or not initial_is_loaded or not history_is_loaded:
    raise ValueError("Failed to load QFT agent or history.")

def eval_policy(agent, env, model_path='untrained'):
    """
    Loads agent and runs a single greedy episode on the Unitary Synthesis environment (QFT).
    """
    if model_path != 'untrained':
        try:
            agent.policy_net.load_state_dict(torch.load(model_path, map_location=DEVICE))
            agent.policy_net.eval()
        except FileNotFoundError:
            print(f"{model_path} not found.")
            return None, None, None 
        except AttributeError as e:
            print(f"AttributeError loading model: {e}")
            return None, None, None
               
    state = env.reset().to(DEVICE)
    
    with torch.no_grad():
        done = False
        while not done:
            action_idx = agent.select_action(state.unsqueeze(0), epsilon=0.0) 
            
            next_state_tensor, _, done = env.step(action_idx)
            state = next_state_tensor.to(DEVICE)
    
    final_circuit = env.current_circuit
    final_unitary = Operator(final_circuit)
    
    d = 2**env.N_QUBITS
    U_target_dag = env.TARGET_UNITARY_OPERATOR.data.conj().T
    dot_product = np.trace(U_target_dag @ final_unitary.data)
    final_fidelity = (np.abs(dot_product)**2) / (d**2)
    
    final_depth = final_circuit.depth()
    
    return final_circuit, final_depth, final_fidelity

print("\nGround Truth Circuit:")

truth = QFT(N_QUBITS_QFT, do_swaps=False)
truth_decomposed = truth.decompose()

d = 2**N_QUBITS_QFT
U_target_dag = TARGET_UNITARY_OPERATOR.data.conj().T
dot_product = np.trace(U_target_dag @ TARGET_UNITARY_OPERATOR.data)
fidelity = (np.abs(dot_product)**2) / (d**2)

print(f'Ground truth depth: {truth_decomposed.depth()}')
print(f'Ground truth number of gates: {truth_decomposed.size()}')
print(f'Ground truth fidelity: {fidelity:.4f}')
print(truth_decomposed.draw())

print("\nInitial Agent Circuit (before learning):")

initial_circuit, initial_depth, initial_fidelity = eval_policy(agent, env, model_path=INITIAL_QFT_PATH)

print(f'Initial policy depth: {initial_depth}')
print(f'Initial policy number of gates: {initial_circuit.size()}')
print(f'Initial policy fidelity: {initial_fidelity:.4f}')
print(initial_circuit.draw())

print("\nBest Agent Circuit (after learning):")

best_circuit, best_depth, best_fidelity = eval_policy(agent, env, model_path=BEST_QFT_PATH)

print(f'Best policy depth: {best_depth}')
print(f'Best policy number of gates: {best_circuit.size()}')
print(f'Best policy fidelity: {best_fidelity:.4f}')
print(best_circuit.draw())