from ..setup import *
from config import *
from learn import *
from mdp import QuantumStateCircuitEnv
from agent import DQNAgent

env = QuantumStateCircuitEnv(
    n_qubits=N_QUBITS_GHZ, 
    target_state=TARGET_STATE_VECTOR, 
    possible_actions=POSSIBLE_ACTIONS_GHZ,
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

best_is_loaded = load_agent(agent, BEST_GHZ_PATH, DEVICE)
initial_is_loaded = load_agent(agent, INITIAL_GHZ_PATH, DEVICE)
history_is_loaded = os.path.exists(HISTORY_GHZ_PATH)

if not best_is_loaded or not initial_is_loaded or not history_is_loaded:
    raise ValueError("Failed to load GHZ agent or history.")

def eval_policy(agent, env, model_path='untrained'):
    """
    Loads agent and runs a single greedy episode.
    """
    if model_path != 'untrained':
        try:
            agent.policy_net.load_state_dict(torch.load(model_path, map_location=DEVICE))
            agent.policy_net.eval()
        except FileNotFoundError:
            print(f"{model_path} not found.")
            return None, None, None 
               
    state = env.reset().to(DEVICE)
    final_circuit = env.current_circuit
    
    with torch.no_grad():
        done = False
        while not done:
            action_idx = agent.select_action(state.unsqueeze(0), epsilon=0.0) 
            
            next_state_tensor, _, done = env.step(action_idx)
            state = next_state_tensor.to(DEVICE)
    
    final_fidelity = state_fidelity(env.TARGET_STATE_VECTOR, Statevector(final_circuit))
    final_depth = final_circuit.depth()
    
    return final_circuit, final_depth, final_fidelity


print("\nGround Truth Circuit:")

truth = QuantumCircuit(N_QUBITS_GHZ)
truth.h(0)
truth.cx(0, 1)
truth.cx(0, 2)

fidelity_truth = state_fidelity(env.TARGET_STATE_VECTOR, Statevector(truth))

print(f'Ground truth depth: {truth.depth()}')
print(f'Ground truth fidelity: {fidelity_truth:.4f}')
print(truth.draw())

print("\nInitial Agent Circuit (before learning):")

initial_circuit, initial_depth, initial_fidelity = eval_policy(agent, env, model_path=INITIAL_GHZ_PATH)

print(f'Initial policy depth: {initial_depth}')
print(f'Initial policy fidelity: {initial_fidelity:.4f}')
print(initial_circuit.draw())

print("\nBest Agent Circuit (after learning):")

best_circuit, best_depth, best_fidelity = eval_policy(agent, env, model_path=BEST_GHZ_PATH)

print(f'Best policy depth: {best_depth}')
print(f'Best policy fidelity: {best_fidelity:.4f}')
print(best_circuit.draw())