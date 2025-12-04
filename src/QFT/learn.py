from ..setup import *
from config import *
from agent import DQNAgent
from mdp import QuantumOperatorCircuitEnv

GAMMA = 0.99 # Discount factor for future rewards
LEARNING_RATE = 1e-4 # Learning rate for the Adam
BATCH_SIZE = 64 # Number of transitions sampled from the replay buffer
TARGET_UPDATE = 100 # Frequency (in steps) to update the target Q-network
REPLAY_CAPACITY = 10000 # Max size of the replay memory

EPS_START = 1.0 # Starting value of epsilon (100% exploration)
EPS_END = 0.1 # Final minimum value of epsilon
EPS_DECAY = 50000 # Number of steps over which epsilon decays linearly
N_EPISODES = 20000 # Total number of training episodes

REWARD_GATE_PENALTY = -0.1
REWARD_SUCCESS = 10.0
FIDELITY_THRESHOLD = 0.99
REWARD_FIDELITY_WEIGHT = 5.0

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

env = QuantumOperatorCircuitEnv(
    n_qubits=N_QUBITS_QFT, 
    target_unitary=TARGET_UNITARY_OPERATOR, 
    possible_actions=POSSIBLE_ACTIONS_QFT,
    REWARD_GATE_PENALTY=REWARD_GATE_PENALTY,
    REWARD_SUCCESS=REWARD_SUCCESS,
    FIDELITY_THRESHOLD=FIDELITY_THRESHOLD,
    REWARD_FIDELITY_WEIGHT=REWARD_FIDELITY_WEIGHT,
    max_gates=20)

STATE_DIM_GHZ = env.STATE_DIM
N_ACTIONS_GHZ = env.N_ACTIONS

agent = DQNAgent(
    state_dim=STATE_DIM_GHZ, 
    action_dim=N_ACTIONS_GHZ, 
    device=DEVICE, 
    gamma=GAMMA, 
    lr=LEARNING_RATE, 
    capacity=REPLAY_CAPACITY)

SCHEDULER_GAMMA = 0.99995 # Exponential decay rate for the Learning Rate
PATIENCE = 50 # Number of 100-episode windows to wait for reward improvement
MIN_AVG_REWARD = 14.0 # Minimum average reward threshold to consider successful

LR_SCHEDULER = optim.lr_scheduler.ExponentialLR(agent.optimizer, gamma=SCHEDULER_GAMMA)

def train_dqn(n_episodes, env, agent, initial_model_path, best_model_path, train_history_path):
    '''
    Trains the agent in the environment, returning rewards and final depths, and saving the best model.
    '''
    all_rewards = []
    final_depths = []
    reward_window = deque(maxlen=100)

    best_avg_reward = -np.inf
    patience_counter = 0 
        
    for episode in range(1, n_episodes + 1):
        
        if episode == 1 and not os.path.exists(initial_model_path):
            os.makedirs(os.path.dirname(initial_model_path), exist_ok=True)

            torch.save(agent.policy_net.state_dict(), initial_model_path)
                        
        state = env.reset().to(DEVICE)
        done = False
        total_reward = 0
        
        while not done:
            epsilon = max(EPS_END, EPS_START - (agent.steps_done / EPS_DECAY))
            
            action = agent.select_action(state.unsqueeze(0), epsilon)
            next_state_tensor, reward, done = env.step(action)
            next_state = next_state_tensor.to(DEVICE)
            
            total_reward += reward
            agent.steps_done += 1
            
            agent.memory.push(state, action, reward, next_state, done)
            state = next_state
            
            if len(agent.memory) > BATCH_SIZE:
                agent.learn(BATCH_SIZE)
                
            if agent.steps_done % TARGET_UPDATE == 0:
                agent.target_net.load_state_dict(agent.policy_net.state_dict())

        LR_SCHEDULER.step() 
        
        all_rewards.append(total_reward)
        final_depths.append(env.gate_count)
        reward_window.append(total_reward)
        
        if episode % 100 == 0:
            avg_reward = np.mean(reward_window)
            current_lr = agent.optimizer.param_groups[0]['lr']
            
            print(f"Episode {episode:6d}/{n_episodes} | "
                  f"Avg Reward: {avg_reward:7.2f} | "
                  f"Max F: {env.max_fidelity:.4f} | "
                  f"Depth: {env.gate_count:3d} | "
                  f"LR: {current_lr:.2e} | "
                  f"Eps: {epsilon:.3f}")

            if avg_reward > best_avg_reward:
                best_avg_reward = avg_reward
                patience_counter = 0
                torch.save(agent.policy_net.state_dict(), best_model_path)
            else:
                patience_counter += 1
                
            if avg_reward >= MIN_AVG_REWARD or patience_counter >= PATIENCE:
                if avg_reward >= MIN_AVG_REWARD:
                    print(f"\nStopping early: target average reward ({MIN_AVG_REWARD}) reached.")
                elif patience_counter >= PATIENCE:
                    print(f"\nStopping: average reward has not improved for {PATIENCE*100} episodes.")
                break 
    
    history = {
        'rewards': all_rewards,
        'depths': final_depths
    }
    torch.save(history, train_history_path)
                            
    return all_rewards, final_depths

def load_agent(agent, model_path, device):
    """
    Checks for a saved model checkpoint and loads its state dict into the agent's policy net.
    """
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    if os.path.exists(model_path):
        try:
            agent.policy_net.load_state_dict(torch.load(model_path, map_location=device))
            agent.target_net.load_state_dict(agent.policy_net.state_dict())
            
            agent.policy_net.eval()
            agent.target_net.eval()
            
            return True
        
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
    else:
        return False
    
best_is_loaded = load_agent(agent, BEST_GHZ_PATH, DEVICE)
initial_is_loaded = load_agent(agent, INITIAL_GHZ_PATH, DEVICE)
history_is_loaded = os.path.exists(HISTORY_GHZ_PATH)

if not best_is_loaded or not initial_is_loaded or not history_is_loaded:
    rewards, depths = train_dqn(N_EPISODES, env, agent, INITIAL_GHZ_PATH, BEST_GHZ_PATH, HISTORY_GHZ_PATH)
else:
    print("Training skipped.")
    
history_data = torch.load(HISTORY_GHZ_PATH)
rewards = history_data['rewards']
depths = history_data['depths']

def plot_training(data, window_size=100, title=""):
    '''
    Plots the raw data and its moving average.
    '''
    window = np.ones(window_size) / window_size
    smoothed_data = np.convolve(data, window, 'valid')
    
    plt.figure(figsize=(12, 5))
    
    plt.plot(data, alpha=0.3, color="tab:blue")
    plt.plot(smoothed_data, color='tab:red')
    plt.title(title)
    plt.xlabel("Episode")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()
    
plot_training(rewards, window_size=100, title="Rewards Over Time")
plot_training(depths, window_size=100, title="Depths Over Time")