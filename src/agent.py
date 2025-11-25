from config import *

class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        '''
        Initializes the Deep Q-Network structure.
        '''
        super(QNetwork, self).__init__()
        
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_dim)
         
        self.relu = nn.ReLU()

    def forward(self, state):
        '''
        Forward pass to compute Q-values for all actions in a given state.
        '''
        x = self.relu(self.fc1(state))
        x = self.relu(self.fc2(x))
        return self.fc3(x)
    
Experience = namedtuple('Experience', ('state', 'action', 'reward', 'next_state', 'done'))

class ReplayBuffer:
    def __init__(self, capacity):
        '''
        Initializes the Replay Buffer with a fixed capacity.
        '''
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        '''
        Saves a transition (s, a, r, s', done) to the buffer.
        '''
        self.memory.append(Experience(*args))

    def sample(self, batch_size):
        '''
        Randomly samples a batch of transitions for learning.
        '''
        if len(self.memory) < batch_size:
            return None

        transitions = random.sample(self.memory, batch_size)
        
        batch = Experience(*zip(*transitions))
        
        state_batch = torch.stack(batch.state)
        action_batch = torch.tensor(batch.action, dtype=torch.long).unsqueeze(1)
        reward_batch = torch.tensor(batch.reward, dtype=torch.float32).unsqueeze(1)
        next_state_batch = torch.stack(batch.next_state)
        done_batch = torch.tensor(batch.done, dtype=torch.float32).unsqueeze(1)
        
        return state_batch, action_batch, reward_batch, next_state_batch, done_batch

    def __len__(self):
        return len(self.memory)

class DQNAgent:
    def __init__(self, state_dim, action_dim, device, gamma, lr, capacity):
        '''
        Initializes the DQN Agent with policy and target networks, optimizer, and replay buffer.
        '''
        self.DEVICE = device
        self.GAMMA = gamma
        
        self.policy_net = QNetwork(state_dim, action_dim).to(device)
        
        self.target_net = QNetwork(state_dim, action_dim).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.memory = ReplayBuffer(capacity)
        
        self.steps_done = 0
    
    def select_action(self, state, epsilon):
        '''
        Selects an action using the epsilon-greedy strategy.
        '''
        if random.random() < epsilon:
            return random.randrange(self.policy_net.fc3.out_features)
        else:
            with torch.no_grad():
                return self.policy_net(state).argmax(1).item()

    def learn(self, batch_size):
        '''
        Performs one optimization step on the policy network using a batch from replay memory.
        '''
        batch = self.memory.sample(batch_size)
        if batch is None:
            return 
        
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = batch
        
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        next_state_values = self.target_net(next_state_batch).max(1)[0].unsqueeze(1)
        next_state_values = next_state_values * (1 - done_batch)
        
        expected_state_action_values = reward_batch + (self.GAMMA * next_state_values)

        loss = F.huber_loss(state_action_values, expected_state_action_values)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()
        
        return loss.item()