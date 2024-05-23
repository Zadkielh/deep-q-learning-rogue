from game import V_HEIGHT, V_WIDTH
import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque
from environment import RogueEnvironment
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns

class DQN(nn.Module):
    def __init__(self, map_input_dim, tile_input_dim, output_dim):
        super(DQN, self).__init__()
        # Convolutional layers for the full map
        self.conv1 = nn.Conv2d(3, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        def conv2d_size_out(size, kernel_size=8, stride=4):
            return (size - (kernel_size - 1) - 1) // stride + 1

        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(map_input_dim[1], 8, 4), 4, 2), 3, 1)
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(map_input_dim[0], 8, 4), 4, 2), 3, 1)
        linear_input_size = convw * convh * 64

        self.map_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(linear_input_size, 256),
            nn.ReLU()
        )
        
        # Fully connected layers for the neighboring tile features
        self.tile_fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(tile_input_dim[0] * tile_input_dim[1], 256),
            nn.ReLU()
        )
        
        # Combining both inputs
        self.combined_fc = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim)
        )

    def forward(self, map_input, tile_input):
        map_output = F.relu(self.conv1(map_input))
        map_output = F.relu(self.conv2(map_output))
        map_output = F.relu(self.conv3(map_output))
        map_output = self.map_head(map_output)
        
        tile_output = self.tile_fc(tile_input)
        
        combined = torch.cat((map_output, tile_output), dim=1)
        return self.combined_fc(combined)
    
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class DQNAgent:
    def __init__(self, map_input_dim, tile_input_dim, output_dim):
        self.model = DQN(map_input_dim, tile_input_dim, output_dim).to(device)
        self.target_model = DQN(map_input_dim, tile_input_dim, output_dim).to(device)
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model.eval()

        self.memory = deque(maxlen=6000)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.0001)
        self.criterion = nn.MSELoss()
        self.gamma = 0.99
        self.batch_size = 128
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.1

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        map_input, tile_input = state
        if np.random.rand() <= self.epsilon:
            return random.randrange(4)
        map_input = torch.FloatTensor(map_input).unsqueeze(0).to(device)
        tile_input = torch.FloatTensor(tile_input).unsqueeze(0).to(device)
        with torch.no_grad():
            q_values = self.model(map_input, tile_input)
        return torch.argmax(q_values[0]).item()

    def replay(self):
        if len(self.memory) < self.batch_size:
            return
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        map_states = torch.FloatTensor(np.array([s[0] for s in states])).to(device)
        tile_states = torch.FloatTensor(np.array([s[1] for s in states])).to(device)
        actions = torch.LongTensor(np.array(actions)).to(device)
        rewards = torch.FloatTensor(np.array(rewards)).to(device)
        map_next_states = torch.FloatTensor(np.array([s[0] for s in next_states])).to(device)
        tile_next_states = torch.FloatTensor(np.array([s[1] for s in next_states])).to(device)
        dones = torch.FloatTensor(np.array(dones)).to(device)

        q_values = self.model(map_states, tile_states)
        next_q_values = self.target_model(map_next_states, tile_next_states)

        q_values = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
        next_q_values = next_q_values.max(1)[0]
        target_q_values = rewards + (1 - dones) * self.gamma * next_q_values

        loss = self.criterion(q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())


def visualize_q_values(agent, state):
    map_input, tile_input = state
    map_input = torch.FloatTensor(map_input).unsqueeze(0).to(device)
    tile_input = torch.FloatTensor(tile_input).unsqueeze(0).to(device)
    with torch.no_grad():
        q_values = agent.model(map_input, tile_input)
    q_values = q_values[0].cpu().numpy()

    sns.heatmap(q_values.reshape((1, -1)), annot=True, cmap="coolwarm")
    plt.xlabel('Actions')
    plt.ylabel('Q-value')
    plt.title('Q-values for Current State')
    plt.show()

# Training the agent

map_input_dim = (V_HEIGHT, V_WIDTH, 3)  # Height, Width, Channels for the game map
tile_input_dim = (4, 3)  # 4 neighboring tiles, each with 3 features (tile type, entity present, entity type)
output_dim = 4  # Number of possible actions

env = RogueEnvironment()
print("Environment initialized")
agent = DQNAgent(map_input_dim, tile_input_dim, output_dim)
print("Agent initialized")
episodes = 100

for e in range(episodes):
    print(f"Starting episode {e}")
    state = env.reset()
    full_map, neighbors = state
    full_map = np.transpose(full_map, (2, 0, 1))  # For channels-first format expected by Conv2D
    state = (full_map, neighbors)
    done = False
    total_reward = 0
    step_count = 0

    while not done:
        action = agent.act(state)
        print(f"Episode: {e}, Step: {step_count}, Action: {action}, Total Reward: {total_reward}")
        next_state, reward, done = env.step(action)
        next_full_map, next_neighbors = next_state
        next_full_map = np.transpose(next_full_map, (2, 0, 1))
        next_state = (next_full_map, next_neighbors)
        agent.remember(state, action, reward, next_state, done)
        state = next_state
        total_reward += reward
        agent.replay()

        env.render()
        step_count += 1

    agent.update_target_model()
    print(f"Episode: {e}/{episodes}, Total Reward: {total_reward}, Epsilon: {agent.epsilon:.2f}")

    if e % 10 == 0:
        torch.save(agent.model.state_dict(), f"dqn_model_{e}.pth")
        visualize_q_values(agent, state)
