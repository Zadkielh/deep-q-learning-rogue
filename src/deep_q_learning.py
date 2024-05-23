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
import cProfile
import pstats

class DQN(nn.Module):
    def __init__(self, map_input_dim, tile_input_dim, output_dim):
        super(DQN, self).__init__()

        map_input_size = map_input_dim[0] * map_input_dim[1]
        
        self.map_fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(map_input_size, 256),
            nn.ReLU()
        )
        
        
        self.tile_fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(tile_input_dim[0] * tile_input_dim[1], 256),
            nn.ReLU()
        )

        self.combined_fc = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim)
        )

    def forward(self, map_input, tile_input):
        map_output = self.map_fc(map_input)
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

        self.memory = deque(maxlen=10000)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.0001)
        self.criterion = nn.MSELoss()
        self.gamma = 0.99
        self.batch_size = 128
        self.epsilon = 1.0
        self.epsilon_decay = 0.9999
        self.epsilon_min = 0.1

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        map_input, tile_input = state
        #print("Tile: ", tile_input)
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

        # Split state into map_states and tile_states for current states and next states
        map_states = np.array([s[0] for s in states])
        tile_states = np.array([s[1] for s in states])
        map_next_states = np.array([s[0] for s in next_states])
        tile_next_states = np.array([s[1] for s in next_states])
        
        # Convert to tensors in a single operation
        map_states = torch.FloatTensor(map_states).to(device)
        tile_states = torch.FloatTensor(tile_states).to(device)
        actions = torch.LongTensor(actions).to(device)
        rewards = torch.FloatTensor(rewards).to(device)
        map_next_states = torch.FloatTensor(map_next_states).to(device)
        tile_next_states = torch.FloatTensor(tile_next_states).to(device)
        dones = torch.FloatTensor(dones).to(device)

        q_values = self.model(map_states, tile_states)  # Feed both map and tile inputs to the model
        next_q_values = self.target_model(map_next_states, tile_next_states)  # Feed both map and tile inputs to the target model

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

# Training the agent
def run_training():
    map_input_dim = (V_HEIGHT, V_WIDTH)  # Height, Width, Channels for the game map
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
        state = (full_map, neighbors)
        done = False
        total_reward = 0
        step_count = 0

        while not done:
            action = agent.act(state)
            print(f"Episode: {e}, Step: {step_count}, Action: {action}, Total Reward: {total_reward}, Epsilon: {agent.epsilon:.2f}")
            next_state, reward, done = env.step(action)
            next_full_map, next_neighbors = next_state
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

# Profile the run_training function
cProfile.run('run_training()', 'profile_output')

# Analyze the profiling results
with open('profile_output.txt', 'w') as f:
    p = pstats.Stats('profile_output', stream=f)
    p.sort_stats('cumulative').print_stats(50)  # Print top 50 cumulative time functions