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
import pandas as pd

class DQN(nn.Module):
    def __init__(self, tile_input_dim, output_dim):
        super(DQN, self).__init__()
        
        self.fc1 = nn.Linear(tile_input_dim[0] * tile_input_dim[1], 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output_dim)

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten the input
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)
    
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class DQNAgent:
    def __init__(self, tile_input_dim, output_dim):
        self.model = DQN(tile_input_dim, output_dim).to(device)
        self.target_model = DQN(tile_input_dim, output_dim).to(device)
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model.eval()

        self.memory = deque(maxlen=100000)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.0001)
        self.criterion = nn.MSELoss()
        self.gamma = 0.99
        self.batch_size = 256
        self.epsilon = 1.0
        self.epsilon_decay = 0.9999
        self.epsilon_min = 0.1

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        if np.random.rand() <= self.epsilon:
            return random.randrange(4)
        with torch.no_grad():
            q_values = self.model(state)
        
        return torch.argmax(q_values[0]).item()

    def replay(self):
        if len(self.memory) < self.batch_size:
            return
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(states).to(device)
        actions = torch.LongTensor(actions).to(device)
        rewards = torch.FloatTensor(rewards).to(device)
        next_states = torch.FloatTensor(next_states).to(device)
        dones = torch.FloatTensor(dones).to(device)

        q_values = self.model(states)
        next_q_values = self.target_model(next_states)

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

def create_test_state(up_tile, down_tile, left_tile, right_tile):
    neighbors = np.array([
        up_tile,    # Up
        down_tile,  # Down
        left_tile,  # Left
        right_tile  # Right
    ])
    return (neighbors)

# Define test states with different tile configurations
void_tile = [0, 0, -1, 1, 1, 1] 
floor_tile = [1, 0, -1, 0, 0, 0] 
wall_tile = [2, 0, -1, 1, 1, 1]  
door_tile = [4, 0, -1, 0, 0, 0] 
tunnel_tile = [5, 0, -1, 0, 0, 0]  
goal_tile = [6, 0, -1, 0, 0, 0] 

visited_floor_tile = [1, 0, -1, 1, 0, 0]
thresh_floor_tile = [1, 0, -1, 1, 1, 0]
last_v_floor_tile = [1, 0, -1, 1, 0, 1]
last_vt_floor_tile = [1, 0, -1, 1, 1, 1]

test_states = [
    create_test_state(void_tile, void_tile, void_tile, void_tile),
    create_test_state(floor_tile, floor_tile, floor_tile, floor_tile),
    create_test_state(visited_floor_tile, visited_floor_tile, visited_floor_tile, visited_floor_tile),
    create_test_state(thresh_floor_tile, thresh_floor_tile, thresh_floor_tile, thresh_floor_tile),

    create_test_state(last_v_floor_tile, last_v_floor_tile, last_v_floor_tile, last_v_floor_tile),
    create_test_state(last_vt_floor_tile, last_vt_floor_tile, last_vt_floor_tile, last_vt_floor_tile),

    create_test_state(wall_tile, wall_tile, wall_tile, wall_tile),
    create_test_state(door_tile, door_tile, door_tile, door_tile),
    create_test_state(tunnel_tile, tunnel_tile, tunnel_tile, tunnel_tile),
    create_test_state(goal_tile, goal_tile, goal_tile, goal_tile),

    create_test_state(wall_tile, floor_tile, floor_tile, wall_tile),
    create_test_state(floor_tile, wall_tile, wall_tile, floor_tile),

    create_test_state(wall_tile, door_tile, floor_tile, tunnel_tile),
    create_test_state(floor_tile, tunnel_tile, tunnel_tile, floor_tile),

    create_test_state(tunnel_tile, tunnel_tile, void_tile, void_tile),

    create_test_state(wall_tile, floor_tile, door_tile, goal_tile),
]

# Training the agent
def run_training():
    map_input_dim = (V_HEIGHT, V_WIDTH)  # Height, Width, Channels for the game map
    tile_input_dim = (4, 6)  # 4 neighboring tiles, each with 3 features (tile type, entity present, entity type)
    output_dim = 4  # Number of possible actions

    env = RogueEnvironment()
    print("Environment initialized")
    agent = DQNAgent(tile_input_dim, output_dim)
    print("Agent initialized")
    episodes = 1000

    for e in range(episodes):
        print(f"Starting episode {e}")
        state = env.reset()
        done = False
        total_reward = 0
        step_count = 0

        while not done:
            action = agent.act(state)
            print(f"Episode: {e}, Step: {step_count}, Action: {action}, Total Reward: {total_reward}, Epsilon: {agent.epsilon:.2f}")
            next_state, reward, done = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            agent.replay()

            env.render()
            step_count += 1

        agent.update_target_model()
        print(f"Episode: {e}/{episodes}, Total Reward: {total_reward}, Epsilon: {agent.epsilon:.2f}")

        #if e % 10 == 0:
            #torch.save(agent.model.state_dict(), f"dqn_model_{e}.pth")

    # Print Q-values for each test state in a table format
    q_values_list = []
    for i, state in enumerate(test_states):
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
        q_values = agent.model(state_tensor).cpu().detach().numpy().flatten()
        q_values_list.append(q_values)

    # Create a DataFrame to display Q-values
    df = pd.DataFrame(q_values_list, columns=['Up', 'Down', 'Left', 'Right'])
    df.index.name = 'Test State'
    print(df)



class SimpleTestEnvironment:
    def __init__(self):
        self.map_grid = np.array([
            [2, 2, 2, 2, 2],
            [2, 1, 1, 1, 2],
            [2, 1, 1, 1, 2],
            [2, 1, 1, 1, 2],
            [2, 2, 2, 2, 2]
        ])  # 2: Wall, 1: Floor, 3: Goal
        self.player_pos = (2, 2)
        self.visited = set()
        self.total_reward = 0

    def reset(self):
        self.player_pos = (2, 2)
        self.visited.clear()
        self.total_reward = 0

        self.map_grid = np.array([
            [2, 2, 2, 2, 2],
            [2, 1, 1, 1, 2],
            [2, 1, 1, 1, 2],
            [2, 1, 1, 1, 2],
            [2, 2, 2, 2, 2]
        ])

        floor_tiles = [(x, y) for x in range(1, 4) for y in range(1, 4)]

        self.goal_pos = floor_tiles[np.random.randint(len(floor_tiles))]
        self.map_grid[self.goal_pos[1], self.goal_pos[0]] = 3
        
        return self._get_state()

    def _get_state(self):
        x, y = self.player_pos
        neighbors = [
            self._get_tile_features(x, y - 1),  # Up
            self._get_tile_features(x, y + 1),  # Down
            self._get_tile_features(x - 1, y),  # Left
            self._get_tile_features(x + 1, y)   # Right
        ]
        return np.array(neighbors)

    def _get_tile_features(self, x, y):
        if 0 <= x < self.map_grid.shape[0] and 0 <= y < self.map_grid.shape[1]:
            tile = self.map_grid[y, x]
            tile_type = tile
            entity_present = 0
            entity_type = -1
            return [tile_type, entity_present, entity_type]
        else:
            return [0, 0, -1]

    def step(self, action):
        x, y = self.player_pos
        if action == 0 and y > 0:  # Up
            self.player_pos = (x, y - 1)
        elif action == 1 and y < self.map_grid.shape[1] - 1:  # Down
            self.player_pos = (x, y + 1)
        elif action == 2 and x > 0:  # Left
            self.player_pos = (x - 1, y)
        elif action == 3 and x < self.map_grid.shape[0] - 1:  # Right
            self.player_pos = (x + 1, y)

        reward, done = self._compute_reward()
        state = self._get_state()
        return state, reward, done

    def _compute_reward(self):
        x, y = self.player_pos
        tile = self.map_grid[y, x]

        if tile == 2:
            return -2, False  # Wall
        elif tile == 3:
            return 15, True  # Goal
        elif tile == 1:
            return 0.1, False  # Floor tile

def run_test_scenario(test_states):
    tile_input_dim = (4, 3)  # 4 neighboring tiles, each with 3 features (tile type, entity present, entity type)
    output_dim = 4  # Number of possible actions

    agent = DQNAgent(tile_input_dim, output_dim)
    
    # Train the agent for a few episodes to ensure it has some learned Q-values
    env = SimpleTestEnvironment()
    episodes = 1000

    for e in range(episodes):
        state = env.reset()
        done = False
        step_count = 0

        while not done:
            action = agent.act(state)
            print(f"Episode: {e}, Step: {step_count}, Action: {action}, Epsilon: {agent.epsilon:.2f}")
            next_state, reward, done = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            agent.replay()

            step_count += 1

        agent.update_target_model()
    
    # Print Q-values for each test state in a table format
    q_values_list = []
    for i, state in enumerate(test_states):
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
        q_values = agent.model(state_tensor).cpu().detach().numpy().flatten()
        q_values_list.append(q_values)

    # Create a DataFrame to display Q-values
    df = pd.DataFrame(q_values_list, columns=['Up', 'Down', 'Left', 'Right'])
    df.index.name = 'Test State'
    print(df)



test = False
if test:
    run_test_scenario(test_states)
else:
    # Profile the run_training function
    cProfile.run('run_training()', 'profile_output')

    # Analyze the profiling results
    with open('profile_output.txt', 'w') as f:
        p = pstats.Stats('profile_output', stream=f)
        p.sort_stats('cumulative').print_stats(50)  # Print top 50 cumulative time functions