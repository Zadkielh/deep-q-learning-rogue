import argparse
from game import V_HEIGHT, V_WIDTH, tiles
import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque
from environment import RogueEnvironment
import torch.nn.functional as F
import cProfile
import pstats
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

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

        self.loss_history = []
        self.epsilon_history = []

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

        self.loss_history.append(loss.item())
        self.epsilon_history.append(self.epsilon)

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

def plot_q_values_over_time(q_values_history, save_path='q_values_over_time.png'):
    plt.figure(figsize=(15, 10))
    average_q_values_history = []

    # Only consider states 1-9 (indices 0-8)
    for state_index in range(10):
        average_q_values_per_state = [
            np.mean(df.iloc[state_index].tolist()) for _, df in q_values_history
        ]
        average_q_values_history.append(average_q_values_per_state)

    for state_index, avg_q_values in enumerate(average_q_values_history):
        plt.plot(
            range(len(q_values_history)), avg_q_values, label=f'State {state_index}'
        )

    plt.xlabel('Episodes')
    plt.ylabel('Average Q-Values')
    plt.title('Average Q-Values Over Time for Test States (0-9)')
    plt.legend()
    plt.savefig(save_path)
    plt.close()

def plot_success_episodes(success_episodes, total_episodes, save_path='success_episodes.png'):
    success_percentage = []
    successful_count = 0

    for episode in range(total_episodes):
        if episode in success_episodes:
            successful_count += 1
        success_percentage.append(successful_count / (episode + 1) * 100)

    plt.figure(figsize=(10, 6))
    plt.plot(range(total_episodes), success_percentage, 'bo-', markersize=4)
    plt.xlabel('Episodes')
    plt.ylabel('Success Percentage')
    plt.title('Percentage of Successful Episodes Over Time')
    plt.savefig(save_path)
    plt.close()

def plot_explored_percentage(explored_percentage, save_path='explored_percentage.png'):
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(explored_percentage)), explored_percentage, 'g-')
    plt.xlabel('Episodes')
    plt.ylabel('Explored Percentage')
    plt.title('Percentage of Map Explored Over Episodes')
    plt.savefig(save_path)
    plt.close()

def plot_loss_over_time(loss_history, save_path='loss_over_time.png'):
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(loss_history)), loss_history, 'r-')
    plt.xlabel('Episodes')
    plt.ylabel('Loss')
    plt.title('Loss Over Time')
    plt.savefig(save_path)
    plt.close()

def plot_epsilon_over_time(epsilon_history, save_path='epsilon_over_time.png'):
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(epsilon_history)), epsilon_history, 'b-')
    plt.xlabel('Episodes')
    plt.ylabel('Epsilon')
    plt.title('Epsilon Over Time')
    plt.savefig(save_path)
    plt.close()

def plot_average_reward_per_episode(reward_history, save_path='average_reward_per_episode.png'):
    average_rewards = [np.mean(reward_history[max(0, i-10):i+1]) for i in range(len(reward_history))]
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(average_rewards)), average_rewards, 'g-')
    plt.xlabel('Episodes')
    plt.ylabel('Average Reward')
    plt.title('Average Reward Per Episode')
    plt.savefig(save_path)
    plt.close()

def plot_action_distribution(action_history, save_path='action_distribution.png'):
    actions = np.array(action_history)
    action_counts = np.bincount(actions, minlength=4)
    plt.figure(figsize=(10, 6))
    plt.bar(range(4), action_counts, tick_label=['Up', 'Down', 'Left', 'Right'])
    plt.xlabel('Actions')
    plt.ylabel('Count')
    plt.title('Action Distribution')
    plt.savefig(save_path)
    plt.close()

def plot_reward_distribution(reward_history, save_path='reward_distribution.png'):
    plt.figure(figsize=(10, 6))
    plt.hist(reward_history, bins=20, color='purple', edgecolor='black')
    plt.xlabel('Reward')
    plt.ylabel('Frequency')
    plt.title('Reward Distribution')
    plt.savefig(save_path)
    plt.close()



# Training the agent
def run_training():
    map_input_dim = (V_HEIGHT, V_WIDTH)
    tile_input_dim = (4, 6)
    output_dim = 4

    env = RogueEnvironment()
    print("Environment initialized")
    agent = DQNAgent(tile_input_dim, output_dim)
    print("Agent initialized")
    episodes = 200

    q_values_history = []
    success_episodes = []
    explored_percentage = []
    action_history = []
    reward_history = []

    for e in range(episodes):
        print(f"Starting episode {e}")
        state = env.reset()
        done = False
        total_reward = 0
        step_count = 0
        explored_tiles = set()

        while not done:
            action = agent.act(state)
            next_state, reward, done = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            agent.replay()

            action_history.append(action)
            reward_history.append(reward)
            player = env.engine_data['player']
            explored_tiles.add((player.x, player.y))

            env.render()
            step_count += 1

        agent.update_target_model()
        print(f"Episode: {e}/{episodes}, Total Reward: {total_reward}, Epsilon: {agent.epsilon:.2f}")

        if env.success:
            success_episodes.append(e)

        total_tiles = V_WIDTH * V_HEIGHT - sum(row.count(tile) for tile in tiles.BLOCKED_TILES for row in env.engine_data['map_grid'])
        explored_percentage.append(len(explored_tiles) / total_tiles * 100)

        q_values_list = []
        for state in test_states:
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            q_values = agent.model(state_tensor).cpu().detach().numpy().flatten()
            q_values_list.append(q_values)

        df = pd.DataFrame(q_values_list, columns=['Up', 'Down', 'Left', 'Right'])
        df.index.name = 'Test State'
        q_values_history.append((e, df))

    plot_q_values_over_time(q_values_history)
    plot_success_episodes(success_episodes, episodes)
    plot_explored_percentage(explored_percentage)
    plot_loss_over_time(agent.loss_history)
    plot_epsilon_over_time(agent.epsilon_history)
    plot_average_reward_per_episode(reward_history)
    plot_action_distribution(action_history)
    plot_reward_distribution(reward_history)

    print("\nFinal Q-values for each test state:")
    for i, state in enumerate(test_states):
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
        q_values = agent.model(state_tensor).cpu().detach().numpy().flatten()
        print(f"Test State {i}: {q_values}")

    with open('q_values_history.pkl', 'wb') as f:
        pickle.dump(q_values_history, f)

    torch.save(agent.model.state_dict(), f"dqn_model{episodes}.pth")

def run_trained_model(model_path, episodes=10):
    env = RogueEnvironment()
    print("Environment initialized")

    agent = DQNAgent(tile_input_dim=(4, 6), output_dim=4)
    agent.model.load_state_dict(torch.load(model_path))
    agent.model.eval()
    agent.epsilon = 0.1

    success_episodes = []
    explored_percentage = []
    action_history = []
    reward_history = []

    for e in range(episodes):
        print(f"Starting episode {e}")
        state = env.reset()
        done = False
        total_reward = 0
        step_count = 0
        explored_tiles = set()

        while not done:
            action = agent.act(state)
            next_state, reward, done = env.step(action)
            state = next_state
            total_reward += reward

            action_history.append(action)
            reward_history.append(reward)
            player = env.engine_data['player']
            explored_tiles.add((player.x, player.y))

            env.render()
            step_count += 1

        print(f"Episode: {e}/{episodes}, Total Reward: {total_reward}, Epsilon: {agent.epsilon:.2f}")

        if env.success:
            success_episodes.append(e)

        total_tiles = V_WIDTH * V_HEIGHT - sum(row.count(tile) for tile in tiles.BLOCKED_TILES for row in env.engine_data['map_grid'])
        explored_percentage.append(len(explored_tiles) / total_tiles * 100)

    plot_success_episodes(success_episodes, episodes)
    plot_explored_percentage(explored_percentage)
    plot_action_distribution(action_history)
    plot_reward_distribution(reward_history)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['train', 'run'], default='train', help="Mode to run the script: 'train' to train a new model, 'run' to run a trained model")
    parser.add_argument('--model_path', type=str, default='dqn_model.pth', help="Path to the trained model file")
    parser.add_argument('--episodes', type=int, default=10, help="Number of episodes to run")

    args = parser.parse_args()

    if args.mode == 'train':
        run_training()
    elif args.mode == 'run':
        run_trained_model(args.model_path, args.episodes)

