from game import V_HEIGHT, V_WIDTH
import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque
from environment import RogueEnvironment
import torch.nn.functional as F

class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        # Calculate the size after convolutions
        def conv2d_size_out(size, kernel_size = 8, stride = 4):
            return (size - (kernel_size - 1) - 1) // stride + 1

        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(V_WIDTH, 8, 4), 4, 2), 3, 1)
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(V_HEIGHT, 8, 4), 4, 2), 3, 1)
        linear_input_size = convw * convh * 64

        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(linear_input_size, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.conv3(x)
        x = F.relu(x)
        return self.head(x)
    
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class DQNAgent:
    def __init__(self, input_dim, output_dim):
        self.model = DQN(input_dim, output_dim).to(device)
        self.target_model = DQN(input_dim, output_dim).to(device)
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model.eval()

        self.memory = deque(maxlen=1000)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.0001)
        self.criterion = nn.MSELoss()
        self.gamma = 0.99
        self.batch_size = 16
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        self.input_dim = input_dim

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(4)
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        with torch.no_grad():
            q_values = self.model(state)
        return torch.argmax(q_values[0]).item()

    def replay(self):
        if len(self.memory) < self.batch_size:
            return
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(np.array(states)).to(device)
        actions = torch.LongTensor(np.array(actions)).to(device)
        rewards = torch.FloatTensor(np.array(rewards)).to(device)
        next_states = torch.FloatTensor(np.array(next_states)).to(device)
        dones = torch.FloatTensor(np.array(dones)).to(device)

        q_values = self.model(states)
        next_q_values = self.target_model(next_states)
        target_q_values = q_values.clone()

        for i in range(self.batch_size):
            target_q_values[i][actions[i]] = rewards[i] + (1 - dones[i]) * self.gamma * torch.max(next_q_values[i])

        loss = self.criterion(q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

# Training the agent
env = RogueEnvironment()
print("Environment initialized")
agent = DQNAgent(input_dim=(3, V_HEIGHT, V_WIDTH), output_dim=4)
print("Agent initialized")
episodes = 100

for e in range(episodes):
    print(f"Starting episode {e}")
    state = env.reset()
    state = np.transpose(state, (2, 0, 1))  # For channels-first format expected by Conv2D
    done = False
    total_reward = 0
    step_count = 0

    while not done:
        action = agent.act(state)
        print(f"Episode: {e}, Step: {step_count}, Action: {action}, Total Reward: {total_reward}")
        next_state, reward, done = env.step(action)
        next_state = np.transpose(next_state, (2, 0, 1))
        agent.remember(state, action, reward, next_state, done)
        state = next_state
        total_reward += reward
        agent.replay()
        
        # Render the game for visualization
        env.render()
        step_count += 1

    agent.update_target_model()
    print(f"Episode: {e}/{episodes}, Total Reward: {total_reward}, Epsilon: {agent.epsilon:.2f}")

    if e % 10 == 0:
        torch.save(agent.model.state_dict(), f"dqn_model_{e}.pth")
