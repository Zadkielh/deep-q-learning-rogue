# train.py

import numpy as np
from rogue_env import RogueEnv
from dql_agent import DQLAgent

EPISODES = 1000

if __name__ == "__main__":
    env = RogueEnv()
    state_shape = env.observation_space
    action_space = env.action_space
    agent = DQLAgent(state_shape, action_space)
    batch_size = 32
    
    for e in range(EPISODES):
        state = env.reset()
        state = np.reshape(state, [1, *state_shape])
        
        for time in range(500):
            action = agent.act(state)
            next_state, reward, done = env.step(action)
            next_state = np.reshape(next_state, [1, *state_shape])
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            if done:
                print(f"episode: {e}/{EPISODES}, score: {time}, e: {agent.epsilon:.2}")
                break
            if len(agent.memory) > batch_size:
                agent.replay(batch_size)
        if e % 50 == 0:
            agent.save("dql_rogue.h5")
