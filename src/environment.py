import numpy as np
import pygame
from game import Engine, draw_game_based_on_visibility, step_game, render_game, ent, tiles, V_WIDTH, V_HEIGHT

class RogueEnvironment:
    def __init__(self):
        print("Initializing environment")
        self.engine_data = Engine()
        self.action_space = 5  # Up, Down, Left, Right, Attack
        self.observation_space = (V_HEIGHT, V_WIDTH, 3)  # RGB screen dimensions
        self.visited = set()  # To track visited tiles
        self.total_reward = 0
        self.action_history = []

    def reset(self):
        print("Resetting environment")
        self.engine_data = Engine()
        self.visited.clear()
        self.total_reward = 0
        self.action_history = []
        return self._get_state()

    def step(self, action):
        print(f"Taking action: {action}")
        self.engine_data = step_game(self.engine_data, action)
        
        self.action_history.append(action)
        if len(self.action_history) > 20:
            self.action_history.pop(0)  # Keep only the last 20 actions
        
        reward, done = self._compute_reward()  # Updated to receive two values
        if done:
            print("Ending episode due to low performance.")
        state = self._get_state()
        return state, reward, done

    def _get_state(self):
        screen = pygame.Surface((V_WIDTH, V_HEIGHT))
        draw_game_based_on_visibility(screen, self.engine_data['map_grid'], self.engine_data['visibility_grid'], self.engine_data['entities_list'])
        state = pygame.surfarray.array3d(screen)
        return state

    def _compute_reward(self):
        player = self.engine_data['player']
        reward = 0

        # Check if the current tile has been visited
        current_pos = (player.x, player.y)
        if current_pos not in self.visited:
            reward += 5  # Increase reward for exploring new tiles
            self.visited.add(current_pos)

        if not player.isAlive:
            reward -= 100  # High negative reward for death
        elif self.engine_data['map_grid'][player.y][player.x] == tiles.STAIRS:
            reward += 100  # High positive reward for reaching stairs

        # Reward for killing an enemy
        for entity in self.engine_data['entities_list']:
            if isinstance(entity, ent.Enemy) and not entity.isAlive:
                reward += 50
                self.engine_data['entities_list'].remove(entity)

        # Reward for collecting an item
        for entity in self.engine_data['entities_list']:
            if isinstance(entity, ent.Item) and not entity.isAlive:
                reward += 10
                self.engine_data['entities_list'].remove(entity)

        # Check for repetitive actions
        if len(set(self.action_history)) == 1 and len(self.action_history) == 20:
            reward -= 50  # Apply a penalty for repetitive actions
            print("Penalty applied for repetitive behavior.")

        self.total_reward += reward
        if self.total_reward <= -100:
            reward -= 30  # Further penalize if the total reward is very low
            return reward, True

        return reward, False

    def render(self):
        render_game(self.engine_data)
