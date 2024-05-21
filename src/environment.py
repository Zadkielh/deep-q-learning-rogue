# environment.py

import numpy as np
import pygame
from game import Engine, draw_game_based_on_visibility, step_game, render_game, ent, tiles, WIDTH, HEIGHT

class RogueEnvironment:
    def __init__(self):
        print("Initializing environment")
        self.engine_data = Engine()
        self.action_space = 4  # Up, Down, Left, Right
        self.observation_space = (HEIGHT, WIDTH, 3)  # RGB screen dimensions

    def reset(self):
        print("Resetting environment")
        self.engine_data = Engine()
        return self._get_state()

    def step(self, action):
        print(f"Taking action: {action}")
        self.engine_data = step_game(self.engine_data, action)
        
        reward = self._compute_reward()
        done = not self.engine_data['running']
        state = self._get_state()
        return state, reward, done

    def _get_state(self):
        screen = pygame.Surface((WIDTH, HEIGHT))
        draw_game_based_on_visibility(screen, self.engine_data['map_grid'], self.engine_data['visibility_grid'], self.engine_data['entities_list'])
        state = pygame.surfarray.array3d(screen)
        return state

    def _compute_reward(self):
        player = self.engine_data['player']
        reward = 0

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
        
        # Small reward for each step taken
        reward += 1
        
        return reward

    def render(self):
        render_game(self.engine_data)
