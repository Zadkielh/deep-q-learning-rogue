# rogue_env.py
# Environment wrapper

import numpy as np
import pygame
from game import Engine
from entities import Player, Goblin

class RogueEnv:
    def __init__(self):
        self.engine = Engine()
        self.action_space = 4  # Up, Down, Left, Right
        self.observation_space = (GRIDHEIGHT, GRIDWIDTH, 1)
        self.reset()
    
    def reset(self):
        self.engine.reset_game()  # Implement reset logic in Engine class
        state = self.get_state()
        return state
    
    def step(self, action):
        done = False
        reward = 0
        
        if action == 0:
            self.engine.player.move(0, -1, self.engine.map_grid, self.engine.entities_list)
        elif action == 1:
            self.engine.player.move(0, 1, self.engine.map_grid, self.engine.entities_list)
        elif action == 2:
            self.engine.player.move(-1, 0, self.engine.map_grid, self.engine.entities_list)
        elif action == 3:
            self.engine.player.move(1, 0, self.engine.map_grid, self.engine.entities_list)
        
        # Update enemies
        enemies = [ent for ent in self.engine.entities_list if isinstance(ent, Goblin)]
        for enemy in enemies:
            enemy.chooseAction(self.engine.map_grid, self.engine.player, self.engine.entities_list)
        
        # Calculate reward
        if self.engine.player.isAlive:
            reward = 1
        else:
            reward = -10
            done = True
        
        state = self.get_state()
        return state, reward, done
    
    def get_state(self):
        state = np.zeros((GRIDHEIGHT, GRIDWIDTH, 1))
        for y in range(GRIDHEIGHT):
            for x in range(GRIDWIDTH):
                if self.engine.map_grid[y][x] == tiles.FLOOR:
                    state[y, x, 0] = 1
                elif self.engine.map_grid[y][x] == tiles.TUNNEL:
                    state[y, x, 0] = 2
                elif self.engine.map_grid[y][x] == tiles.WALL:
                    state[y, x, 0] = 3
                elif self.engine.map_grid[y][x] == tiles.WALL_TORCH:
                    state[y, x, 0] = 4
                elif self.engine.map_grid[y][x] == tiles.DOOR:
                    state[y, x, 0] = 5
        state[self.engine.player.y, self.engine.player.x, 0] = 6
        for enemy in self.engine.entities_list:
            if isinstance(enemy, Goblin):
                state[enemy.y, enemy.x, 0] = 7
        return state
