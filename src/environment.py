import numpy as np
import time
import pygame
from game import Engine, draw_game_based_on_visibility, step_game, render_game, ent, tiles, V_WIDTH, V_HEIGHT
from objects.items import Food, HealthPotion
from objects.ent_constants import SLOT_LHAND, SLOT_RHAND, SLOT_TWOHAND

class RogueEnvironment:
    def __init__(self):
        print("Initializing environment")
        self.engine_data = Engine()
        self.action_space = 4  # Up, Down, Left, Right
        self.feature_size = 3
        self.observation_space = ((V_HEIGHT, V_WIDTH, 3), (4, self.feature_size))
        self.visited = set()  # To track visited tiles
        self.visit_counts = {}
        self.total_reward = 0
        self.action_history = []

    def reset(self):
        print("Resetting environment")
        self.engine_data = Engine()
        self.visited.clear()
        self.visit_counts.clear()
        self.total_reward = 0
        self.action_history = []
        return self._get_state()

    def step(self, action):
        print(f"Taking action: {action}")
        player = self.engine_data['player']
        self.engine_data = step_game(self.engine_data, action)
        next_tile = self._get_tile_from_action(player)
        
        self.action_history.append(action)
        if len(self.action_history) > 20:
            self.action_history.pop(0)

        self._check_and_equip_items()  # Check and equip items after picking them up
        self._check_and_use_health_items()  # Check and use health-restoring items
        
        reward, done = self._compute_reward(next_tile)  # Updated to receive two values
        if done:
            print("Ending episode due to low performance.")
        state = self._get_state()

        self._smooth_delay(0.001)

        return state, reward, done
    
    def _get_tile_from_action(self, player):
        return self.engine_data['map_grid'][player.y][player.x]
    
    def _smooth_delay(self, delay_time):
        end_time = time.time() + delay_time
        while time.time() < end_time:
            pygame.event.pump()
            time.sleep(0.001)  # Sleep for 10 milliseconds

    def _get_state(self):
        player = self.engine_data['player']

        screen = pygame.Surface((V_WIDTH, V_HEIGHT))
        self.engine_data['old_visibility_grid'] = self.engine_data['visibility_grid']
        draw_game_based_on_visibility(screen, self.engine_data['map_grid'], self.engine_data['visibility_grid'], self.engine_data['entities_list'])
        full_map = pygame.surfarray.array3d(screen)

        # Get features of the neighboring tiles
        neighbors = [
            self._get_tile_features(player.x, player.y - 1),  # Up
            self._get_tile_features(player.x, player.y + 1),  # Down
            self._get_tile_features(player.x - 1, player.y),  # Left
            self._get_tile_features(player.x + 1, player.y)   # Right
        ]

        return (full_map, np.array(neighbors))
    
    def _get_tile_features(self, x, y):
        if 0 <= x < V_WIDTH and 0 <= y < V_HEIGHT:
            tile = self.engine_data['map_grid'][y][x]
            tile_type = tile  # Assuming tile itself is an integer between 0-6
            
            # Check if an entity is present
            entity_present = 0
            entity_type = -1  # Default to -1 if no entity is present
            for entity in self.engine_data['entities_list']:
                if entity.x == x and entity.y == y:
                    entity_present = 1
                    entity_type = entity.type  # Assuming each entity has a 'type' attribute
                    break

            return [tile_type, entity_present, entity_type]
        else:
            # Handle out of bounds as a non-traversable or special tile
            return [-1, 0, -1]

    def _compute_reward(self, tile):
        player = self.engine_data['player']
        reward = 0

        # Check if the current tile has been visited
        current_pos = (player.x, player.y)
        if current_pos not in self.visited:
            reward += 5  # Increase reward for exploring new tiles
            # Give additional reward if visited a door for the first time
            if tile == tiles.DOOR:
                reward += 25
            # Give additional smaller reward for visiting a tunnel tile for the first time
            elif tile == tiles.TUNNEL:
                reward += 5

            self.visited.add(current_pos)
            self.visit_counts[current_pos] = 1
        else:
            self.visit_counts[current_pos] += 1
            if self.visit_counts[current_pos] > 50:  # Threshold for penalizing repetitive visits
                reward -= 0.1  # Penalize revisiting the same tile

        # Negative Reward for staying at the same coordinates
        last_pos = (player.lastx, player.lasty)
        if (current_pos == last_pos):
            reward -= 0.1

        # Reward for revealing more tiles
        old_data = sum(len(sublist) for sublist in self.engine_data['old_visibility_grid'])
        new_data = sum(len(sublist) for sublist in self.engine_data['visibility_grid'])
        if (new_data > old_data):
            diff = new_data - old_data
            # Reward 5 points for each tile revealed
            reward += int(diff)*5

        if not player.isAlive:
            reward -= 1000  # High negative reward for death
        elif self.engine_data['map_grid'][player.y][player.x] == tiles.STAIRS:
            reward += 1000  # High positive reward for reaching stairs

        # Reward for hurting an enemy
        for entity in self.engine_data['entities_list']:
            if isinstance(entity, ent.Enemy) and entity.isAlive:
                if entity.lastHealth < entity.health:
                    reward += 25

        # Reward for killing an enemy
        for entity in self.engine_data['entities_list']:
            if isinstance(entity, ent.Enemy) and not entity.isAlive:
                reward += 100
                self.engine_data['entities_list'].remove(entity)

        # Reward for collecting an item
        for entity in self.engine_data['entities_list']:
            if isinstance(entity, ent.Item) and not entity.isAlive:
                reward += 50
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
    
    def _check_and_equip_items(self):
        player = self.engine_data['player']
        for item in player.inventory:
            if item.canWield:
                currently_equipped = player.equipped[item.slot]
                if currently_equipped:
                    if item.name == currently_equipped.name: continue
                    other_equipped = None
                    # Two hand items
                    if item.slot == SLOT_TWOHAND:
                        # Check if we have a two-hand item equipped first
                        currently_equipped = player.equipped[item.slot]
                        if item.name == currently_equipped.name: continue
                        # If we don't check the hands
                        if currently_equipped == None:
                            currently_equipped = player.equipped[SLOT_RHAND]
                            other_equipped = player.equipped[SLOT_LHAND]

                    weights = {
                        'damage': 2,
                        'armor': 4,
                        'strength': 3,
                        'dexterity': 2,
                        'agility': 2
                    }

                    value = 0
                    for stat, weight in weights.items():
                        stat_value = getattr(item, stat, 0)
                        value += stat_value * weight

                    old_value = 0
                    for stat, weight in weights.items():
                        stat_value = getattr(currently_equipped, stat, 0)
                        old_value += stat_value * weight

                    # Get value of other hand too
                    if other_equipped:
                        for stat, weight in weights.items():
                            stat_value = getattr(other_equipped, stat, 0)
                            old_value += stat_value * weight

                    wield = False
                    if value > old_value:
                        wield = True
                else:
                    wield = True
                if wield:
                    player.Equip(item, self.engine_data['notification_manager'])
                    return

    def _check_and_use_health_items(self):
        player = self.engine_data['player']
        if player.health <= player.maxHealth * 0.5:
            for i, item in enumerate(player.inventory):
                if isinstance(item, (Food, HealthPotion)):
                    item.OnUse(player, i, self.engine_data['notification_manager'])
                    break

    def render(self):
        render_game(self.engine_data)#
