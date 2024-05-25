import numpy as np
import time
import pygame
from game import Engine, draw_game_based_on_visibility, step_game, render_game, ent, tiles, V_WIDTH, V_HEIGHT
from objects.items import Food, HealthPotion
from objects.ent_constants import SLOT_LHAND, SLOT_RHAND, SLOT_TWOHAND
from objects.tiles import BLOCKED_TILES

class RogueEnvironment:
    def __init__(self):
        print("Initializing environment")
        self.engine_data = Engine()
        self.action_space = 4  # Up, Down, Left, Right
        self.feature_size = 5
        self.observation_space = ((4, self.feature_size))
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
        self.acummulated_penalty = 0
        self.action_history = []
        return self._get_state()

    def step(self, action):
        print(f"Taking action: {action}")
        player = self.engine_data['player']
        self.engine_data = step_game(self.engine_data, action)
        next_tile = self._get_tile_from_action(player.x, player.y)

        action_type = self.get_tile_valid(next_tile, player.x, player.y)  # Determine the type of action
        
        self.action_history.append(action)
        if len(self.action_history) > 20:
            self.action_history.pop(0)

        self._check_and_equip_items()  # Check and equip items after picking them up
        self._check_and_use_health_items()  # Check and use health-restoring items
        
        reward, done = self._compute_reward(next_tile, action_type)  # Updated to receive two values
        if done:
            print("Ending episode.")
        state = self._get_state()

        self._smooth_delay(0.001)

        return state, reward, done
    
    def get_tile_valid(self, tile, x, y):
        if self.is_valid_move(tile, x, y):
            return True
        return False
    
    def _get_tile_from_action(self, x, y):
        return self.engine_data['map_grid'][y][x]
    
    def is_valid_move(self, target_tile, x, y):
        if 0 <= x < V_WIDTH and 0 <= y < V_HEIGHT:
            if target_tile in BLOCKED_TILES:
                return False
            return True
        return False
    
    def _smooth_delay(self, delay_time):
        end_time = time.time() + delay_time
        while time.time() < end_time:
            pygame.event.pump()
            time.sleep(0.001)

    def _convert_to_tile_based_representation(self):
        map_grid = self.engine_data['map_grid']
        visibility_grid = self.engine_data['visibility_grid']

        tile_based_representation = self.engine_data['agent_grid']
        
        for y in range(len(map_grid)):
            for x in range(len(map_grid[0])):
                if visibility_grid[y][x]:  # If the tile is in the visibility grid
                    tile_based_representation[y][x] = map_grid[y][x]
        
        return tile_based_representation

    def _get_state(self):
        player = self.engine_data['player']
        
        # Get features of the neighboring tiles
        neighbors = [
            self._get_tile_features(player.x, player.y - 1),  # Up
            self._get_tile_features(player.x, player.y + 1),  # Down
            self._get_tile_features(player.x - 1, player.y),  # Left
            self._get_tile_features(player.x + 1, player.y)   # Right
        ]

        return (np.array(neighbors))
    
    def _get_tile_features(self, x, y):
        if 0 <= x < V_WIDTH and 0 <= y < V_HEIGHT:
            tile = self.engine_data['map_grid'][y][x]
            tile_type = tile

            threshold = False

            visited = True if (x, y) in self.visited else False
            if tile in tiles.BLOCKED_TILES:
                visited = True
                threshold = True

            
            visit_count = self.visit_counts.get((x, y), 0)
            if visited and not threshold:
                threshold = visit_count > 20
            
            entity_present = 0
            entity_type = -1
            for entity in self.engine_data['entities_list']:
                if entity.x == x and entity.y == y:
                    entity_present = 1
                    entity_type = entity.type
                    break

            return [tile_type, entity_present, entity_type, visited, threshold]
        else:
            return [0, 0, -1, False, False, False]

    def _compute_reward(self, tile, action_type):
        player = self.engine_data['player']
        reward = 0
        # Give negative reward for attempting a non-valid action (like moving into a wall or a void)
        if not action_type: 
            reward -= 4
            # Higher penalty for void as it is less common
            if tile == tiles.VOID: reward -= 0

        # Check if the current tile has been visited
        current_pos = (player.x, player.y)
        if current_pos not in self.visited:
            reward += 1  # Increase reward for exploring new tiles
            # Give additional reward if visited a door for the first time
            if tile == tiles.DOOR:
                reward += 0
            # Give additional smaller reward for visiting a tunnel tile for the first time
            elif tile == tiles.TUNNEL:
                reward += 0

            self.visited.add(current_pos)
            self.visit_counts[current_pos] = 1
        else:
            self.visit_counts[current_pos] += 1
            if self.visit_counts[current_pos] > 20:  # Threshold for penalizing repetitive visits
                reward -= 0.02
            # Give very small penalty for revisiting tiles
            reward -= 0.01  # Penalize revisiting the same tile

        # Reward for revealing more tiles
        old_data = sum(len(sublist) for sublist in self.engine_data['old_visibility_grid'])
        new_data = sum(len(sublist) for sublist in self.engine_data['visibility_grid'])
        if (new_data > old_data):
            diff = new_data - old_data
            # Reward 5 points for each tile revealed
            reward += int(diff)*0

        if not player.isAlive:
            reward -= 300  # High negative reward for death
        elif tile == tiles.STAIRS:
            reward += 10  # High positive reward for reaching stairs
            return reward, True

        # Reward for hurting an enemy
        for entity in self.engine_data['entities_list']:
            if isinstance(entity, ent.Enemy) and entity.isAlive:
                if entity.lastHealth < entity.health:
                    reward += 0

        # Reward for killing an enemy
        for entity in self.engine_data['entities_list']:
            if isinstance(entity, ent.Enemy) and not entity.isAlive:
                reward += 0
                self.engine_data['entities_list'].remove(entity)

        # Reward for collecting an item
        for entity in self.engine_data['entities_list']:
            if isinstance(entity, ent.Item) and not entity.isAlive:
                reward += 0
                self.engine_data['entities_list'].remove(entity)

        if reward < 0:
            self.acummulated_penalty += reward
        else:
            self.acummulated_penalty = 0
        
        self.total_reward += reward
        # End if accumulated penalty gets too high, probably means it got stuck.
        if self.acummulated_penalty < 0:
            print("Accumulated Penalty: ", self.acummulated_penalty)
        if self.acummulated_penalty <= -200:
            return reward, True

        return reward, False
    
    def _check_and_equip_items(self):
        player = self.engine_data['player']
        for item in player.inventory:
            if item.canWield:
                currently_equipped = player.equipped[item.slot]

                # Check if the item is one handed
                if item.slot == SLOT_LHAND or item.slot == SLOT_RHAND:
                    # If we don't have anything equipped here, check two handed
                    if not currently_equipped: currently_equipped = player.equipped[SLOT_TWOHAND]
                other_equipped = None
                wield = False
                
                # Special handling for two-handed items
                if item.slot == SLOT_TWOHAND:
                    currently_equipped = player.equipped[item.slot]
                    if currently_equipped and item.name == currently_equipped.name:
                        continue
                    if currently_equipped is None:
                        currently_equipped = player.equipped[SLOT_RHAND]
                        other_equipped = player.equipped[SLOT_LHAND]

                if currently_equipped:
                    if item.name == currently_equipped.name:
                        continue

                    weights = {
                        'damage': 2,
                        'armor': 4,
                        'strength': 3,
                        'dexterity': 2,
                        'agility': 2
                    }

                    value = sum(getattr(item, stat, 0) * weight for stat, weight in weights.items())
                    old_value = sum(getattr(currently_equipped, stat, 0) * weight for stat, weight in weights.items())

                    if other_equipped:
                        old_value += sum(getattr(other_equipped, stat, 0) * weight for stat, weight in weights.items())

                    if value > old_value:
                        wield = True
                else:
                    wield = True

                if wield:
                    player.Equip(item, self.engine_data['notification_manager'])
                    return  # Equip only one item per call to avoid conflicts

    def _check_and_use_health_items(self):
        player = self.engine_data['player']
        if player.health <= player.maxHealth * 0.5:
            for i, item in enumerate(player.inventory):
                if isinstance(item, (Food, HealthPotion)):
                    item.OnUse(player, i, self.engine_data['notification_manager'])
                    break

    def render(self):
        render_game(self.engine_data)#
