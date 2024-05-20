from objects import tiles
from objects.ent_constants import *
import random


def IsTileBlocked(x, y, grid, entities, caller=None):
    if grid[y][x] in tiles.BLOCKED_TILES:
        return True, grid[y][x]
    
    for entity in entities:
        if entity is caller:
            continue
        if entity.x == x and entity.y == y and entity.type in ENTITIES_COLLISION:
            return True, entity
    return False, None

def PlaceEnemy(x, y, grid, entities):
    if IsTileBlocked(x, y, grid, entities)[0]:
        for dx in range(-5, 6):
            for dy in range(-5, 6):
                if not IsTileBlocked(x + dx, y + dy, grid, entities)[0]:
                    return x + dx, y + dy
    return x, y

class Entity:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.isAlive = True
        self.health = 1
        self.armor = 0
        self.strength = 0
        self.dexterity = 0
        self.agility = 0
        self.isHostile = False
        self.viewDistance = VIEW_DISTANCE_LOW

    def move(self, dx, dy, grid, entities):
        blocked, type = IsTileBlocked(self.x + dx, self.y + dy, grid, entities)
        if not blocked:
            # Move
            self.x += dx
            self.y += dy
        else:
            if not isinstance(type, int):
                type.interact(self)
                
    def set_pos(self, x, y):
        self.x = x
        self.y = y

    def interact(self, caller):
        pass

class Player(Entity):
    def __init__(self, x, y, name="Hero"):
        super().__init__(x, y)
        self.name = name
        self.health = 10
        self.type = PLAYER
        self.color = (0,0,255)
        self.viewDistance = VIEW_DISTANCE_LOW
        
class Enemy(Entity):
    def __init__(self, x, y):
        super().__init__(x, y)
        self.isHostile = True
        self.color = (255,0,0)
        self.viewDistance = VIEW_DISTANCE_MEDIUM

    def chase(self, target_x, target_y, grid, entities):
        step_x = step_y = 0
        if self.x < target_x:
            step_x = 1
        elif self.x > target_x:
            step_x = -1
        elif self.y < target_y:
            step_y = 1
        elif self.y > target_y:
            step_y = -1

        self.move(step_x, step_y, grid, entities)

    def wander(self, grid, entities):
        wanderChance = random.randint(0, 100)
        if wanderChance <= 50:
            x = 0
            y = 0
            direction = random.randint(0, 3)
            if direction == 0:
                x = -1
            elif direction == 1:
                x = 1
            elif direction == 2:
                y = -1
            elif direction == 3:
                y = 1
            
            self.move(x, y, grid, entities)

    def canDetectPlayer(self, player):
        
        # Check tiles in a 5x5 radius around the entity
        lbound, ubound = self.viewDistance
        for dy in range(lbound, ubound):
            for dx in range(lbound, ubound):
                ny, nx = max(0, self.y + dy), max(0, self.x + dx)
                #print(nx, ny, player.x, player.y)

                if player.x == nx and player.y == ny:
                    return True

        return False
    
    def chooseAction(self, grid, player, entities):
        if self.canDetectPlayer(player):
            self.chase(player.x, player.y, grid, entities)
        else:
            self.wander(grid, entities)

    def performCombatRound(self, caller):
        if caller.type == PLAYER:
            self.isAlive = False

    def interact(self, caller):
        self.performCombatRound(caller)

class Friendly(Entity):
    def __init__(self, x, y):
        super().__init__(x, y)
        self.isHostile = False