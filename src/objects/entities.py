from objects import tiles
import random
# Constants
PLAYER = 0
GOBLIN = 1

# View Distance
VIEW_DISTANCE_LOW = (-1,2)
VIEW_DISTANCE_MEDIUM = (-2,3)
VIEW_DISTANCE_HIGH = (-3,4)
VIEW_DISTANCE_MAX = (-100, 100)

ENTITIES_COLLISION = [
    PLAYER,
    GOBLIN
]

#--------------------------------------------#

ENEMY_TIER_1 = [
    GOBLIN
]

def GetEnemyFromTier(tier):
    if tier < 2:
        return random.choice(ENEMY_TIER_1)

def CreateEnemy(tier, room, grid, entities):
    x, y = room.center()
    x, y = PlaceEnemy(x, y, grid, entities)
    type = GetEnemyFromTier(tier)
    if type == GOBLIN:
        enemy = Goblin(x,y)


    return enemy

def IsTileBlocked(x, y, grid, entities, caller=None):
    for type in tiles.BLOCKED_TILES:
        if grid[y][x] == type:
            return True, type
    
    for entity in entities:
        if entity is caller:
            continue
        
        if entity.x == x and entity.y == y:
            for collision in ENTITIES_COLLISION:
                if entity.type == collision:
                    return True, entity
        

    return False, None

def PlaceEnemy(x, y, grid, entities):
    if IsTileBlocked(x, y, grid, entities):
        for x2 in range(-5, 5):
            for y2 in range(-5, 5):
                if not IsTileBlocked(x + x2, y + y2, grid, entities):
                    x += x2
                    y += y2

    return x, y

class Entity:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.isAlive = True
        self.health = 1
        self.isHostile = False

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
    def __init__(self, x, y, name):
        super().__init__(x, y)
        self.playerName = name
        self.health = 10
        self.type = PLAYER
        self.color = (0,0,255)
        self.viewDistance = VIEW_DISTANCE_LOW
        
class Enemy(Entity):
    def __init__(self, x, y):
        super().__init__(x, y)
        self.isHostile = True
        self.color = (255,0,0)

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
        for dy in range(-5, 5):
            for dx in range(-5, 5):
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

class Goblin(Enemy):
    def __init__(self, x, y):
        super().__init__(x, y)
        self.health = 5
        self.type = GOBLIN
        self.color = (0,255,0)

class Friendly(Entity):
    def __init__(self, x, y):
        super().__init__(x, y)
        self.isHostile = False