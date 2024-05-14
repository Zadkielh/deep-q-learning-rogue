from objects import tiles
import random
# Constants
PLAYER = 0
GOBLIN = 1


#--------------------------------------------#

ENEMY_TIER_1 = [
    GOBLIN
]

def GetEnemyType(tier):
    if tier < 2:
        return random.choice(ENEMY_TIER_1)

def CreateEnemy(tier, room, grid):
    x, y = room.center()
    x, y = PlaceEnemy(x, y, grid)
    type = GetEnemyType(tier)
    if type == GOBLIN:
        enemy = Goblin(x,y)


    return enemy

def IsTileBlocked(x, y, grid):
    for type in tiles.BLOCKED_TILES:
        if grid[y][x] == type:
            return True
        
    return False

def PlaceEnemy(x, y, grid):
    if IsTileBlocked(x, y, grid):
        for x2 in range(-5, 5):
            for y2 in range(-5, 5):
                if not IsTileBlocked(x + x2, y + y2):
                    x += x2
                    y += y2

    return x, y

class Entity:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.health = 1

    def move(self, dx, dy, grid):
        if not IsTileBlocked(self.x + dx, self.y + dy, grid):
            # Move
            self.x += dx
            self.y += dy

    def set_pos(self, x, y):
        self.x = x
        self.y = y

class Player(Entity):
    def __init__(self, x, y, name):
        super().__init__(x, y)
        self.playerName = name
        self.health = 10
        
class Enemy(Entity):
    def __init__(self, x, y):
        super().__init__(x, y)
        self.isHostile = True

    def chase(self, target_x, target_y, grid):
        step_x = step_y = 0
        if self.x < target_x:
            step_x = 1
        elif self.x > target_x:
            step_x = -1
        elif self.y < target_y:
            step_y = 1
        elif self.y > target_y:
            step_y = -1

        self.move(step_x, step_y, grid)

    def wander(self, grid):
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
            
            self.move(x, y, grid)

class Goblin(Enemy):
    def __init__(self, x, y):
        super().__init__(x, y)
        self.health = 5

class Friendly(Entity):
    def __init__(self, x, y):
        super().__init__(x, y)
        self.isHostile = False