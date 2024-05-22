import random

# Identifiers
DEBUG = -1
VOID = 0
FLOOR = 1
WALL = 2
WALL_TORCH = 3
DOOR = 4
TUNNEL = 5
STAIRS = 6

BLOCKED_TILES = [
        VOID,
        WALL,
]

class Room:
    def __init__(self, x, y, w, h):
        self.x1 = x
        self.y1 = y
        self.x2 = x + w
        self.y2 = y+ h
        self.torchChance = random.randint(0, 100)
        self.spawnChance = random.randint(0, 100)
        self.hasTorch = False

    def center(self):
        center_x = (self.x1 + self.x2) // 2
        center_y = (self.y1 + self.y2) // 2
        return (center_x, center_y)
    
    def intersects(self, other):
        return (self.x1 <= other.x2 and self.x2 >= other.x1 and
                self.y1 <= other.y2 and self.y2 >= other.y1)
    
    def isWall(self, x, y):
        if x == self.x1 or x == self.x2-1:
            return True
        elif y == self.y1 or y == self.y2-1:
            return True
        
    def isCorner(self, x, y):
        if x == self.x1 or x == self.x2 or y == self.y1 or y == self.y2:
            return True
        
        return False
    
    def create_room(self, grid):
        # Create Floor and Walls
        for x in range(self.x1, self.x2):
            for y in range(self.y1, self.y2):
                if self.isWall(x, y):
                    grid[y][x] = WALL
                else:
                    grid[y][x] = FLOOR

    def contains(self, x, y):
        return self.x1 < x < self.x2-1 and self.y1 < y < self.y2-1
    
    def hasDoor(self, grid):
        for x in range(self.x1, self.x2):
            for y in range(self.y1, self.y2):
                if grid[y][x] == DOOR:
                    return True
                
        return False
