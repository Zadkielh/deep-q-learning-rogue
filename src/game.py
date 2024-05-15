import pygame
import sys
import random
from objects import tiles
from objects import entities

# Constants
WIDTH = 1600
HEIGHT = 900
TILESIZE = 20
GRIDWIDTH = WIDTH // TILESIZE
GRIDHEIGHT = HEIGHT // TILESIZE

# Colors
BLACK = (0,0,0)
WHITE = (255,255,255)
DARK_GREY = (50,50,50)
GREY = (100,100,100)
BROWN = (200,200,150)
DARK_BROWN = (150,150,100)
ORANGE = (255,200,100)
BLUE = (0,0,255)
RED = (255,0,0)
GREEN = (0,255,0)

def create_h_tunnel(x1, x2, y, grid):
    for x in range(min(x1, x2), max(x1, x2) + 1):
        
        # Stop when reaching a room
        isWall = grid[y][x] == tiles.WALL
        if isWall:
            # Check if we should make a door
            isNotCorner = (grid[y-1][x] == tiles.WALL and grid[y+1][x] == tiles.WALL) or (grid[y][x-1] == tiles.WALL and grid[y][x+1] == tiles.WALL)
            AdjacentDoor = (grid[y-1][x] == tiles.DOOR or grid[y+1][x] == tiles.DOOR) or (grid[y][x-1] == tiles.DOOR or grid[y][x+1] == tiles.DOOR)
        
            if isNotCorner and not AdjacentDoor:
                grid[y][x] = tiles.DOOR

            # Check if next is also a wall
            if grid[y][x+1] == tiles.WALL:
                return
        
        elif grid[y][x] == 0:
                grid[y][x] = tiles.TUNNEL

def create_v_tunnel(y1, y2, x, grid):
    for y in range(min(y1, y2), max(y1, y2) + 1):
        # Stop when reaching a room
        isWall = grid[y][x] == tiles.WALL
        if isWall:
            # Check if we should make a door
            isNotCorner = (grid[y-1][x] == tiles.WALL and grid[y+1][x] == tiles.WALL) or (grid[y][x-1] == tiles.WALL and grid[y][x+1] == tiles.WALL)
            AdjacentDoor = (grid[y-1][x] == tiles.DOOR or grid[y+1][x] == tiles.DOOR) or (grid[y][x-1] == tiles.DOOR or grid[y][x+1] == tiles.DOOR)
        
            if isNotCorner and not AdjacentDoor:
                grid[y][x] = tiles.DOOR

            # Check if next is also a wall
            if grid[y+1][x] == tiles.WALL:
                return
        
        elif grid[y][x] == 0:
                grid[y][x] = tiles.TUNNEL


def make_map(max_rooms, room_min_size, room_max_size):
    grid = [[0 for x in range(GRIDWIDTH)] for y in range(GRIDHEIGHT)]
    rooms = []

    for r in range(max_rooms):
        w = random.randint(room_min_size, room_max_size)
        h = random.randint(room_min_size, room_max_size)
        x = random.randint(0, GRIDWIDTH - w - 1)
        y = random.randint(0, GRIDHEIGHT - h - 1)

        new_room = tiles.Room(x, y, w, h)
        if any(new_room.intersects(other_room) for other_room in rooms):
            continue  # If a room intersects, skip
        new_room.create_room(grid)
        (new_x, new_y) = new_room.center()

        if rooms:
            (prev_x, prev_y) = rooms[-1].center()
            if random.randint(0, 1):
                create_h_tunnel(prev_x, new_x, prev_y, grid)
                create_v_tunnel(prev_y, new_y, new_x, grid)
            else:
                create_v_tunnel(prev_y, new_y, prev_x, grid)
                create_h_tunnel(prev_x, new_x, new_y, grid)

        rooms.append(new_room)
    return grid, rooms

def place_statics(grid, rooms, floor):
    # Place Torches
    torchMod = 95 - (5*floor)
    for room in rooms:
        if torchMod > room.torchChance:
            x, y = room.center()
            grid[y][x] = tiles.WALL_TORCH
            room.hasTorch = True

def place_entities(grid, rooms, floor, player, list):
    # Place Player
    x, y = rooms[0].center()
    player.set_pos(x, y)

    # Select difficulty level
    EnemyTier = min(1, 1 * (floor / 5))
    MaxEnemiesPerRoom = 1 + int(EnemyTier)

    for room in rooms:
        # Spawn Enemies
        spawnMod = min(20 + (1*floor), 100)
        if spawnMod > room.spawnChance:
            # Place Enemies
            enemyCount = random.randint(1, MaxEnemiesPerRoom)
            for _ in range(enemyCount):
                # Select Type
                enemy = entities.CreateEnemy(EnemyTier, room, grid, list)
                list.append(enemy)

    return list

def update_vision_normal(player_x, player_y, grid, visibility_grid):
    for dy in range(-1, 2):
        for dx in range(-1, 2):
            nx, ny = player_x + dx, player_y + dy
            if 0 <= nx < len(grid[0]) and 0 <= ny < len(grid):
                visibility_grid[ny][nx] = True  # This grid tracks visible tiles

def update_vision_lit_rooms(player_x, player_y, rooms, visibility_grid):
    for room in rooms:
        if room.hasTorch and room.contains(player_x, player_y):
            for y in range(room.y1, room.y2 + 1):
                for x in range(room.x1, room.x2 + 1):
                    visibility_grid[y][x] = True

def draw_tile(x, y, grid, screen):
    rect = pygame.Rect(x*TILESIZE, y*TILESIZE, TILESIZE, TILESIZE)
    if grid[y][x] == tiles.FLOOR:
        pygame.draw.rect(screen, WHITE, rect)
    elif grid[y][x] == tiles.TUNNEL:
        pygame.draw.rect(screen, DARK_GREY, rect)
    elif grid[y][x] == tiles.WALL:
        pygame.draw.rect(screen, DARK_BROWN, rect)
    elif grid[y][x] == tiles.WALL_TORCH:
        pygame.draw.rect(screen, ORANGE, rect)
    elif grid[y][x] == tiles.DOOR:
        pygame.draw.rect(screen, BROWN, rect)
    else:
        pygame.draw.rect(screen, BLACK, rect)

def draw_game_based_on_visibility(screen, map_grid, visibility_grid, entities_list):
    for y in range(len(map_grid)):
        for x in range(len(map_grid[0])):
            if visibility_grid[y][x]:  # Only draw if the tile is visible
                draw_tile(x, y, map_grid, screen)  # Implement drawing based on tile type
    # Draw player and enemies if they are in visible tiles
    for ent in entities_list:
            if visibility_grid[ent.y][ent.x]:
                pygame.draw.rect(screen, ent.color, pygame.Rect(ent.x*20, ent.y*20, 20, 20))

# Initiliaze Game
pygame.init()

# Set Screen Size
screen = pygame.display.set_mode((WIDTH,HEIGHT))
pygame.display.set_caption('Rogue')

# Game Loop
def Engine():

    # Engine Condition
    running = True
    clock = pygame.time.Clock()
    floor = 1
    map_grid, rooms = make_map(30, 6, 10)
    player = entities.Player(0, 0, "Player")
    place_statics(map_grid, rooms, floor)

    entities_list = []
    entities_list.append(player)
    entities_list = place_entities(map_grid, rooms, floor, player, entities_list)

    visibility_grid = [[False for _ in range(len(map_grid[0]))] for _ in range(len(map_grid))]

    while running:
        
        update_vision_normal(player.x, player.y, map_grid, visibility_grid)
        update_vision_lit_rooms(player.x, player.y, rooms, visibility_grid)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
            if event.type == pygame.KEYDOWN:
                playerUsedTurn = False
                if event.key == pygame.K_LEFT:
                    player.move(-1, 0, map_grid, entities_list)
                    playerUsedTurn = True
                elif event.key == pygame.K_RIGHT:
                    player.move(1, 0, map_grid, entities_list)
                    playerUsedTurn = True
                elif event.key == pygame.K_UP:
                    player.move(0, -1, map_grid, entities_list)
                    playerUsedTurn = True
                elif event.key == pygame.K_DOWN:
                    player.move(0, 1, map_grid, entities_list)
                    playerUsedTurn = True

                if playerUsedTurn:
                    # After the player moves, enemies take their turn
                    enemies = [ent for ent in entities_list if ent.isHostile]
                    for enemy in enemies:
                        enemy.chooseAction(map_grid, player, entities_list)
        
        screen.fill(BLACK)
        draw_game_based_on_visibility(screen, map_grid, visibility_grid, entities_list)

        pygame.display.flip()
        clock.tick(60)

Engine()

pygame.quit()
sys.exit()