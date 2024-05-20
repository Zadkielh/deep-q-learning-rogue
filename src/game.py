import pygame
import sys
import random
from objects import tiles
from objects import entities as ent
from objects import enemies as enm

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
ALMOND = (234, 221, 202)
ORANGE = (255,200,100)
BLUE = (0,0,255)
RED = (255,0,0)
GREEN = (0,255,0)


ENEMY_TIER_1 = [
    ent.GOBLIN,
    ent.HOBGOBLIN,
    ent.SKELETON_ARMORED,
    ent.SKELETON_BOW,
    ent.SKELETON_SWORD_SHIELD,
    ent.SKELETON_UNARMED,
    ##### Test
    ent.SKELETON_KNIGHT,
    ent.SKELETON_MAGE,
    ent.GHOST,
    ent.GHOUL,
    ent.ORC,
    ent.ORC_BOW,
    ent.ORC_BRUTE,
    ent.OGRE,
    ent.OGRE_ARMORED,
    ent.OGRE_BERSERKER
]

ENEMY_TIER_2 = [
    ent.HOBGOBLIN,
    ent.SKELETON_ARMORED,
    ent.SKELETON_BOW,
    ent.SKELETON_SWORD_SHIELD,
]

def draw_name_tag(screen, font, entity, offset_y=10):
    text = font.render(entity.name, True, WHITE)
    text_width = text.get_width()
    text_rect = text.get_rect(center=(entity.x*20 + 10, entity.y*20 - offset_y))
    screen.blit(text, (entity.x*20 + 10 - text_width // 2, entity.y*20 - offset_y))

def GetEnemyFromTier(tier):
    if tier < 2:
        return random.choice(ENEMY_TIER_1)
    
def CreateEnemy(tier, room, grid, entities):
    x, y = room.center()
    x, y = ent.PlaceEnemy(x, y, grid, entities)
    enemy_type = GetEnemyFromTier(tier)
    enemy_class = enm.ENEMY_CLASSES.get(enemy_type)
    if enemy_class:
        enemy = enemy_class(x, y)
        return enemy
    return None

def create_tunnel_to_room(grid, room, target_room):
    start_x, start_y = room.center()
    end_x, end_y = target_room.center()

    DIR_X = 1
    DIR_Y = 0

    def adjacent_door(x, y):
        adjacent_door = (grid[y-1][x] == tiles.DOOR if y > 0 else False) or \
                        (grid[y+1][x] == tiles.DOOR if y < len(grid) - 1 else False) or \
                        (grid[y][x-1] == tiles.DOOR if x > 0 else False) or \
                        (grid[y][x+1] == tiles.DOOR if x < len(grid[0]) - 1 else False)
        
        return adjacent_door
    
    def place_door(x, y, direction, step, force=False):
        # Check if there is a floor behind or infront of the wall
        if direction == DIR_X:
            if grid[y][x+step] == tiles.FLOOR or grid[y][x-step] == tiles.FLOOR:
                if not adjacent_door(x, y) and not force:
                    grid[y][x] = tiles.DOOR
                    return True
        else:
            if grid[y+step][x] == tiles.FLOOR or grid[y-step][x] == tiles.FLOOR:
                if not adjacent_door(x, y) and not force:
                    grid[y][x] = tiles.DOOR
                    return True

        return False

    direction = True if abs(end_x - start_x) > abs(end_y - start_y) else False # True = Horizontal | False = Vertical
    i = 0
    while start_x != end_x or start_y != end_y:
        i += 1
        #print(i, start_x, end_x, start_y, end_y, direction)
        if direction:
            if start_x == end_x: 
                direction = False
                continue
            step = 1 if start_x < end_x else -1
            current_tile = grid[start_y][start_x]
            if current_tile == tiles.VOID:
                grid[start_y][start_x] = tiles.TUNNEL
            elif current_tile == tiles.WALL:
                door_placed = place_door(start_x, start_y, DIR_X, step)
                if not door_placed:
                    # We need to move in another direction
                    if start_y + 1 < len(grid) and grid[start_y + 1][start_x] == tiles.WALL:
                        if grid[start_y + 1][start_x + step] == tiles.WALL: # Likely a corner
                            # We need to make an L around the corner
                            start_y += 1
                            if grid[start_y][start_x] == tiles.VOID:
                                grid[start_y][start_x] = tiles.TUNNEL
                            start_y += 1
                            if grid[start_y][start_x] == tiles.VOID:
                                grid[start_y][start_x] = tiles.TUNNEL
                            start_x += step
                            if grid[start_y][start_x] == tiles.VOID:
                                grid[start_y][start_x] = tiles.TUNNEL
                            start_x += step
                            if grid[start_y][start_x] == tiles.VOID:
                                grid[start_y][start_x] = tiles.TUNNEL
                        elif adjacent_door(start_x, start_y):
                            place_door(start_x, start_y, DIR_X, step, True)
                        else:
                            start_y += 1
                            start_x -= step
                            if grid[start_y][start_x] == tiles.VOID:
                                grid[start_y][start_x] = tiles.TUNNEL
                    elif start_y - 1 >= 0 and grid[start_y - 1][start_x] == tiles.WALL:
                        if grid[start_y - 1][start_x + step] == tiles.WALL: # Likely a corner
                            # We need to make an L around the corner
                            start_y -= 1
                            if grid[start_y][start_x] == tiles.VOID:
                                grid[start_y][start_x] = tiles.TUNNEL
                            start_y -= 1
                            if grid[start_y][start_x] == tiles.VOID:
                                grid[start_y][start_x] = tiles.TUNNEL
                            start_x += step
                            if grid[start_y][start_x] == tiles.VOID:
                                grid[start_y][start_x] = tiles.TUNNEL
                            start_x += step
                            if grid[start_y][start_x] == tiles.VOID:
                                grid[start_y][start_x] = tiles.TUNNEL
                        elif adjacent_door(start_x, start_y):
                            place_door(start_x, start_y, DIR_X, step, True)
                        else:
                            start_y -= 1
                            start_x -= step
                            if grid[start_y][start_x] == tiles.VOID:
                                grid[start_y][start_x] = tiles.TUNNEL
            start_x += step

        else:
            if start_y == end_y: 
                direction = True
                continue
            step = 1 if start_y < end_y else -1
            current_tile = grid[start_y][start_x]
            if current_tile == tiles.VOID:
                grid[start_y][start_x] = tiles.TUNNEL
            elif current_tile == tiles.WALL:
                door_placed = place_door(start_x, start_y, DIR_Y, step)
                if not door_placed:
                    # We need to move in another direction
                    if start_x + 1 < len(grid[0]) and grid[start_y][start_x + 1] == tiles.WALL:
                        if grid[start_y + step][start_x + 1] == tiles.WALL: # Likely a corner
                            # We need to make an L around the corner
                            start_x += 1
                            if grid[start_y][start_x] == tiles.VOID:
                                grid[start_y][start_x] = tiles.TUNNEL
                            start_x += 1
                            if grid[start_y][start_x] == tiles.VOID:
                                grid[start_y][start_x] = tiles.TUNNEL
                            start_y += step
                            if grid[start_y][start_x] == tiles.VOID:
                                grid[start_y][start_x] = tiles.TUNNEL
                            start_y += step
                            if grid[start_y][start_x] == tiles.VOID:
                                grid[start_y][start_x] = tiles.TUNNEL
                        elif adjacent_door(start_x, start_y):
                            place_door(start_x, start_y, DIR_Y, step, True)
                        else:
                            start_x += 1
                            start_y -= step
                            if grid[start_y][start_x] == tiles.VOID:
                                grid[start_y][start_x] = tiles.TUNNEL
                    elif start_x - 1 >= 0 and grid[start_y][start_x - 1] == tiles.WALL:
                        if grid[start_y + step][start_x - 1] == tiles.WALL: # Likely a corner
                            # We need to make an L around the corner
                            start_x -= 1
                            if grid[start_y][start_x] == tiles.VOID:
                                grid[start_y][start_x] = tiles.TUNNEL
                            start_x -= 1
                            if grid[start_y][start_x] == tiles.VOID:
                                grid[start_y][start_x] = tiles.TUNNEL
                            start_y += step
                            if grid[start_y][start_x] == tiles.VOID:
                                grid[start_y][start_x] = tiles.TUNNEL
                            start_y += step
                            if grid[start_y][start_x] == tiles.VOID:
                                grid[start_y][start_x] = tiles.TUNNEL
                        elif adjacent_door(start_x, start_y):
                            # Force a door here
                            place_door(start_x, start_y, DIR_Y, step, True)
                        else:
                            start_x -= 1
                            start_y -= step
                            if grid[start_y][start_x] == tiles.VOID:
                                grid[start_y][start_x] = tiles.TUNNEL
            
            start_y += step
            
        if i > 1000:  # Arbitrary large number to prevent infinite loops
            print("Infinite loop detected, breaking out of the loop.", start_x, end_x, start_y, end_y, direction)
            grid[start_y][start_x] = tiles.DEBUG
            grid[end_y][end_x] = tiles.DEBUG
            break



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
        
        rooms.append(new_room)

    for i, room in enumerate(rooms):
        if i == 0:
            last_room = room
            continue
        create_tunnel_to_room(grid, last_room, room)
        last_room = room

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
    MaxEnemiesPerRoom = 3 + int(EnemyTier)

    for room in rooms:
        # Spawn Enemies
        spawnMod = min(20 + (1*floor), 100)
        if room.contains(player.x, player.y):
            spawnMod = 0
        if spawnMod > room.spawnChance:
            # Place Enemies
            enemyCount = random.randint(1, MaxEnemiesPerRoom)
            for _ in range(enemyCount):
                # Select Type
                enemy = CreateEnemy(EnemyTier, room, grid, list)
                list.append(enemy)

    return list

def update_vision_normal(player_x, player_y, grid, visibility_grid, radius):
    minvalue, maxvalue = radius
    for dy in range(minvalue, maxvalue):
        for dx in range(minvalue, maxvalue):
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
    current_tile = grid[y][x]
    if current_tile == tiles.FLOOR:
        pygame.draw.rect(screen, ALMOND, rect)
    elif current_tile == tiles.DEBUG:
        pygame.draw.rect(screen, RED, rect)
    elif current_tile == tiles.TUNNEL:
        pygame.draw.rect(screen, DARK_GREY, rect)
    elif current_tile == tiles.WALL:
        pygame.draw.rect(screen, DARK_BROWN, rect)
    elif current_tile == tiles.WALL_TORCH:
        pygame.draw.rect(screen, ORANGE, rect)
    elif current_tile == tiles.DOOR:
        pygame.draw.rect(screen, BROWN, rect)
    else:
        pygame.draw.rect(screen, BLACK, rect)

def draw_game_based_on_visibility(screen, map_grid, visibility_grid, entities_list):
    for y in range(len(map_grid)):
        for x in range(len(map_grid[0])):
            if visibility_grid[y][x]:  # Only draw if the tile is visible
                draw_tile(x, y, map_grid, screen)  # Implement drawing based on tile type
    # Draw player and enemies if they are in visible tiles
    for entity in entities_list:
            if visibility_grid[entity.y][entity.x]:
                pygame.draw.rect(screen, entity.color, pygame.Rect(entity.x*20, entity.y*20, 20, 20))
                draw_name_tag(screen, font, entity)

# Initiliaze Game
pygame.init()

# Set Screen Size
screen = pygame.display.set_mode((WIDTH,HEIGHT))
pygame.display.set_caption('Rogue-like')

# Font
font = pygame.font.Font(None, 14)

# Game Loop
def Engine():

    # Engine Condition
    running = True
    clock = pygame.time.Clock()
    floor = 1
    map_grid, rooms = make_map(30, 6, 10)
    player = ent.Player(0, 0)
    place_statics(map_grid, rooms, floor)

    entities_list = []
    entities_list.append(player)
    entities_list = place_entities(map_grid, rooms, floor, player, entities_list)

    visibility_grid = [[False for _ in range(len(map_grid[0]))] for _ in range(len(map_grid))]

    while running:
        
        update_vision_normal(player.x, player.y, map_grid, visibility_grid, player.viewDistance)
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

                if event.key == pygame.K_q:
                    player.viewDistance = ent.VIEW_DISTANCE_MAX

                if playerUsedTurn:
                    for id, entity in enumerate(entities_list):
                        if not entity.isAlive:
                            entities_list.pop(id)

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