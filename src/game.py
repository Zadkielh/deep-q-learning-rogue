import pygame
import sys
import random
from objects import tiles
from objects import entities as ent
from objects import enemies as enm
from objects import items as itm

# Constants
WIDTH = 1000
HEIGHT = 600
TILESIZE = 20

HUD_SIZE = 100

game_area_height = HEIGHT

GRIDWIDTH = WIDTH // TILESIZE
GRIDHEIGHT = HEIGHT // TILESIZE

MAX_ROOMS = 30
ROOM_SIZE_MIN = 6
ROOM_SIZE_MAX = 10

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
BLUE_GREY = ( 96, 125, 139 )

OUTLINE_COLOR = (255, 255, 255)


ENEMY_TIER_1 = [
    ent.GOBLIN,
    ent.HOBGOBLIN,
    ent.SKELETON_ARMORED,
    ent.SKELETON_BOW,
    ent.SKELETON_SWORD_SHIELD,
    ent.SKELETON_UNARMED,
]

ENEMY_TIER_2 = [
    ent.HOBGOBLIN,
    ent.SKELETON_ARMORED,
    ent.SKELETON_BOW,
    ent.SKELETON_SWORD_SHIELD,
    ent.SKELETON_MAGE,
    ent.OGRE,
    ent.OGRE_ARMORED,
    ent.ORC,
    ent.ORC_BOW,
]

ENEMY_TIER_3 = [
    ent.SKELETON_ARMORED,
    ent.SKELETON_MAGE,
    ent.OGRE,
    ent.OGRE_ARMORED,
    ent.ORC,
    ent.ORC_BOW,
    ent.SKELETON_KNIGHT,
    ent.GHOST,
    ent.GHOUL,
    ent.ORC_BRUTE,
    ent.OGRE_BERSERKER
]

class NotificationManager:
    def __init__(self, font):
        self.font = font
        self.notifications = []
        self.duration = 10 * 1000 # x1000 for seconds

    def add_notification(self, message):
        timestamp = pygame.time.get_ticks()  # Get the current time
        self.notifications.append((message, timestamp))

    def update(self):
        current_time = pygame.time.get_ticks()
        self.notifications = [
            (message, timestamp) for message, timestamp in self.notifications
            if current_time - timestamp < self.duration
        ]

    def draw(self, screen):
        y_offset = 10
        for message, _ in self.notifications:
            text = self.font.render(message, True, WHITE)
            screen.blit(text, (10, y_offset))
            y_offset += text.get_height() + 5


def draw_name_tag(screen, font, entity, offset_y=10):
    text = font.render(entity.name, True, WHITE)
    text_width = text.get_width()
    text_rect = text.get_rect(center=(entity.x*20 + 10, entity.y*20 - offset_y))
    screen.blit(text, (entity.x*20 + 10 - text_width // 2, entity.y*20 - offset_y))

def GetEnemyFromTier(tier):
    if tier >= 3:
        return random.choice(ENEMY_TIER_3)
    elif tier >= 2:
        return random.choice(ENEMY_TIER_2)
    else:
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

def CreateItem(tier, room, grid, entities):
    x, y = room.center()
    x, y = ent.PlaceItem(x, y, grid, entities)

    item_classes = list(itm.ITEM_CLASSES.values())
    spawn_chances = [item_class.get_spawn_chance() for item_class in item_classes]

    total_spawn_chance = sum(spawn_chances)

    probabilities = [chance / total_spawn_chance for chance in spawn_chances]

    item_class = random.choices(item_classes, probabilities)[0]
    item = item_class(x, y)

    while item.tier > tier:
        item_class = random.choices(item_classes, probabilities)[0]
        item = item_class(x, y)

    return item


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

def place_statics(grid, rooms, floor, player):
    # Place Torches
    torchMod = 95 - (5*floor)
    for room in rooms:
        if torchMod > room.torchChance:
            x = random.choice(range(room.x1, room.x2))
            y = random.choice(range(room.y1, room.y2))
            while not room.contains(x, y):
                x = random.choice(range(room.x1, room.x2))
                y = random.choice(range(room.y1, room.y2))
            
            grid[y][x] = tiles.WALL_TORCH
            room.hasTorch = True


    # Pick a random room that the player doesn't start in
    room = random.choice(rooms)
    while room.contains(player.x, player.y):
        room = random.choice(rooms)

    # Pick a random position thats within the boundary of the room
    x, y = room.center()
    x = random.choice(range(x-1, x+1))
    y = random.choice(range(y-1, y+1))
    
    # Place Stairs
    grid[y][x] = tiles.STAIRS



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
        if room.contains(player.x, player.y):
            spawnMod = 0
        if spawnMod > room.spawnChance:
            # Place Enemies
            enemyCount = random.randint(1, MaxEnemiesPerRoom)
            for _ in range(enemyCount):
                # Select Type
                enemy = CreateEnemy(EnemyTier, room, grid, list)
                list.append(enemy)

        # Spawn Items
        itemSpawnMod = 20
        if itemSpawnMod > random.randint(0, 100):
            item = CreateItem(EnemyTier, room, grid, list)
            list.append(item)


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
    elif current_tile == tiles.STAIRS:
        pygame.draw.rect(screen, BLUE_GREY, rect)
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
            pygame.draw.rect(screen, entity.color, pygame.Rect(entity.x * TILESIZE, entity.y * TILESIZE, TILESIZE, TILESIZE))
            if not isinstance(entity, itm.Item):
                draw_name_tag(screen, font, entity)

def draw_hud(screen, font, player, floor):
    hud_rect = pygame.Rect(0, game_area_height, WIDTH, HUD_SIZE)

    outline_thickness = 3
    outline_rect = pygame.Rect(
        hud_rect.x - outline_thickness,
        hud_rect.y - outline_thickness,
        hud_rect.width + 2 * outline_thickness,
        hud_rect.height + 2 * outline_thickness,
    )
    pygame.draw.rect(screen, OUTLINE_COLOR, outline_rect)

    pygame.draw.rect(screen, BLACK, hud_rect)

    health_text = font.render(f'Health: {player.health} / {player.maxHealth}', True, WHITE)
    health_rect = health_text.get_rect(topleft=(10, game_area_height + 10))

    level_text = font.render(f'Level: {player.level}', True, WHITE)
    level_rect = level_text.get_rect(topleft=(10, game_area_height + 40))

    xp_text = font.render(f'XP: {player.xp} / {player.maxXp}', True, WHITE)
    xp_rect = level_text.get_rect(topleft=(10, game_area_height + 70))

    ####

    armor_text = font.render(f'Armor: {player.GetArmor()}', True, WHITE)
    armor_rect = armor_text.get_rect(topleft=(210, game_area_height + 10))

    sight = "Normal"
    if player.viewDistance == ent.VIEW_DISTANCE_MEDIUM:
        sight = "Great"
    elif player.viewDistance == ent.VIEW_DISTANCE_HIGH:
        sight = "Perfect"
    elif player.viewDistance == ent.VIEW_DISTANCE_MAX:
        sight = "All Seeing"
    view_text = font.render(f'Sight: {sight}', True, WHITE)
    view_rect = view_text.get_rect(topleft=(210, game_area_height + 40))

    damage_text = font.render(f'Damage: {player.GetDamage()}', True, WHITE)
    damage_rect = damage_text.get_rect(topleft=(210, game_area_height + 70))

    ###

    str_text = font.render(f'Strength: {player.GetStrength()}', True, WHITE)
    str_rect = str_text.get_rect(topleft=(410, game_area_height + 10))

    dex_text = font.render(f'Dexterity: {player.GetDexterity()}', True, WHITE)
    dex_rect = dex_text.get_rect(topleft=(410, game_area_height + 40))

    agi_text = font.render(f'Agility: {player.GetAgility()}', True, WHITE)
    agi_rect = agi_text.get_rect(topleft=(410, game_area_height + 70))

    ###

    floor_text = font.render(f'Floor: {floor}', True, WHITE)
    floor_rect = floor_text.get_rect(topleft=(610, game_area_height + 40))

    screen.blit(health_text, health_rect)
    screen.blit(level_text, level_rect)
    screen.blit(xp_text, xp_rect)
    #
    screen.blit(armor_text, armor_rect)
    screen.blit(view_text, view_rect)
    screen.blit(damage_text, damage_rect)
    #
    screen.blit(str_text, str_rect)
    screen.blit(dex_text, dex_rect)
    screen.blit(agi_text, agi_rect)
    #

    screen.blit(floor_text, floor_rect)


def draw_inventory(screen, font, player):
    inventory_text = font.render("Inventory:", True, WHITE)
    screen.blit(inventory_text, (710, game_area_height + 10))
    y_offset = game_area_height + 25
    x_offset = 710
    for i, item in enumerate(player.inventory):
        if y_offset > HEIGHT + HUD_SIZE - 20:
            x_offset += 100
            y_offset = game_area_height + 25
         
        item_text = font.render(f"{i} - {item.name}", True, WHITE)
        screen.blit(item_text, (x_offset, y_offset))
        y_offset += item_text.get_height() + 5


def NextLevel(player, floor, entities_list, notification_manager):
    floor += 1
    map_grid, rooms = make_map(MAX_ROOMS, ROOM_SIZE_MIN, ROOM_SIZE_MAX)
    entities_list = []
    entities_list.append(player)

    entities_list = place_entities(map_grid, rooms, floor, player, entities_list)
    place_statics(map_grid, rooms, floor, player)

    notification_manager.add_notification(f"You move further down..")

    visibility_grid = [[False for _ in range(len(map_grid[0]))] for _ in range(len(map_grid))]

    return map_grid, rooms, entities_list, floor, visibility_grid

# Initialize Game
pygame.init()

# Set Screen Size
screen = pygame.display.set_mode((WIDTH, HEIGHT + HUD_SIZE))
pygame.display.set_caption('Rogue-like')

# Font
font = pygame.font.Font(None, 14)
hud_font = pygame.font.Font(None, 24)
noti_font = pygame.font.Font(None, 18)

# Game Loop
def Engine():
    engine_data = {}
    engine_data['running'] = True
    engine_data['clock'] = pygame.time.Clock()
    engine_data['notification_manager'] = NotificationManager(hud_font)
    engine_data['floor'] = 1
    engine_data['map_grid'], engine_data['rooms'] = make_map(MAX_ROOMS, ROOM_SIZE_MIN, ROOM_SIZE_MAX)
    engine_data['player'] = ent.Player(0, 0)

    # Give starter items to player
    weapon = itm.IronSword(0, 0)
    engine_data['player'].interact(weapon, engine_data['notification_manager'])
    engine_data['player'].Equip(weapon, engine_data['notification_manager'])

    armor = itm.ChainMail(0, 0)
    engine_data['player'].interact(armor, engine_data['notification_manager'])
    engine_data['player'].Equip(armor, engine_data['notification_manager'])

    food = itm.Food(0, 0)
    engine_data['player'].interact(food, engine_data['notification_manager'])

    engine_data['entities_list'] = []
    engine_data['entities_list'].append(engine_data['player'])
    engine_data['entities_list'] = place_entities(engine_data['map_grid'], engine_data['rooms'], engine_data['floor'], engine_data['player'], engine_data['entities_list'])
    place_statics(engine_data['map_grid'], engine_data['rooms'], engine_data['floor'], engine_data['player'])

    engine_data['visibility_grid'] = [[False for _ in range(len(engine_data['map_grid'][0]))] for _ in range(len(engine_data['map_grid']))]

    engine_data['notification_manager'].add_notification(f"Your descent starts..")

    return engine_data

def step_game(engine_data, action):
    player = engine_data['player']
    playerUsedTurn = False
    
    if action == 0:  # Up
        player.move(0, -1, engine_data['map_grid'], engine_data['entities_list'], engine_data['notification_manager'])
        playerUsedTurn = True
    elif action == 1:  # Down
        player.move(0, 1, engine_data['map_grid'], engine_data['entities_list'], engine_data['notification_manager'])
        playerUsedTurn = True
    elif action == 2:  # Left
        player.move(-1, 0, engine_data['map_grid'], engine_data['entities_list'], engine_data['notification_manager'])
        playerUsedTurn = True
    elif action == 3:  # Right
        player.move(1, 0, engine_data['map_grid'], engine_data['entities_list'], engine_data['notification_manager'])
        playerUsedTurn = True

    if playerUsedTurn:
        for entity in engine_data['entities_list']:
            if entity.isHostile:
                entity.chooseAction(engine_data['map_grid'], player, engine_data['entities_list'], engine_data['notification_manager'])
                
    update_vision_normal(player.x, player.y, engine_data['map_grid'], engine_data['visibility_grid'], player.viewDistance)
    update_vision_lit_rooms(player.x, player.y, engine_data['rooms'], engine_data['visibility_grid'])

    if not player.isAlive:
        engine_data['running'] = False

    if engine_data['map_grid'][player.y][player.x] == tiles.STAIRS:
        # Go to next level
        engine_data['map_grid'], engine_data['rooms'], engine_data['entities_list'], engine_data['floor'], engine_data['visibility_grid'] = NextLevel(player, engine_data['floor'], engine_data['entities_list'], engine_data['notification_manager'])
        
    return engine_data

def render_game(engine_data):
    screen.fill(BLACK)
    draw_game_based_on_visibility(screen, engine_data['map_grid'], engine_data['visibility_grid'], engine_data['entities_list'])
    draw_hud(screen, hud_font, engine_data['player'], engine_data['floor'])
    engine_data['notification_manager'].draw(screen)
    pygame.display.flip()
