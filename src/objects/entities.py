from objects import tiles
from objects.ent_constants import *
import random


def IsTileBlocked(x, y, grid, entities, caller=None):
    if not (0 <= y < len(grid) and 0 <= x < len(grid[0])):
        return False, None
    if grid[y][x] in tiles.BLOCKED_TILES:
        return True, grid[y][x]
    
    for entity in entities:
        if entity is caller:
            continue
        if entity.x == x and entity.y == y:
            return True, entity
    return False, None

def PlaceEnemy(x, y, grid, entities):
    if IsTileBlocked(x, y, grid, entities)[0]:
        for dx in range(-5, 6):
            for dy in range(-5, 6):
                if not IsTileBlocked(x + dx, y + dy, grid, entities)[0]:
                    return x + dx, y + dy
    return x, y

def PlaceItem(x, y, grid, entities):
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

        self.level = 1
        self.xp = 0
        self.xpValue = 0

    def move(self, dx, dy, grid, entities, notification_manager):
        blocked, type = IsTileBlocked(self.x + dx, self.y + dy, grid, entities)
        if not blocked:
            # Move
            self.x += dx
            self.y += dy
        else:
            if not isinstance(type, int):
                self.interact(type, notification_manager)
                
    def set_pos(self, x, y):
        self.x = x
        self.y = y

    def interact(self, caller):
        pass

    def GetDamage(self):
        return self.strength
    
    def GetArmor(self):
        return self.armor
    
    def GetHitChance(self):
        return self.dexterity*2
    
    def GetDodgeChance(self):
        return self.agility*2
    
    def DoMitigation(self, damage):
        return damage
    
    def DoDamageAdjustment(self, damage):
        return damage
    
    def Die(self, caller, notification_manager):
        notification_manager.add_notification(f"{self.name} has been slain by {caller.name}!")
        caller.xp += self.xpValue
        self.isAlive = False

class Item(Entity):
    def __init__(self, x, y, name="Item"):
        super().__init__(x, y)
        self.name = name
        self.health = 1
        self.type = ITEM
        self.color = (0,255,255)

        ###

        self.spawnChance = 1
        self.tier = 0

        self.canWield = False

        self.damage = 0
        self.armor = 0
        self.heal = 0
        self.strength = 0
        self.dexterity = 0
        self.agility = 0

    def interact(self, caller, notification_manager):
        pass

    def OnUse(self, caller, index, notification_manager):
        self.removeFromInventory(caller, index)

    def addToInventory(self, caller, notification_manager):
        if isinstance(caller, Player):
            caller.inventory.append(self)
            notification_manager.add_notification(f"You picked up {self.name}.")
    
    def removeFromInventory(self, caller, index):
        if isinstance(caller, Player):
            caller.inventory.pop(index)

    @classmethod
    def get_spawn_chance(cls):
        return cls.spawnChance
    
class Player(Entity):
    def __init__(self, x, y, name="Hero"):
        super().__init__(x, y)
        self.name = name
        self.health = 10
        self.maxHealth = 10
        self.maxXp = 4
        self.type = PLAYER
        self.color = (0,0,255)
        self.viewDistance = VIEW_DISTANCE_LOW
        self.inventory = []
        self.equipped = {
            SLOT_HEAD : None,
            SLOT_BODY : None,
            SLOT_RHAND : None,
            SLOT_LHAND : None,
            SLOT_BELT : None,
            SLOT_TWOHAND : None
        }

        self.strength = 2
        self.agility = 2
        self.dexterity = 2
        self.lastx = x
        self.lasty = y

    def Equip(self, item, notification_manager):
        if not item in self.inventory: return False
        if item.canWield:
            slot = item.slot
            if slot == SLOT_TWOHAND:
                # Check if we have equipped in lhand or rhand
                if self.equipped[SLOT_RHAND] != None:
                    # Unequip before equipping
                    self.equipped[SLOT_RHAND] = None
                if self.equipped[SLOT_LHAND] != None:
                    self.equipped[SLOT_LHAND] = None

                self.equipped[slot] = item
                notification_manager.add_notification(f"You have equipped {item.name}.")
                return True
            elif slot == SLOT_RHAND or slot == SLOT_LHAND:
                self.equipped[SLOT_TWOHAND] = None
                self.equipped[slot] = item
            else:
                self.equipped[slot] = item
                notification_manager.add_notification(f"You have equipped {item.name}.")
                return True

    def StatSummary(self):
        print("Base Stats:", self.health, self.armor, self.strength, self.dexterity, self.agility, self.level, self.xp)
        item_list = []
        for item in self.inventory:
            item_list.append(item.name)

        print("Inventory:", item_list)

        equipped_list = []
        for key in self.equipped:
            if self.equipped[key]:
                equipped_list.append(self.equipped[key].name)

        print("Equipped:", equipped_list)

    def UseItem(self, item, notification_manager):
        if item.OnUse:
            item.OnUse(self, notification_manager)

    def LevelUp(self, notification_manager):
        if self.xp >= self.maxXp:
            self.level += 1
            self.xp = 0
            self.maxXp = self.level * 4

            self.maxHealth = 5 + self.level * 5
            self.health = self.maxHealth

            self.strength += 1
            self.dexterity += 1
            self.agility += 1

            notification_manager.add_notification(f"You leveled up! You are now {self.level}")

    def GetDamage(self):
        damage = self.strength
        for key in self.equipped:
            if self.equipped[key]:
                damage += self.equipped[key].damage
                damage += self.equipped[key].strength

        return damage
    
    def GetStrength(self):
        str = self.strength
        for key in self.equipped:
            if self.equipped[key]:
                str += self.equipped[key].strength

        return str
    
    def GetDexterity(self):
        dex = self.dexterity
        for key in self.equipped:
            if self.equipped[key]:
                dex += self.equipped[key].dexterity

        return dex
    
    def GetAgility(self):
        agi = self.agility
        for key in self.equipped:
            if self.equipped[key]:
                agi += self.equipped[key].agility

        return agi
    
    def GetArmor(self):
        armor = self.armor
        for key in self.equipped:
            if self.equipped[key]:
                armor += self.equipped[key].armor

        return armor
    
    def GetHitChance(self):
        dex = self.dexterity
        for key in self.equipped:
            if self.equipped[key]:
                dex += self.equipped[key].dexterity

        return dex * 2
    
    def GetDodgeChance(self):
        agi = self.agility
        for key in self.equipped:
            if self.equipped[key]:
                agi += self.equipped[key].agility

        return agi * 2
    
    def DoMitigation(self, damage):
        return damage
    
    def DoDamageAdjustment(self, damage):
        return damage

    def interact(self, caller, notification_manager):
        if isinstance(caller, Item):
            caller.addToInventory(self, notification_manager)
            caller.isAlive = False
        elif isinstance(caller, Enemy):
            self.DoCombat(caller, notification_manager)
    
    def DoCombat(self, target, notification_manager):
        if isinstance(target, Enemy):
            # Attacker Stats
            damage = self.GetDamage()
            hitchance = self.GetHitChance()

            # Defender Stats
            armor = target.GetArmor()
            dodgechance = target.GetDodgeChance()

            # Check if we hit target
            hit = min(random.randint(0, 100) + hitchance, 100)
            dodge = min(random.randint(0, 100) + dodgechance, 100)

            # We hit
            if hit >= dodge:
                # Do damage increases if applicable
                damage = self.DoDamageAdjustment(damage)
                # Flat damage reduction from armor
                damage -= armor
                # Do extra damage adjustments if applicable
                damage = target.DoMitigation(damage)

                target.health = target.health - max(0, damage)
                notification_manager.add_notification(f"{self.name} struck {target.name} dealing {damage} damage!")
            else:
                notification_manager.add_notification(f"{self.name} misses {target.name}!")

            if (target.health <= 0):
                target.isAlive = False
                target.Die(self, notification_manager)
                self.LevelUp(notification_manager)

    def Die(self, caller, notification_manager):
        notification_manager.add_notification(f"You have perished!")
        self.isAlive = False

        
class Enemy(Entity):
    def __init__(self, x, y):
        super().__init__(x, y)
        self.isHostile = True
        self.color = (255,0,0)
        self.viewDistance = VIEW_DISTANCE_MEDIUM

    def chase(self, target_x, target_y, grid, entities, notification_manager):
        step_x = step_y = 0
        if self.x < target_x:
            step_x = 1
        elif self.x > target_x:
            step_x = -1
        elif self.y < target_y:
            step_y = 1
        elif self.y > target_y:
            step_y = -1

        self.move(step_x, step_y, grid, entities, notification_manager)

    def wander(self, grid, entities, notification_manager):
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
            
            self.move(x, y, grid, entities, notification_manager)

    def canDetectPlayer(self, player):
        
        # Check tiles in a 5x5 radius around the entity
        lbound, ubound = self.viewDistance
        for dy in range(lbound, ubound):
            for dx in range(lbound, ubound):
                ny, nx = max(0, self.y + dy), max(0, self.x + dx)

                if player.x == nx and player.y == ny:
                    return True

        return False
    
    def chooseAction(self, grid, player, entities, notification_manager):
        if self.canDetectPlayer(player):
            self.chase(player.x, player.y, grid, entities, notification_manager)
        else:
            self.wander(grid, entities, notification_manager)

    def interact(self, caller, notification_manager):
        self.DoCombat(caller, notification_manager)
    
    def DoCombat(self, target, notification_manager):
        if isinstance(target, Player):
            # Attacker Stats
            damage = self.GetDamage()
            hitchance = self.GetHitChance()

            # Defender Stats
            armor = target.GetArmor()
            dodgechance = target.GetDodgeChance()

            # Check if we hit target
            hit = min(random.randint(0, 100) + hitchance, 100)
            dodge = min(random.randint(0, 100) + dodgechance, 100)

            # We hit
            if hit >= dodge:
                # Do damage increases if applicable
                damage = self.DoDamageAdjustment(damage)
                # Flat damage reduction from armor
                damage -= armor
                # Do extra damage adjustments if applicable
                damage = target.DoMitigation(damage)

                target.health = target.health - max(0, damage)
                notification_manager.add_notification(f"{self.name} struck {target.name} dealing {damage} damage!")
            else:
                notification_manager.add_notification(f"{self.name} misses {target.name}!")

            if (target.health <= 0):
                target.isAlive = False
                target.Die(self, notification_manager)

class Friendly(Entity):
    def __init__(self, x, y):
        super().__init__(x, y)
        self.isHostile = False
