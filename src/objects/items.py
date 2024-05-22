from objects.entities import *

class IronSword(Item):
    spawnChance = 1

    def __init__(self, x, y, name="Iron Sword"):
        super().__init__(x, y)
        self.name = name
        self.health = 1
        self.type = ITEM_IRON_SWORD
        self.color = (0,255,255)

        ###

        self.canWield = True
        self.slot = SLOT_RHAND
        self.damage = 3

class SteelSword(Item):
    spawnChance = 1

    def __init__(self, x, y, name="Steel Sword"):
        super().__init__(x, y)
        self.name = name
        self.health = 1
        self.type = ITEM_STEEL_SWORD
        self.color = (0,255,255)

        ###
        self.tier = 2

        self.canWield = True
        self.slot = SLOT_RHAND
        self.damage = 6

class ChainMail(Item):
    spawnChance = 1

    def __init__(self, x, y, name="Chain Mail"):
        super().__init__(x, y)
        self.name = name
        self.health = 1
        self.type = ITEM_CHAIN_MAIL
        self.color = (0,255,255)

        ###

        self.canWield = True
        self.slot = SLOT_BODY
        self.armor = 2

class Food(Item):
    spawnChance = 1.5

    def __init__(self, x, y, name="Hearty Meal"):
        super().__init__(x, y)
        self.name = name
        self.health = 1
        self.type = ITEM_FOOD
        self.color = (0,255,255)

        ###
        self.spawnChance = 1.5

        self.heal = 5
    
    def OnUse(self, caller, index, notification_manager):
        caller.health = min(caller.health + self.heal, caller.maxHealth)
        notification_manager.add_notification(f"You regain {self.heal} health!")
        self.removeFromInventory(caller, index)

class HealthPotion(Item):
    spawnChance = 1.5

    def __init__(self, x, y, name="Health Potion"):
        super().__init__(x, y)
        self.name = name
        self.health = 1
        self.type = ITEM_HEALTH_POTION
        self.color = (0,255,255)

        ###
        self.spawnChance = 1.5

        self.heal = 10
    
    def OnUse(self, caller, index, notification_manager):
        caller.health = min(caller.health + self.heal, caller.maxHealth)
        notification_manager.add_notification(f"You regain {self.heal} health!")
        self.removeFromInventory(caller, index)

class PlateArmor(Item):
    spawnChance = 1

    def __init__(self, x, y, name="Plate Armor"):
        super().__init__(x, y)
        self.name = name
        self.health = 1
        self.type = ITEM_PLATE_ARMOR
        self.color = (0,255,255)

        ###

        self.canWield = True
        self.slot = SLOT_BODY
        self.armor = 4

class ReinforcedPlateArmor(Item):
    spawnChance = 1

    def __init__(self, x, y, name="Reinforced Plate Armor"):
        super().__init__(x, y)
        self.name = name
        self.health = 1
        self.type = ITEM_REINFORCED_PLATE_ARMOR
        self.color = (0,255,255)

        ###
        self.tier = 2

        self.canWield = True
        self.slot = SLOT_BODY
        self.armor = 8
        self.dexterity = -1
        self.agility = -1

class LeatherArmor(Item):
    spawnChance = 1

    def __init__(self, x, y, name="Leather Armor"):
        super().__init__(x, y)
        self.name = name
        self.health = 1
        self.type = ITEM_LEATHER_ARMOR
        self.color = (0,255,255)

        ###

        self.canWield = True
        self.slot = SLOT_BODY
        self.armor = 1
        self.dexterity = 1
        self.agility = 1

class HideArmor(Item):
    spawnChance = 1

    def __init__(self, x, y, name="Hide Armor"):
        super().__init__(x, y)
        self.name = name
        self.health = 1
        self.type = ITEM_HIDE_ARMOR
        self.color = (0,255,255)

        ###
        self.tier = 2

        self.canWield = True
        self.slot = SLOT_BODY
        self.armor = 2
        self.dexterity = 3
        self.agility = 3

class RoundShield(Item):
    spawnChance = 1

    def __init__(self, x, y, name="Round Shield"):
        super().__init__(x, y)
        self.name = name
        self.health = 1
        self.type = ITEM_ROUND_SHIELD
        self.color = (0,255,255)

        ###

        self.canWield = True
        self.slot = SLOT_LHAND
        self.armor = 2

class TowerShield(Item):
    spawnChance = 1

    def __init__(self, x, y, name="Tower Shield"):
        super().__init__(x, y)
        self.name = name
        self.health = 1
        self.type = ITEM_TOWER_SHIELD
        self.color = (0,255,255)

        ###

        self.canWield = True
        self.slot = SLOT_LHAND
        self.armor = 4
        self.dexterity = -1
        self.agility = -2

class IronHalberd(Item):
    spawnChance = 1

    def __init__(self, x, y, name="Iron Halberd"):
        super().__init__(x, y)
        self.name = name
        self.health = 1
        self.type = ITEM_STEEL_HALBERD
        self.color = (0,255,255)

        ###

        self.canWield = True
        self.slot = SLOT_TWOHAND
        self.damage = 10
        self.dexterity = -1

class SteelHalberd(Item):
    spawnChance = 1

    def __init__(self, x, y, name="Steel Halberd"):
        super().__init__(x, y)
        self.name = name
        self.health = 1
        self.type = ITEM_STEEL_HALBERD
        self.color = (0,255,255)

        ###
        self.tier = 2

        self.canWield = True
        self.slot = SLOT_TWOHAND
        self.damage = 15
        self.dexterity = -1

class Moonblade(Item):
    spawnChance = 0.01

    def __init__(self, x, y, name="Moonblade"):
        super().__init__(x, y)
        self.name = name
        self.health = 1
        self.type = ITEM_MOONBLADE
        self.color = (0,255,255)

        ###
        self.spawnChance = 0.01

        self.canWield = True
        self.slot = SLOT_TWOHAND
        self.damage = 15
        self.dexterity = 5
        self.strength = 5

class IronGreatsword(Item):
    spawnChance = 1

    def __init__(self, x, y, name="Iron Greatsword"):
        super().__init__(x, y)
        self.name = name
        self.health = 1
        self.type = ITEM_IRON_GREATSWORD
        self.color = (0,255,255)

        ###

        self.canWield = True
        self.slot = SLOT_TWOHAND
        self.damage = 7

class SteelGreatsword(Item):
    spawnChance = 1

    def __init__(self, x, y, name="Steel Greatsword"):
        super().__init__(x, y)
        self.name = name
        self.health = 1
        self.type = ITEM_STEEL_GREATSWORD
        self.color = (0,255,255)

        ###
        self.tier = 2

        self.canWield = True
        self.slot = SLOT_TWOHAND
        self.damage = 10

ITEM_CLASSES = {
    ITEM_IRON_SWORD : IronSword,
    ITEM_CHAIN_MAIL : ChainMail,
    ITEM_FOOD : Food,
    ITEM_HEALTH_POTION : HealthPotion,
    ITEM_PLATE_ARMOR : PlateArmor,
    ITEM_ROUND_SHIELD : RoundShield,
    ITEM_STEEL_HALBERD : SteelHalberd,
    ITEM_TOWER_SHIELD : TowerShield,
    ITEM_MOONBLADE : Moonblade,
    ITEM_STEEL_SWORD : SteelSword,
    ITEM_REINFORCED_PLATE_ARMOR : ReinforcedPlateArmor,
    ITEM_LEATHER_ARMOR : LeatherArmor,
    ITEM_HIDE_ARMOR : HideArmor,
    ITEM_IRON_GREATSWORD : IronGreatsword,
    ITEM_STEEL_GREATSWORD : SteelGreatsword

}