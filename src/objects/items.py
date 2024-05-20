from objects.entities import *

class IronSword(Item):
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

class ChainMail(Item):
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
    def __init__(self, x, y, name="Hearty Meal"):
        super().__init__(x, y)
        self.name = name
        self.health = 1
        self.type = ITEM_FOOD
        self.color = (0,255,255)

        ###

        self.heal = 5
    
    def OnUse(self, caller, notification_manager):
        caller.health = min(caller.health + self.heal, caller.maxHealth)
        notification_manager.add_notification(f"You regain {self.heal} health!")


ITEM_CLASSES = {
    ITEM_IRON_SWORD : IronSword,
    ITEM_CHAIN_MAIL : ChainMail,
    ITEM_FOOD : Food
}