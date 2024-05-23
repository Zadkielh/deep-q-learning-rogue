from objects.entities import *

class Goblin(Enemy):
    def __init__(self, x, y):
        super().__init__(x, y)
        self.name = "Goblin"
        self.health = 5
        self.strength = 2
        self.dexterity = 5
        self.agility = 5
        self.type = GOBLIN
        self.color = (100,230,5)

        self.lastHealth = self.health

        self.xpValue = 2

class HobGoblin(Enemy):
    def __init__(self, x, y):
        super().__init__(x, y)
        self.name = "Hob-Goblin"
        self.health = 10
        self.armor = 2
        self.strength = 5
        self.dexterity = 2
        self.agility = 2
        self.type = HOBGOBLIN
        self.color = (82,83,4)

        self.lastHealth = self.health

        self.xpValue = 3

class Skeleton(Enemy):
    def __init__(self, x, y):
        super().__init__(x, y)
        self.name = "Shambling Skeleton"
        self.health = 5
        self.armor = 1
        self.strength = 3
        self.dexterity = 3
        self.agility = 3
        self.type = SKELETON_UNARMED
        self.color = (200,200,200)

        self.lastHealth = self.health

        self.xpValue = 2

class SkeletonSS(Skeleton):
    def __init__(self, x, y):
        super().__init__(x, y)
        self.name = "Warrior Skeleton"
        self.health = 5
        self.armor = 2
        self.strength = 5
        self.dexterity = 3
        self.agility = 3
        self.type = SKELETON_SWORD_SHIELD
        self.color = (200,200,200)

        self.lastHealth = self.health

        self.xpValue = 3

class SkeletonBow(Skeleton):
    def __init__(self, x, y):
        super().__init__(x, y)
        self.name = "Archer Skeleton"
        self.health = 3
        self.strength = 2
        self.dexterity = 5
        self.agility = 5
        self.type = SKELETON_BOW
        self.color = (200,200,200)

        self.lastHealth = self.health

        self.xpValue = 3

class SkeletonArmored(Skeleton):
    def __init__(self, x, y):
        super().__init__(x, y)
        self.name = "Hulking Skeleton"
        self.health = 10
        self.armor = 3
        self.strength = 5
        self.dexterity = 1
        self.agility = 1
        self.type = SKELETON_ARMORED
        self.color = (200,200,200)

        self.lastHealth = self.health

        self.xpValue = 4

# Tier 2 SPECIFIC Enemies

class SkeletonMage(Skeleton):
    def __init__(self, x, y):
        super().__init__(x, y)
        self.name = "Skeleton Wizard"
        self.health = 15
        self.armor = 3
        self.strength = 3
        self.dexterity = 5
        self.agility = 3
        self.type = SKELETON_MAGE
        self.color = (200,200,230)
        
        self.lastHealth = self.health

        self.xpValue = 4

class Ogre(HobGoblin):
    def __init__(self, x, y):
        super().__init__(x, y)
        self.name = "Ogre"
        self.health = 30
        self.armor = 0
        self.strength = 10
        self.dexterity = 5
        self.agility = 1
        self.type = OGRE
        self.color = (230,211,132)

        self.lastHealth = self.health

        self.xpValue = 6

class OgreArmored(HobGoblin):
    def __init__(self, x, y):
        super().__init__(x, y)
        self.name = "Armored Ogre"
        self.health = 30
        self.armor = 5
        self.strength = 10
        self.dexterity = 3
        self.agility = 1
        self.type = OGRE_ARMORED
        self.color = (230,211,132)
        
        self.lastHealth = self.health

        self.xpValue = 7

class Orc(HobGoblin):
    def __init__(self, x, y):
        super().__init__(x, y)
        self.name = "Orc"
        self.health = 15
        self.armor = 3
        self.strength = 7
        self.dexterity = 5
        self.agility = 5
        self.type = ORC
        self.color = (230,250,152)

        self.lastHealth = self.health

        self.xpValue = 5

class OrcBow(HobGoblin):
    def __init__(self, x, y):
        super().__init__(x, y)
        self.name = "Orc Sharpshooter"
        self.health = 10
        self.armor = 1
        self.strength = 9
        self.dexterity = 7
        self.agility = 7
        self.type = ORC_BOW
        self.color = (230,250,152)

        self.lastHealth = self.health

        self.xpValue = 5

# Tier 3 Specific
class Ghoul(Enemy):
    def __init__(self, x, y):
        super().__init__(x, y)
        self.name = "Ghoul"
        self.health = 15
        self.armor = 3
        self.strength = 10
        self.dexterity = 8
        self.agility = 8
        self.type = GHOUL
        self.color = (233, 116, 81)

        self.lastHealth = self.health

        self.xpValue = 10

class OrcBrute(Enemy):
    def __init__(self, x, y):
        super().__init__(x, y)
        self.name = "Orc Brute"
        self.health = 25
        self.armor = 5
        self.strength = 13
        self.dexterity = 5
        self.agility = 5
        self.type = ORC_BRUTE
        self.color = (230,250,152)

        self.lastHealth = self.health

        self.xpValue = 10

class OgreBerserker(Enemy):
    def __init__(self, x, y):
        super().__init__(x, y)
        self.name = "Ogre Berserker"
        self.health = 40
        self.armor = 0
        self.strength = 20
        self.dexterity = 7
        self.agility = 5
        self.type = OGRE_BERSERKER
        self.color = (230,211,132)

        self.lastHealth = self.health

        self.xpValue = 12

class SkeletonKnight(Skeleton):
    def __init__(self, x, y):
        super().__init__(x, y)
        self.name = "Undead Knight"
        self.health = 20
        self.armor = 10
        self.strength = 10
        self.dexterity = 10
        self.agility = 5
        self.type = SKELETON_KNIGHT
        self.color = (175,175,175)

        self.lastHealth = self.health

        self.xpValue = 12

class Ghost(Skeleton):
    def __init__(self, x, y):
        super().__init__(x, y)
        self.name = "Ghost"
        self.health = 1
        self.armor = 0
        self.strength = 10
        self.dexterity = 10
        self.agility = 25
        self.type = GHOST
        self.color = (200,200,255)

        self.lastHealth = self.health
        
        self.xpValue = 10

# Bosses
class DeathKnight(Enemy):
    def __init__(self, x, y):
        super().__init__(x, y)
        self.name = "Death Knight"
        self.health = 1
        self.armor = 1
        self.strength = 1
        self.dexterity = 1
        self.agility = 1
        self.type = DEATH_KNIGHT
        self.color = (0, 0, 0)

class OrcWarlord(Enemy):
    def __init__(self, x, y):
        super().__init__(x, y)
        self.name = "Orc Warlord"
        self.health = 1
        self.armor = 1
        self.strength = 1
        self.dexterity = 1
        self.agility = 1
        self.type = ORC_WARLORD
        self.color = (0, 0, 0)

class OgreWarlord(Enemy):
    def __init__(self, x, y):
        super().__init__(x, y)
        self.name = "Ogre Warlord"
        self.health = 1
        self.armor = 1
        self.strength = 1
        self.dexterity = 1
        self.agility = 1
        self.type = OGRE_WARLORD
        self.color = (0, 0, 0)

class Necromancer(Enemy):
    def __init__(self, x, y):
        super().__init__(x, y)
        self.name = "Necromancer"
        self.health = 1
        self.armor = 1
        self.strength = 1
        self.dexterity = 1
        self.agility = 1
        self.type = NECROMANCER
        self.color = (0, 0, 0)

ENEMY_CLASSES = {
    GOBLIN: Goblin,
    HOBGOBLIN: HobGoblin,
    SKELETON_UNARMED: Skeleton,
    SKELETON_SWORD_SHIELD: SkeletonSS,
    SKELETON_ARMORED: SkeletonArmored,
    SKELETON_BOW: SkeletonBow,
    SKELETON_MAGE: SkeletonMage,
    OGRE: Ogre,
    OGRE_ARMORED: OgreArmored,
    ORC: Orc,
    ORC_BOW: OrcBow,
    GHOUL: Ghoul,
    ORC_BRUTE: OrcBrute,
    OGRE_BERSERKER: OgreBerserker,
    SKELETON_KNIGHT: SkeletonKnight,
    GHOST: Ghost,
    # Bosses
    DEATH_KNIGHT: DeathKnight,
    ORC_WARLORD: OrcWarlord,
    OGRE_WARLORD: OgreWarlord,
    NECROMANCER: Necromancer
}