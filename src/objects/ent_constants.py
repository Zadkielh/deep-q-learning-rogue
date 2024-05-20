# Constants
PLAYER = 0
# Enemy Constants
# --------------------------------------------------------
GOBLIN = 1
HOBGOBLIN = 2
SKELETON_UNARMED = 3
SKELETON_SWORD_SHIELD = 4
SKELETON_BOW = 5
SKELETON_ARMORED = 6
# Tier 2 Specific
SKELETON_MAGE = 7
OGRE = 8
OGRE_ARMORED = 9
ORC = 10
ORC_BOW = 11
# Tier 3 Specific
GHOUL = 12
ORC_BRUTE = 13
OGRE_BERSERKER = 14
SKELETON_KNIGHT = 15
GHOST = 16
# Bosses
DEATH_KNIGHT = 17
ORC_WARLORD = 18
OGRE_WARLORD = 19
NECROMANCER = 20
# --------------------------------------------------------

# View Distance
VIEW_DISTANCE_LOW = (-1,2)
VIEW_DISTANCE_MEDIUM = (-2,3)
VIEW_DISTANCE_HIGH = (-3,4)
VIEW_DISTANCE_MAX = (-100, 100)

ENTITIES_COLLISION = [
    PLAYER,
    GOBLIN
]