
# Pheremone types:
NO_PHER = 0
FOOD_PHER = 1
NO_FOOD_PHER = 2

# Intensity: 0-10, 0 being no pheremone, 10 being recent pheremone

"""
Pheremone rules:
- Can only exist in air
- Pheremone grid only use is to iterate thru all pheremones?
"""
BLACK = (0, 0, 0)  # Background
WHITE = (255, 255, 255)  # Border for tunnels
YELLOW = (255,255,0)  # Food
BROWN = (139,69,19)
RED = (255,0,0)
GRAY = (100,100,100)
GREEN = (0,255,0)
BLUE = (0,0,255)
GRID_SIZE = 40

food_pheremone_grid = [[(0,x,y) for x in range(GRID_SIZE)] for y in range(GRID_SIZE)]  # 0-10 intensity, x of source, y of source
no_food_pheremone_grid = [[(1,x,y) for x in range(GRID_SIZE)] for y in range(GRID_SIZE)]

pheremone_list = {
    FOOD_PHER:food_pheremone_grid,
    NO_FOOD_PHER:no_food_pheremone_grid
}
def pheremone_to_color(pheremone_type):
    if pheremone_type == FOOD_PHER:
        return GREEN
    elif pheremone_type == NO_FOOD_PHER:
        return BLUE


def return_pheremone_color(x,y):
    colors = []
    for key,item in pheremone_list.items():
        if item[x][y][0] > 0:
            colors.append(pheremone_to_color(key))
    return colors
print(return_pheremone_color(0,0))