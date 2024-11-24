"""
Goals:
- Simulate an ant colony using machine learning, see how complex machine learning can get/how closely it can replicate real ant behavior
- A queen, workers


Random things:
- Food collection
- Ants digging
- Ants can only travel on the edge and cannot be adjacent to each other, so they can't all be crammed together and must build tunnels to traverse
- Ants can carry food with them in their node
- Ants can only see one node in all directions, but can transmit information three nodes in all directions (except thru dirt), so a hive mind is possible

Ant goals:
1. Expand the hive
2. 


Queens can:
- Make more ants


Ants can:
- See 1 node in all directions
- Dig earth nodes
- Pick up food nodes



Structure:
- Build it with nodes
- Ants are classes?
"""

import pygame
import math
import os
from textreader import read_txt,return_empty_grid

# Pygame Initialization
pygame.init()

# Simulation Parameters
GRID_SIZE = 40  # Number of nodes per row/column // 40
NODE_SIZE = 15  # Size of each node in pixels // 15
WIDTH, HEIGHT = GRID_SIZE * NODE_SIZE, GRID_SIZE * NODE_SIZE
holding = False  # Holding down mouse
global spacebar_wait
spacebar_wait = False  # Waiting on next input for spacebar to place food
dig_wait = False
FOG_OF_WAR = False
ant_sight = 2
mouse_intensity = 10
pheremone_lifespan = .1  # how many frames per intensity decrease
mouse_radius = 1




# Colors
BLACK = (0, 0, 0)  # Background
WHITE = (255, 255, 255)  # Border for tunnels
YELLOW = (255,255,0)  # Food
BROWN = (139,69,19)
RED = (255,0,0)
GRAY = (100,100,100)
GREEN = (0,255,0)
BLUE = (0,0,255)
#comment

# Grid Initialization
# Start with all nodes as undug earth (represented as 0)
grid = []

for _ in range(GRID_SIZE):  # Create physical grid
    row = []
    for _ in range(GRID_SIZE): # Create columns
        row.append(0)  # Initialize as undug earth
    grid.append(row)

EMPTY_GRID = False
if not EMPTY_GRID:
    maps = {}  # number:name of map
    for index in range(len(os.listdir('./colony_maps'))):
        # print(index)
        # print(os.listdir('./colony_maps'))
        # print(len(os.listdir('./colony_maps')))
        name = os.listdir('./colony_maps')[index]
        maps.update({str(index):name})
        print(name + ":",index)
    map_index = input("Please enter the number of the map you'd like to use: ")
    map_index = int(map_index)
    map_name = maps[str(map_index)]
    grid = read_txt(GRID_SIZE,map_name)




pheremone_grid = []

for _ in range(GRID_SIZE):  # Create physical grid
    row = []
    for _ in range(GRID_SIZE): # Create columns
        row.append((0,0))  # Initialize as (0,0); no pheremone, no intensity
    pheremone_grid.append(row)

# Pygame Setup
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Ant Colony Simulation")
clock = pygame.time.Clock()  # For framerate




# Node types:

GROUND = 0
AIR = 1
FOOD = 2
ANT = 3
DEBUG_NODE = 10

# Pheremone types:
NO_PHER = 0
FOOD_PHER = 1
NO_FOOD_PHER = 2


# Intensity: 0-10, 0 being no pheremone, 10 being recent pheremone

def average_colors(color1,color2,intensity):
    color3 = []
    for x in range(len(color1)):
        color3.append(math.floor(((10 * color1[x]) + (intensity * color2[x]))/(10 + intensity)))
    return tuple(color3)

def return_grid_color(x,y):
    if grid[x][y] == GROUND:  # Undug earth
        return WHITE
    elif grid[x][y] == AIR:  # Air
        return BLACK
    elif grid[x][y] == FOOD:
        return YELLOW
    elif grid[x][y] == ANT:
        return BROWN
    elif grid[x][y] == DEBUG_NODE:
        return RED

def return_pheremone_color(x,y):
    pher_color = None
    if pheremone_grid[x][y][0] == NO_PHER:
        return return_grid_color(x,y)
    elif pheremone_grid[x][y][0] == FOOD_PHER:
        pher_color = GREEN
    elif pheremone_grid[x][y][0] == NO_FOOD_PHER:
        pher_color = BLUE
    return average_colors(return_grid_color(x,y),pher_color,pheremone_grid[x][y][1])

def pixel_to_index(pixel):
    return pixel // NODE_SIZE

def index_to_pixel(index):
    return index * NODE_SIZE



# Function to draw the grid
def draw_grid():
    start_pos_bound_coords = []
    end_pos_bound_coords = []


    def draw_white_bounds(x,y):
        if y>0 and grid[x][y-1] == GROUND:  # Top neighbor
            start_pos_bound_coords.append((index_to_pixel(x), index_to_pixel(y)))
            end_pos_bound_coords.append((index_to_pixel(x+1), index_to_pixel(y)))
        if y<GRID_SIZE - 1 and grid[x][y+1] == GROUND:  # bottom neighbor // works
            start_pos_bound_coords.append((index_to_pixel(x), index_to_pixel(y+1)))
            end_pos_bound_coords.append((index_to_pixel(x+1), index_to_pixel(y+1)))
        if x>0 and grid[x-1][y] == GROUND:  # Left neighbor
            start_pos_bound_coords.append((index_to_pixel(x), index_to_pixel(y)))
            end_pos_bound_coords.append((index_to_pixel(x), index_to_pixel(y+1)))
        if x<GRID_SIZE-1 and grid[x+1][y] == GROUND:  # Left neighbor
            start_pos_bound_coords.append((index_to_pixel(x+1), index_to_pixel(y)))
            end_pos_bound_coords.append((index_to_pixel(x+1), index_to_pixel(y+1)))


    
    if not FOG_OF_WAR or len(Ant.instances) == 0:
        for x in range(GRID_SIZE):
            for y in range(GRID_SIZE):
                rect = pygame.Rect(index_to_pixel(x), index_to_pixel(y), NODE_SIZE, NODE_SIZE)
                if grid[x][y] == GROUND:  # Undug earth
                    pygame.draw.rect(screen, BLACK, rect)  # Earth
                elif grid[x][y] == AIR:  # Air
                    pygame.draw.rect(screen, return_pheremone_color(x,y), rect)  # Air
                    draw_white_bounds(x,y)
                elif grid[x][y] == FOOD:
                    pygame.draw.rect(screen, YELLOW, rect)  # Yellow
                    draw_white_bounds(x,y)
                elif grid[x][y] == ANT:
                    pygame.draw.rect(screen, BROWN, rect)  # Brown
                    draw_white_bounds(x,y)
                elif grid[x][y] == DEBUG_NODE:
                    pygame.draw.rect(screen, RED, rect)  # Brown
        for z in range(len(start_pos_bound_coords)):
            pygame.draw.line(screen,WHITE,start_pos_bound_coords[z],end_pos_bound_coords[z], 1)

    if FOG_OF_WAR and len(Ant.instances) != 0:
        for curr_ant in range(len(Ant.instances)):
            x = Ant.instances[curr_ant].x
            y = Ant.instances[curr_ant].y
            for x1 in range(max(0,x - ant_sight), min(x + ant_sight + 1,GRID_SIZE)):
                for y1 in range(max(0,y - ant_sight), min(y + ant_sight + 1,GRID_SIZE)):
                    rect = pygame.Rect(index_to_pixel(x1), index_to_pixel(y1), NODE_SIZE, NODE_SIZE)
                    if grid[x1][y1] == GROUND:  # Undug earth
                        pygame.draw.rect(screen, BLACK, rect)  # Earth
                    elif grid[x1][y1] == AIR:  # Air
                        pygame.draw.rect(screen, return_pheremone_color(x1,y1), rect)  # Air
                        draw_white_bounds(x1,y1)
                    elif grid[x1][y1] == FOOD:
                        pygame.draw.rect(screen, YELLOW, rect)  # Yellow
                        draw_white_bounds(x1,y1)
                    elif grid[x1][y1] == ANT or (Ant.instances[current_ant].x == x1 and Ant.instances[current_ant].y == y1):
                        pygame.draw.rect(screen, BROWN, rect)  # Brown
                        draw_white_bounds(x1,y1)
                    elif grid[x1][y1] == DEBUG_NODE:
                        pygame.draw.rect(screen, RED, rect)  # Brown
                    else:
                        pygame.draw.rect(screen,GRAY,rect)
            for z in range(len(start_pos_bound_coords)):
                pygame.draw.line(screen,WHITE,start_pos_bound_coords[z],end_pos_bound_coords[z], 1)
                    


# (x - h)² + (y - k)² = r²
running = True
def draw_node(mouse_pos,node_type,rad):
    x, y = mouse_pos
    grid_x = pixel_to_index(x)
    grid_y = pixel_to_index(y)

    if 0 <= grid_x < GRID_SIZE and 0 <= grid_y < GRID_SIZE:
        grid[grid_x][grid_y] = node_type
    for x1 in range(max(0,grid_x - rad), min(grid_x + rad,GRID_SIZE - 1) + 1):
        for y1 in range(max(0,grid_y - rad), min(grid_y + rad,GRID_SIZE - 1) + 1):
            grid[x1][y1] = node_type


def draw_pheremone(mouse_pos,rad,pheremone_type, intensity):
    print("pheremone thing")
    x, y = mouse_pos
    grid_x = pixel_to_index(x)
    grid_y = pixel_to_index(y)

    if 0 <= grid_x < GRID_SIZE and 0 <= grid_y < GRID_SIZE:
        pheremone_grid[grid_x][grid_y] = (pheremone_type,intensity)
    for x1 in range(max(0,grid_x - rad), min(grid_x + rad,GRID_SIZE - 1) + 1):
        for y1 in range(max(0,grid_y - rad), min(grid_y + rad,GRID_SIZE - 1) + 1):
            pheremone_grid[x1][y1] = (pheremone_type, intensity)


def check_bounds(x,y,rad):  # Check nodes w/in radius (2 for 3x3, 1 for 2x2, 0 for 1x1), return list of node types(?)
    # draw_node((x * NODE_SIZE,y * NODE_SIZE),DEBUG_NODE,rad)
    grid1 = []
    for x1 in range(max(0,x - rad), min(x + rad,GRID_SIZE - 1) + 1):
        row1 = []
        for y1 in range(max(0,y - rad), min(y + rad,GRID_SIZE - 1) + 1):
            if x1 == x and y1 == y:
                row1.append(AIR)
            else:
                row1.append(grid[x1][y1])
        grid1.append(row1)
    print(grid1)
    return grid1



# Ant class
ants = {}  # List of ants and the node reference of what they're holding; "0,1":1 if it's an ant at 0,1 holding air
class Ant:
    instances = []
    def __init__(self,x,y):
        valid = False
        if grid[x][y] == AIR:
            for z in check_bounds(x,y,1):
                if GROUND in z:
                    valid = True
                if ANT in z:
                    valid = False
                    print('ant found')
                    break
            if valid:
                self.x = x
                self.y = y
                self.id = len(ants) + 1
                self.holding = AIR
                grid[x][y] = ANT
                ants.update({self.id:[self.x,self.y,self.holding]})
                Ant.instances.append(self)
    def move(self,direction):  # Directions: UP, DOWN, LEFT, RIGHT
        global dig_wait
        if direction == "UP":
            check_x = self.x
            check_y = self.y - 1
        elif direction == "DOWN":
            check_x = self.x
            check_y = self.y + 1
        elif direction == "LEFT":
            check_x = self.x - 1
            check_y = self.y
        elif direction == "RIGHT":
            check_x = self.x + 1
            check_y = self.y
        valid = False
        ant_count = 0
        for z in check_bounds(check_x,check_y,1):
            if GROUND in z:
                valid = True
            if ANT in z:
                ant_count +=1
        if valid and ant_count < 2 and GRID_SIZE > check_x and GRID_SIZE > check_y and check_y >= 0 and check_x >= 0:
            if grid[check_x][check_y] == AIR or (dig_wait and grid[check_x][check_y] == GROUND):
                if dig_wait and grid[check_x][check_y] == GROUND:
                    dig_wait = False
                    self.holding = GROUND
                grid[self.x][self.y] = AIR
                self.x = check_x
                self.y = check_y
                grid[check_x][check_y] = ANT
                ants.update({self.id:[self.x,self.y,self.holding]})
            elif grid[check_x][check_y] == FOOD:
                grid[self.x][self.y] = AIR
                self.x = check_x
                self.y = check_y
                self.holding = FOOD
                grid[check_x][check_y] = ANT
                ants.update({self.id:[self.x,self.y,self.holding]})
        test = list(ants.keys())
        print(ants[test[0]][2])
    def place_food(self,direction):
        if direction == "UP":
            check_x = self.x
            check_y = self.y - 1
        elif direction == "DOWN":
            check_x = self.x
            check_y = self.y + 1
        elif direction == "LEFT":
            check_x = self.x - 1
            check_y = self.y
        elif direction == "RIGHT":
            check_x = self.x + 1
            check_y = self.y
        valid = False
        ant_count = 0
        for z in check_bounds(check_x,check_y,1):
            if GROUND in z:
                valid = True
            if ANT in z:
                ant_count +=1
        if valid and ant_count < 3 and grid[check_x][check_y] == AIR:
            grid[check_x][check_y] = self.holding
            self.holding = AIR
            ants.update({self.id:[self.x,self.y,self.holding]})
    def suicide(self):
        global current_ant
        grid[self.x][self.y] == AIR
        ants.pop(self.id)
        Ant.instances.pop(current_ant)
        current_ant = max(0,current_ant - 1)
        draw_node((index_to_pixel(self.x),index_to_pixel(self.y)),AIR,0)
        del self

    def leave_pheremone(self,node, type, intensity):  # Types of pheremones: FOOD_PHER,NO_FOOD_PHER, leave_pheremone((0,1),"FOOD_PHER", 10)
        pheremone_grid[node[0]][node[1]] = (type,intensity)


def save_file():
    string = str(GRID_SIZE) + " "
    for x in range(GRID_SIZE):
        row = ""
        for y in range(GRID_SIZE):
            row += str(grid[x][y])
        string += row
    print(string)
    filename = input("Map title: ") + ".txt"
    file = open("colony_maps/" + filename,'w')
    file.write(string)


current_ant = 0  # Ant currently being controlled

def send_keys(key):
    global spacebar_wait
    if spacebar_wait:
        Ant.instances[current_ant].place_food(key)
        spacebar_wait = False
    elif key == "BACKSPACE":
        Ant.instances[current_ant].suicide()
    else:  
        Ant.instances[current_ant].move(key)




# Main Loop
while running:
    screen.fill(GRAY)
    draw_grid()
    pygame.display.flip()
    clock.tick(30)
    for x in range(GRID_SIZE):
        for y in range(GRID_SIZE):
            pheremone_grid[x][y] = (pheremone_grid[x][y][0],max(pheremone_grid[x][y][1] - pheremone_lifespan,0))

    # Event Handling
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.MOUSEBUTTONDOWN:  # Detect mouse click
            holding = True
        elif event.type == pygame.MOUSEBUTTONUP:
            holding = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_DOWN:
                send_keys("DOWN")
            elif event.key == pygame.K_UP:
                send_keys("UP")
            elif event.key == pygame.K_LEFT:
                send_keys("LEFT")
            elif event.key == pygame.K_RIGHT:
                send_keys("RIGHT")
            elif event.key == pygame.K_SPACE and (Ant.instances[current_ant].holding == FOOD or Ant.instances[current_ant].holding == GROUND):
                spacebar_wait = True
            elif event.key == pygame.K_q and Ant.instances[current_ant].holding != FOOD:
                dig_wait = True
            elif event.key == pygame.K_e:
                if current_ant >= len(Ant.instances) - 1:
                    current_ant = 0
                else:
                    current_ant += 1
            elif event.key == pygame.K_BACKSPACE:
                send_keys("BACKSPACE")
            elif event.key == pygame.K_f:
                FOG_OF_WAR = not FOG_OF_WAR
            elif event.key == pygame.K_s:
                save_file()
            elif event.key == pygame.K_c:
                grid = return_empty_grid(GRID_SIZE)

    if holding:
        if pygame.mouse.get_pressed(3)[0]:  # Left click // make air
            draw_node(pygame.mouse.get_pos(),AIR,mouse_radius)
        if pygame.mouse.get_pressed(3)[2]:  # Right click // make food
            draw_node(pygame.mouse.get_pos(),GROUND,0)
        if pygame.mouse.get_pressed(3)[1]:  # Middle click // make ant
            Ant(pixel_to_index(pygame.mouse.get_pos()[0]),pixel_to_index(pygame.mouse.get_pos()[1]))
        if pygame.mouse.get_pressed(5)[4]:
            draw_pheremone(pygame.mouse.get_pos(),mouse_radius,FOOD_PHER,mouse_intensity)

pygame.quit()