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


TODO:
- Make nodes into a singular class (such a good idea)
"""

import pygame
import math
import os
from textreader import read_txt,return_empty_grid,Node

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
ant_sight = 1
mouse_intensity = 1
pheremone_lifespan = .1  # how many frames per intensity decrease
mouse_radius = 3




# Colors
BLACK = (0, 0, 0)  # Background
WHITE = (255, 255, 255)  # Border for tunnels
YELLOW = (255,255,0)  # Food
BROWN = (139,69,19)
RED = (255,0,0)
GRAY = (100,100,100)
GREEN = (0,255,0)
BLUE = (0,0,255)

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

# Intensity: 0-1, 0 being no pheremone, 1 being recent pheremone



# TODO: Create Node class, include all relevant pheremone, physical information





# Grid Initialization
# Start with all nodes as undug earth (represented as 0)
grid = [[Node(x,y,GROUND,[],[],[]) for x in range(GRID_SIZE)] for y in range(GRID_SIZE)]

EMPTY_GRID = False
if not EMPTY_GRID:
    maps = {}  # number:name of map
    for index in range(len(os.listdir('Ant Colony\\colony_maps'))):
        # print(index)
        # print(os.listdir('./colony_maps'))
        # print(len(os.listdir('./colony_maps')))
        name = os.listdir('Ant Colony\\colony_maps')[index]
        maps.update({str(index):name})
        print(name + ":",index)
    map_index = input("Please enter the number of the map you'd like to use (or enter 'c' if you'd like a clear map): ")
    if not map_index == 'c':
        map_index = int(map_index)
        map_name = maps[str(map_index)]
        grid = read_txt(GRID_SIZE,map_name)


# Pygame Setup
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Ant Colony Simulation")
clock = pygame.time.Clock()  # For framerate





import statistics
def average_colors(colors,intensities):  # list of color tuples; [grid node color, pheremone color 1,...],list of intensities
    # VERY INOPTIMAL AND SLOW
    new_intensities = [sum(intensities)] + intensities  # incl. gride intensity as 1/2 of all others
    print(new_intensities)
    denominator = max(1,sum(new_intensities))
    final_color = [0 for _ in range(3)]
    for rgb_index in range(3):
        current_val = 0
        for index,color in enumerate(colors):
            print(color)
            current_val += (color[rgb_index]) * new_intensities[index]
        final_color[rgb_index] = (math.floor((current_val)/denominator))
    return tuple(final_color)


def return_grid_color(x,y):
    if grid[x][y].node_type == GROUND:  # Undug earth
        return BLACK
    elif grid[x][y].node_type == AIR:  # Air
        return BLACK
    elif grid[x][y].node_type == FOOD:
        return YELLOW
    elif grid[x][y].node_type == ANT:
        return BROWN
    elif grid[x][y].node_type == DEBUG_NODE:
        return RED

def pheremone_to_color(pheremone_type):
    if pheremone_type == FOOD_PHER:
        return GREEN
    elif pheremone_type == NO_FOOD_PHER:
        return BLUE

def return_pheremone_color(x,y):
    colors = [pheremone_to_color(grid[x][y].pheremone_types[index]) for index in range(len(grid[x][y].pheremone_types))]
    if len(colors) == 0:
        return return_grid_color(x,y)
    colors = [return_grid_color(x,y)] + colors
    intensities = grid[x][y].pheremone_intensities
    intensities = [sum(intensities)] + intensities
    denominator = max(1,sum(intensities))
    final_color = [0 for _ in range(3)]
    for rgb_index in range(3):
        current_val = 0
        for index,color in enumerate(colors):
            current_val += color[rgb_index] * intensities[index]
        final_color[rgb_index] = (math.floor((current_val)/denominator))
    return tuple(final_color)

def pixel_to_index(pixel):
    return pixel // NODE_SIZE

def index_to_pixel(index):
    return index * NODE_SIZE



# Function to draw the grid
def draw_grid():
    start_pos_bound_coords = []
    end_pos_bound_coords = []


    def draw_white_bounds(x,y):
        if y>0 and grid[x][y-1].node_type == GROUND:  # Top neighbor
            start_pos_bound_coords.append((index_to_pixel(x), index_to_pixel(y)))
            end_pos_bound_coords.append((index_to_pixel(x+1), index_to_pixel(y)))
        if y<GRID_SIZE - 1 and grid[x][y+1].node_type == GROUND:  # bottom neighbor // works
            start_pos_bound_coords.append((index_to_pixel(x), index_to_pixel(y+1)))
            end_pos_bound_coords.append((index_to_pixel(x+1), index_to_pixel(y+1)))
        if x>0 and grid[x-1][y].node_type == GROUND:  # Left neighbor
            start_pos_bound_coords.append((index_to_pixel(x), index_to_pixel(y)))
            end_pos_bound_coords.append((index_to_pixel(x), index_to_pixel(y+1)))
        if x<GRID_SIZE-1 and grid[x+1][y].node_type == GROUND:  # Left neighbor
            start_pos_bound_coords.append((index_to_pixel(x+1), index_to_pixel(y)))
            end_pos_bound_coords.append((index_to_pixel(x+1), index_to_pixel(y+1)))





    if not FOG_OF_WAR or len(Ant.instances) == 0:  # aren't actively watching ant pov
        for x in range(GRID_SIZE):
            for y in range(GRID_SIZE):
                rect = pygame.Rect(index_to_pixel(x), index_to_pixel(y), NODE_SIZE, NODE_SIZE)
                if len(grid[x][y].pheremone_types) == 0:
                    pygame.draw.rect(screen,return_grid_color(x,y),rect)
                else:
                    pygame.draw.rect(screen,return_pheremone_color(x,y),rect)
                if grid[x][y].node_type in [AIR,FOOD,ANT]:
                    draw_white_bounds(x,y)
                else:
                    grid[x][y].remove_all_pheremones()
                element = grid[x][y]
                for test_index,pheremone_type in enumerate(element.pheremone_types):
                    pheremone_index = element.pheremone_types.index(pheremone_type)
                    update_pheremone(element.x,element.y,pheremone_type,max(0,element.pheremone_intensities[pheremone_index] - pheremone_lifespan),element.pheremone_sources[pheremone_index][0],element.pheremone_sources[pheremone_index][1])
                    if element.pheremone_intensities[pheremone_index] <= 0:
                        element.remove_pheremone(pheremone_type)
                        if len(element.pheremone_types) == 0:
                            element.remove_all_pheremones()
                        
        # for key,item in enumerate(nonzero_pheremone_list):
        #     # item is a node
        #     x,y = item.x,item.y
        #     rect = pygame.Rect(index_to_pixel(x), index_to_pixel(y), NODE_SIZE, NODE_SIZE)
        #     if grid[x][y].node_type == AIR:
        #         pygame.draw.rect(screen, return_pheremone_color(x,y), rect)  # Air
        #     elif grid[x][y].node_type == GROUND:
        #         grid[x][y].remove_all_pheremones()
        #         pygame.draw.rect(screen, return_grid_color(x,y), rect)

        for z in range(len(start_pos_bound_coords)):
            pygame.draw.line(screen,WHITE,start_pos_bound_coords[z],end_pos_bound_coords[z], 1)

    if FOG_OF_WAR and len(Ant.instances) != 0:
        for curr_ant in range(len(Ant.instances)):
            x = Ant.instances[curr_ant].x
            y = Ant.instances[curr_ant].y
            for x1 in range(max(0,x - ant_sight), min(x + ant_sight + 1,GRID_SIZE)):
                for y1 in range(max(0,y - ant_sight), min(y + ant_sight + 1,GRID_SIZE)):
                    rect = pygame.Rect(index_to_pixel(x1), index_to_pixel(y1), NODE_SIZE, NODE_SIZE)
                    pygame.draw.rect(screen,return_pheremone_color(x1,y1),rect)
                    if grid[x1][y1].node_type in [AIR,FOOD,ANT]:
                        draw_white_bounds(x1,y1)

            # for key,item in enumerate(nonzero_pheremone_list):
            #     # item is a node
            #     rect = pygame.Rect(index_to_pixel(x), index_to_pixel(y), NODE_SIZE, NODE_SIZE)
            #     x,y = item.x,item.y
            #     if grid[x][y].node_type == AIR:
            #         pygame.draw.rect(screen, return_pheremone_color(x,y), rect)  # Air
            #     elif grid[x][y].node_type == GROUND:
            #         grid[x][y].remove_all_pheremones()
            #         pygame.draw.rect(screen, return_grid_color(x,y), rect)  # Air
            for z in range(len(start_pos_bound_coords)):
                pygame.draw.line(screen,WHITE,start_pos_bound_coords[z],end_pos_bound_coords[z], 1)
                    


# (x - h)² + (y - k)² = r²
running = True
def draw_node(mouse_pos,node_type,rad):
    x, y = mouse_pos[0],mouse_pos[1]
    grid_x = pixel_to_index(x)
    grid_y = pixel_to_index(y)

    if 0 <= grid_x < GRID_SIZE and 0 <= grid_y < GRID_SIZE:
        grid[grid_x][grid_y].node_type = node_type
    for x1 in range(max(0,grid_x - rad), min(grid_x + rad,GRID_SIZE - 1) + 1):
        for y1 in range(max(0,grid_y - rad), min(grid_y + rad,GRID_SIZE - 1) + 1):
            grid[x1][y1].node_type = node_type

def update_pheremone(x,y,pheremone_type,intensity,source_x,source_y):
    if grid[x][y].node_type == GROUND:
        grid[x][y].remove_all_pheremones()
    if not pheremone_type in grid[x][y].pheremone_types:
        grid[x][y].pheremone_types.append(pheremone_type)
        grid[x][y].pheremone_sources.append((source_x,source_y))
        grid[x][y].pheremone_intensities.append(intensity)
        # grid[x][y].nonzeropheremonelistindex = len(nonzero_pheremone_list)
        # nonzero_pheremone_list.append(grid[x][y])
    else:
        index = grid[x][y].pheremone_types.index(pheremone_type)
        grid[x][y].pheremone_sources[index] = (x,y)
        grid[x][y].pheremone_intensities[index] = intensity


def draw_pheremone(pos,rad,pheremone_type, intensity,mouse_used,source_x,source_y):
    print("pheremone thing")
    if mouse_used:
        grid_x,grid_y = max(0,min(GRID_SIZE-1,pixel_to_index(pos[0]))),max(0,min(GRID_SIZE-1,pixel_to_index(pos[1])))
    else:
        grid_x, grid_y = pos[0],pos[1]
    if grid[grid_x][grid_y].node_type == AIR:
        if 0 <= grid_x < GRID_SIZE and 0 <= grid_y < GRID_SIZE:
            update_pheremone(grid_x,grid_y,pheremone_type,intensity,source_x,source_y)
            rect = pygame.Rect(index_to_pixel(grid_x), index_to_pixel(grid_y), NODE_SIZE, NODE_SIZE)
            pygame.draw.rect(screen,return_pheremone_color(grid_x,grid_y),rect)
    for x1 in range(max(0,grid_x - rad), min(grid_x + rad,GRID_SIZE - 1) + 1):
        for y1 in range(max(0,grid_y - rad), min(grid_y + rad,GRID_SIZE - 1) + 1):
            if grid[x1][y1].node_type == AIR:
                update_pheremone(x1,y1,pheremone_type,intensity,source_x,source_y)
                rect = pygame.Rect(index_to_pixel(x1), index_to_pixel(y1), NODE_SIZE, NODE_SIZE)
                pygame.draw.rect(screen,return_pheremone_color(x1,y1),rect)



def check_bounds(x,y,rad):  # Check nodes w/in radius (2 for 3x3, 1 for 2x2, 0 for 1x1), return list of node types(?
    # draw_node((x * NODE_SIZE,y * NODE_SIZE),DEBUG_NODE,rad)
    grid1 = []
    for x1 in range(max(0,x - rad), min(x + rad,GRID_SIZE - 1) + 1):
        row1 = []
        for y1 in range(max(0,y - rad), min(y + rad,GRID_SIZE - 1) + 1):
            if x1 == x and y1 == y:
                row1.append(AIR)
            else:
                row1.append(grid[x1][y1].node_type)
        grid1.append(row1)
    print(grid1)
    return grid1



# Ant class
ants = {}  # List of ants and the node reference of what they're holding; "0,1":1 if it's an ant at 0,1 holding air
class Ant:
    instances = []
    def __init__(self,x,y):
        valid = False
        if grid[x][y].node_type == AIR:
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
                grid[x][y].node_type = ANT
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
            if grid[check_x][check_y].node_type == AIR or (dig_wait and grid[check_x][check_y].node_type == GROUND):
                if dig_wait and grid[check_x][check_y].node_type == GROUND:
                    dig_wait = False
                    self.holding = GROUND
                grid[self.x][self.y].node_type = AIR
                self.x = check_x
                self.y = check_y
                grid[check_x][check_y].node_type = ANT
                ants.update({self.id:[self.x,self.y,self.holding]})
            elif grid[check_x][check_y].node_type == FOOD:
                grid[self.x][self.y].node_type = AIR
                self.x = check_x
                self.y = check_y
                self.holding = FOOD
                grid[check_x][check_y].node_type = ANT
                ants.update({self.id:[self.x,self.y,self.holding]})
        test = list(ants.keys())
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
        if valid and ant_count < 3 and grid[check_x][check_y].node_type == AIR:
            grid[check_x][check_y].node_type = self.holding
            self.holding = AIR
            ants.update({self.id:[self.x,self.y,self.holding]})
    def suicide(self):
        global current_ant
        grid[self.x][self.y].node_type == AIR
        ants.pop(self.id)
        Ant.instances.pop(current_ant)
        current_ant = max(0,current_ant - 1)
        draw_node((index_to_pixel(self.x),index_to_pixel(self.y)),AIR,0)
        del self

    def leave_pheremone(self,node, type, intensity):  # Types of pheremones: FOOD_PHER,NO_FOOD_PHER, leave_pheremone((0,1),"FOOD_PHER", 10)
        draw_pheremone((node[0],node[1]),0,type,intensity,False,node[0],node[1])


def save_file():
    string = str(GRID_SIZE) + " "
    for x in range(GRID_SIZE):
        row = ""
        for y in range(GRID_SIZE):
            row += str(grid[x][y].node_type)
        string += row
    print(string)
    filetitle = input("Map title: ")
    filename = filetitle + ".txt"
    file = open("Ant Colony\\colony_maps\\" + filename,'w')
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
    screen.fill(BLACK)
    draw_grid()
    pygame.display.flip()
    clock.tick(30)
    # for index,element in enumerate(nonzero_pheremone_list):
    #     # element is a node
    #     for index_test,pheremone_type in enumerate(element.pheremone_types):
    #         pheremone_index = element.pheremone_types.index(pheremone_type)
    #         # print(element.pheremone_types,element.pheremone_intensities)
    #         # update_pheremone(x,y,pheremone_type,intensity,source_x,source_y)
    #         update_pheremone(element.x,element.y,pheremone_type,max(0,element.pheremone_intensities[pheremone_index] - pheremone_lifespan),element.pheremone_sources[pheremone_index][0],element.pheremone_sources[pheremone_index][1])
    #         if element.pheremone_intensities[pheremone_index] <= 0:
    #             element.remove_pheremone(pheremone_type)
    #         # if len(element.pheremone_types) == 0 and element in nonzero_pheremone_list:
    #         #     print("element in non zero thing!")
    #         #     nonzero_pheremone_list.pop(element.nonzeropheremonelistindex)
    #         #     element.nonzeropheremonelistindex = None

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
            print(pygame.mouse.get_pos())
            draw_pheremone(pygame.mouse.get_pos(),mouse_radius,FOOD_PHER,mouse_intensity,True,pixel_to_index(pygame.mouse.get_pos()[0]),pixel_to_index(pygame.mouse.get_pos()[1]))
        if pygame.mouse.get_pressed(5)[3]:
            draw_pheremone(pygame.mouse.get_pos(),mouse_radius,NO_FOOD_PHER,mouse_intensity,True,pixel_to_index(pygame.mouse.get_pos()[0]),pixel_to_index(pygame.mouse.get_pos()[1]))

pygame.quit()