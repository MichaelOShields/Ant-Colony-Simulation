

#region Imports
import numpy as np
import pygame


#endregion


#region Screen Variables
GRID_SIZE = 25  # Size of grid (obv) // 25
NODE_SIZE = 30  # Pixel size of nodes // 30
SCREEN_WIDTH = GRID_SIZE * NODE_SIZE
SCREEN_HEIGHT = GRID_SIZE * NODE_SIZE

#endregion


#region Colors
# Colors
WHITE = (255, 255, 255)
BLACK = (0,   0,   0)
RED   = (255, 0,   0)
GREEN = (0,   255, 0)
BLUE  = (0,   0,   255)
YELLOW = (255,255,0)  # Food
BROWN = (139,69,19)
GRAY = (100,100,100)
DARK_BROWN = (92, 64, 51)
#endregion


#region Node Types, Colors, Etc
# Node types:
GROUND = 0
AIR = 1
FOOD = 2
ANT = 3
CENTER = 4
DEBUG = 10

node_colors = {
    GROUND:BLACK,
    AIR:BLACK,
    FOOD:YELLOW,
    ANT:BROWN,
    CENTER:DARK_BROWN
}

node_titles = {
    AIR:"AIR",
    GROUND:"GROUND",
    FOOD:"FOOD",
    ANT:"ANT"
}

#endregion


#region Game Constants

DEBUG = True

count = 0
running = True
holding = False  # holding down the mouse
current_ant = 0  # current ant being controlled

ant_pheromone_strength = 5.0

changing_nodes = set()  # List of coordinate tuples of nodes to be rerendered in next frame

score = 0

#endregion


#region Mouse Variables

mouse_rad = 1
mouse_intensity = 5

#endregion


#region  Diffusion variables

evaporation_num = 0.000000000001  # 0.01
cutoff_num = 0.005

decay_pheromone = lambda x : x - evaporation_num if x - evaporation_num > 0 else 0


pheromone_surface = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.SRCALPHA)
zeroed_pheromones = set()

remove_nonzero = set()

nonzero_pheromones = set()

diffusion_factor = .2


#endregion


#region Initialize Pygame

# Initialize Pygame
pygame.init()
pygame.font.init()
font = pygame.font.SysFont('Courier New', 15)
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Pheremone Dispersion Simulation")
clock = pygame.time.Clock()

#endregion


#region Directions

six_directions = [
    (0,1),
    (0,-1),
    (-1,0),
    (1,0),
    (1,1),
    (-1,1),
    (-1,-1),
    (1,-1)
]

cardinal_directions = [  # used for gas dispersion
    (0,1),
    (0,-1),
    (-1,0),
    (1,0),
]

render_directions = [
        (0, -1,lambda x, y: [(x,y),(x + 1, y)]),  # top edge // return coords of line
        (0, 1, lambda x, y: [(x,y+1),(x + 1, y + 1)]),  # bottom edge
        (-1, 0, lambda x, y: [(x, y),(x, y+1)]),  # Left edge
        (1, 0, lambda x, y: [(x+1, y),(x+1, y+1)])  # Right edge
    ]
    
#endregion





#region Initializing grids


grid = np.full((GRID_SIZE,GRID_SIZE),GROUND)  # node grid
pheromone_grid = np.full((GRID_SIZE,GRID_SIZE),0.0)  # Pheromone grid



#endregion


#region Functions

def return_bounds(coord):
    if type(coord) == int:
        return max(0,min(GRID_SIZE-1,coord))
    else:
        final = []
        for coordinate in coord:
            final.append(return_bounds(coordinate))
        return tuple(final)

def return_node_radius(x,y,radius):
    x_min = return_bounds(x-radius)
    x_max = return_bounds(x+radius)
    y_min = return_bounds(y-radius)
    y_max = return_bounds(y+radius)
    return x_min,x_max,y_min,y_max


def pixel_to_index(pixel):
    if type(pixel) == int:
        return return_bounds(pixel // NODE_SIZE)
    else:
        final = []
        for px in pixel:
            final.append(pixel_to_index(px))
        return tuple(final)

def index_to_pixel(index):
    if type(index) == int:
        return index * NODE_SIZE
    else:
        final = []
        for ind in index:
            final.append(index_to_pixel(ind))
        return tuple(final)



def return_grid_color(ntype):
    return node_colors[ntype]


def update_node(position,node_type):  # used to update nodes, iterates thru cardinal directions to avoid rendering issues w/ boundaries
    x,y = return_bounds(position)
    if not grid[x,y] == CENTER:
        grid[x,y] = node_type
        changing_nodes.add((x,y))
        for z in six_directions:  # loop thru cardinal directions of nodes drawn & add them to changing nodes
            changing_nodes.add(return_bounds((x+z[0],y+z[1])))




def fps_counter():
    fps = str(int(clock.get_fps()))
    fps_t = font.render(fps , 1, pygame.Color("RED"))
    fps_rect = pygame.Rect(0,0,NODE_SIZE * 3, NODE_SIZE * 2)
    pygame.draw.rect(screen,BLACK,fps_rect)
    screen.blit(fps_t,(0,0))




def send_keys(direction_tuple):  # Used to send commands to ants
    if len(ants) > 0:
        if ants[current_ant].placing and ants[current_ant].holding != AIR:  # if about to place a node
            ants[current_ant].place(direction_tuple)
        else:
            ants[current_ant].move(direction_tuple)


def count_neighbors(position):
    x, y = position
    num_neighbors = 0
    if grid[return_bounds(x + 1), y] != GROUND and pheromone_grid[return_bounds(x + 1), y] < pheromone_grid[x, y]:
        num_neighbors += 1
    if grid[return_bounds(x - 1), y] != GROUND and pheromone_grid[return_bounds(x - 1), y] < pheromone_grid[x, y]:
        num_neighbors += 1
    if grid[x, return_bounds(y - 1)] != GROUND and pheromone_grid[x, return_bounds(y - 1)] < pheromone_grid[x, y]:
        num_neighbors += 1
    if grid[x, return_bounds(y + 1)] != GROUND and pheromone_grid[x, return_bounds(y + 1)] < pheromone_grid[x, y]:
        num_neighbors += 1
    return num_neighbors



#endregion



#region Ant

ant_lifespan = GRID_SIZE ** 2  # number of moves they can make until they die

ants = []  # List of all ants
ant_positions = {}  # (x,y):ant memory reference // used to destroy ants that are drawn over
class Ant:
    def __init__(self,position,holding,start_digging,placing):

        x_min,x_max,y_min,y_max = return_node_radius(position[0],position[1],1)  # Get radius of ant spawn point

        if not ANT in grid[x_min:x_max + 1,y_min:y_max + 1] and GROUND in grid[x_min:x_max + 1,y_min:y_max + 1]:  # Continue; ant location is valid

            self.position = position  # (x,y) tuple
            self.holding = holding  # what it's holding // 1 if holding nothing (air)
            self.digging = start_digging  # If ant is currently digging
            self.placing = placing

            grid[position] = ANT  # sets grid value to ant
            changing_nodes.add(position)  # tells pygame to render ant node next frame
            ants.append(self)
            print(ants)
            ant_positions.update({self.position:self})
            print(ant_positions)
            self.lifetime = 0

            x,y = position
            for z in six_directions:  # ensure boundaries are drawn correctly
                changing_nodes.add(return_bounds((x+z[0],y+z[1])))
            
            self.sight = grid[x_min:x_max,y_min:y_max]
    
    def move(self,direction_tuple):  # digging is a bool, direction_tuple is a (dx,dy) tuple denoting which direction to travel
        nx, ny = return_bounds((self.position[0] + direction_tuple[0],self.position[1] + direction_tuple[1])) # New x, new y
        if (nx, ny) == self.position:  # Don't allow to leave bounds
            return False
        print(nx,ny)

        current_digging = self.digging  # if we're currently digging

        x_min,x_max,y_min,y_max = return_node_radius(nx,ny,1)
        grid_to_check = grid[x_min:x_max + 1,y_min:y_max + 1]  # grid area the ant is moving to


        # First: validate movement (are there any ants in the vicinity this ant is moving to?)
        valid = np.count_nonzero(grid_to_check == ANT) <= 1 and (GROUND in grid[x_min:x_max + 1,y_min:y_max + 1] or FOOD in grid[x_min:x_max + 1,y_min:y_max + 1])  # no extra ants around destination node, ground is present around desination node

        if current_digging and (grid[nx,ny] == GROUND or grid[nx, ny] == FOOD) and grid[nx, ny] != CENTER and self.holding != FOOD:  # currently digging AND grid to be dug is ground; if so, can safely check if there's atleast 2 ground nodes in vicinity
                valid = valid and (np.count_nonzero(grid_to_check == GROUND) > 1 or np.count_nonzero(grid_to_check == FOOD) > 1)  # Makes sure ant will still be adjacent to ground node after digging
        else:
            valid = valid and grid[nx,ny] == AIR
            current_digging = False

        if valid:
            self.lifetime += 1
            if current_digging:
                self.holding = grid[nx,ny]
            del ant_positions[self.position]  # Delete ant reference from positions dictionary
            update_node((nx,ny),ANT)
            update_node(self.position,AIR)
            self.position = (nx,ny)
            ant_positions.update({self.position:self})   
            self.sight = grid_to_check
            if self.lifetime > ant_lifespan:
                self.suicide()

    def place(self,direction_tuple):
        plx, ply = self.position[0] + direction_tuple[0],self.position[1] + direction_tuple[1] # Place x, place y

        if self.holding != AIR and grid[plx,ply] == AIR:  # Validate placement; placing onto empty node and able to place nodes
            update_node((plx,ply),self.holding)
            self.holding = AIR
            self.placing = False

        elif self.holding == FOOD and grid[plx,ply] == CENTER:  # Placing food into center
            self.holding = AIR
            self.placing = False
            global score
            score += 1
            print("score:", score)
            self.lifetime = 0  # Refresh ant life


    def release_pheromone(self,intensity):
        draw_pheromone(position=self.position,intensity=intensity,rad=0)


    
    def suicide(self):
        global current_ant  # access current_ant variable
        ants.remove(self)  # remove self from ants list
        grid[self.position] = AIR  # set grid value to air
        changing_nodes.add(self.position)  # tell pygame to render deleted node next frame
        current_ant = max(current_ant - 1,0)  # change current_ant value
        if ant_positions[self.position]:
            del ant_positions[self.position]
        print(ant_positions)
        del self

#endregion

















#region Pheromone functions


def update_pheromones():
    global pheromone_grid
    global count
    for x, y in nonzero_pheromones.copy():
        if count % 15 == 0:
            pheromone_grid[x, y] = max(pheromone_grid[x, y] - evaporation_num, 0)
            if pheromone_grid[x, y] <= cutoff_num:
                zeroed_pheromones.add((x,y))
                nonzero_pheromones.discard((x, y))
            else:
                zeroed_pheromones.discard((x,y))

        if grid[x, y] == GROUND:

            pheromone_grid[x, y] = 0.0
            zeroed_pheromones.add((x,y))
            nonzero_pheromones.discard((x, y))

        else:
            num_neighbors = count_neighbors((x, y))
            
            if num_neighbors > 0:

                amount_to_diffuse = pheromone_grid[x, y] * diffusion_factor
                if pheromone_grid[x, y] - amount_to_diffuse > cutoff_num:

                    for direction in cardinal_directions:
                        nx, ny = return_bounds(x + direction[0]), return_bounds(y + direction[1])  # New x, new y
                        if grid[nx, ny] != GROUND and pheromone_grid[nx, ny] + amount_to_diffuse / num_neighbors < pheromone_grid[x, y]:
                            pheromone_grid[nx, ny] += amount_to_diffuse / num_neighbors
                            nonzero_pheromones.add((nx, ny))
                    
                    pheromone_grid[x, y] -= amount_to_diffuse
                    # to_be_updated.add((x, y))



def update_pheromone_display():
    pheromone_surface.fill(pygame.Color(0,0,0,0))
    for x, y in nonzero_pheromones.copy():
        # print(x, y)
        concentration = pheromone_grid[x, y]
        alpha = max(0, min(int(concentration * 255), 255))
        color = (255, 255, 0, alpha) # Yellow with transparency
        rect = pygame.Rect(index_to_pixel(x), index_to_pixel(y), NODE_SIZE, NODE_SIZE)
        pygame.draw.rect(pheromone_surface, color, rect)
        # print(pheromone_grid[np.where(pheromone_grid != 0)])
        if pheromone_grid[x, y] <= cutoff_num:
            pheromone_grid[x, y] = 0.0
            nonzero_pheromones.discard((x, y))
            zeroed_pheromones.add((x,y))
            for direction in cardinal_directions:
                nx, ny = x + direction[0], y + direction[1]
                if 0 <= nx < GRID_SIZE and 0 <= ny < GRID_SIZE:
                    changing_nodes.add((nx,ny))

    
    for x, y in zeroed_pheromones:
        rect = pygame.Rect(index_to_pixel(x), index_to_pixel(y), NODE_SIZE, NODE_SIZE)
        pygame.draw.rect(pheromone_surface, return_grid_color(grid[x, y]), rect)  # Fully transparent
    screen.blit(pheromone_surface,(0,0))

    for x, y in zeroed_pheromones:
        render_bounds(x,y)
    zeroed_pheromones.clear()
    # zeroed_pheromones.clear()








#endregion









#region Draw grid

def clear_zeroed_pheromones():
    for x, y in zeroed_pheromones:
        rect = pygame.Rect(index_to_pixel(x), index_to_pixel(y), NODE_SIZE, NODE_SIZE)
        pygame.draw.rect(pheromone_surface, return_grid_color(grid[x, y]), rect)  # Fully transparent
    zeroed_pheromones.clear()
    screen.blit(pheromone_surface,(0,0))
    render_bounds(x,y)


def render_bounds(x, y):
    for (dx, dy, line_func) in render_directions:  # Change in x, change in y, lambda function to return coordinates of line
            nx = x + dx  # Index of neighboring x node
            ny = y + dy  # Index of neighboring y node
            if 0 <= nx < GRID_SIZE and 0 <= ny < GRID_SIZE:  # Making sure neighbor is within bounds
                current_is_ground = (grid[x,y] == GROUND)  # Boolean checking if current node is a ground node
                neighbor_is_ground = (grid[nx,ny] == GROUND)  # Boolean chekcing if neighboring node is a ground node
                if current_is_ground ^ neighbor_is_ground:  # If one is ground and the other isn't; ^ is a XOR operator which basically returns True if both are different
                    start_pos = index_to_pixel(line_func(x,y)[0])  # Start position in coordinate values, using index_to_pixel to convert grid indexes to pixel values
                    end_pos = index_to_pixel(line_func(x,y)[1])
                    pygame.draw.line(screen, WHITE, start_pos,end_pos,1)



def draw_grid(iterable):  # consolidate grid drawing; iterable is either changing_nodes, entire empty grid, or just ant vision (add ant_vision boolean parameter later -- add gray background if yes?)



    for (x, y) in iterable:
        #region Render physical nodes
        rect = pygame.Rect(index_to_pixel(x), index_to_pixel(y), NODE_SIZE, NODE_SIZE)  # Creates a rectangle encompassing the node
        pygame.draw.rect(screen, return_grid_color(grid[x,y]), rect)  # Draw the node

        render_bounds(x, y)
        #endregion


    #region Pheromones

    update_pheromones()

    update_pheromone_display()
    # clear_zeroed_pheromones()

    #endregion



    
    changing_nodes.clear()
    
    fps_counter()



#endregion


def draw_node(mouse_pos,node_type,rad):
    grid_x = pixel_to_index(mouse_pos[0])
    grid_y = pixel_to_index(mouse_pos[1])

    x_min,x_max,y_min,y_max = return_node_radius(grid_x,grid_y,rad)
    for x in range(x_min,x_max+1):  # Iterate through x's in radius
        for y in range(y_min,y_max+1):
            update_node((x,y),node_type)



#region Initialize game
def reset():
    global ants, grid, pheromone_grid, changing_nodes
    print('reset)')
    for ant in ants:
        ant.suicide()
    grid = np.full((GRID_SIZE,GRID_SIZE),GROUND)
    grid[GRID_SIZE // 2, GRID_SIZE // 2] = CENTER
    pheromone_grid = np.full((GRID_SIZE,GRID_SIZE),0.0)
    changing_nodes.clear()
    draw_grid([(x,y) for x in range(GRID_SIZE) for y in range(GRID_SIZE)])


import random
def randomize_food():
    probability = 0.005  # 5% chance of any given node being a food pocket
    big_probability = 0.003
    small_probability = 0.02
    edge_range = GRID_SIZE // 5  # Edge range to spawn food in
    possible_rads = [0,1]
    for x in range(0, edge_range):
        for y in range(0, GRID_SIZE):
            if random.random() < big_probability:
                draw_node(index_to_pixel((x, y)), FOOD, 1)
            elif random.random() < small_probability:
                draw_node(index_to_pixel((x, y)), FOOD, 0)

    for x in range(GRID_SIZE - edge_range, GRID_SIZE):
        for y in range(0, GRID_SIZE):
            if random.random() < big_probability:
                draw_node(index_to_pixel((x, y)), FOOD, 1)
            elif random.random() < small_probability:
                draw_node(index_to_pixel((x, y)), FOOD, 0)


    for y in range(GRID_SIZE - edge_range, GRID_SIZE):
        for x in range(0, GRID_SIZE):
            if random.random() < big_probability:
                draw_node(index_to_pixel((x, y)), FOOD, 1)
            elif random.random() < small_probability:
                draw_node(index_to_pixel((x, y)), FOOD, 0)
    
    for y in range(0, edge_range):
        for x in range(0, GRID_SIZE):
            if random.random() < big_probability:
                draw_node(index_to_pixel((x, y)), FOOD, 1)
            elif random.random() < small_probability:
                draw_node(index_to_pixel((x, y)), FOOD, 0)


reset()
randomize_food()



#endregion






def draw_pheromone(position, intensity,rad):
    grid_x,grid_y = position
    

    x_min,x_max,y_min,y_max = return_node_radius(grid_x,grid_y,rad)
    for x in range(x_min,x_max+1):  # Iterate through x's in radius
        for y in range(y_min,y_max+1):
            pheromone_grid[x, y] = intensity
            nonzero_pheromones.add((x, y))
    



while running:
    # print("Nonzero", nonzero_pheromones)
    # print("zero",zeroed_pheromones)
    changing_nodes.update(nonzero_pheromones)
    changing_nodes.update(zeroed_pheromones)
    draw_grid(changing_nodes)         # Draw the current grid

    pygame.display.flip()  # Update the display
    clock.tick(30)         # Limit to 30 FPS

    # DEBUG TRACKERS:
    count += 1
    if count % 100 == 0 and DEBUG:
        print(grid[pixel_to_index(pygame.mouse.get_pos()[0]),pixel_to_index(pygame.mouse.get_pos()[0])])  # Printing the node the mouse is over for debug purposes
        if len(ants) > 0:
            print("Holding: " + node_titles[ants[current_ant].holding])
            print("Currently digging: " + str(ants[current_ant].digging))
            print("Waiting to place: " + str(ants[current_ant].placing))
    # Event Handling
    for event in pygame.event.get():

        if event.type == pygame.QUIT:

            running = False

        elif event.type == pygame.MOUSEBUTTONDOWN:  # Detect mouse click

            holding = True

        elif event.type == pygame.MOUSEBUTTONUP:
            
            holding = False

        elif event.type == pygame.KEYDOWN:  # Handle keyboard inputs

            if event.key == pygame.K_c:
                reset()
                randomize_food()
            
            elif event.key == pygame.K_r:  # Refresh grid
                draw_grid([(x,y) for x in range(GRID_SIZE) for y in range(GRID_SIZE)])
            
            elif event.key == pygame.K_BACKSPACE and len(ants) > 0:  # ant suicide
                ants[current_ant].suicide()
                current_ant -= 1
            
            elif event.key == pygame.K_q and len(ants) > 0:  # digging toggle
                ants[current_ant].digging = not ants[current_ant].digging
            
            elif event.key == pygame.K_UP:
                send_keys((0,-1))
            

            elif event.key == pygame.K_DOWN:
                send_keys((0,1))
            

            elif event.key == pygame.K_LEFT:
                send_keys((-1,0))


            elif event.key == pygame.K_RIGHT:
                send_keys((1,0))
            
            elif event.key == pygame.K_SPACE and len(ants) > 0 and ants[current_ant].holding != AIR:  # Start to place a node
                ants[current_ant].placing = not ants[current_ant].placing
            
            elif event.key == pygame.K_TAB and len(ants) > 0:
                if current_ant + 1 >= len(ants):  # If current ant index will exceed total # of ants
                    current_ant = 0
                else:
                    current_ant += 1
            
            elif event.key == pygame.K_p and len(ants) > 0:
                ants[current_ant].release_pheromone(ant_pheromone_strength)



    if holding:
        if pygame.mouse.get_pressed(3)[0]:  # Left click // make air        
            draw_node(pygame.mouse.get_pos(),AIR,mouse_rad)
        if pygame.mouse.get_pressed(3)[2]:  # Right click // make ground
            draw_node(pygame.mouse.get_pos(),GROUND,mouse_rad)
        if pygame.mouse.get_pressed(3)[1]:  # Middle click // make ant
            if not grid[pixel_to_index(pygame.mouse.get_pos())] == CENTER:
                Ant(position=pixel_to_index(pygame.mouse.get_pos()),holding=AIR,start_digging=False,placing=False)
        if pygame.mouse.get_pressed(5)[4]:
            draw_node(pygame.mouse.get_pos(),FOOD,mouse_rad)
        if pygame.mouse.get_pressed(5)[3]:  # Backward click // make pheromones
            draw_pheromone(pixel_to_index(pygame.mouse.get_pos()),intensity=mouse_intensity,rad=0)

pygame.quit()
