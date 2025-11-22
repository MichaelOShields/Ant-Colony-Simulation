
#region Imports
import numpy as np
import pygame
import random
import math


#endregion


#region Screen Variables
GRID_SIZE = 25  # Size of grid (obv) // 25
NODE_SIZE = 30  # Pixel size of nodes // 30
SCREEN_WIDTH = GRID_SIZE * NODE_SIZE
SCREEN_HEIGHT = GRID_SIZE * NODE_SIZE

FPS = 200  # FPS

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
BORDER = 5

node_colors = {
    GROUND:BLACK,
    AIR:BLACK,
    FOOD:YELLOW,
    ANT:BROWN,
    CENTER:DARK_BROWN,
    BORDER:GRAY
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

# ant_lifespan = GRID_SIZE ** 2  # number of moves they can make until they die
ant_lifespan = GRID_SIZE ** 2

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
diffusion_factor = .2


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


#region Functions

def return_new_list(array, target):
    if array.count(target) > 0:
        target_index = array.index(target)
        return array[0:target_index]
    else:
        return array



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




#endregion







class antColony:
    def __init__(self):
        # Initialize Pygame
        pygame.init()
        pygame.font.init()
        self.font = pygame.font.SysFont('Courier New', 15)
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("Ant Colony Reinforcement Learning")
        self.clock = pygame.time.Clock()
        self.reset()




    def render_bounds(self, x, y):
        for (dx, dy, line_func) in render_directions:  # Change in x, change in y, lambda function to return coordinates of line
                nx = x + dx  # Index of neighboring x node
                ny = y + dy  # Index of neighboring y node
                if 0 <= nx < GRID_SIZE and 0 <= ny < GRID_SIZE:  # Making sure neighbor is within bounds
                    current_is_ground = (self.grid[x,y] == GROUND)  # Boolean checking if current node is a ground node
                    neighbor_is_ground = (self.grid[nx,ny] == GROUND)  # Boolean chekcing if neighboring node is a ground node
                    if current_is_ground ^ neighbor_is_ground:  # If one is ground and the other isn't; ^ is a XOR operator which basically returns True if both are different
                        start_pos = index_to_pixel(line_func(x,y)[0])  # Start position in coordinate values, using index_to_pixel to convert grid indexes to pixel values
                        end_pos = index_to_pixel(line_func(x,y)[1])
                        pygame.draw.line(self.screen, WHITE, start_pos,end_pos,1)



    def draw_grid(self,iterable):  # consolidate grid drawing; iterable is either changing_nodes, entire empty grid, or just ant vision (add ant_vision boolean parameter later -- add gray background if yes?)
        for (x, y) in iterable:
            #region Render physical nodes
            rect = pygame.Rect(index_to_pixel(x), index_to_pixel(y), NODE_SIZE, NODE_SIZE)  # Creates a rectangle encompassing the node
            pygame.draw.rect(self.screen, return_grid_color(self.grid[x,y]), rect)  # Draw the node

            self.render_bounds(x, y)
            #endregion
        
        self.changing_nodes.clear()


    def draw_node(self,mouse_pos,node_type,rad):
        grid_x = pixel_to_index(mouse_pos[0])
        grid_y = pixel_to_index(mouse_pos[1])

        x_min,x_max,y_min,y_max = return_node_radius(grid_x,grid_y,rad)
        for x in range(x_min,x_max+1):  # Iterate through x's in radius
            for y in range(y_min,y_max+1):
                self.update_node((x,y),node_type)


    def reset(self):  # Reset system
        self.grid = np.full((GRID_SIZE,GRID_SIZE),GROUND)  # node grid

        self.score = 0
        self.game_over = False

        self.ants = []  # List of all ants
        self.ant_positions = {}  # (x,y):ant memory reference // used to destroy ants that are drawn over
        self.changing_nodes = set()

        self.grid = np.full((GRID_SIZE,GRID_SIZE),GROUND)
        self.grid[GRID_SIZE // 2, GRID_SIZE // 2] = CENTER
        self.changing_nodes.clear()

        # Randomly spawn ant w/in 1 node radius of center 
        ant_x_possible = [GRID_SIZE //2 - 1,GRID_SIZE // 2, GRID_SIZE // 2 + 1]
        ant_y_possible = [GRID_SIZE //2 - 1,GRID_SIZE // 2, GRID_SIZE // 2 + 1]



        ant_x = random.choice(ant_x_possible)
        ant_y = random.choice(ant_y_possible)
        while ant_x == ant_y:
            ant_x = random.choice(ant_x_possible)
        
        self.ant = self.Ant(self, (ant_x, ant_y), AIR)
        self.randomize_food()

        # Draw border nodes
        self.grid[0:GRID_SIZE-1,0] = BORDER
        self.grid[0:GRID_SIZE-1,GRID_SIZE-1] = BORDER
        self.grid[0,0:GRID_SIZE-1] = BORDER
        self.grid[GRID_SIZE-1,0:GRID_SIZE-1] = BORDER
        self.grid[GRID_SIZE - 1, GRID_SIZE - 1] = BORDER

        self.draw_grid([(x,y) for x in range(GRID_SIZE) for y in range(GRID_SIZE)])





    def update_node(self,position,node_type):  # used to update nodes, iterates thru cardinal directions to avoid rendering issues w/ boundaries
        x,y = return_bounds(position)
        if not self.grid[x,y] == CENTER:
            self.grid[x,y] = node_type
            self.changing_nodes.add((x,y))
            for z in six_directions:  # loop thru cardinal directions of nodes drawn & add them to changing nodes
                self.changing_nodes.add(return_bounds((x+z[0],y+z[1])))



    class Ant:
        def __init__(self,outer_class,position,holding):

            x_min,x_max,y_min,y_max = return_node_radius(position[0],position[1],1)  # Get radius of ant spawn point

            if not ANT in outer_class.grid[x_min:x_max + 1,y_min:y_max + 1] and GROUND in outer_class.grid[x_min:x_max + 1,y_min:y_max + 1]:  # Continue; ant location is valid

                self.position = position  # (x,y) tuple
                self.holding = holding  # what it's holding // 1 if holding nothing (air)

                outer_class.grid[position] = ANT  # sets grid value to ant
                outer_class.changing_nodes.add(position)  # tells pygame to render ant node next frame
                outer_class.ants.append(self)
                outer_class.ant_positions.update({self.position:self})
                self.lifetime = 0

                x,y = position
                for z in six_directions:  # ensure boundaries are drawn correctly
                    outer_class.changing_nodes.add(return_bounds((x+z[0],y+z[1])))
                
                self.last_positions = []
                self.last_position = self.position
        

        def check_valid(self, outer_class, nx, ny):  # Check if a movement is valid


            x_min,x_max,y_min,y_max = return_node_radius(nx,ny,1)
            grid_to_check = outer_class.grid[x_min:x_max + 1,y_min:y_max + 1]  # grid area the ant is moving to


            valid = outer_class.grid[nx, ny] != BORDER and np.count_nonzero(grid_to_check == ANT) <= 1 and (GROUND in outer_class.grid[x_min:x_max + 1,y_min:y_max + 1] or FOOD in outer_class.grid[x_min:x_max + 1,y_min:y_max + 1] or BORDER in outer_class.grid[x_min:x_max + 1,y_min:y_max + 1])  # no extra ants around destination node, ground is present around desination node
            
            current_digging = True  # Generally want ant to dig
            # current_digging = self.holding != FOOD

            if self.holding == FOOD and outer_class.grid[nx, ny] == FOOD:
                valid = False  # Don't let ant eat food if it's already found food & hasn't deposited it yet
            
            if current_digging and (outer_class.grid[nx,ny] == GROUND or outer_class.grid[nx, ny] == FOOD):  # currently digging AND grid to be dug is ground; if so, can safely check if there's atleast 2 ground nodes in vicinity
                
                normal_grid_to_check = list(grid_to_check.flatten())

                around_valid_nodes = normal_grid_to_check.count(FOOD) > 1 or normal_grid_to_check.count(GROUND) > 1 or normal_grid_to_check.count(BORDER) > 1
                
                valid = valid and around_valid_nodes  # Makes sure ant will still be adjacent to ground node after digging
            else:
                valid = valid and outer_class.grid[nx,ny] == AIR  # If not digging, only allow ant to move to air nodes


            if CENTER in grid_to_check:  # Make sure ant can't move into center, update rewards

                if outer_class.grid[nx, ny] == CENTER:
                    valid = False
            
            return valid, current_digging





        def move(self,outer_class,direction_tuple):  # digging is a bool, direction_tuple is a (dx,dy) tuple denoting which direction to travel
            
            reward = -1  # Start w/ -1 reward

            nx, ny = return_bounds((self.position[0] + direction_tuple[0],self.position[1] + direction_tuple[1])) # New x, new y

            x_min,x_max,y_min,y_max = return_node_radius(nx,ny,1)
            grid_to_check = outer_class.grid[x_min:x_max + 1,y_min:y_max + 1]  # grid area the ant is moving to


            valid, current_digging = self.check_valid(outer_class, nx, ny)

            if valid:
                self.lifetime += 1

                # Increment reward if ant is holding food & moves closer to center
                # if self.holding == FOOD and math.sqrt(nx**2 + ny **2) < math.sqrt(self.position[0] ** 2 + self.position[1] ** 2):
                #     reward += 3
                if self.holding == FOOD and (nx, ny) in self.last_positions:
                    reward += 3


                if self.holding == FOOD and CENTER in grid_to_check:
                    # Increment score by 1

                    self.lifetime = 0  # Refresh ant lifetime

                    # Increment reward
                    reward += 100

                    self.holding = AIR
                    outer_class.score += 1
                    print("Score:", outer_class.score)

                    # Check if this is the last piece of food
                    if not FOOD in outer_class.grid:
                        outer_class.game_over = True
                        return (reward, outer_class.game_over, outer_class.score)


                if current_digging:
                    if outer_class.grid[nx, ny] == FOOD and self.holding != FOOD:
                        reward += 10  # Grabs food, increment reward

                    
                    if not self.holding == FOOD:  # only change self.holding if it isn't carrying food
                        self.holding = outer_class.grid[nx,ny]

                del outer_class.ant_positions[self.position]  # Delete ant reference from positions dictionary
                outer_class.update_node((nx,ny),ANT)
                outer_class.update_node(self.position,AIR)


                if not (nx, ny) in self.last_positions:
                    self.last_positions.append(self.position)
                else:
                    self.last_positions = return_new_list(self.last_positions, (nx, ny))
                
                if len(self.last_positions) == 0:
                    self.last_position = self.position
                else:
                    self.last_position = self.last_positions[-1]
                


                self.position = (nx,ny)
                outer_class.ant_positions.update({self.position:self})   
                self.sight = grid_to_check
                if self.lifetime > ant_lifespan:    
                    outer_class.game_over = True
                    # reward -= 20
                    return (reward, True, outer_class.score)

                return (reward, outer_class.game_over, outer_class.score)
            else:
                reward -= 10  # Disincentivize invalid movements
                return (reward, outer_class.game_over, outer_class.score)

    

    def place_random(self, big_prob, small_prob,x_start, x_end, y_start, y_end):
        for x in range(x_start, x_end):
            for y in range(y_start, y_end):
                if random.random() < big_prob:
                    self.draw_node(index_to_pixel((x, y)), FOOD, 1)
                elif random.random() < small_prob:
                    self.draw_node(index_to_pixel((x, y)), FOOD, 0)

    def randomize_food(self):
        big_probability = 0.003
        small_probability = 0.02
        edge_range = GRID_SIZE // 5  # Edge range to spawn food in

        self.place_random(big_probability, small_probability, 0, edge_range, 0, GRID_SIZE)

        self.place_random(big_probability, small_probability, GRID_SIZE - edge_range, GRID_SIZE, 0, GRID_SIZE)

        self.place_random(big_probability, small_probability, 0, GRID_SIZE, GRID_SIZE - edge_range, GRID_SIZE)
        
        self.place_random(big_probability, small_probability, 0, GRID_SIZE, 0, edge_range)


    def take_action(self, action, source):
        if source == "HUMAN":
            keys = pygame.key.get_pressed()
            if keys[pygame.K_LEFT]:
                self.ant.move(self,(-1,0))
            elif keys[pygame.K_RIGHT]:
                self.ant.move(self,(1,0))
            elif keys[pygame.K_UP]:
                self.ant.move(self,(0,-1))
            elif keys[pygame.K_DOWN]:
                self.ant.move(self,(0,1))
        else:  # RL source
            if np.array_equal(action,[1,0,0,0]):  # Left

                return self.ant.move(self,(-1,0))
            
            elif np.array_equal(action,[0,1,0,0]):  # Right

                return self.ant.move(self,(1,0))
            
            elif np.array_equal(action,[0,0,1,0]):  # Up

                return self.ant.move(self,(0,1))
            
            elif np.array_equal(action,[0,0,0,1]):  # Down

                return self.ant.move(self,(0,-1))
                


    def update_ui(self):
        self.draw_grid(self.changing_nodes)
        pygame.display.flip()  # Update the display
        self.clock.tick(FPS)  # FPS

    def step(self, action, source):

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
    
        self.update_ui()
        return self.take_action(action,source)  # reward, done, score




#  UNCOMMENT FOR HUMAN-PLAYED GAME

game = antColony()

running = True
while running:
    for event in pygame.event.get():

        if event.type == pygame.QUIT:

            running = False
    game.step(pygame.key.get_pressed(),"HUMAN")
    print(game.grid)

pygame.quit()