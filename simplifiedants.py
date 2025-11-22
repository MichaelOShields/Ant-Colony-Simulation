
#region Imports
import numpy as np
import pygame
import random
import math


#endregion


#region Screen Variables
GRID_SIZE = 30  # Size of grid (obv) // 25
NODE_SIZE = 20  # Pixel size of nodes // 30
SCREEN_WIDTH = GRID_SIZE * NODE_SIZE
SCREEN_HEIGHT = GRID_SIZE * NODE_SIZE

FPS = 1000  # FPS

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
ant_lifespan = max(GRID_SIZE ** 2, GRID_SIZE * 25)

#endregion


#region Mouse Variables

mouse_rad = 1
mouse_intensity = 5

#endregion


#region  Diffusion variables

evaporation_num = .1  # 0.01
cutoff_num = 1

decay_pheromone = lambda x : x - evaporation_num if x - evaporation_num > 0 else 0


pheromone_surface = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.SRCALPHA)
diffusion_factor = .2


food_intensity = 50  # Strength of pheromones released from food
max_pheromone = food_intensity

use_pheromones = False


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

cardinal_directions = [  # Down Up Left Right
    (0,1),
    (0,-1),
    (-1,0),
    (1,0)
]

render_directions = [
        (0, -1,lambda x, y: [(x,y),(x + 1, y)]),  # top edge // return coords of line
        (0, 1, lambda x, y: [(x,y+1),(x + 1, y + 1)]),  # bottom edge
        (-1, 0, lambda x, y: [(x, y),(x, y+1)]),  # Left edge
        (1, 0, lambda x, y: [(x+1, y),(x+1, y+1)])  # Right edge
    ]
    
#endregion


#region Functions

def count_neighbors(position, grid, pheromone_grid):
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


def distance(pos1, pos2):
    return math.sqrt( ( pos1[0] - pos2[0] ) ** 2 + (pos1[1] - pos2[1]) ** 2 )

def return_closest_point(list_of_points, point):

    # print(list_of_points)
    closest_point = random.choice(list_of_points)
    closest_distance = distance(point, closest_point)

    for x,y in list_of_points:
        if distance(point,(x, y)) < closest_distance:
            closest_point = (x, y)
            closest_distance = distance(point, (x, y))
    
    # print(closest_point)
    return closest_point




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



def dist_to_center(x, y):
    return math.sqrt((x - GRID_SIZE // 2)**2 + (y - GRID_SIZE // 2)**2) 


#endregion







class antColony:
    def __init__(self):
        # Initialize Pygame
        pygame.init()
        pygame.font.init()
        self.font = pygame.font.SysFont('Courier New', 15)
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("Simple Ant Colony Reinforcement Learning")
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
        renderer = self.ant.return_sight_table(self.ant.position)
        renderer.append((GRID_SIZE//2,GRID_SIZE//2))
        # renderer.extend(self.ant.permanent_previous_positions)
        for (x, y) in iterable:
            #region Render physical nodes

            # if (x, y) in self.ant.path:

            #     rect = pygame.Rect(index_to_pixel(x), index_to_pixel(y), NODE_SIZE, NODE_SIZE)  # Creates a rectangle encompassing the node
            #     pygame.draw.rect(self.screen, RED, rect)  # Draw the node
            
            # elif (x, y) in self.ant.previous_positions:

            #     rect = pygame.Rect(index_to_pixel(x), index_to_pixel(y), NODE_SIZE, NODE_SIZE)  # Creates a rectangle encompassing the node
            #     pygame.draw.rect(self.screen, GREEN, rect)  # Draw the node

            # else:

            rect = pygame.Rect(index_to_pixel(x), index_to_pixel(y), NODE_SIZE, NODE_SIZE)  # Creates a rectangle encompassing the node
            pygame.draw.rect(self.screen, return_grid_color(self.grid[x,y]), rect)  # Draw the node

            self.render_bounds(x, y)

            #endregion
        if use_pheromones:
            self.update_pheromone_display()
        self.changing_nodes.clear()


    def draw_node(self,mouse_pos,node_type,rad):
        grid_x = pixel_to_index(mouse_pos[0])
        grid_y = pixel_to_index(mouse_pos[1])

        x_min,x_max,y_min,y_max = return_node_radius(grid_x,grid_y,rad)
        for x in range(x_min,x_max+1):  # Iterate through x's in radius
            for y in range(y_min,y_max+1):
                self.update_node((x,y),node_type)


    #region Reset

    def reset(self):  # Reset system
        self.food_positions = []
        self.grid = np.full((GRID_SIZE,GRID_SIZE),GROUND)  # node grid
        self.pheromone_grid = np.full((GRID_SIZE,GRID_SIZE),0.0)  # Make pheromones automatically emit from food?

        self.score = 0
        self.game_over = False

        self.ants = []  # List of all ants
        self.ant_positions = {}  # (x,y):ant memory reference // used to destroy ants that are drawn over
        self.changing_nodes = set()
        self.nonzero_pheromones = set()

        self.grid[GRID_SIZE // 2, GRID_SIZE // 2] = CENTER
        self.changing_nodes.clear()

        while len(self.food_positions) == 0:
            self.randomize_food()


        # Randomly spawn ant w/in 1 node radius of center 
        ant_x_possible = [GRID_SIZE //2 - 1,GRID_SIZE // 2, GRID_SIZE // 2 + 1]
        ant_y_possible = [GRID_SIZE //2 - 1,GRID_SIZE // 2, GRID_SIZE // 2 + 1]



        ant_x = random.choice(ant_x_possible)
        ant_y = random.choice(ant_y_possible)
        while ant_x == ant_y == GRID_SIZE //2 :
            ant_x = random.choice(ant_x_possible)
        
        self.ant = self.Ant(self, (ant_x, ant_y), GROUND)




        self.num_food = len(self.food_positions)

        # Draw border nodes
        self.grid[0:GRID_SIZE-1,0] = BORDER
        self.grid[0:GRID_SIZE-1,GRID_SIZE-1] = BORDER
        self.grid[0,0:GRID_SIZE-1] = BORDER
        self.grid[GRID_SIZE-1,0:GRID_SIZE-1] = BORDER
        self.grid[GRID_SIZE - 1, GRID_SIZE - 1] = BORDER

        self.draw_grid([(x,y) for x in range(GRID_SIZE) for y in range(GRID_SIZE)])


    #endregion

    #region Pheromones

    def remove_pheromone(self,position):
        x,y = position

        self.pheromone_grid[x, y] = 0
        self.changing_nodes.add((x, y))  # To be erased
        self.nonzero_pheromones.discard((x, y))

    def update_pheromones(self):
        # print(self.nonzero_pheromones)
        for (x, y) in self.nonzero_pheromones.copy():
            if self.grid[x, y] == GROUND or self.grid[x, y] == CENTER or self.grid[x, y] == BORDER:
                self.remove_pheromone((x, y))
            
            if self.pheromone_grid[x, y] > max_pheromone:
                self.pheromone_grid[x, y] = max_pheromone - evaporation_num

            # Decay
            self.pheromone_grid[x, y] = decay_pheromone(self.pheromone_grid[x, y])
            if self.pheromone_grid[x, y] <= 0:
                self.remove_pheromone((x, y))

            else:

                # Diffuse
                num_neighbors = count_neighbors((x, y), self.grid, self.pheromone_grid)

                if num_neighbors > 0:
                    amount_to_diffuse = self.pheromone_grid[x, y] * diffusion_factor

                    if self.pheromone_grid[x, y] - amount_to_diffuse > cutoff_num:
                        for z in cardinal_directions:
                            nx, ny = return_bounds(x + z[0]), return_bounds(y + z[1])
                            if self.grid[nx, ny] != GROUND and self.grid[nx, ny] != CENTER and self.grid[nx, ny] != BORDER and self.pheromone_grid[x, y] > self.pheromone_grid[nx, ny]:
                                self.emit_pheromone((nx, ny), self.pheromone_grid[nx, ny] - evaporation_num + (amount_to_diffuse / num_neighbors))

                        self.pheromone_grid[x, y] -= amount_to_diffuse + evaporation_num


    def update_pheromone_display(self):
        pheromone_surface.fill(pygame.Color(0,0,0,0))
        print(max(self.pheromone_grid.flatten()))
        max_limiter = max(self.pheromone_grid.flatten())
        max_limiter = max_pheromone
        for (x, y) in self.nonzero_pheromones.copy():
            if self.grid[x, y] != ANT:
                concentration = self.pheromone_grid[x, y] / max(1,max_limiter)

                alpha = max(0, min(int(concentration * 255), 255))
                # print(max(self.pheromone_grid.flatten()))

                color = (255, 255, 0, alpha)
                rect = pygame.Rect(index_to_pixel(x), index_to_pixel(y), NODE_SIZE, NODE_SIZE)
                pygame.draw.rect(pheromone_surface, color, rect)
        self.screen.blit(pheromone_surface,(0,0))

    def emit_pheromone(self,position,intensity):
        self.pheromone_grid[position] = intensity
        x, y = position
        self.nonzero_pheromones.add((x, y))


    def update_food_pheromones(self):  # Emit pheromones from each food source
        if len(self.food_positions) > 0:
            for (x, y) in self.food_positions:
                if self.pheromone_grid[x, y] < .8 * food_intensity:
                    # print(x,y)
                    self.emit_pheromone((x,y), food_intensity)



    #endregion

    def update_node(self,position,node_type):  # used to update nodes, iterates thru cardinal directions to avoid rendering issues w/ boundaries
        x,y = return_bounds(position)
        if not self.grid[x,y] == CENTER and not self.grid[x,y] == BORDER:
            self.grid[x,y] = node_type
            self.changing_nodes.add((x,y))
            for z in six_directions:  # loop thru cardinal directions of nodes drawn & add them to changing nodes
                self.changing_nodes.add(return_bounds((x+z[0],y+z[1])))


    #region Ant
    class Ant:
        def __init__(self,outer_class,position,holding):

            x_min,x_max,y_min,y_max = return_node_radius(position[0],position[1],1)  # Get radius of ant spawn point

            if not ANT in outer_class.grid[x_min:x_max + 1,y_min:y_max + 1]:  # Continue; ant location is valid

                self.position = position  # (x,y) tuple
                self.holding = holding  # what it's holding // 1 if holding nothing (air)

                outer_class.grid[position] = ANT  # sets grid value to ant
                outer_class.changing_nodes.add(position)  # tells pygame to render ant node next frame
                outer_class.ants.append(self)
                outer_class.ant_positions.update({self.position:self})
                self.lifetime = 0

                self.eaten_food = 0
                self.outer_class = outer_class

                
                self.closest_food = self.find_closest_food(outer_class)  # Have a defined closest food to pursue across movements

                x,y = position
                for z in six_directions:  # ensure boundaries are drawn correctly
                    outer_class.changing_nodes.add(return_bounds((x+z[0],y+z[1])))
                
                self.path = []  # Path
                self.previous_positions = set()  # All nodes it's visited in this "retrieval round"
                self.permanent_previous_positions = set()  # All nodes it's visited ever
        

        def increment_direction(self, direction, position):
            x, y = position
            nx = x + direction[0]
            ny = y + direction[1]
            return (nx, ny)

        def cast_ray(self, direction, position):  # Cast a ray in the direction given, return the node type at the end of the ray and how long the ray is
            length = 0
            current_position = position
            current_position = self.increment_direction(direction, current_position)
            # print(self.outer_class.grid[current_position])
            # print(current_position)
            while self.outer_class.grid[current_position] != CENTER and self.outer_class.grid[current_position] != BORDER and self.outer_class.grid[current_position] != FOOD and self.outer_class.grid[current_position] != GROUND:
                # print(current_position)
                current_position = self.increment_direction(direction, current_position)
                length += 1
            return length, self.outer_class.grid[current_position].item()



        def find_closest_food(self, outer_class):

            return return_closest_point(outer_class.food_positions, self.position)  # Have a defined closest food to pursue across movements // return closest_food

        def check_valid(self, outer_class, nx, ny):  # Check if a movement is valid

            current_digging = self.holding != FOOD  # Only dig if not holding food
            # current_digging = True

            valid = True
            if outer_class.grid[nx, ny] == BORDER or outer_class.grid[nx, ny] == CENTER:
                valid = False
            
            if outer_class.grid[nx, ny] in [GROUND, FOOD] and not current_digging:  # If trying to dig ground but not digging
                valid = False
            
            if self.holding == FOOD and outer_class.grid[nx, ny] == FOOD:
                valid = False
            
            return valid
            

        def dist_to_seen_food(self, ant_sight, position):
            if FOOD in ant_sight:
                x, y = position
                for z in cardinal_directions:
                    nx, ny = x + z[0], y + z[1]
                    if self.cast_ray(z, position)[1] == FOOD:
                        return self.cast_ray(z, position)[0]
            else:
                return GRID_SIZE * 2


        def get_sight(self, position):
            ant_sight = []
            x, y = position
            for z in cardinal_directions:
                ant_sight.append(self.cast_ray(z, position)[1])
            return ant_sight

        def get_big_sight(self, position):
            ant_sight = []
            x, y = position
            for z in cardinal_directions:
                ant_sight.extend(self.cast_ray(z, position))
            return ant_sight

        def return_sight_table(self, position):
            sight_positions = [position]
            x, y = position
            for z in cardinal_directions:
                nx = z[0] * (self.cast_ray(z, position)[0] + 1) + x
                ny = z[1] * (self.cast_ray(z, position)[0] + 1) + y
                sight_positions.append((nx, ny))
            return sight_positions


        def move(self,outer_class,direction_tuple):  # digging is a bool, direction_tuple is a (dx,dy) tuple denoting which direction to travel

            outer_class.score = self.eaten_food / (outer_class.num_food)
            
            reward = -2  # Start w/ -1 reward
            x, y = self.position

            # Check ant life
            if self.lifetime > ant_lifespan:    

                outer_class.game_over = True
                reward -= 100


            nx, ny = return_bounds((self.position[0] + direction_tuple[0],self.position[1] + direction_tuple[1])) # New x, new y

            x_min,x_max,y_min,y_max = return_node_radius(nx,ny,1)
            grid_to_check = outer_class.grid[x_min:x_max + 1,y_min:y_max + 1]  # grid area the ant is moving to

            centerxmin, centerxmax, centerymin, centerymax = return_node_radius(GRID_SIZE // 2,GRID_SIZE // 2,1)
            center_grid = outer_class.grid[centerxmin:centerxmax + 1, centerymin:centerymax + 1]



            valid = self.check_valid(outer_class, nx, ny)

            if valid:

                ant_sight = self.get_sight((nx, ny))
                
                small_sight = []
                for (dx,dy) in cardinal_directions:
                    neighboring_pos = return_bounds((nx + dx, ny + dy))
                    small_sight.append(outer_class.grid[neighboring_pos])

                print(self.get_big_sight((nx, ny)))


                self.previous_positions.add((x, y))
                self.permanent_previous_positions.add((x, y))
                self.lifetime += 1


                """
                REWARDS:

                IF HAS FOOD:
                + GOING CLOSER TO CENTER
                + GOING BACKWARDS IN PATH
                - EXPLORING NEW AREAS

                IF DOESN'T HAVE FOOD:
                + EXPLORING NEW AREAS
                + DIGGING
                - GOING CLOSER TO CENTER
                - GOING TO PREVIOUSLY VISITED AREA
                --- NOT GOING CLOSER TO SEEN FOOD
                ++ GOING CLOSER TO SEEN FOOD

                BOTH:
                - SEES BORDER

                
                """


                if self.holding == FOOD:
                    if dist_to_center(nx, ny) < dist_to_center(x, y) or (nx, ny) in self.path:  # going closer to center
                        reward += 1
                    else:  # - going away from center; time is critical
                        reward -= 2
                
                if self.holding != FOOD:
                    # if dist_to_center(nx, ny) < dist_to_center(x, y):  # going closer to center
                    #     reward -= 1
                    
                    if (nx, ny) in self.path:
                        reward -= 1

                    if outer_class.grid[nx, ny] == GROUND:
                        reward += 2  # Digging
                    
                    if FOOD in self.get_sight((x, y)):  # Check if previously saw food
                        if FOOD in self.get_sight((nx, ny)):
                            if self.dist_to_seen_food(self.get_sight((nx, ny)), (nx, ny)) > self.dist_to_seen_food(self.get_sight((x, y)), (x, y)):  # Going away from food
                                print("going away from food")
                                reward -= 20
                        else:
                            if outer_class.grid[nx, ny] != FOOD:
                                print("ignoring food")
                                reward -= 20  # Ignoring food
                    
                
                if BORDER in small_sight:
                    reward -= 1
                
                
                # if distance((nx, ny), self.closest_food) <= distance(self.position, self.closest_food) and self.holding != FOOD:  # Reward for going towards food
                #     reward += 3
                


                # if outer_class.grid[nx, ny] == FOOD and self.holding != FOOD:


                if outer_class.grid[nx, ny] == FOOD:
                    self.holding = FOOD
                    reward += 50
                    outer_class.food_positions.remove((nx, ny))
                    # print('picked up food')

                    if (nx, ny) == self.closest_food and len(outer_class.food_positions) > 0:

                        outer_class.changing_nodes.add(self.closest_food)

                        self.closest_food = self.find_closest_food(outer_class)  # Have a defined closest food to pursue across movements

                        outer_class.changing_nodes.add(self.closest_food)
                



                if self.holding == FOOD and ANT in center_grid:
                    # Increment score by 1

                    self.lifetime = 0  # Refresh ant lifetime

                    # Increment reward
                    reward += 100

                    self.holding = AIR
                    self.eaten_food += 1
                    print("Score:", outer_class.score)

                    outer_class.changing_nodes.update(self.path)
                    self.path = []  # Reset last positions so it has to go and find another path
                    self.previous_positions = set()

                    # outer_class.draw_grid([(x,y) for x in range(GRID_SIZE) for y in range(GRID_SIZE)])  # Re-render
                    
                    # if not FOOD in outer_class.grid.flatten() or len(outer_class.food_positions) < 1:
                    if len(outer_class.food_positions) < 1:
                        outer_class.game_over = True
                    



                del outer_class.ant_positions[self.position]  # Delete ant reference from positions dictionary
                outer_class.update_node((nx,ny),ANT)
                outer_class.update_node((x, y),AIR)




                


                if not (nx, ny) in self.path:
                    self.path.append(self.position)
                else:
                    outer_class.changing_nodes.update(self.path)
                    self.path = return_new_list(self.path, (nx, ny))
                


                self.position = (nx,ny)
                outer_class.ant_positions.update({self.position:self})

                

                outer_class.score = self.eaten_food / (outer_class.num_food)

                if FOOD in outer_class.grid.flatten():
                    outer_class.changing_nodes.add(self.closest_food)

                    self.closest_food = self.find_closest_food(outer_class)  # Have a defined closest food to pursue across movements

                    outer_class.changing_nodes.add(self.closest_food)

            else:
                reward -= 15  # Disincentivize invalid movements
                self.lifetime += 1
                print("invalid")
                # outer_class.game_over = True
            
            return (reward, outer_class.game_over, outer_class.score)

    #endregino
    

    def place_random(self, big_prob, small_prob,x_start, x_end, y_start, y_end):
        num_food = (self.grid.flatten() == FOOD).sum()
        # print(num_food)
        # if num_food < 1:

        for x in range(x_start, x_end):
            for y in range(y_start, y_end):
                if random.random() < big_prob:
                    self.draw_node(index_to_pixel((x, y)), FOOD, 1)
                    self.food_positions.append((x, y))
                elif random.random() < small_prob:
                    self.draw_node(index_to_pixel((x, y)), FOOD, 0)
                    self.food_positions.append((x, y))


    def randomize_food(self):
        big_probability = 0.000  # 0.003
        small_probability = 0.02  # 0.02
        edge_range = GRID_SIZE // 5  # Edge range to spawn food in

        self.place_random(big_probability, small_probability, 1, edge_range -1, 1, GRID_SIZE-1)

        self.place_random(big_probability, small_probability, GRID_SIZE - edge_range, GRID_SIZE - 1, 1, GRID_SIZE - 1)

        self.place_random(big_probability, small_probability, 1, GRID_SIZE - 1, GRID_SIZE - edge_range, GRID_SIZE - 1)
        
        self.place_random(big_probability, small_probability, 1, GRID_SIZE - 1, 1, edge_range - 1)

        # print(self.food_positions)


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
            if np.array_equal(action,[1,0,0,0]):  # Down

                return self.ant.move(self,(0,1))
            
            elif np.array_equal(action,[0,1,0,0]):  # Up

                return self.ant.move(self,(0,-1))
            
            elif np.array_equal(action,[0,0,1,0]):  # Left

                return self.ant.move(self,(-1,0))
            
            elif np.array_equal(action,[0,0,0,1]):  # Right

                return self.ant.move(self,(1,0))
                


    def update_ui(self):
        if use_pheromones:
            self.update_food_pheromones()
            self.update_pheromones()
            self.changing_nodes.update(self.nonzero_pheromones)
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
    # print(game.ant.last_positions)

pygame.quit()