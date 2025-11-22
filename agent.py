import torch
import random
import numpy as np
from collections import deque
import math
import pygame

from simplifiedants import antColony,return_bounds, six_directions, cardinal_directions, GRID_SIZE, ant_lifespan, dist_to_center

from model import Linear_QNet, QTrainer

from helper import plot


# Node types
GROUND = 0
AIR = 1
FOOD = 2
ANT = 3
CENTER = 4
BORDER = 5

MAX_MEMORY = 100_000
BATCH_SIZE = 1_000
LEARNING_RATE = 0.001  #  0.001



def distance(pos1, pos2):
    return math.sqrt((pos1[0]-pos2[0])**2+(pos1[1]-pos2[1])**2)

def return_closest_point(list_of_points, point):

    closest_point = random.choice(list_of_points)
    closest_distance = distance(point, closest_point)

    for x,y in list_of_points:
        if distance(point,(x, y)) < closest_distance:
            closest_point = (x, y)
            closest_distance = distance(point, (x, y))
    
    return closest_point, closest_distance




class DQNAgent:
    def __init__(self):
        self.num_games = 0
        self.epsilon = 0  # Randomness
        self.gamma = 0.9  # Discount rate // must be <1
        self.memory = deque(maxlen=MAX_MEMORY)  # If exceeds memory, forgets first memories (popleft())
        self.model = Linear_QNet(15, 128, 4)
        self.trainer = QTrainer(self.model, learning_rate=LEARNING_RATE, gamma=self.gamma)



    def get_state(self,game):  # Represent the state of the game; relative positional values normalized to [0,1] range

        state = []  # 4 closest food bools, 4 closest path bools, 1 holding food bool

        #State:
        # Grid type of six direction nodes using return_bounds
        # 0 (if not holding food) or 1 (if holding food)
        # bool if any of 4 directions is last_position

        ant = game.ant
        x,y = ant.position


        # Giving it the closest food:
        # if ant.holding != FOOD:
        # # First 4 bools
        #     if len(game.food_positions) > 0:
        #         closest_food = ant.closest_food

        #         state = [0,0,0,0]  # Four booleans representing the direction & if that's in the direction of food
        #         best_direction = random.randint(0, 3)  # Initialize best move as random move
        #         best_x, best_y = x + cardinal_directions[best_direction][0], y + cardinal_directions[best_direction][1]

        #         for z in cardinal_directions:
        #             nx, ny = x + z[0], y + z[1]
        #             if distance(closest_food, (nx, ny)) < distance(closest_food, (best_x, best_y)):  # If moving in this direction will bring us closer to the nearest food

        #                 best_x, best_y = nx, ny
        #                 best_direction = cardinal_directions.index(z)
                
        #         state[best_direction] = 1
                
        #     else:
        #         state = [0,0,0,0]
                
        # else:
        #     # Second 4 bools
            # if len(ant.path) == 0:
            #     state.extend([0,0,0,0])
            # else:
            #     added_array = [0,0,0,0]
            #     last_pos = ant.path[-1]

            #     # nx = ox + dx
            #     # dx = nx - ox
                
            #     dir_closest_to_path = (
            #         # x - last_pos[0], y - last_pos[1]
            #         last_pos[0] - x, last_pos[1] - y
            #     )
            #     print(dir_closest_to_path)
            #     best_move_back = cardinal_directions.index(dir_closest_to_path)
                
            #     for z in cardinal_directions:
            #         nx, ny = x + z[0], y + z[1]
            #         if (nx, ny) in ant.path and ant.path.index((nx, ny)) < ant.path.index(last_pos):
            #             last_pos = (nx, ny)
            #             best_move_back = cardinal_directions.index((x - nx, y - ny))
                
            #     added_array[best_move_back] = 1
            #     state.extend(added_array)

        

        # Not giving it the closest food (giving it 4 node types and 4 cardinal directions and if they're in the most recent direction or whatever and if it's holding food):

        state = []

        for z in cardinal_directions:
            nx, ny = x + z[0], y + z[1]
            state.append(round(ant.cast_ray(z, ant.position)[0] / GRID_SIZE,2))
            state.append(ant.cast_ray(z, ant.position)[1] / 5)
        
        if len(ant.path) == 0:
            state.extend([0,0,0,0])
        else:
            added_array = [0,0,0,0]
            last_pos = ant.path[-1]

            # nx = ox + dx
            # dx = nx - ox

            dir_closest_to_path = (
                # x - last_pos[0], y - last_pos[1]
                last_pos[0] - x, last_pos[1] - y
            )
            print(dir_closest_to_path)
            best_move_back = cardinal_directions.index(dir_closest_to_path)

            for z in cardinal_directions:
                nx, ny = x + z[0], y + z[1]
                if (nx, ny) in ant.path and ant.path.index((nx, ny)) < ant.path.index(last_pos):
                    last_pos = (nx, ny)
                    best_move_back = cardinal_directions.index((x - nx, y - ny))
            
            added_array[best_move_back] = 1
            state.extend(added_array)
        
        # for z in cardinal_directions:
        #     nx, ny = x + z[0], y + z[1]
            
        #     if (nx, ny) in ant.permanent_previous_positions:
        #         state.append(1)
        #     else:
        #         state.append(0)  # New area to explore

        # for z in cardinal_directions:
        #     nx, ny = x + z[0], y + z[1]
        #     state.append(game.grid[nx, ny].item() / 5)  # Normalizing it to [0,1]
        
        # for z in cardinal_directions:
        #     nx, ny = x + z[0], y + z[1]
        #     if (nx, ny) in ant.path:
        #         state.append(1)
        #     else:
        #         state.append(0)  # Unexplored area (for this retrieval "round")
        
        # for z in cardinal_directions:  # Distance to center
        #     nx, ny = x + z[0], y + z[1]
        #     if game.grid[nx, ny] == AIR and dist_to_center(nx, ny) <= dist_to_center(x, y):  # If it's closer and valid to go to
        #         state.append(1)
        #     else:
        #         state.append(0)
        
        # state.append(round(ant.lifetime / ant_lifespan,2))  # If it gets too high it'll die

        state.append(round(x / GRID_SIZE, 2))
        state.append(round(y / GRID_SIZE, 2))

        if ant.holding == FOOD:
            state.append(1)
        else:
            state.append(0)
        
        


        print(state)
        return np.array(state,int)

    def remember(self, state, action, reward, next_state, game_over):
        self.memory.append((state, action, reward, next_state, game_over))  # popleft if MAX_MEMORY is reached



    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)  # Returns a list of tuples
        else:
            mini_sample = self.memory
        
        states,actions,rewards,next_states,game_overs = zip(*mini_sample,strict=False)
        self.trainer.train_step(states, actions, rewards, next_states, game_overs)

    def train_short_memory(self, state, action, reward, next_state, game_over):
        self.trainer.train_step(state, action, reward, next_state, game_over)
    


    def get_action(self, state, game):
        # random moves in the beginning // tradeoff btwn exploration & exploitation
        self.epsilon = 120 - self.num_games  # Hard-coded 80 games

        final_move = [0,0,0,0]
        

        # Initial move determination
        # if random.randint(0, 200) < self.epsilon or random.random() < 0.07:  # 7% of all moves are random
        if random.randint(0,200) < self.epsilon:
            move = random.randint(0,len(final_move) - 1)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1
        
        # LRDU -> Move order
        # xxDU
        # DULR -> Cardinal directions order
        # if np.array_equal(state, [0,0,0,0]):
        #     move = random.randint(0,len(final_move) - 1)
        #     final_move[move] = 1
        # else:
        #     final_move = state
        


        


        # if np.array_equal(final_move,[1,0,0,0]):  # Left

        #     direction_tuple = (-1,0)
        
        # elif np.array_equal(final_move,[0,1,0,0]):  # Right

        #     direction_tuple = (1,0)
        
        # elif np.array_equal(final_move,[0,0,1,0]):  # Up

        #     direction_tuple = (0,1)
        
        # elif np.array_equal(final_move,[0,0,0,1]):  # Down

        #     direction_tuple = (0,-1)


        # x, y = game.ant.position
        # nx, ny = return_bounds((x + direction_tuple[0], y + direction_tuple[1]))

        # while game.ant.check_valid(game, nx, ny) == False:  # Only let ant make a move if it's a valid move
        #     # print("INVALID")

        #     final_move = [0,0,0,0]
        #     if random.randint(0, 200) < self.epsilon:
        #         move = random.randint(0,len(final_move) - 1)
        #         final_move[move] = 1
        #     else:
        #         state0 = torch.tensor(state, dtype=torch.float)
        #         prediction = self.model(state0)
        #         move = torch.argmax(prediction).item()
        #         final_move[move] = 1
            


        #     if np.array_equal(final_move,[1,0,0,0]):  # Left

        #         direction_tuple = (-1,0)
            
        #     elif np.array_equal(final_move,[0,1,0,0]):  # Right

        #         direction_tuple = (1,0)
            
        #     elif np.array_equal(final_move,[0,0,1,0]):  # Up

        #         direction_tuple = (0,1)
            
        #     elif np.array_equal(final_move,[0,0,0,1]):  # Down

        #         direction_tuple = (0,-1)


        #     x, y = game.ant.position
        #     nx, ny = return_bounds((x + direction_tuple[0], y + direction_tuple[1]))

        return final_move


def train():
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record_score = 0
    agent = DQNAgent()
    game = antColony()
    while True:  # Run forever until stop training
        # Get old state
        state_old = agent.get_state(game)
        # print(agent.get_state(game))

        # Get move based on current state
        final_move = agent.get_action(state_old, game)

        # Perform move and get new state
        reward, done, score = game.step(final_move, "agent")
        state_new = agent.get_state(game)
        # print(reward)
        # print(state_new)

        # Train short memory (1 step)
        agent.train_short_memory(state_old,final_move,reward,state_new,done)
        
        # Remember
        agent.remember(state_old, final_move, reward, state_new, done)

        if done:
            # Train long memory/replay memory, plot result
            game.reset()
            agent.num_games += 1
            agent.train_long_memory()

            if score > record_score:
                record_score = score
                agent.model.save()
            
            print("Game", agent.num_games, "Score", score, "Record", record_score)
            
            
            plot_scores.append(score)
            total_score += score
            mean_score = total_score/agent.num_games
            plot_mean_scores.append(mean_score)

            plot(plot_scores,plot_mean_scores)


if __name__ == '__main__':
    train()
