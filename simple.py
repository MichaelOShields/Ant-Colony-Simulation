import pygame
import random
from enum import Enum
import numpy as np


# Screen variables
SCREEN_WIDTH = 1000
SCREEN_HEIGHT = 100

background_color = (0,0,0)  # Black
sprite_color = (255,255,255)
frame_speed = 1000

# Try: remove still option
class Movement(Enum):
    LEFT = 1
    # STILL = 2
    RIGHT = 2



"""

[1, 0, 0] -> Move left
[0, 1, 0] -> Don't move
[0, 0, 1] -> Move right

"""

class ObstacleGame:

    def __init__(self,screen_width=500,screen_height=500):
        self.screen_width = screen_width
        self.screen_height = screen_height

        # Initialize display
        self.display = pygame.display.set_mode((self.screen_width,self.screen_height))
        pygame.display.set_caption("Find food")
        self.clock = pygame.time.Clock()
        self.reset()
    
    def reset(self):
        self.food_size = 10
        self.food_pos = random.randint(30, self.screen_width - 30)

        self.player_width = 35  # Player width
        self.player_height = 35  # Player height
        self.player_speed = 10  # Player speed

        self.x_pos = self.screen_width // 2 - self.player_width // 2
        self.y_pos = self.screen_height * .5 - self.player_height // 2
        self.frame = 0
        self.score = 0
        



    def move(self, action):
        if action == "HUMAN":
            keys = pygame.key.get_pressed()
            if keys[pygame.K_LEFT]:
                self.x_pos = max(0, self.x_pos - self.player_speed)
            elif keys[pygame.K_RIGHT]:
                self.x_pos =  min(self.screen_width - self.player_width, self.x_pos + self.player_speed)
        else:
            if np.array_equal(action,[1,0]):  # Left
                self.x_pos = max(0, self.x_pos - self.player_speed)
            elif np.array_equal(action,[0,1]):  # Right
                self.x_pos =  min(self.screen_width - self.player_width, self.x_pos + self.player_speed)



    def step(self, action):
        self.frame += 1
    

        # Get user input
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()


        self.move(action)

        # Update, handle reset, reward
        game_over = False
        reward = -0.5

        foodRect = pygame.Rect(self.food_pos, 0, self.food_size, self.screen_height)
        hugging_wall = self.x_pos <= 5 or self.x_pos + self.player_width >= self.screen_width - 5


        playerRect = rect = pygame.Rect(self.x_pos,self.y_pos,self.player_width,self.player_height)
        

        if hugging_wall:  # just kill the mf
            game_over = True
            reward -= 50
            if action == "HUMAN":
                self.reset()
            return reward,game_over,self.score

        if foodRect.colliderect(playerRect):  # Food found
            reward += 100
            self.score += 1

            while foodRect.colliderect(playerRect):
                self.food_pos = random.randint(30, self.screen_width - 30)
                foodRect = pygame.Rect(self.food_pos, 0, self.food_size, self.screen_height)
        

        if self.frame % 1000 == 0:
            reward += 50
            game_over = True
            return reward, game_over,self.score

        # Update UI, clock
        self.update_ui()
        self.clock.tick(frame_speed)
        return reward, game_over,self.score
    
    
        # return self.get_state(), reward, game_over


    def update_ui(self):
        self.display.fill(background_color)
    
        
        foodRect = pygame.Rect(self.food_pos, 0, self.food_size, self.screen_height)
        pygame.draw.rect(self.display,(0,255,0),foodRect)

        # Draw player
        rect = pygame.Rect(self.x_pos,self.y_pos,self.player_width,self.player_height)
        pygame.draw.rect(self.display,sprite_color,rect)
        pygame.display.update()

# game = ObstacleGame()
# while True:
#     game.step("HUMAN")