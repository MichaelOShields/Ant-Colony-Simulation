import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os

class Linear_QNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        # Define the layers of the network
        self.linear1 = nn.Linear(input_size, hidden_size)  # First hidden layer
        self.linear2 = nn.Linear(hidden_size, hidden_size)  # Second hidden layer
        self.linear3 = nn.Linear(hidden_size, output_size)  # Output layer

        # Define the activation function
        self.relu = nn.ReLU()

        # Load existing model weights if available
        if os.path.exists('./classes/model/model.pth'):
            self.load_state_dict(torch.load('./classes/model/model.pth'))
            print('Model loaded')

    def forward(self, x):
        # Pass input through each layer with activation in between
        x = self.relu(self.linear1(x))  # Input to first hidden layer
        x = self.relu(self.linear2(x))
        x = self.linear3(x)  # Third hidden layer to output layer
        return x
    
    def save(self, file_name='model.pth'):
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)
        file_name = os.path.join(model_folder_path, file_name)

        torch.save(self.state_dict(), file_name)

        
class QTrainer:
    def __init__(self, model, learning_rate, gamma):
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.model = model

        self.optimizer = optim.Adam(model.parameters(),lr=self.learning_rate)
        self.criterion = nn.MSELoss()

    def train_step(self, state,action,reward,next_state,done):
        state = torch.tensor(state, dtype=torch.float)
        next_state = torch.tensor(next_state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float)
        # size (n, x)

        if len(state.shape) == 1:
            # size (1, x)
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            done = (done, )

        # 1: predicted Q values with current state
        prediction = self.model(state)

        target = prediction.clone()
        for index in range(len(done)):
            Q_new = reward[index]
            if not done[index]:
                Q_new = reward[index] + self.gamma * torch.max(self.model(next_state[index]))
        
            target[index][torch.argmax(action).item()] = Q_new


        # 2: Q_new = reward + gamma * max(next_predicted Q value) -> only do this if not done
        # prediction.clone()
        # predictions[argmax(action)] = Q_new
        self.optimizer.zero_grad()
        loss = self.criterion(target, prediction)
        loss.backward()

        self.optimizer.step()
