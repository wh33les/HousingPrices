# %%
import numpy as np
import torch
import torch.nn as nn

class RegressionNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RegressionNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.lrelu = nn.LeakyReLU()#nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.optimizer = torch.optim.SGD(self.parameters(), lr=1e-6)

    def forward(self, x):
        x = self.lrelu(self.fc1(x))
        x = self.fc2(x)
        return x

    def fit(self, X_train, y_train):
        # Convert to tensors
        
        X_train = torch.tensor(X_train.values, dtype=torch.float32)
        y_train = torch.tensor(y_train.values, dtype=torch.float32)
        
        # Variables
    
        epochs = 10000
        criterion = nn.MSELoss()
    
        # Training loop

        for epoch in range(epochs):
            # Forward pass

            y_pred = self(X_train)#.forward(X_train)
            loss = criterion(y_pred, y_train) 
            if epoch % 1000 == 0: 
                print(f"y_pred at epoch {epoch}: {y_pred}")
                print(f"loss at epoch {epoch}: {loss}")

            # Backward pass and optimization

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def predict(self, X_test):
        X_test = torch.tensor(X_test.values, dtype=torch.float32)
        #with torch.no_grad():
        y_pred = self(X_test)#.forward(X_test)
        return y_pred.detach().numpy()


