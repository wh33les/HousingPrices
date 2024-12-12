import torch
import torch.nn as nn
import torch.optim as optim

class MyNeuralNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(163, 8, bias=True)
        self.layer2 = nn.Linear(8, 1, bias=True)
        self.loss = nn.MSELoss()
        self.compile_()

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        return x.squeeze()

    def fit(self, x, y):
        x = torch.tensor(x.values, dtype=torch.float32)
        y = torch.tensor(y.values, dtype=torch.float32)
        losses = []
        for epoch in range(100):
            ## Inference
            res = self.forward(x)#self(self,x)
            loss_value = self.loss(res,y)

            ## Backpropagation
            self.opt.zero_grad()  # flush previous epoch's gradient
            loss_value.backward() # compute gradient
            self.opt.step()       # Perform iteration using gradient above

            ## Logging
            losses.append(loss_value.item())
        
    def compile_(self):
        self.opt = optim.SGD(self.parameters(), lr=0.0001)
       
    def predict(self, x_test):
        x_test = torch.tensor(x_test.values, dtype=torch.float32)
        self.eval()
        y_test_hat = self(x_test)
        return y_test_hat.detach().numpy()
        #self.train()