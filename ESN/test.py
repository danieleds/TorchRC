import torch
import math
import random
from esn import ESN

dataset = torch.Tensor( [ math.sin(x*0.5) + 2 * round(math.cos(x*0.5)) for x in range(2000) ] )
dataset = dataset / dataset.abs().max()

training_x = dataset[0:1200].view(-1,1)
training_y = dataset[1:1201]
test_x = dataset[1200:-1].view(-1,1)
test_y = dataset[1201:]

model = ESN(1, 50, contractivity_coeff=1.2, density=0.9)

# Collect states
X = model(training_x)

# Washout
X = X[200:]
Y = training_y[200:]

# Train the model
W = X.pinverse() @ Y

# Test
X_test = model(test_x)
predicted_test = X_test @ W

loss = torch.nn.MSELoss()
print("MSE:", loss(test_y, predicted_test).item())