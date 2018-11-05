import torch
import math
from esn import ESN

# Prepare a synthetic dataset
dataset = torch.Tensor( [ math.sin(x*0.5) + 2 * round(math.cos(x*0.5)) for x in range(2000) ] )
dataset = dataset / dataset.abs().max()

# Washout length
washout = 200

# Split training set and test set
training_x = dataset[0:1200].view(-1,1)
training_y = dataset[1:1201]
test_x = dataset[1200:-1].view(-1,1)
test_y = dataset[1201:]

model = ESN(1, reservoir_size=50, contractivity_coeff=1.2, density=0.9)

# Collect states for training set
X = model(training_x)

# Washout
X = X[washout:]
Y = training_y[washout:]

# Train the model by Moore-Penrose pseudoinversion.
W = X.pinverse() @ Y

# Evaluate the model on the test set
# We pass the latest training state in order to avoid the need for another washout
X_test = model(test_x, X[-1])
predicted_test = X_test @ W

# Compute and print the loss
loss = torch.nn.MSELoss()
print("MSE:", loss(test_y, predicted_test).item())