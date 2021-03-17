import torch
from torch.utils.data import DataLoader, TensorDataset
import torch_rc.nn
import torch_rc.optim
import math

# Build training/validation dataset
ds_x = [torch.sin(torch.arange(0, 20, 0.1) + 2*math.pi*torch.rand(1)).view(-1, 1) if i % 2 == 0 else
        torch.sin(torch.arange(0, 40, 0.2) + 2*math.pi*torch.rand(1)).view(-1, 1)
        for i in range(1000)]
ds_y = [i % 2 for i in range(1000)]
ds_x = torch.stack(ds_x)
ds_y = torch.as_tensor(ds_y)

train_ds = TensorDataset(ds_x[:600], ds_y[:600])
val_ds = TensorDataset(ds_x[600:], ds_y[600:])

# Define the model
esn = torch_rc.nn.esn.LeakyESN(1, 64, scale_rec=0.99)
readout = torch_rc.nn.Linear(64, 2)

# Train the model (here we do it in minibatches)
train_dl = DataLoader(train_ds, batch_size=8)
optimizer = torch_rc.optim.optim.RidgeIncrementalClassifier(readout)
for x, y in train_dl:
    h, _ = esn(x.transpose(1, 0))
    optimizer.fit_step(h[-1], y)
optimizer.fit_end()

# Validate the model (here we do it example by example)
n_correct = 0
for x, y in val_ds:
    h, _ = esn(x.unsqueeze(1))
    out = readout(h[-1])
    if out.argmax(-1) == y:
        n_correct += 1
print(f"Validation accuracy: {100 * n_correct / len(val_ds)}%")
