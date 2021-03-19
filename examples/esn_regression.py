import torch
import torch_rc.nn
import torch_rc.optim

# Build training/validation dataset
t_max = 50
t_step = 0.1
t = torch.arange(0, t_max, t_step)
ds_x = torch.sin(t + torch.cos(t)**2).view(-1, 1)
ds_y = ds_x[1:]  # Next-step prediction

n_train = int(0.8 * len(t))  # 80% of the dataset is for training, the rest for validation
train_x, train_y = ds_x[:n_train], ds_y[:n_train]
val_x, val_y = ds_x[n_train:-1], ds_y[n_train:]

# Try different reservoir sizes
for reservoir_size in [4, 8, 16, 32, 64]:
    mse = []
    # Average the performance over multiple random initializations
    for trial in range(10):
        # Define the model
        esn = torch_rc.nn.LeakyESN(1, reservoir_size, scale_rec=0.99)
        readout = torch_rc.nn.Linear(reservoir_size, 1)

        # Train the model (here we do it in full batch)
        optimizer = torch_rc.optim.RidgeRegression(readout, l2_reg=1e-2)
        n_washout = 20
        h, _ = esn(train_x.unsqueeze(1))
        optimizer.fit(h[n_washout:, 0, :], train_y[n_washout:])

        # Validate the model
        h, _ = esn(val_x.unsqueeze(1))
        out = readout(h[:, 0, :])
        mse.append(torch.nn.functional.mse_loss(out, val_y))
    print(f"Reservoir size: {reservoir_size} -> avg MSE loss: {sum(mse)/len(mse)}")
