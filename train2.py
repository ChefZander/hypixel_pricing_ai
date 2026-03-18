import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

data = np.load("training_data.npz")

X = torch.from_numpy(data['x']).float()
raw_y = torch.from_numpy(data['y']).float().view(-1, 1)

y_mean = raw_y.mean().item()
y_std = raw_y.std().item()

y = (raw_y - y_mean) / y_std

print(f"Loaded {len(X)} samples.")
print(f"Target Stats -> Mean: {y_mean:.4f}, Std: {y_std:.4f}")

train_size = int(0.8 * len(X))
train_X, test_X = X[:train_size], X[train_size:]
train_y, test_y = y[:train_size], y[train_size:]

train_loader = DataLoader(TensorDataset(train_X, train_y), batch_size=128, shuffle=True)
test_loader = DataLoader(TensorDataset(test_X, test_y), batch_size=128)

class ResBlock(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(size, size),
            nn.BatchNorm1d(size),
            nn.ReLU(),
            nn.Linear(size, size),
            nn.BatchNorm1d(size)
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(x + self.net(x))

class SkyblockPriceNet(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.initial = nn.Sequential(
            nn.Linear(input_size, 1024),
            nn.ReLU()
        )
        self.res1 = ResBlock(1024)
        self.res2 = ResBlock(1024)
        self.res3 = ResBlock(1024)
        
        self.output = nn.Sequential(
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, x):
        x = self.initial(x)
        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
        return self.output(x)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SkyblockPriceNet(input_size=16384).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=1, factor=0.5)

print(f"Training on {device}...")

for epoch in range(3):
    model.train()
    train_loss = 0
    
    for batch_X, batch_y in train_loader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
        
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
    
    avg_train_loss = train_loss / len(train_loader)

    model.eval()
    test_loss = 0
    test_coin_error = 0
    test_pct_error = 0
    
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            test_loss += loss.item()

            pred_log = (outputs * y_std) + y_mean
            act_log = (batch_y * y_std) + y_mean
            
            pred_coins = 10 ** pred_log
            act_coins = 10 ** act_log
            
            test_coin_error += torch.abs(pred_coins - act_coins).mean().item()
            
            batch_pct_err = torch.abs((pred_coins - act_coins) / (act_coins + 1e-6)).mean().item()
            test_pct_error += batch_pct_err * 100

    avg_test_loss = test_loss / len(test_loader)
    avg_test_coin_err = test_coin_error / len(test_loader)
    avg_test_pct_err = test_pct_error / len(test_loader)
    
    scheduler.step(avg_test_loss)
    
    current_lr = optimizer.param_groups[0]['lr']
    print(f"Epoch {epoch+1:02d} | Train MSE: {avg_train_loss:.4f} | Test MSE: {avg_test_loss:.4f} | "
          f"Err: {avg_test_coin_err:,.0f} Coins ({avg_test_pct_err:.2f}%) | LR: {current_lr:.6f}")

torch.save({
    'model_state': model.state_dict(),
    'y_mean': y_mean,
    'y_std': y_std,
    'vector_size': 16384
}, "skyblock_model_v2.pth")