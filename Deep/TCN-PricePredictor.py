import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt


# Preprocessing with sliding window and cyclic encoding
def preprocess_long_data(df, window_size=168, min_price=0.1, max_price=250):
    df['Date'] = pd.to_datetime(df['Date'])
    df['Price'] = np.clip(df['Price'], min_price, max_price)
    scaler = MinMaxScaler()
    df['Price'] = scaler.fit_transform(df[['Price']])
    df['Hour_sin'] = np.sin(2 * np.pi * df['Hour'] / 24)
    df['Hour_cos'] = np.cos(2 * np.pi * df['Hour'] / 24)
    df['Month'] = df['Date'].dt.month
    df['Month_sin'] = np.sin(2 * np.pi * df['Month'] / 12)
    df['Month_cos'] = np.cos(2 * np.pi * df['Month'] / 12)
    df['Day_of_week'] = df['Date'].dt.dayofweek
    feature_columns = ['Price', 'Hour_sin', 'Hour_cos', 'Month_sin', 'Month_cos', 'Day_of_week']
    feature_data = df[feature_columns].values.astype(np.float32)
    X, y = [], []
    for i in range(len(feature_data) - window_size):
        X.append(feature_data[i:i + window_size])
        y.append(feature_data[i + window_size, 0])  # Target is the price
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32)
    return X, y, scaler


# Load and preprocess the dataset
df = pd.read_excel('train.xlsx', sheet_name='prices_2')
long_df = df.melt(id_vars=['PRICES'], var_name='Hour', value_name='Price')
long_df.dropna(subset=['Price'], inplace=True)
long_df.rename(columns={'PRICES': 'Date'}, inplace=True)
long_df['Hour'] = long_df['Hour'].str.extract(r'(\d+)').astype(int)
long_df = long_df.sort_values(by=['Date', 'Hour']).reset_index(drop=True)

window_size = 168
X, y, scaler = preprocess_long_data(long_df, window_size=window_size)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

batch_size = 32
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


# Define the TCN model
class CausalConvNet(nn.Module):
    def __init__(self, input_size, window_size):
        super(CausalConvNet, self).__init__()
        self.causal_conv1 = nn.Conv1d(in_channels=input_size, out_channels=32, kernel_size=3, padding=2, dilation=1)
        self.conv_blocks = nn.Sequential(
            self._conv_block(32, 32, dilation=1),
            self._conv_block(32, 64, dilation=2),
            self._conv_block(64, 128, dilation=4),
            self._conv_block(128, 128, dilation=8),
            self._conv_block(128, 128, dilation=16),
            self._conv_block(128, 128, dilation=32)
        )
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(128, 128)
        self.fc2 = nn.Linear(128, 1)

    def _conv_block(self, in_channels, out_channels, dilation):
        return nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=dilation, dilation=dilation),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.causal_conv1(x)
        x = self.conv_blocks(x)
        x = self.global_avg_pool(x).squeeze(-1)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)


input_size = X_train.shape[2]
model = CausalConvNet(input_size=input_size, window_size=window_size)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.00003)


# Training function with early stopping and learning rate scheduler
def denormalize(scaler, values):
    return scaler.inverse_transform(values.reshape(-1, 1)).flatten()


def train_model_with_scheduler_and_early_stopping(
        model, train_loader, test_loader, criterion, optimizer, scaler, epochs=30, patience=5
):
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5)

    best_test_loss = float('inf')
    early_stop_counter = 0

    all_train_losses = []
    all_test_losses = []

    for epoch in range(epochs):
        model.train()
        train_losses = []
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs.squeeze(), targets)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        train_loss = np.mean(train_losses)
        all_train_losses.append(train_loss)

        model.eval()
        test_losses = []
        all_predictions, all_targets = [], []
        with torch.no_grad():
            for inputs, targets in test_loader:
                outputs = model(inputs)
                loss = criterion(outputs.squeeze(), targets)
                test_losses.append(loss.item())
                all_predictions.append(outputs.squeeze().cpu().numpy())
                all_targets.append(targets.cpu().numpy())

        test_loss = np.mean(test_losses)
        all_test_losses.append(test_loss)
        scheduler.step(test_loss)

        all_predictions = np.concatenate(all_predictions, axis=0)
        all_targets = np.concatenate(all_targets, axis=0)
        denorm_predictions = denormalize(scaler, all_predictions)
        denorm_targets = denormalize(scaler, all_targets)

        rmse = np.sqrt(np.mean((denorm_predictions - denorm_targets) ** 2))
        mape = np.mean(np.abs((denorm_targets - denorm_predictions) / denorm_targets)) * 100

        print(
            f"Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss:.6f}, Test Loss: {test_loss:.6f}, RMSE: {rmse:.2f}, MAPE: {mape:.2f}%")

        if test_loss < best_test_loss:
            best_test_loss = test_loss
            early_stop_counter = 0
            torch.save(model.state_dict(), "../best_tcn_model.pth")
        else:
            early_stop_counter += 1

        if early_stop_counter >= patience:
            print(f"Early stopping triggered after {epoch + 1} epochs.")
            break

    print("Training complete. Best model saved as 'best_tcn_model.pth'.")

    # Visualizations
    plt.figure(figsize=(12, 6))
    plt.plot(all_train_losses, label="Train Loss")
    plt.plot(all_test_losses, label="Test Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Training and Testing Loss Over Epochs")
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.plot(denorm_targets[:200], label='Actual')
    plt.plot(denorm_predictions[:200], label='Predicted')
    plt.title("Actual vs Predicted Prices")
    plt.xlabel("Time Steps")
    plt.ylabel("Price")
    plt.legend()
    plt.show()


# Train the model
train_model_with_scheduler_and_early_stopping(model, train_loader, test_loader, criterion, optimizer, scaler, epochs=30)
