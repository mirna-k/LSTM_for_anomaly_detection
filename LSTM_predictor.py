import torch
import pandas as pd
import numpy as np
import torch.nn as nn
import torch.optim as optim
import plotly.graph_objects as go
from tqdm import tqdm
from sklearn.metrics import classification_report, mean_squared_error, mean_absolute_error, roc_curve, auc, precision_recall_curve
import math
from torch.utils.data import DataLoader, TensorDataset

from helper import crop_datetime, extract_signal_and_anomaly_array

MODEL_PATH = 'model_LSTM_predictor.pth'
DATA_PATH = 'DRIVE/MyDrive/ESA-AD_data/channel_44.csv'
TRAINED_MODEL_PATH = 'DRIVE/MyDrive/ESA-AD_data/model_LSTM_predictor.pth'
MODEL_STATE_DICT_PATH = 'DRIVE/MyDrive/ESA-AD_data/model_LSTM_predictor_state_dict.pth'
CHANNEL = "channel_44"
WINDOW_SIZE = 1000
PRED_SIZE = 100  # how many future steps to predict
STEP = 10  # slide step
BATCH_SIZE = 128
LEARNING_RATE = 1e-3
EPOCHS = 20

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    #df = get_scaled_data_values("esa_data_my_csvs/channel_37.csv", CHANNEL)
    df = pd.read_csv(DATA_PATH)

    start_datetime="2000-01-01T00:00:00.000Z"
    end_datetime="2001-01-01T00:00:00.000Z"
    filtered_df = crop_datetime(df, start_datetime, end_datetime)

    # Extract arrays
    data, labels = extract_signal_and_anomaly_array(filtered_df, CHANNEL)

    # Use only normal data for training
    normal_mask = labels == 0
    normal_data = data[normal_mask]

    # Normalize safety check
    assert normal_data.min() >= 0 and normal_data.max() <= 1

    X_train, Y_train = create_windows(normal_data, input_len=WINDOW_SIZE, pred_len=PRED_SIZE, step=STEP)

    # Reshape: (samples, timesteps, features)
    X_train = X_train[..., np.newaxis]
    Y_train = Y_train[..., np.newaxis]

    # Convert to PyTorch tensors
    X_tensor = torch.tensor(X_train, dtype=torch.float32)
    Y_tensor = torch.tensor(Y_train, dtype=torch.float32)

    # Build dataset and dataloader
    train_dataset = TensorDataset(X_tensor, Y_tensor)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    print(f"Prepared {len(train_dataset)} training sequences.")

    # Instantiate model
    model = LSTMPredictor(
        input_size=1,
        hidden_size=128,
        num_layers=3,
        pred_len=PRED_SIZE,
        dropout=0.3
    )

    trained_model = train_model(model, train_loader, epochs=EPOCHS, learning_rate=LEARNING_RATE)

    torch.save(trained_model, MODEL_PATH)


def main_eval():
    trained_model = torch.load(TRAINED_MODEL_PATH, weights_only=False, map_location=torch.device('cpu'))

    # Recreate your model architecture
    trained_model = LSTMPredictor(
        input_size=1,
        hidden_size=128,
        num_layers=3,
        pred_len=PRED_SIZE,
        dropout=0.3
    )

    # Load weights
    trained_model.load_state_dict(torch.load(MODEL_STATE_DICT_PATH, map_location="cpu"))

    trained_model=trained_model.to(DEVICE)

    df = pd.read_csv(DATA_PATH)

    start_datetime="2000-01-01T00:00:00.000Z"
    end_datetime="2001-01-01T00:00:00.000Z"
    filtered_df = crop_datetime(df, start_datetime, end_datetime)

    data, labels = extract_signal_and_anomaly_array(filtered_df, CHANNEL)

    errors, pred_labels, true_labels, threshold = evaluate_model(
        model=trained_model,
        data=data,            # full scaled signal
        labels=labels,             # original anomaly labels
        window_size=WINDOW_SIZE,
        pred_size=PRED_SIZE,
    )

    # Plot 1: Prediction Error and Threshold
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(
        y=errors,
        mode='lines',
        name='Prediction Error'
    ))

    fig1.add_trace(go.Scatter(
        y=[threshold] * len(errors),
        mode='lines',
        name='Threshold',
        line=dict(color='red', dash='dash')
    ))

    fig1.update_layout(
        title='Prediction Error and Anomaly Threshold',
        xaxis_title='Time Step',
        yaxis_title='Prediction Error',
    )
    fig1.show()
    fig1.write_html("pred_err_anomal_threshold.html")

    # Plot 2: Signal with Anomalies
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(
        y=data,
        mode='lines',
        name='Signal'
    ))

        # Detected anomalies (red)
    fig2.add_trace(go.Scatter(
        y=np.where(pred_labels == 1, data.max(), np.nan),
        mode='markers',
        name='Detected Anomaly',
        marker=dict(color='red', size=5),
    ))

    # Ground truth anomalies (green)
    fig2.add_trace(go.Scatter(
        y=np.where(true_labels == 1, data.max(), np.nan),
        mode='markers',
        name='True Anomaly',
        marker=dict(color='green', size=3),
    ))

    # Layout
    fig2.update_layout(
        title='Anomaly Detection Result',
        xaxis_title='Time Step',
        yaxis_title='Signal Value',
        legend=dict(x=0, y=1.1, orientation="h")
    )
    fig2.show() 
    fig2.write_html("anomaly_detection_result.html")

    print(classification_report(true_labels, pred_labels, digits=4))

    # Additional metrics
    mse = mean_squared_error(true_labels, pred_labels)
    mae = mean_absolute_error(true_labels, pred_labels)
    rmse = math.sqrt(mse)
    print(f"MSE: {mse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")

    # Compute ROC curve and ROC area
    label_for_roc = true_labels[WINDOW_SIZE + PRED_SIZE - 1 : WINDOW_SIZE + PRED_SIZE - 1 + len(errors)]
    fpr, tpr, _ = roc_curve(label_for_roc, errors)
    roc_auc = auc(fpr, tpr)

    # Plot ROC curve
    fig_roc = go.Figure()
    fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f'ROC curve (AUC = {roc_auc:.4f})'))
    fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Random', line=dict(dash='dash')))
    fig_roc.update_layout(
        title='Receiver Operating Characteristic (ROC)',
        xaxis_title='False Positive Rate',
        yaxis_title='True Positive Rate',
        showlegend=True
    )
    fig_roc.show()
    fig_roc.write_html("roc_curve.html")

    # Compute Precision-Recall curve and area
    precision, recall, _ = precision_recall_curve(true_labels, errors[:len(true_labels)])
    pr_auc = auc(recall, precision)

    # Plot Precision-Recall curve
    fig_pr = go.Figure()
    fig_pr.add_trace(go.Scatter(x=recall, y=precision, mode='lines', name=f'PR curve (AUC = {pr_auc:.4f})'))
    fig_pr.update_layout(
        title='Precision-Recall Curve',
        xaxis_title='Recall',
        yaxis_title='Precision',
        showlegend=True
    )
    fig_pr.show()
    fig_pr.write_html("pr_curve.html")


def train_model(model, train_loader, epochs=25, learning_rate=1e-3, device='cuda' if torch.cuda.is_available() else 'cpu'):
    print(device)
    model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    model.train()
    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        for batch_x, batch_y in tqdm(train_loader, desc=f"Epoch {epoch}", leave=False):
            batch_x = batch_x.to(device)  # shape: (batch, window, 1)
            batch_y = batch_y.to(device)  # shape: (batch, pred_len, 1)

            optimizer.zero_grad()
            output = model(batch_x)       # shape: (batch, pred_len, 1)
            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch}/{epochs} | Loss: {avg_loss:.6f}")

    return model


def evaluate_model(model, data, labels, window_size, pred_size, threshold=0.001, device='cuda' if torch.cuda.is_available() else 'cpu'):
    print(device)
    model.eval()
    model.to(device)

    # Generate sliding windows
    X_eval, Y_eval = [], []
    for i in range(0, len(data) - window_size - pred_size + 1):
        x = data[i : i + window_size]
        y = data[i + window_size : i + window_size + pred_size]
        X_eval.append(x)
        Y_eval.append(y)

    X_eval = np.array(X_eval)[..., np.newaxis]  # (samples, window, 1)
    Y_eval = np.array(Y_eval)[..., np.newaxis]  # (samples, pred_len, 1)

    X_tensor = torch.tensor(X_eval, dtype=torch.float32).to(device)
    Y_tensor = torch.tensor(Y_eval, dtype=torch.float32).to(device)

    batch_size = 256
    eval_loader = DataLoader(TensorDataset(X_tensor, Y_tensor), batch_size=batch_size, shuffle=False)

    errors = []

    with torch.no_grad():
        for x_batch, y_batch in eval_loader:
            y_pred = model(x_batch)
            batch_errors = torch.mean((y_pred - y_batch) ** 2, dim=(1, 2))  # MSE over all timesteps and features
            errors.extend(batch_errors.cpu().numpy())

    errors = np.array(errors)

    # Threshold: if not given, use e.g., 99th percentile of errors as threshold
    if threshold is None:
        threshold = np.percentile(errors, 99) * 10
        print(f"Auto-calculated threshold (99th percentile): {threshold:.6f}")

    # Create anomaly predictions
    preds = (errors > threshold).astype(int)

    # Align prediction with the timeline
    padding = window_size + pred_size - 1
    full_preds = np.zeros_like(data, dtype=int)
    full_preds[padding:padding + len(preds)] = preds

    # Align labels too
    true_labels = labels[:len(full_preds)]

    return errors, full_preds, true_labels, threshold


# Prepare sliding windows for training
def create_windows(arr, input_len, pred_len, step=1):
    X, Y = [], [] # X - input len, Y - input + pred len
    total_len = input_len + pred_len
    for i in range(0, len(arr) - total_len + 1, step):
        x_win = arr[i:i + input_len]
        y_win = arr[i + input_len:i + total_len]
        X.append(x_win)
        Y.append(y_win)
    return np.array(X), np.array(Y)


class LSTMPredictor(nn.Module):
    """
    LSTM-based sequence prediction model with convolutional feature extraction.
    Given a window of length `input_len`, predicts the next `pred_len` steps for each feature.
    """
    def __init__(self,
                 input_size: int,
                 hidden_size: int = 64,
                 num_layers: int = 2,
                 pred_len: int = 1,
                 cnn_channels: int = 16,
                 kernel_size: int = 3,
                 dropout: float = 0.2):
        
        super(LSTMPredictor, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.pred_len = pred_len

        # CNN for feature extraction: applies across time dimension
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=input_size, out_channels=cnn_channels, kernel_size=kernel_size, padding=kernel_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(in_channels=cnn_channels, out_channels=input_size, kernel_size=1),
            nn.ReLU()
        )

        # LSTM for sequence modeling
        self.lstm = nn.LSTM(input_size=input_size,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            batch_first=True,
                            dropout=dropout)

        # Decoder: map hidden state to prediction
        self.decoder = nn.Linear(hidden_size, input_size * pred_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: Tensor of shape (batch_size, seq_len, input_size)
        :return: Tensor of shape (batch_size, pred_len, input_size)
        """
        batch_size, seq_len, input_size = x.shape

        # CNN expects input shape: (batch, input_size, seq_len)
        x_cnn = x.permute(0, 2, 1)  # (B, C, T)
        x_cnn = self.cnn(x_cnn)     # (B, C, T)
        x = x_cnn.permute(0, 2, 1)  # (B, T, C) — back to LSTM input shape

        # Pass through LSTM
        lstm_out, (h_n, _) = self.lstm(x)
        last_hidden = h_n[-1]  # (batch, hidden_size)

        # Decode
        decoded = self.decoder(last_hidden)  # (batch, input_size * pred_len)
        preds = decoded.view(batch_size, self.pred_len, self.input_size)
        return preds


if __name__ == "__main__":
    main_eval()
