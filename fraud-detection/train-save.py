import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

data_path = "/opt/app-root/src/mlops-ws/data/fraud-detection.csv"
model_path = "/opt/app-root/src/mlops-ws/models/fraud-detection.onnx"

class LogisticRegressionTorch(nn.Module):
    def __init__(self, input_dim):
        super(LogisticRegressionTorch, self).__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x):
        return torch.sigmoid(self.linear(x))

def main():
    # Load and preprocess data
    df = pd.read_csv(data_path)
    for col in df.columns:
        df[col] = df[col].astype(float)
    
    target = "fraud"
    features = [col for col in df.columns if col != target]
    x_train, x_test, y_train, y_test = train_test_split(df[features], df[target], test_size=0.2, random_state=1234)
    
    # Convert data to PyTorch tensors
    x_train_tensor = torch.tensor(x_train.values, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
    x_test_tensor = torch.tensor(x_test.values, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32)

    # Define and train the PyTorch logistic regression model
    model = LogisticRegressionTorch(input_dim=len(features))
    criterion = nn.BCELoss()  # Binary Cross Entropy for binary classification
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    # Training loop
    num_epochs = 100
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(x_train_tensor)
        loss = criterion(outputs, y_train_tensor)
        loss.backward()
        optimizer.step()

    # Evaluation
    model.eval()
    with torch.no_grad():
        train_pred = (model(x_train_tensor) > 0.5).float()
        test_pred = (model(x_test_tensor) > 0.5).float()

    train_accuracy = accuracy_score(y_train, train_pred.numpy())
    test_accuracy = accuracy_score(y_test, test_pred.numpy())
    print(f"Train Accuracy: {train_accuracy}\nTest Accuracy: {test_accuracy}")

    # Convert to ONNX
    dummy_input = torch.randn(1, len(features), dtype=torch.float32)
    torch.onnx.export(model, dummy_input, model_path, input_names=["float_input"], output_names=["output"], opset_version=11)
    print(f"Model saved to {model_path}")

if __name__ == "__main__":
    main()
