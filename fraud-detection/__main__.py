import pandas as pd
import pickle
from skl2onnx import to_onnx

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

pd.set_option("display.precision", 3)

data_path = "data/fraud-detection.csv"
model_path = "models/fraud-detection.onnx"


def main():
    df = pd.read_csv(data_path)
    seed = 1234

    # Split
    target = "fraud"
    features = [col for col in df.columns if col != target]
    x_train, x_test, y_train, y_test = train_test_split(df[features], df[target], test_size=0.2)

    # Train
    model = LogisticRegression(random_state=seed)
    model.fit(x_train, y_train)

    # Predict
    train_pred = pd.DataFrame({"true": y_train, "pred": model.predict(x_train)})
    test_pred = pd.DataFrame({"true": y_test, "pred": model.predict(x_test)})

    # Score
    train_accuracy = accuracy_score(train_pred["true"], train_pred["pred"])
    test_accuracy = accuracy_score(test_pred["true"], test_pred["pred"])
    print(f"\n{train_accuracy=}\n{test_accuracy=}")

    # Save
    with open(model_path, "wb") as f:
        onx = to_onnx(model, df[features].iloc[:1])
        pickle.dump(onx, f)
        print(f"Model saved to {model_path=}")


if __name__ == "__main__":
    main()
