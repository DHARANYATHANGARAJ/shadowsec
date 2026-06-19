"""
train_model.py
-----------------
Optional training script for the *production* Deep Learning IDS path.

The Streamlit app (app.py) runs perfectly well without this script — it
auto-trains and caches a scikit-learn RandomForest fallback the first time
it needs a model (see model_utils.py). This script is for anyone who wants
the "real" LSTM described in the spec: install TensorFlow, then run

    python train_model.py
    python train_model.py --epochs 30 --n-per-class 600
    python train_model.py --csv path/to/CICIDS2017.csv --label-col Label

It writes the three artifacts model_utils.py looks for first:

    models/lstm_model.h5
    models/scaler.pkl
    models/label_encoder.pkl
    models/feature_list.pkl
    models/training_history.pkl   (epoch accuracy/loss, used by the
                                    Model Performance page's training curve)

   # FUTURE SCOPE: support resuming/fine-tuning an existing checkpoint
   # instead of always training from scratch, and add early stopping /
   # learning-rate scheduling for larger real-world datasets.
"""

import argparse
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

import config
import data_utils


def _check_tensorflow():
    try:
        import tensorflow as tf  # noqa: F401
        return True
    except Exception as exc:  # noqa: BLE001
        print("=" * 70)
        print("TensorFlow is not installed in this environment.")
        print("Install it first, e.g.:  pip install tensorflow")
        print(f"(import error: {exc})")
        print("=" * 70)
        print("The Streamlit app does NOT need this script to run — it will")
        print("automatically use a scikit-learn RandomForest fallback instead.")
        return False


def _load_data(csv_path, label_col, n_per_class):
    if csv_path:
        df = pd.read_csv(csv_path)
        label_col = label_col or "label"
        if label_col not in df.columns:
            raise ValueError(f"Label column '{label_col}' not found in {csv_path}. "
                              f"Available columns: {list(df.columns)}")
    else:
        df = data_utils.generate_synthetic_dataset(n_per_class=n_per_class)
        label_col = data_utils.LABEL_COLUMN
    return df, label_col


def main():
    parser = argparse.ArgumentParser(description="Train the ShadowSec LSTM intrusion-detection model.")
    parser.add_argument("--csv", default=None, help="Path to a real CICIDS2017/NSL-KDD CSV (optional).")
    parser.add_argument("--label-col", default=None, help="Label column name (required if --csv is used).")
    parser.add_argument("--n-per-class", type=int, default=500, help="Rows per class for the synthetic dataset.")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    if not _check_tensorflow():
        sys.exit(1)

    import tensorflow as tf
    from tensorflow.keras import layers, models, callbacks

    print("Loading dataset...")
    df, label_col = _load_data(args.csv, args.label_col, args.n_per_class)
    feature_cols = [c for c in df.columns if c != label_col]

    X = df[feature_cols].copy()
    cat_cols = [c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c])]
    for c in cat_cols:
        X[c] = pd.factorize(X[c])[0]

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df[label_col].astype(str))
    n_classes = len(label_encoder.classes_)
    print(f"Classes: {list(label_encoder.classes_)}")

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)

    # Reshape for the LSTM: (samples, timesteps=1, features)
    n_features = X_train_s.shape[1]
    X_train_seq = X_train_s.reshape((-1, 1, n_features))
    X_val_seq = X_val_s.reshape((-1, 1, n_features))
    X_test_seq = X_test_s.reshape((-1, 1, n_features))

    print("Building LSTM model...")
    model = models.Sequential([
        layers.Input(shape=(1, n_features)),
        layers.LSTM(64, return_sequences=True),
        layers.Dropout(0.3),
        layers.LSTM(32),
        layers.Dropout(0.3),
        layers.Dense(32, activation="relu"),
        layers.Dense(n_classes, activation="softmax"),
    ])
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    model.summary()

    early_stop = callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)

    print("Training...")
    history = model.fit(
        X_train_seq, y_train,
        validation_data=(X_val_seq, y_val),
        epochs=args.epochs, batch_size=args.batch_size,
        callbacks=[early_stop], verbose=2,
    )

    print("Evaluating on held-out test split...")
    proba_test = model.predict(X_test_seq, verbose=0)
    y_pred = np.argmax(proba_test, axis=1)
    print(f"Test accuracy : {accuracy_score(y_test, y_pred):.4f}")
    print(f"Test precision: {precision_score(y_test, y_pred, average='macro', zero_division=0):.4f}")
    print(f"Test recall   : {recall_score(y_test, y_pred, average='macro', zero_division=0):.4f}")
    print(f"Test F1       : {f1_score(y_test, y_pred, average='macro', zero_division=0):.4f}")

    print("Saving artifacts to models/ ...")
    model.save(config.LSTM_MODEL_PATH)
    joblib.dump(scaler, config.SCALER_PATH)
    joblib.dump(label_encoder, config.LABEL_ENCODER_PATH)
    joblib.dump(feature_cols, config.FEATURE_LIST_PATH)
    joblib.dump(dict(history.history), config.MODELS_DIR + "/training_history.pkl")

    print("Done! app.py will now automatically use the trained LSTM on next launch.")


if __name__ == "__main__":
    main()
