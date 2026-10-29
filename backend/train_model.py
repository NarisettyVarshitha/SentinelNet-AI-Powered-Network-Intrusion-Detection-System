from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"
TRAIN_PATH = DATA_DIR / "KDDTrain+.txt"
TEST_PATH = DATA_DIR / "KDDTest+.txt"
MODEL_PATH = MODEL_DIR / "rf_model.pkl"
TEST_SELECTED_PATH = DATA_DIR / "test_selected.csv"

COLUMNS = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes",
    "land", "wrong_fragment", "urgent", "hot", "num_failed_logins",
    "logged_in", "num_compromised", "root_shell", "su_attempted", "num_root",
    "num_file_creations", "num_shells", "num_access_files", "num_outbound_cmds",
    "is_host_login", "is_guest_login", "count", "srv_count", "serror_rate",
    "srv_serror_rate", "rerror_rate", "srv_rerror_rate", "same_srv_rate",
    "diff_srv_rate", "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count",
    "dst_host_same_srv_rate", "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate",
    "dst_host_serror_rate", "dst_host_srv_serror_rate", "dst_host_rerror_rate",
    "dst_host_srv_rerror_rate", "label", "difficulty_level",
]
FEATURE_COLUMNS = COLUMNS[:-2]
CATEGORICAL_COLUMNS = ["protocol_type", "service", "flag"]
NUMERIC_COLUMNS = [
    column for column in FEATURE_COLUMNS if column not in CATEGORICAL_COLUMNS
]


def load_dataset(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(
            f"Dataset not found: {path}. Put KDDTrain+.txt and KDDTest+.txt in the project's data folder."
        )
    return pd.read_csv(path, header=None, names=COLUMNS)


def binary_labels(labels: pd.Series) -> pd.Series:
    normalized = labels.astype(str).str.strip().str.rstrip(".").str.lower()
    return (normalized != "normal").astype("int8")


def main() -> None:
    print(f"Loading training data from {TRAIN_PATH}")
    training_data = load_dataset(TRAIN_PATH)
    test_data = load_dataset(TEST_PATH)

    x_train = training_data[FEATURE_COLUMNS]
    y_train = binary_labels(training_data["label"])
    x_test = test_data[FEATURE_COLUMNS]
    y_test = binary_labels(test_data["label"])

    preprocessing = ColumnTransformer(
        transformers=[
            ("categorical", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_COLUMNS),
            ("numeric", "passthrough", NUMERIC_COLUMNS),
        ]
    )
    model = Pipeline(
        steps=[
            ("preprocessing", preprocessing),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=100,
                    class_weight="balanced_subsample",
                    n_jobs=-1,
                    random_state=42,
                ),
            ),
        ]
    )

    print(f"Training on {len(training_data):,} rows...")
    model.fit(x_train, y_train)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)

    seed_data = x_test.copy()
    seed_data["label"] = y_test
    seed_data.to_csv(TEST_SELECTED_PATH, index=False)
    print(f"Saved trained model to {MODEL_PATH}")
    print(f"Saved simulation seed data to {TEST_SELECTED_PATH}")


if __name__ == "__main__":
    main()
