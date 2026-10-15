import matplotlib.pyplot as plt
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.calibration import calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import OneHotEncoder

SEED = 67

df = pd.read_csv("output/dataset.csv")


def split_by_patient(df, size):
    # Split data by patient (anon_id)
    splitter = GroupShuffleSplit(n_splits=1, test_size=size, random_state=SEED)
    a, b = next(splitter.split(df, groups=df["anon_id"]))
    return df.iloc[a], df.iloc[b]


# Split data so that training gets 70%, validation gets 15% and testing gets 15%
train, rest = split_by_patient(df, size=0.30)
val, test = split_by_patient(rest, size=0.50)

for name, part in [("train", train), ("val", val), ("test", test)]:
    print(
        f"{name}: {round(100 * len(part) / len(df), 1)}% of patients, resistant share: {round(1 - part['label'].mean(), 3)}"
    )

# Define feature column type
CATEGORICAL = [
    "antibiotic",
    "organism",
    "culture_description",
    "ordering_mode",
    "gender",
    "age",
]
NUMERIC = [
    "hosp_ward_IP",
    "hosp_ward_OP",
    "hosp_ward_ER",
    "hosp_ward_ICU",
    "prior_resistant",
]

# Convert data into a format that the models can use
preprocess = ColumnTransformer(
    [
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
        ("num", "passthrough", NUMERIC),
    ]
)


def features(part):
    # Remove identifiers from data
    features = part[CATEGORICAL + NUMERIC].copy()

    # Convert gender column to text and replace null values with the string "missing"
    features["gender"] = features["gender"].map({0: "0", 1: "1"})
    features[CATEGORICAL] = features[CATEGORICAL].astype(object).fillna("missing")
    return features, part["label"]


# Get features and answers from data
X_train, y_train = features(train)
X_val, y_val = features(val)
X_test, y_test = features(test)

# Learn encoding rules from training data and apply the rules to validation and test
X_train_t = preprocess.fit_transform(X_train)
X_val_t = preprocess.transform(X_val)
X_test_t = preprocess.transform(X_test)


def evaluate(y, p):
    # p is the predicted probability of being susceptible
    # Convert the probability to a one or zero (required for some metrics)
    pred = (p >= 0.5).astype(int)

    # Return a dictionary of results for all the required metrics
    return {
        "macro AUROC": roc_auc_score(y, p),
        "macro AUPRC": (
            average_precision_score(y, p) + average_precision_score(1 - y, 1 - p)
        )
        / 2,
        "balanced accuracy": balanced_accuracy_score(y, pred),
        "resistant precision": precision_score(y, pred, pos_label=0),
        "resistant recall": recall_score(y, pred, pos_label=0),
        "Brier score": brier_score_loss(y, p),
    }


# Create models, use balanced class weights to handle imbalances in data
models = {
    "logistic regression": LogisticRegression(class_weight="balanced", max_iter=1000),
    "LightGBM": LGBMClassifier(class_weight="balanced", random_state=SEED),
}

results = {}
val_probs = {}
test_probs = {}

for name, model in models.items():
    # Train model
    model.fit(X_train_t, y_train)

    # Run predictions
    val_probs[name] = model.predict_proba(X_val_t)[:, 1]
    test_probs[name] = model.predict_proba(X_test_t)[:, 1]

    # Evaluate results
    results[(name, "val")] = evaluate(y_val, val_probs[name])
    results[(name, "test")] = evaluate(y_test, test_probs[name])

results = pd.DataFrame(results).round(3)
print("\nresults:")
print(results)
results.to_csv("output/results.csv")


def plot_calibration(y, probs, title, path):
    fig, ax = plt.subplots()

    # Plot a perfect calibration target line
    ax.plot([0, 1], [0, 1], "--", color="#9a9990", label="perfect calibration")

    # Plot both the logistic regression and LightGBM calibration results
    for (name, p), color in zip(probs.items(), ["#2a78d6", "#eb6834"]):
        # Compute the calibration curve and plot it
        observed, predicted = calibration_curve(y, p)
        ax.plot(
            predicted,
            observed,
            "-o",
            color=color,
            label=name,
        )
    ax.set_xlabel("Predicted probability of susceptible")
    ax.set_ylabel("Observed share susceptible")
    ax.set_title(title)
    ax.grid()
    ax.legend()
    fig.tight_layout()
    fig.savefig(path)
    print("saved", path)


# Plot both the validation set and test set
plot_calibration(
    y_val, val_probs, "Calibration (validation set)", "output/calibration_val.png"
)
plot_calibration(
    y_test, test_probs, "Calibration (test set)", "output/calibration_test.png"
)
