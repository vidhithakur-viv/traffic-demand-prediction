import pandas as pd
import numpy as np

from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# ==================================================
# LOAD DATA
# ==================================================

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

# ==================================================
# MISSING VALUES
# ==================================================

train["RoadType"] = train["RoadType"].fillna("Unknown")
test["RoadType"] = test["RoadType"].fillna("Unknown")

train["Weather"] = train["Weather"].fillna("Unknown")
test["Weather"] = test["Weather"].fillna("Unknown")

temp_median = train["Temperature"].median()

train["Temperature"] = train["Temperature"].fillna(temp_median)
test["Temperature"] = test["Temperature"].fillna(temp_median)

# ==================================================
# GEOHASH MEAN ENCODING
# ==================================================

global_mean = train["demand"].mean()

geo_mean = (
    train.groupby("geohash")["demand"]
    .mean()
)

train["geo_mean_demand"] = (
    train["geohash"]
    .map(geo_mean)
)

test["geo_mean_demand"] = (
    test["geohash"]
    .map(geo_mean)
    .fillna(global_mean)
)

# ==================================================
# TIME FEATURES
# ==================================================

for df in [train, test]:

    df["hour"] = (
        df["timestamp"]
        .str.split(":")
        .str[0]
        .astype(int)
    )

    df["minute"] = (
        df["timestamp"]
        .str.split(":")
        .str[1]
        .astype(int)
    )

    df["time_slot"] = (
        df["hour"] * 60 +
        df["minute"]
    )

    df["sin_time"] = np.sin(
        2 * np.pi * df["time_slot"] / 1440
    )

    df["cos_time"] = np.cos(
        2 * np.pi * df["time_slot"] / 1440
    )

# ==================================================
# DAY FEATURE
# ==================================================

for df in [train, test]:
    df["is_day_49"] = (
        df["day"] == 49
    ).astype(int)

# ==================================================
# FEATURES / TARGET
# ==================================================

X = train.drop(
    ["Index", "demand", "timestamp"],
    axis=1
)

y = train["demand"]

X_test = test.drop(
    ["Index", "timestamp"],
    axis=1
)

# ==================================================
# CATEGORICAL FEATURES
# ==================================================

cat_features = [
    "geohash",
    "RoadType",
    "LargeVehicles",
    "Landmarks",
    "Weather"
]

# ==================================================
# VALIDATION SPLIT
# ==================================================

X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# ==================================================
# MODEL
# ==================================================

model = CatBoostRegressor(
    iterations=5000,
    learning_rate=0.03,
    depth=8,
    loss_function="RMSE",
    eval_metric="R2",
    random_seed=42,
    verbose=200
)

# ==================================================
# TRAIN
# ==================================================

model.fit(
    X_train,
    y_train,
    cat_features=cat_features,
    eval_set=(X_val, y_val),
    use_best_model=True
)

# ==================================================
# VALIDATION SCORE
# ==================================================

val_preds = model.predict(X_val)

print(
    "Validation R2:",
    r2_score(y_val, val_preds)
)

# ==================================================
# FEATURE IMPORTANCE
# ==================================================

importance = model.feature_importances_

print("\nFeature Importance:\n")

for feature, score in sorted(
    zip(X.columns, importance),
    key=lambda x: x[1],
    reverse=True
):
    print(
        f"{feature}: {score:.4f}"
    )

# ==================================================
# RETRAIN ON FULL DATA
# ==================================================

final_model = CatBoostRegressor(
    iterations=model.get_best_iteration(),
    learning_rate=0.03,
    depth=8,
    loss_function="RMSE",
    random_seed=42,
    verbose=200
)

final_model.fit(
    X,
    y,
    cat_features=cat_features
)

# ==================================================
# PREDICT TEST
# ==================================================

predictions = final_model.predict(X_test)

# ==================================================
# SUBMISSION
# ==================================================

submission = pd.DataFrame({
    "Index": test["Index"],
    "demand": predictions
})

submission.to_csv(
    "submission.csv",
    index=False
)

print("\nSubmission file created!")