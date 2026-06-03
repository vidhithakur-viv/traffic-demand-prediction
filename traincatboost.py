import pandas as pd
from catboost import CatBoostRegressor

# Load data
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

# Fill missing values
train["RoadType"] = train["RoadType"].fillna("Unknown")
test["RoadType"] = test["RoadType"].fillna("Unknown")

train["Weather"] = train["Weather"].fillna("Unknown")
test["Weather"] = test["Weather"].fillna("Unknown")

train["Temperature"] = train["Temperature"].fillna(train["Temperature"].median())
test["Temperature"] = test["Temperature"].fillna(train["Temperature"].median())

# Features and target
X = train.drop(["Index", "demand"], axis=1)
y = train["demand"]

X_test = test.drop(["Index"], axis=1)

# Categorical columns
cat_features = [
    "geohash",
    "timestamp",
    "RoadType",
    "LargeVehicles",
    "Landmarks",
    "Weather"
]

# Train model
model = CatBoostRegressor(
    iterations=500,
    learning_rate=0.05,
    depth=8,
    loss_function="RMSE",
    verbose=100
)

model.fit(X, y, cat_features=cat_features)

# Predict
predictions = model.predict(X_test)

# Submission
submission = pd.DataFrame({
    "Index": test["Index"],
    "demand": predictions
})

submission.to_csv("submission.csv", index=False)

print("Submission file created!")