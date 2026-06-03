import pandas as pd

train = pd.read_csv("train.csv")

print("Unique geohashes:", train["geohash"].nunique())