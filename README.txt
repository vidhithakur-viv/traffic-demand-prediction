Traffic Demand Prediction Hackathon

Author: Vidhi Thakur

Approach:
This solution uses CatBoost to predict traffic demand from the provided dataset.

Steps Followed:
1. Loaded the training and test datasets.
2. Performed exploratory data analysis to understand the features.
3. Applied preprocessing and handled categorical features using CatBoost.
4. Trained a CatBoost model on the training dataset.
5. Generated predictions for the test dataset.
6. Saved the predictions in submission.csv.

Feature Engineering:
- Utilized the features provided in the dataset.
- Handled categorical variables using CatBoost's built-in capabilities.
- Applied basic data cleaning and preprocessing where required.

Tools and Libraries:
- Python
- Pandas
- NumPy
- CatBoost
- Scikit-learn

Files Included:
- traincatboost.py : Model training and prediction script.
- explore.py : Exploratory data analysis.
- README.txt : Description of the solution and methodology.