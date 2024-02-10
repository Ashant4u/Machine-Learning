##############################################################################

#  Grid Search - Basic Template


##############################################################################

# Import the required Packages

from sklearn.ensemble import RandomForestRegressor
import pandas as pd
from sklearn.model_selection import GridSearchCV

# Import Sample Data

my_df = pd.read_csv("data/sample_data_regression.csv")


# Split Data into input and output Object

X = my_df.drop(["output"],axis = 1)
y = my_df["output"]


# Instantiate our GridSearch Object

gscv = GridSearchCV(
    estimator=RandomForestRegressor(random_state=42),
    param_grid= {"n_estimators" : [10,50,100,500],
                 "max_depth" : [1,2,3,4,5,6,7,8,9,10,None]},
    cv = 5,
    scoring= "r2",
    n_jobs= -1)

# Fit the data

gscv.fit(X,y)


# Get the best CV Score (mean)

gscv.best_score_


# Optimal Parameters

gscv.best_params_

# Create Optimal Model Object

regressor = gscv.best_estimator_