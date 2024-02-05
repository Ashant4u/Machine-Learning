##############################################################################

#  Regression Tree - Basic Template


##############################################################################

# Import the required Packages

from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import pandas as pd

# Import Sample Data

my_df = pd.read_csv("data/sample_data_regression.csv")


# Split Data into input and output Object

X = my_df.drop(["output"],axis = 1)
y = my_df["output"]



# Split Data into training and test sets

X_train , X_test, y_train , y_test = train_test_split(X, y, test_size = 0.2 , random_state= 42)


# Instantiate our Model Object

regressor = DecisionTreeRegressor(min_samples_leaf = 7)

# Train our Model

regressor.fit(X_train, y_train)


# Access Model Accuracy

y_pred = regressor.predict(X_test)

r2_score(y_test,y_pred)




# Demostration of overfitting


y_pred_training = regressor.predict(X_train)

r2_score(y_train,y_pred_training)





# Plot our Decision Tree

import matplotlib.pyplot as plt
from sklearn.tree import plot_tree


plt.figure(figsize=(25,15))
tree = plot_tree(regressor,
                 feature_names=X.columns.tolist(), # or list(X.columns)
                 filled= True,
                 rounded= True,
                 fontsize= 24)














