##############################################################################

# Logistic Regression - Basic Template


##############################################################################

# Import the required Packages

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score , confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Import Sample Data

my_df = pd.read_csv("data/sample_data_classification.csv")


# Split Data into input and output Object

X = my_df.drop(["output"],axis = 1)
y = my_df["output"]



# Split Data into training and test sets

X_train , X_test, y_train , y_test = train_test_split(X, y, test_size = 0.2 , random_state= 42,stratify=y )


# Instantiate our Model Object

clf = LogisticRegression(random_state=42)

# Train our Model

clf.fit(X_train, y_train)


# Access Model Accuracy

y_pred = clf.predict(X_test)

accuracy_score(y_test,y_pred)



y_pred_prob = clf.predict_proba(X_test)



# Confusion Matrix

conf_matrix = confusion_matrix(y_test,y_pred)

print(conf_matrix)

plt.style.available

plt.style.use("seaborn-v0_8-poster")
plt.matshow(conf_matrix, cmap ="coolwarm")
plt.gca().xaxis.tick_bottom()
plt.title("Confusion Matrix")
plt.ylabel("Actual Class")
plt.xlabel("Predicted Class")
for (i, j), corr_value in np.ndenumerate(conf_matrix):
    plt.text(j, i, corr_value, ha = "center",va ="center", fontsize =20)
plt.show()






