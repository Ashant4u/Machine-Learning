##############################################################################
# Logistic Regression - ABC Grocery Task
##############################################################################


##############################################################################
# Import Required Packages
##############################################################################

import pandas as pd
import pickle
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score,f1_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_selection import RFECV


##############################################################################
# Import Sample Data
##############################################################################

# Import

data_for_model = pd.read_pickle(("data/abc_classification_modelling.p"))


# Remove Unnecessary Columns

data_for_model.drop("customer_id",axis =1 , inplace =True)

# Shuffle Data

data_for_model = shuffle(data_for_model,random_state=42)

data_for_model["signup_flag"].value_counts(normalize = True)

##############################################################################
# Deal with Missing Values
##############################################################################

data_for_model.isna().sum()

data_for_model.dropna(how = "any", inplace = True)


##############################################################################
# Deal with Outliers
##############################################################################


outlier_investigation = data_for_model.describe()

# Box Plot Approach 

outlier_columns =["distance_from_store","total_sales","total_items"]


for column in outlier_columns:
    
    lower_quartile =data_for_model[column].quantile(0.25)
    upper_quartile =data_for_model[column].quantile(0.75)
    iqr = upper_quartile - lower_quartile
    extended_iqr = iqr * 2
    min_border = lower_quartile - extended_iqr
    max_border = upper_quartile + extended_iqr
    
    outliers = data_for_model[(data_for_model[column]< min_border) | (data_for_model[column]> max_border)].index
    print(f"{len(outliers)} Outliers dectected in column {column}")
    
    data_for_model.drop(outliers,inplace=True)


##############################################################################
# Split Input and Output Variables
##############################################################################

X = data_for_model.drop(["signup_flag"],axis = 1 )

y = data_for_model["signup_flag"]


##############################################################################
# Split out Traing and Test sets
##############################################################################


X_train,X_test,y_train,y_test = train_test_split(X, y,test_size=0.2,random_state=42, stratify=y)


##############################################################################
# Deal with Categorical Values
##############################################################################


categorical_vars = ["gender"]

one_hot_encoder = OneHotEncoder(sparse_output=False,drop="first")

X_train_encoded = one_hot_encoder.fit_transform(X_train[categorical_vars])
X_test_encoded = one_hot_encoder.transform(X_test[categorical_vars])

encoder_feature_names = one_hot_encoder.get_feature_names_out(categorical_vars)



X_train_encoded = pd.DataFrame(X_train_encoded, columns=encoder_feature_names)
X_train = pd.concat([X_train.reset_index(drop=True),X_train_encoded.reset_index(drop=True)], axis= 1)
X_train.drop(categorical_vars, axis = 1 , inplace =True)

X_test_encoded = pd.DataFrame(X_test_encoded, columns=encoder_feature_names)
X_test = pd.concat([X_test.reset_index(drop=True),X_test_encoded.reset_index(drop=True)], axis= 1)
X_test.drop(categorical_vars, axis = 1 , inplace =True)


##############################################################################
# Feature Selection
##############################################################################


clf = LogisticRegression(random_state= 42, max_iter= 1000)
feature_selector = RFECV(clf)

fit = feature_selector.fit(X_train,y_train)

optimal_feature_count = feature_selector.n_features_
print(f"Optimal number of features : {optimal_feature_count}")

X_train = X_train.loc[:,feature_selector.get_support()]
X_test = X_test.loc[:,feature_selector.get_support()]


plt.plot(range(1, len(fit.cv_results_['mean_test_score']) + 1), fit.cv_results_['mean_test_score'], marker = "o")
plt.ylabel("Model Score")
plt.xlabel("Number of Features")
plt.title(f"Feature Selection using RFE \n Optimal number of features is {optimal_feature_count} (at score of {round(max(fit.cv_results_['mean_test_score']),4)})")
plt.tight_layout()
plt.show()


################################################################################
# Model Training
################################################################################

clf = LogisticRegression(random_state= 42, max_iter= 1000)
clf.fit(X_train,y_train)



################################################################################
# Model Assessment
################################################################################


# Access Model Accuracy

y_pred_class = clf.predict(X_test)

y_pred_prob = clf.predict_proba(X_test)[:,1]


# Confusion Matrix

conf_matrix = confusion_matrix(y_test,y_pred_class)

#plt.style.available

plt.style.use("seaborn-v0_8-poster")
plt.matshow(conf_matrix, cmap ="coolwarm")
plt.gca().xaxis.tick_bottom()
plt.title("Confusion Matrix")
plt.ylabel("Actual Class")
plt.xlabel("Predicted Class")
for (i, j), corr_value in np.ndenumerate(conf_matrix):
    plt.text(j, i, corr_value, ha = "center",va ="center", fontsize =20)
plt.show()



# Accuracy (the number of correct classifications out of all attempted classifications)

accuracy_score(y_test, y_pred_class)


# Precision Score ( of all the observations that were predicted as positive, how many were actually positive)

precision_score(y_test, y_pred_class)

# Recall Score ( of all the positive observations, how many did we predict  positive)

recall_score(y_test, y_pred_class)

# F1 Score (Harmonic Mean of Precision and Recall Score)

f1_score(y_test, y_pred_class)


################################################################################
# Finding the optimal Threshold
################################################################################


thresholds = np.arange(0,1,0.01)


precision_scores =[]
recall_scores =[]
f1_scores =[]

for threshold in thresholds:
    
    pred_class = (y_pred_prob >= threshold) * 1
    
    precision = precision_score(y_test, pred_class ,zero_division= 0)
    precision_scores.append(precision)
    
    recall = recall_score(y_test, pred_class)
    recall_scores.append(recall)
       
    f1 = f1_score(y_test, pred_class)
    f1_scores.append(f1)
    
    
max_f1 = max(f1_scores)
max_f1_idx = f1_scores.index(max_f1)


plt.style.use("seaborn-v0_8-poster")
plt.plot(thresholds,precision_scores,label ="Precision",linestyle ="--")
plt.plot(thresholds,recall_scores,label ="Recall",linestyle ="--")
plt.plot(thresholds,f1_scores,label ="F1",linewidth = 5)
plt.title(f"Finding the Optimal threshold for Classification Model \n Max F1: {round(max_f1,2)}(Threshold = {round(thresholds[max_f1_idx],2)})")
plt.xlabel("Threshold")
plt.ylabel("Assessment Score")
plt.legend(loc ="lower left")
plt.tight_layout()
plt.show()


optimal_threshold = 0.44

y_pred_class_opt_thresh = (y_pred_prob >= optimal_threshold) * 1
