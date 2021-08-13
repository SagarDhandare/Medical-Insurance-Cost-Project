# Importing Libraries:
import pandas as pd
import pickle

# Loading Train dataset:
train_data = pd.read_csv('Train_Data.csv')

# Rounding up & down Age:
train_data['age'] = round(train_data['age'])

# Encoding:
train_data = pd.get_dummies(train_data, drop_first=True)

# Rearranging columns to see better:
train_data = train_data[['age','sex_male','smoker_yes','bmi','children','region_northwest','region_southeast','region_southwest','charges']]

# Splitting Independent & Dependent Feature:
X = train_data.iloc[:, :-1]
y = train_data.iloc[:, 1]

# Train Test Split:
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=0)

# Random Forest Regressor:
from sklearn.ensemble import RandomForestRegressor
RandomForest = RandomForestRegressor()
RandomForest = RandomForest.fit(X_train, y_train)

# Creating a pickle file for the classifier
filename = 'MedicalInsuranceCost.pkl'
pickle.dump(RandomForest, open(filename, 'wb'))