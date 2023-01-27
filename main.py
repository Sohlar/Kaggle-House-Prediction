import pandas as pd
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_absolute_error, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt

train_data = pd.read_csv('data/train.csv')

test_data = pd.read_csv('data/test.csv')


#Split training data into numerical and categorical dataframes
categorical_train_data = train_data.select_dtypes(include=['object'])
numerical_train_data = train_data.select_dtypes(include=['int64'])
Sale_price_data = train_data.SalePrice

print(train_data.shape)

#Converts cateogircal dataframe into indicator variables
encoded_categorical_train_data = pd.get_dummies(categorical_train_data)
print(train_data.shape)
#Drops NaN values
encoded_categorical_train_data = encoded_categorical_train_data.dropna(axis=0)
numerical_train_data = numerical_train_data.dropna(axis=0)
print(encoded_categorical_train_data.shape)
print(numerical_train_data.shape)


numerical_feature_list = [feature for feature in numerical_train_data.columns if feature != 'SalePrice' ]
categorical_feature_list = [feature for feature in encoded_categorical_train_data.columns]

#This might be wrong, Since we split the dataframe I may not need to signify columns because they only exist within the dataframe
#Except with numerical data we do not want saleprice
#We could potentially improve speed by dropping saleprice from df therefore no longer needing to check when building feature list
numerical_x = numerical_train_data[numerical_feature_list]
categorical_x = encoded_categorical_train_data[categorical_feature_list]

y = train_data.SalePrice

#Numerical Regression
train_numerical_x, val_numerical_x, train_numerical_y, val_numerical_y = train_test_split(numerical_x, y, test_size=0.2, random_state=0)
forest_regression_model = RandomForestRegressor(random_state = 1)
forest_regression_model.fit(train_numerical_x, train_numerical_y)
regression_predictions = forest_regression_model.predict(val_numerical_x)
print(len(val_numerical_y))
print(len(regression_predictions))

print('Regression MAE: ' + str(mean_absolute_error(val_numerical_y, regression_predictions)))
plt.subplot(1, 2, 1)
plt.plot(regression_predictions, color ='r')
plt.plot(val_numerical_y.values, color ='b')

#categorical Classification
train_categorical_x, val_categorical_x, train_categorical_y, val_categorical_y = train_test_split(categorical_x, y, test_size=0.2, random_state=0)
forest_classification_model = RandomForestClassifier(n_estimators=100, random_state=0)
forest_classification_model.fit(train_categorical_x, train_categorical_y)
classification_predictions = forest_classification_model.predict(val_categorical_x)
print(len(val_categorical_y))
print(len(classification_predictions))

print('Classification MAE: ' + str(mean_absolute_error(val_categorical_y, classification_predictions)))
plt.subplot(1, 2, 2)
plt.plot(classification_predictions, color ='r')
plt.plot(val_categorical_y.values, color ='b')
plt.show()

ovr_pred = (classification_predictions + regression_predictions) / 2

