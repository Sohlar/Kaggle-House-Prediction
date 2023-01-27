import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

train_data = pd.read_csv('data/train.csv')

test_data = pd.read_csv('data/test.csv')





encoder = OneHotEncoder()
print(train_data.shape)
train_data = pd.get_dummies(train_data, columns=train_data.select_dtypes(include='object').columns)
train_data = train_data.dropna(axis=0)
print(train_data.shape)
'''
for feature in feature_list:
    print(train_data[feature].dtype)
    if train_data[feature].dtype == 'int64':
        train_data[feature] = encoder.fit_transform(train_data[[feature]])
'''

#feature_list_encoded = [encoder.fit_transform(train_data[[feature]]) for feature in feature_list if train_data[feature].dtype == object]
feature_list = [feature for feature in train_data.columns if feature != 'SalePrice' ]
x = train_data[feature_list]

y = train_data.SalePrice


train_x, val_x, train_y, val_y = train_test_split(x, y, test_size=0.2, random_state=0)

forest_model = RandomForestRegressor(random_state = 1)
forest_model.fit(train_x, train_y)
predictions = forest_model.predict(val_x)

print(mean_absolute_error(val_y, predictions))