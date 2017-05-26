# House Pricing Prediction

# Polynomial Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
training_set = pd.read_csv('Data/train.csv')
X_train = training_set.iloc[:, :-1]
y = training_set.iloc[:, -1].values
X_test = pd.read_csv('Data/test.csv')
X = X_train.append(X_test)

# Filling null values
"""
from sklearn.preprocessing import Imputer
imp_mean = Imputer(missing_values="NaN", strategy='mean', axis = 0)
imp_most_frequent = Imputer(missing_values="NaN", strategy='most_frequent', axis = 0)
imp_median = Imputer(missing_values="NaN", strategy='median', axis = 0)
"""

X.loc[:, 'LotFrontage'] = X.loc[:, 'LotFrontage'].fillna(X.loc[:, 'LotFrontage'].mean())
X.loc[:, 'Alley'] = X.loc[:, 'Alley'].fillna(X.loc[:, 'Alley'].value_counts().idxmax()) # Lots of NaN's, consider removing
X.loc[:, 'Electrical'] = X.loc[:, 'Electrical'].fillna(X.loc[:, 'Electrical'].value_counts().idxmax())
# All other NaN's are actual values, so they need to be converted
X = X.fillna('i')
X.loc[:, 'GarageYrBlt'] = X.loc[:, 'GarageYrBlt'].replace('i', 0)
X.loc[:, 'MasVnrArea'] = X.loc[:, 'MasVnrArea'].replace('i', 0)

X = pd.get_dummies(X, drop_first=True,
                           columns=['MSSubClass', 'MSZoning', 'Street', 'Alley',
                                   'LotShape', 'LandContour', 'Utilities', 
                                   'LotConfig', 'LandSlope', 'LandSlope', 
                                   'LandSlope', 'LandSlope', 'Neighborhood', 
                                   'Condition1', 'Condition2', 'BldgType', 
                                   'HouseStyle', 'OverallQual', 'OverallCond', 
                                   'YearBuilt', 'YearRemodAdd', 'RoofStyle', 
                                   'RoofMatl', 'Exterior1st', 'Exterior2nd', 
                                   'MasVnrType', 'ExterQual', 'ExterCond', 
                                   'Foundation', 'BsmtQual', 'BsmtCond', 
                                   'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 
                                   'Heating', 'MSSubClass', 'HeatingQC', 
                                   'CentralAir', 'Electrical', 'KitchenQual', 
                                   'Functional', 'FireplaceQu', 'GarageType', 
                                   'GarageType', 'GarageYrBlt', 'GarageFinish', 
                                   'GarageQual', 'GarageCond', 'PavedDrive', 
                                   'PoolQC', 'Fence', 'MiscFeature', 
                                   'MoSold', 'YrSold', 'SaleType',
                                   'SaleCondition'])

# Splitting into training examples and test examples
X_train = X.iloc[:1460, :]
X_test = X.iloc[1460:, :]

"""
# Encoding categorical data in X
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
X.loc[:, 'MSSubClass'] = LabelEncoder().fit_transform(X.loc[:, 'MSSubClass'])
X.loc[:, 'MSZoning'] = LabelEncoder().fit_transform(X.loc[:, 'MSZoning'])
X.loc[:, 'Street'] = LabelEncoder().fit_transform(X.loc[:, 'Street'])
X.loc[:, 'Alley'] = LabelEncoder().fit_transform(X.loc[:, 'Alley'])
X.loc[:, 'LotShape'] = LabelEncoder().fit_transform(X.loc[:, 'LotShape'])
X.loc[:, 'LandContour'] = LabelEncoder().fit_transform(X.loc[:, 'LandContour'])
X.loc[:, 'Utilities'] = LabelEncoder().fit_transform(X.loc[:, 'Utilities'])
X.loc[:, 'LotConfig'] = LabelEncoder().fit_transform(X.loc[:, 'LotConfig'])
X.loc[:, 'LandSlope'] = LabelEncoder().fit_transform(X.loc[:, 'LandSlope'])
X.loc[:, 'Neighborhood'] = LabelEncoder().fit_transform(X.loc[:, 'Neighborhood'])
X.loc[:, 'Condition1'] = LabelEncoder().fit_transform(X.loc[:, 'Condition1'])
X.loc[:, 'Condition2'] = LabelEncoder().fit_transform(X.loc[:, 'Condition2'])
X.loc[:, 'BldgType'] = LabelEncoder().fit_transform(X.loc[:, 'BldgType'])
X.loc[:, 'HouseStyle'] = LabelEncoder().fit_transform(X.loc[:, 'HouseStyle'])
X.loc[:, 'OverallQual'] = LabelEncoder().fit_transform(X.loc[:, 'OverallQual'])
X.loc[:, 'OverallCond'] = LabelEncoder().fit_transform(X.loc[:, 'OverallCond'])
X.loc[:, 'YearBuilt'] = LabelEncoder().fit_transform(X.loc[:, 'YearBuilt'])
X.loc[:, 'YearRemodAdd'] = LabelEncoder().fit_transform(X.loc[:, 'YearRemodAdd'])
X.loc[:, 'RoofStyle'] = LabelEncoder().fit_transform(X.loc[:, 'RoofStyle'])
X.loc[:, 'RoofMatl'] = LabelEncoder().fit_transform(X.loc[:, 'RoofMatl'])
X.loc[:, 'Exterior1st'] = LabelEncoder().fit_transform(X.loc[:, 'Exterior1st'])
X.loc[:, 'Exterior2nd'] = LabelEncoder().fit_transform(X.loc[:, 'Exterior2nd'])
X.loc[:, 'MasVnrType'] = LabelEncoder().fit_transform(X.loc[:, 'MasVnrType'])
X.loc[:, 'ExterQual'] = LabelEncoder().fit_transform(X.loc[:, 'ExterQual'])
X.loc[:, 'ExterCond'] = LabelEncoder().fit_transform(X.loc[:, 'ExterCond'])
X.loc[:, 'Foundation'] = LabelEncoder().fit_transform(X.loc[:, 'Foundation'])
X.loc[:, 'BsmtQual'] = LabelEncoder().fit_transform(X.loc[:, 'BsmtQual'])
X.loc[:, 'BsmtCond'] = LabelEncoder().fit_transform(X.loc[:, 'BsmtCond'])
X.loc[:, 'BsmtExposure'] = LabelEncoder().fit_transform(X.loc[:, 'BsmtExposure'])
X.loc[:, 'BsmtFinType1'] = LabelEncoder().fit_transform(X.loc[:, 'BsmtFinType1'])
X.loc[:, 'BsmtFinType2'] = LabelEncoder().fit_transform(X.loc[:, 'BsmtFinType2'])
X.loc[:, 'Heating'] = LabelEncoder().fit_transform(X.loc[:, 'Heating'])
X.loc[:, 'Heating'] = LabelEncoder().fit_transform(X.loc[:, 'MSSubClass'])
X.loc[:, 'HeatingQC'] = LabelEncoder().fit_transform(X.loc[:, 'HeatingQC'])
X.loc[:, 'CentralAir'] = LabelEncoder().fit_transform(X.loc[:, 'CentralAir'])
X.loc[:, 'Electrical'] = LabelEncoder().fit_transform(X.loc[:, 'Electrical'])
X.loc[:, 'KitchenQual'] = LabelEncoder().fit_transform(X.loc[:, 'KitchenQual'])
X.loc[:, 'Functional'] = LabelEncoder().fit_transform(X.loc[:, 'Functional'])
X.loc[:, 'FireplaceQu'] = LabelEncoder().fit_transform(X.loc[:, 'FireplaceQu'])
X.loc[:, 'GarageType'] = LabelEncoder().fit_transform(X.loc[:, 'GarageType'])
X.loc[:, 'GarageYrBlt'] = LabelEncoder().fit_transform(X.loc[:, 'GarageYrBlt'])
X.loc[:, 'GarageFinish'] = LabelEncoder().fit_transform(X.loc[:, 'GarageFinish'])
X.loc[:, 'GarageQual'] = LabelEncoder().fit_transform(X.loc[:, 'GarageQual'])
X.loc[:, 'GarageCond'] = LabelEncoder().fit_transform(X.loc[:, 'GarageCond'])
X.loc[:, 'PavedDrive'] = LabelEncoder().fit_transform(X.loc[:, 'PavedDrive'])
X.loc[:, 'PoolQC'] = LabelEncoder().fit_transform(X.loc[:, 'PoolQC'])
X.loc[:, 'Fence'] = LabelEncoder().fit_transform(X.loc[:, 'Fence'])
X.loc[:, 'MiscFeature'] = LabelEncoder().fit_transform(X.loc[:, 'MiscFeature'])
X.loc[:, 'MoSold'] = LabelEncoder().fit_transform(X.loc[:, 'MoSold'])
X.loc[:, 'YrSold'] = LabelEncoder().fit_transform(X.loc[:, 'YrSold'])
X.loc[:, 'SaleType'] = LabelEncoder().fit_transform(X.loc[:, 'SaleType'])
X.loc[:, 'SaleCondition'] = LabelEncoder().fit_transform(X.loc[:, 'SaleCondition'])
"""

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train1, X_train2, y_train1, y_train2 = train_test_split(X_train, y, test_size = 0.2, random_state = 0)
X_train1 = X_train1.values
X_train2 = X_train2.values

# Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)"""

# Fitting Linear Regression to the dataset
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X_train1, y_train1)
y_pred1 = lin_reg.predict(X_train1)
y_pred2 = lin_reg.predict(X_train2)

# Fitting Polynomial Regression to the dataset
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 2) # Muy mal, probar con SGD?
X_poly = poly_reg.fit_transform(X_train1)
X_poly2 = poly_reg.fit_transform(X_train2)
poly_reg.fit(X_poly, y_train1)
lin_reg = LinearRegression()
lin_reg.fit(X_poly, y_train1)
y_pred1 = lin_reg.predict(X_poly)
y_pred2 = lin_reg.predict(X_poly2)


# Evaluating the model
from sklearn.metrics import r2_score
R_sq1 = r2_score(y_train1, y_pred1)
R_sq2 = r2_score(y_train2, y_pred2)

# Visualising the Linear Regression results
plt.scatter(X, y, color = 'red')
plt.plot(X, lin_reg.predict(X), color = 'blue')
plt.title('Truth or Bluff (Linear Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

# Visualising the Polynomial Regression results
plt.scatter(X, y, color = 'red')
plt.plot(X, lin_reg_2.predict(poly_reg.fit_transform(X)), color = 'blue')
plt.title('Truth or Bluff (Polynomial Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

# Visualising the Polynomial Regression results (for higher resolution and smoother curve)
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, lin_reg_2.predict(poly_reg.fit_transform(X_grid)), color = 'blue')
plt.title('Truth or Bluff (Polynomial Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()