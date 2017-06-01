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

continuous_features = ['LotFrontage', 'LotArea', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', 
                       'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 
                       'LowQualFinSF', 'GrLivArea', 'GarageArea', 'WoodDeckSF', 
                       'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 
                       'ScreenPorch', 'PoolArea', 'MiscVal']

# Filling null values
X[continuous_features] = X[continuous_features].fillna(X[continuous_features].mean())

X.loc[:, 'Alley'] = X.loc[:, 'Alley'].fillna(X.loc[:, 'Alley'].value_counts().idxmax()) # Lots of NaN's, consider removing
X.loc[:, 'Electrical'] = X.loc[:, 'Electrical'].fillna(X.loc[:, 'Electrical'].value_counts().idxmax())
# All other NaN's are actual values, so they need to be converted
X = X.fillna('i')
X.loc[:, 'GarageYrBlt'] = X.loc[:, 'GarageYrBlt'].replace('i', 0)
X.loc[:, 'MasVnrArea'] = X.loc[:, 'MasVnrArea'].replace('i', 0)

X = pd.get_dummies(X, drop_first=True,
                           columns=['MSSubClass', 'MSZoning', 'Street', 'Alley',
                                   'LotShape', 'LandContour', 'Utilities', 
                                   'LotConfig', 'LandSlope', 'Neighborhood', 
                                   'Condition1', 'Condition2', 'BldgType', 
                                   'HouseStyle', 'OverallQual', 'OverallCond', 
                                   'YearBuilt', 'YearRemodAdd', 'RoofStyle', 
                                   'RoofMatl', 'Exterior1st', 'Exterior2nd', 
                                   'MasVnrType', 'ExterQual', 'ExterCond', 
                                   'Foundation', 'BsmtQual', 'BsmtCond', 
                                   'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 
                                   'Heating', 'HeatingQC', 'CentralAir', 
                                   'Electrical', 'BsmtFullBath', 'BsmtHalfBath', 
                                   'FullBath', 'HalfBath', 'BedroomAbvGr', 
                                   'KitchenAbvGr', 'KitchenQual', 'TotRmsAbvGrd', 
                                   'Functional', 'Fireplaces', 'FireplaceQu', 
                                   'GarageType', 'GarageYrBlt', 'GarageFinish', 
                                   'GarageCars', 'GarageQual', 'GarageCond', 
                                   'PavedDrive', 'PoolQC', 'Fence', 'MiscFeature', 
                                   'MoSold', 'YrSold', 'SaleType', 'SaleCondition'])

# Feature Scaling
from sklearn.preprocessing import StandardScaler
X[continuous_features] = StandardScaler().fit_transform(X[continuous_features])

# Splitting into training examples and test examples
X_train = X.iloc[:1460, :]
X_test = X.iloc[1460:, :]

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train1, X_train2, y_train1, y_train2 = train_test_split(X_train, y, test_size = 0.2, random_state = 0)
X_train1 = X_train1.values
X_train2 = X_train2.values

# Fitting SVR
from sklearn.svm import SVR
import datetime
print(datetime.datetime.now())
svr_poly = SVR(kernel='poly', degree=3)
svr_rbf = SVR(kernel='rbf')
print(datetime.datetime.now())
y_poly1 = svr_poly.fit(X_train1, y_train1).predict(X_train1)
y_poly2 = svr_poly.fit(X_train1, y_train1).predict(X_train2)
y_rbf1 = svr_rbf.fit(X_train1.loc[:, 'LotArea'].to_frame(), y_train1).predict(X_train1.loc[:, 'LotArea'].to_frame())
y_rbf2 = svr_rbf.fit(X_train1, y_train1).predict(X_train2)

# Fitting Linear Regression to the dataset
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X_train1, y_train1)
y_pred1 = lin_reg.predict(X_train1)
y_pred2 = lin_reg.predict(X_train2)

# Fitting Polynomial Regression to the dataset
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 2) # Muy mal comportamiento
X_poly = poly_reg.fit_transform(X_train1)
X_poly2 = poly_reg.fit_transform(X_train2)
poly_reg.fit(X_poly, y_train1)
lin_reg = LinearRegression()
lin_reg.fit(X_poly, y_train1)
y_pred1 = lin_reg.predict(X_poly)
y_pred2 = lin_reg.predict(X_poly2)


# Evaluating the model
from sklearn.metrics import r2_score
R_sq1 = r2_score(y_train1, y_rbf1)
R_sq2 = r2_score(y_train2, y_rbf2)
R_sq1 = r2_score(y_train1, y_pred1)
R_sq2 = r2_score(y_train2, y_pred2)

# Visualising the Linear Regression results
plt.scatter(X, y, color = 'red')
plt.plot(X, lin_reg.predict(X), color = 'blue')
plt.title('Truth or Bluff (Linear Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

""" Diario de resultados:
    La regresión lineal es el algoritmo que mejor funciona: R^2 para test de 0.3865
    Ni SVR con kernel rbf ni regresiones polinómicas se ajustan siquiera a los ejemplos de entrenamiento
    ¿Qué va mal?
"""

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