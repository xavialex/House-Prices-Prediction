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

continuous_features = ['LotFrontage', 'LotArea', 'MasVnrArea', 'BsmtFinSF1', 
                       'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF', 
                       '2ndFlrSF', 'LowQualFinSF', 'GrLivArea', 'GarageArea', 
                       'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', 
                       '3SsnPorch', 'ScreenPorch', 'PoolArea', 'MiscVal']

categorical_features = ['MSSubClass', 'MSZoning', 'Street', 'Alley', 'LotShape', 
                        'LandContour', 'Utilities', 'LotConfig', 'LandSlope', 
                        'Neighborhood', 'Condition1', 'Condition2', 'BldgType', 
                        'HouseStyle', 'OverallQual', 'OverallCond', 'YearBuilt', 
                        'YearRemodAdd', 'RoofStyle', 'RoofMatl', 'Exterior1st', 
                        'Exterior2nd', 'MasVnrType', 'ExterQual', 'ExterCond', 
                        'Foundation', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 
                        'BsmtFinType1', 'BsmtFinType2', 'Heating', 'HeatingQC', 
                        'CentralAir', 'Electrical', 'BsmtFullBath', 'BsmtHalfBath', 
                        'FullBath', 'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 
                        'KitchenQual', 'TotRmsAbvGrd', 'Functional', 'Fireplaces', 
                        'FireplaceQu', 'GarageType', 'GarageYrBlt', 'GarageFinish', 
                        'GarageCars', 'GarageQual', 'GarageCond', 'PavedDrive', 
                        'PoolQC', 'Fence', 'MiscFeature', 'MoSold', 'YrSold', 
                        'SaleType', 'SaleCondition']

# Filling null values
X[continuous_features] = X[continuous_features].fillna(X[continuous_features].mean())

X.loc[:, 'Alley'] = X.loc[:, 'Alley'].fillna(X.loc[:, 'Alley'].value_counts().idxmax()) # Lots of NaN's, consider removing
X.loc[:, 'Electrical'] = X.loc[:, 'Electrical'].fillna(X.loc[:, 'Electrical'].value_counts().idxmax())
# All other NaN's are actual values, so they need to be converted
X = X.fillna('i')
X.loc[:, 'GarageYrBlt'] = X.loc[:, 'GarageYrBlt'].replace('i', 0)
X.loc[:, 'MasVnrArea'] = X.loc[:, 'MasVnrArea'].replace('i', 0)

X = pd.get_dummies(X, drop_first=True, columns=categorical_features)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
X[continuous_features] = StandardScaler().fit_transform(X[continuous_features])

# Splitting into training examples and test examples
X_train = X.iloc[:1460, :]
X_test = X.iloc[1460:, :]

# Removing Id column
X_train.update(X.drop('Id', axis=1, inplace=True))

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

y_rbf2 = svr_rbf.fit(X_train1[:, 5].reshape(-1, 1), y_train1).predict(X_train2[:, 2].reshape(-1, 1))

y_rbf1 = svr_rbf.fit(X_train1.loc[:, 'GarageArea'].to_frame(), y_train1).predict(X_train1.loc[:, 'GarageArea'].to_frame())
y_rbf1 = svr_rbf.fit(X_train1.loc[:, 'LotFrontage'].to_frame(), y_train1).predict(X_train1.loc[:, 'LotFrontage'].to_frame())
y_rbf2 = svr_rbf.fit(X_train1, y_train1).predict(X_train2)

""" 
SVR siempre devuelve 163000, ¿qué pasa?
"""

# Fitting Decision Tree Regressor
from sklearn import tree
clf = tree.DecisionTreeRegressor()
clf = clf.fit(X_train1, y_train1)
a = clf.predict(X_train1)
b = clf.predict(X_train2)

# Fitting Random Forest Regressor
from sklearn.ensemble import RandomForestRegressor
regr_rf = RandomForestRegressor().fit(X_train1, y_train1)
y_pred1 = regr_rf.predict(X_train1)
y_pred2 = regr_rf.predict(X_train2)
# R^2 en test de 0.86

# Fitting Linear Regression to the dataset
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X_train1, y_train1)
y_pred1 = lin_reg.predict(X_train1)
y_pred2 = lin_reg.predict(X_train2)

# Fitting Polynomial Regression to the dataset
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 2) # Muy mal comportamiento
# Idea, expandir polinómicamente sólo las vbles continuas, en proceso, ¿cómo le agrego luego las discretas?
X_poly = poly_reg.fit_transform(X.loc[:, continuous_features])
X_poly = X.loc[:, categorical_features].join(pd.DataFrame(X_poly))
X_poly = pd.get_dummies(X_poly, drop_first=True, columns=categorical_features)
X_poly_train = X_poly[:1460]
X_poly_test = X_poly[1460:]
X_poly1, X_poly2, y_train1, y_train2 = train_test_split(X_poly_train, y, test_size = 0.2, random_state = 0)
poly_reg.fit(X_poly1, y_train1)
lin_reg = LinearRegression()
lin_reg.fit(X_poly1, y_train1)
y_pred1 = lin_reg.predict(X_poly1)
y_pred2 = lin_reg.predict(X_poly2)

plt.scatter(np.arange(0, 2919), y_pred1)
plt.scatter(np.arange(0, 1168), y_train1, edgecolors='red')
plt.ylim(0, 500000)
plt.show()

# Fitting XGBoost to the Training set
import os

mingw_path = 'C:\\Program Files\\mingw-w64\\x86_64-5.3.0-posix-seh-rt_v4-rev0\\mingw64\\bin'

os.environ['PATH'] = mingw_path + ';' + os.environ['PATH']

from xgboost import XGBClassifier
classifier = XGBClassifier()
classifier.fit(X_train1, y_train1) # Nunca acaba de entrenarse

# Predicting the Test set results
y_pred = classifier.predict(X_train1)
y_pred2 = classifier.predict(X_train2)


# Evaluating the model
from sklearn.metrics import r2_score
R_sq1 = r2_score(y_train1, y_rbf1)
R_sq2 = r2_score(y_train2, y_rbf2)
R_sq1 = r2_score(y_train1, y_pred1)
R_sq2 = r2_score(y_train2, y_pred2)

""" Diario de resultados:
    La regresión lineal es el algoritmo que mejor funciona: R^2 para test de 0.3865
    Ni SVR con kernel rbf ni regresiones polinómicas se ajustan siquiera a los ejemplos de entrenamiento
    ¿Qué va mal?
    ---
    Decision Tree Regressor: R^2 de 0.719. No está mal, pero mucho margen de mejora
"""

# Visualising the Polynomial Regression results
plt.scatter(training_set.loc[:, 'LotFrontage'], y)
plt.scatter(training_set.loc[:, 'LotArea'], y)

# Writting the predictions
submission = pd.DataFrame(columns=['Id', 'SalePrice'])
submission.iloc[:, 0] = X_test.loc[:, 'Id']
regr_rf = RandomForestRegressor().fit(X_train, y) # Entrenar con todos los ejemplos da peores resultados!!!
submission.iloc[:, 1] = regr_rf.predict(X_test)
df_csv = submission.to_csv('Random Forest Regressor Submission.csv', index=False)
