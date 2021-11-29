
# import dependencies
import warnings
import numpy as np
import pandas as pd
import seaborn as sns
from datetime import datetime as dt
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import norm, skew
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error as mse
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
import matplotlib as mpl
from __future__ import print_function
warnings.filterwarnings("ignore")
import psycopg2
from sqlalchemy import create_engine
from sklearn.metrics import accuracy_score, confusion_matrix
import warnings

# import datat set
df = pd.read_csv('Melbourne_housing_FULL.csv')
df.head()

# Learn more about our data
df.info()

# Drop unnecessary columns
df.drop(["SellerG","Method","BuildingArea","YearBuilt","Lattitude","Longtitude","Bedroom2","Address","CouncilArea","Suburb"], axis=1, inplace=True)
df.shape

# Rename columns
melbourne_df = df.rename(columns={"Landsize": "Land Size","Regionname": "Region","Propertycount": "Property Count"})

# Identify missing data
melbourne_df.isna().sum()

# drop missing value
melbourne_df.dropna(inplace = True)

# Identify missing data
melbourne_df.isna().sum()


# check duplicate value
melbourne_df.drop_duplicates(inplace = True)


# change format
melbourne_df['Date'] = pd.to_datetime(melbourne_df['Date'])
melbourne_df['year'] = melbourne_df['Date'].dt.year
melbourne_df.drop(['Date'], axis = 1, inplace = True)


# heatmap
plt.figure(figsize=(15,8))
sns.heatmap(melbourne_df.corr(), annot=True, cmap='coolwarm')
plt.savefig('heatmap.png')


# Base on the heatmap the dropping of some columns are needed
melbourne_df.drop(['Postcode', 'year', 'Land Size', 'Property Count'], axis=1, inplace=True)
melbourne_df

# Describe data
melbourne_df.describe()

# Check the histogram and probability plot to see whether the target feature is normally distributed
sns.distplot(melbourne_df["Price"], fit=norm)
fig = plt.figure()
prob = stats.probplot(melbourne_df["Price"], plot=plt)


# Since the probability plot looks like log distribution, we can transform it with np.log()
melbourne_df["LogPrice"] = np.log(melbourne_df["Price"])
dist_price = sns.distplot(melbourne_df["LogPrice"], fit=norm)
fig = plt.figure()
prob_log = stats.probplot(melbourne_df["LogPrice"], plot=plt)
plt.show()


#### outliers

# Value < Q1 - 1,5*IQR OR Value > Q3 + 1,5 * IQR
def finding_outliers(data, variable_name) :
    iqr = data[variable_name].quantile(0.75) - data[variable_name].quantile(0.25)
    lower =  data[variable_name].quantile(0.25) -1.5*iqr
    upper =  data[variable_name].quantile(0.75) + 1.5*iqr
    return data [(data[variable_name] < lower) | (data[variable_name] > upper)]


# Price boxplot
plt.figure(figsize=(8,8))
sns.boxplot(y="Price", data=melbourne_df)

## # Price outliers
finding_outliers(melbourne_df, "Price").sort_values("Price")

# For price
iqr_price = melbourne_df["Price"].quantile(0.75) - melbourne_df["Price"].quantile(0.25)
melbourne_df["Price"].quantile(0.75) + 1.5 * iqr_price
melbourne_df.loc[(finding_outliers(melbourne_df, "Price").index, "Price")] = melbourne_df["Price"].quantile(0.75) + 1.5 * iqr_price

# Price boxplot
plt.figure(figsize=(8,8))
sns.boxplot(y="Price", data=melbourne_df)


# room boxplot
plt.figure(figsize=(8,8))
sns.boxplot(y="Rooms", data=melbourne_df)

# room outliers
finding_outliers(melbourne_df, "Rooms").sort_values("Rooms")

# For Rooms
iqr_price = melbourne_df["Rooms"].quantile(0.75) - melbourne_df["Rooms"].quantile(0.25)
melbourne_df["Rooms"].quantile(0.75) + 1.5 * iqr_price
melbourne_df.loc[(finding_outliers(melbourne_df, "Rooms").index, "Rooms")] = melbourne_df["Rooms"].quantile(0.75) + 1.5 * iqr_price

# room boxplot
plt.figure(figsize=(8,8))
sns.boxplot(y="Rooms", data=melbourne_df)


# Bathroom boxplot
plt.figure(figsize=(8,8))
sns.boxplot(y="Bathroom", data=melbourne_df)


# Bathroom outliers
finding_outliers(melbourne_df, "Bathroom").sort_values("Bathroom")

# bathroom
iqr_price = melbourne_df["Bathroom"].quantile(0.75) - melbourne_df["Bathroom"].quantile(0.25)
melbourne_df["Bathroom"].quantile(0.75) + 1.5 * iqr_price
melbourne_df.loc[(finding_outliers(melbourne_df, "Bathroom").index, "Bathroom")] = melbourne_df["Bathroom"].quantile(0.75) + 1.5 * iqr_price


# Bathroom boxplot
plt.figure(figsize=(8,8))
sns.boxplot(y="Bathroom", data=melbourne_df)

## Plot Bathroom vs Price
plt.figure(figsize=(15,8))
sns.boxplot(x="Bathroom", y="Price", data=melbourne_df)

# Plot rooms vs Price
plt.figure(figsize=(15,8))
sns.boxplot(x="Rooms", y="Price", data=melbourne_df)

# Plot Price, rooms and Bathroom
plt.figure(figsize=(15,8))
sns.boxplot(x="Rooms", y="Price", hue="Bathroom", data=melbourne_df)

# Plot each numerical attribute
melbourne_df.hist(figsize=(15, 10))
plt.show()


### countplot

# Plot Bathroom
plt.figure(figsize=(15,8))
sns.countplot(x="Bathroom", data=melbourne_df)

# Plot rooms
plt.figure(figsize = (15,8))
sns.countplot(x="Rooms", data=melbourne_df)

# Plot Type
plt.figure(figsize = (15,8))
sns.countplot(x="Type", data=melbourne_df)

# Plot Car 
plt.figure(figsize = (15,8))
sns.countplot(x="Car", data=melbourne_df)

# Plot Region 
plt.figure(figsize = (15,8))
ax = sns.countplot(x="Region", data=melbourne_df)
ax.set_xticklabels(ax.get_xticklabels(),rotation = 35)

# Plot Region vs Price
plt.figure(figsize=(15,8))
sns.barplot(x="Region", y="Price", data=melbourne_df)
#ax.set_xticklabels(ax.get_xticklabels(),rotation = 35)
plt.xticks(rotation=45)

melbourne_df.groupby('Region')['Price'].mean()

###################### Data preparation

#save as csv after cleaning
melbourne_df.to_csv('melbourne.csv', index = False)

melbourne_df.columns = [c.lower() for c in melbourne_df.columns]


# connect to postgres and Create an engine instance
engine = create_engine(f'postgresql://postgres:Blome00228@localhost:5433/Housing')
conn = engine.connect()

# load or import the table into sql
melbourne_df.to_sql("melbourne", conn,  if_exists='replace', index = False)

housing_df = pd.read_sql("select * from \"melbourne\"", conn)

conn.close()

housing_df



### encoding
encode = LabelEncoder().fit(housing_df['type'])
carpet = {x: i for i, x in enumerate(encode.classes_)}
carpet

encoder = LabelEncoder().fit(housing_df['region'])
carp = {x: i for i, x in enumerate(encoder.classes_)}
carp

# convert to numerical variable 
housing_df['type'] = LabelEncoder().fit_transform(housing_df['type'])
housing_df['type']

housing_df['region'] = LabelEncoder().fit_transform(housing_df['region'])
housing_df['region']



## Convert categorical data to numeric and separate target feature for training data
X = housing_df.drop(["logprice", 'price'],  axis = 1)

y = housing_df['price']

X


# split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

#Scala data
scaler = StandardScaler().fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

## Fit the Linear Regression model with data
modelR = LinearRegression().fit(X_train_scaled, y_train)

# Calculate training and testing score 
training_score = modelR.score(X_train_scaled, y_train)
testing_score = modelR.score(X_test_scaled, y_test)

# Print the training and testing score
print(f"Training Score: {training_score}")
print(f"Testing Score: {testing_score}")


# Fit the Random Forest model with data
model_rf = RandomForestRegressor(n_estimators = 100, criterion = 'mse',random_state = 42, max_depth = 2).fit(X_train, y_train)

## Calculate the training and testing score
training_score = model_rf.score(X_train, y_train)
testing_score = model_rf.score(X_test_scaled, y_test)

# Calculate the training and testing score
training_score = model_rf.score(X_train, y_train)
testing_score = model_rf.score(X_test_scaled, y_test)

# Print the training and testing score
print(f"Training Score: {training_score}")
print(f"Testing Score: {testing_score}")


# Fit the Decision Tree model with data
model_tree = DecisionTreeRegressor(criterion='squared_error', \
                                splitter='best', max_depth=None, 
                                  min_samples_split=2,min_samples_leaf=1, 
                                  min_weight_fraction_leaf=0.0,max_features=None, 
                                  random_state= 42, max_leaf_nodes=None, 
                                  min_impurity_decrease=0.0, ccp_alpha=0.0,).fit(X_train, y_train)


# Calculate training and testing score 
training_score = model_tree.score(X_train, y_train)
testing_score = model_tree.score(X_test, y_test)


# Print the training testing score
print(f"Training Score: {training_score}")
print(f"Testing Score: {testing_score}")


## Fit the Randomized Search model with data
param_dists = {'criterion' : ['mean_squared_error', 'friedman_mse',], 
                       'max_depth': [3,4,7, None],
                        'min_samples_split':np.arange(0.1, 1.1, 0.1),
                        'min_samples_leaf' : list(range(1, 21)), 
                        'max_features' : ['auto', 'sqrt', 'log2', None]}

model_cv = RandomizedSearchCV(estimator = RandomForestRegressor(random_state= 42), 
                              param_distributions = param_dists,  n_iter=200, 
                              scoring= 'neg_mean_squared_error',
                              cv=5, random_state= 42).fit(X_train_scaled, y_train)

# Calculate training and testing score 
training_score = model_cv.score(X_train_scaled, y_train)
testing_score = model_cv.score(X_test_scaled, y_test)

# Print the training testing score
print(f"Training Score: {training_score}")
print(f"Testing Score: {testing_score}")

# fit SVR model
from sklearn.svm import SVR
regressor = SVR(kernel = "rbf").fit(X_train_scaled, y_train)

# Calculate training and testing score 
training_score = regressor.score(X_train_scaled, y_train)
testing_score = regressor.score(X_test_scaled, y_test)

# Print the training testing score
print(f"Training Score: {training_score}")
print(f"Testing Score: {testing_score}")

#fit the lasso model with data
model_lasso = Lasso(alpha =1.0 , max_iter = 1000).fit(X_train_scaled, y_train)

# Calculate training and testing score 
training_score = model_lasso.score(X_train_scaled, y_train)
testing_score = model_lasso.score(X_test_scaled, y_test)

# Print the training testing score
print(f"Training Score: {training_score}")
print(f"Testing Score: {testing_score}")


# Fit the Ridge model with data
model_Ridge = Ridge(alpha = 100).fit(X_train, y_train)


# Calculate training and testing score 
training_score = model_Ridge.score(X_train_scaled, y_train)
testing_score = model_Ridge.score(X_test_scaled, y_test)

# Print the training testing score
print(f"Training Score: {training_score}")
print(f"Testing Score: {testing_score}")


# Predict the price with linear regression
y_pred = modelR.predict(X_test)
pd.DataFrame({"Prediction": y_pred, "Actual": y_test})


# Predict the price wLasso model 
y_pred = model_lasso.predict(X_test)
pd.DataFrame({"Prediction": y_pred, "Actual": y_test})


# Predict the price 
y_pred = model_tree.predict(X_test)
pd.DataFrame({"Prediction": y_pred, "Actual": y_test})

import pickle
# Saving model
pickle.dump(model_tree, open('model.pkl','wb'))











