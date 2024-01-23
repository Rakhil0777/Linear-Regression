#!/usr/bin/env python
# coding: utf-8

# **Supervised Learning Project- Recell**

# In[60]:


#Import all libraries
#We are importing first libraries for visualisation 
import matplotlib.pyplot as plt
import seaborn as sns


# In[61]:


#Importing libraries for reading and manipulating the data
import pandas as pd
import numpy as np


# In[62]:


#Importing libraries for stats 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor


# **Problem statement- The rising potential of this comparatively under-the-radar market fuels the need for an ML-based solution to develop a dynamic pricing strategy for used and refurbished devices. ReCell, a startup aiming to tap the potential in this market, has hired you as a data scientist. They want you to analyze the data provided and build a linear regression model to predict the price of a used phone/tablet and identify factors that significantly influence it**

# In[63]:


#load dataset
db=pd.read_csv('used_device_data.csv')


# *The data contains the different attributes of used/refurbished phones and tablets. The data was collected in the year 2021. The detailed data dictionary is given below*

# In[64]:


db


# In[65]:


db.head()
#Displaying the first few rows of the data set


# In[66]:


db.shape
#Checking the shape of the dataset


# In[67]:


db.info()
#Lets check the data type in the dataset


# In[68]:


db.describe().T
#lets see the statistical summary of the dataset now


# In[69]:


db.duplicated().count
#Looking for duplicates in the data
#As per the results given, data does not have any duplicates 


# In[70]:


db.isnull().sum()
#As per the results, we have some missing values in the column main_camera_mp, selfie_camera_mp,int_memory,ram,battery,weight


# In[71]:


db.isnull().sum().sum()
#We can see total of 202 values are missing in the dataset


# In[72]:


df=db.copy()
#creating a copy of the original data so as to keep the original one unchanged


# In[73]:


df


# In[74]:


# Converting object data types to category
category_col = df.select_dtypes(exclude=np.number).columns.tolist()
df[category_col] = df[category_col].astype("category")


# In[75]:


df.dtypes


# **Lets begin by Univariate analysis of the dataset**

# In[76]:


df.groupby('brand_name').size().plot(kind='pie', subplots=True ,figsize=(10,10))
plt.title('Various Phone brands')


# **Most number of devices fall in category others, followed by Samsung**

# In[77]:


df.groupby('os').size().plot(kind='pie', subplots=True ,figsize=(5,5))
plt.title('os')


# In[78]:


df.groupby('release_year').size().plot(kind='pie',subplots=True ,figsize=(10,10));
#This pie chart shows us the number of releases in each year with 2013-2014-2015 with approx. same number of releases. 


# In[79]:


sns.violinplot(df,x="days_used")
#Median age of the devices is around 700 days.


# **Most devices use Android OS**

# In[80]:


sns.displot(df['screen_size'], kde=False, color='Black',bins=10);


# **Most common screensize is between 10-12**

# In[83]:


df.boxplot(figsize=(20,10))


# **Conclusion- The following columns have outliers**
# 1. main_camera_mp
# 2.selfie_camera_mp
# 3.int_memory
# 4.ram
# 5.battery
# 6.weight.
# **It is important to note here that the battery has considerable numbers of outliers in the data**
# 

# **Now lets start with bivariate analysis of the data**

# In[84]:


plt.figure(figsize=(15,5))
plt.title('brandname vs weight')
sns.boxplot(df,x='brand_name',y='weight')
plt.xticks(rotation=90)
plt.show()
#Box plot depicts various brands and their weights with median weight here


# In[85]:


plt.figure(figsize=(12, 5))
sns.lineplot(df,x='release_year',y='normalized_used_price') 
plt.show()
#Normalized used price increased with the release year, hence we can deduce there is a positive relationship between both


# In[86]:


plt.figure(figsize=(10, 4)),
plt.subplot(121),
plt.title('4g phone vs normalized phone price')
sns.boxplot(data=df, x="4g", y="normalized_used_price"),
plt.subplot(122)
plt.title('5g phone vs normalized used price')
sns.boxplot(data=df, x="5g", y="normalized_used_price"),
plt.show()
#Boxplot depicting used prices against both 4g and non 4g, 5g and non 5g phone
#Used price is the high for both 4g and 5g phone with 5g phone having high median used price


# In[87]:


plt.figure(figsize=(12,12))
sns.pairplot(df,kind='scatter',diag_kind='kde')
#view the results by zooming in


# **There seems to be a positive relationship between normalized_used_price with screensize, selfie camera mp,ram,weight, battery. There is a negative correlation with days used and used price**

# **Positive correlation. As the screensize increases, the normalized used price of the phone is going up**

# **There is a negative correlation. The phones used for more days will have a relatively lesser price**

# In[88]:


corr=df.corr()
corr


# In[89]:


plt.figure(figsize=(10,10))
sns.heatmap(corr,annot=True)


# **There is a high positive correlation between used price and ram, main camera mp, screen size and battery. While the only negative correlation for used price lies with the days used**

# **Lets now have a look at the preprocessing of the data. We need to make sure that there are no duplicate values**

# In[90]:


df.duplicated().sum()


# **No duplicated found in the data**

# **Missing value check and treatment**

# In[91]:


df.isnull().sum()


# **Impute the missing values using groupby and the median**

# In[92]:


cols_impute = ["main_camera_mp","selfie_camera_mp","int_memory","ram","battery","weight"]

for col in cols_impute:
    df[col] = df[col].fillna(value=df.groupby(['brand_name','release_year'])[col].transform("median"))


# In[93]:


df.isnull().sum()


# **As shown, two columns dont have any missing values now, but we still have the four columns with the missing values**

# In[94]:


cols_impute = ["main_camera_mp","selfie_camera_mp","battery","weight"]

for col in cols_impute:
    df[col] = df[col].fillna(value=df.groupby(['brand_name'])[col].transform("median"))


# **Using groupby with only brand name column this time to impute the rest missing values**

# In[95]:


df.isnull().sum()


# In[96]:


df["main_camera_mp"] = df["main_camera_mp"].fillna(df["main_camera_mp"].median())


# **Finally we use the median function to fill the missing values in the main_camera_mp column**

# In[97]:


df.isnull().sum()


# **There are no missing values in the dataset now**

# ### Feature Engineering

# **Let's create a new column `years_since_release` from the `release_year` column,
#     "- We will consider the year of data collection, 2021, as the baseline
#     "- We will drop the `release_year` column**

# In[98]:


df['years_since_release']=2021-df['release_year']


# In[99]:


df.drop('release_year',axis=1,inplace=True)


# In[100]:


df['years_since_release'].describe


# **We have now replaced the column year_released with years_since_release in the dataset**

# ## Data preparation for modelling

# We want to predict the normalized price of used devices
# Before we proceed to build a model, we'll have to encode categorical features
# We'll split the data into train and test to be able to evaluate the model that we build on the train data
# We will build a Linear Regression model using the train data and then check it's performance

# In[101]:


# Create dummy variables for the categorical variables
df = pd.get_dummies(df, columns=["brand_name", "os", "4g", "5g"], drop_first=True)


# In[102]:


df.head()


# In[118]:


# defining X and y variables
X = df.drop(["normalized_used_price"], axis=1)
Y = df["normalized_used_price"]


# In[119]:


X


# In[120]:


Y


#  **Splitting the data in 70:30 ratio for train to test data**
# 

# In[124]:


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.30 , random_state=1)


# In[125]:


X_train, X_test, Y_train, Y_test


# In[126]:


# check shape of the train and test data
print("Number of rows in train data =", X_train.shape[0])
print("Number of rows in test data =", X_test.shape[0])


# ## Model Building - Linear Regression

# In[127]:


regression_model = LinearRegression()
regression_model.fit(X_train, Y_train)


# In[128]:


regression_model.coef_


# In[129]:


regression_model.intercept_


# In[130]:


# dataframe to show the model coefficients and intercept
coef_df = pd.DataFrame(
    np.append(regression_model.coef_, regression_model.intercept_),
    index=X_train.columns.tolist() + ["Intercept"],
    columns=["Coefficients"],
)

coef_df


# **We have calculated the coefficients for the equation(Y=A+B1X1+B2X2+...........B52X52) Where intercept A is equal to 1.288484 and coefficients B1=0.00244,B2=0.00208.......**

# In[149]:


# function to compute adjusted R-squared

def adj_r2_score(predictors, targets, predictions):
    r2 = r2_score(targets, predictions)
    n = predictors.shape[0]
    k = predictors.shape[1]
    return 1 - ((1 - r2) * (n - 1) / (n - k - 1))


# function to compute MAPE
def mape_score(targets, predictions):
    return np.mean(np.abs(targets - predictions) / targets) * 100


# function to compute different metrics to check performance of a regression model
def model_performance_regression(model, predictors, target):
    """
    Function to compute different metrics to check regression model performance

    model: regressor
    predictors: independent variables
    target: dependent variable
    """

    # predicting using the independent variables
    pred = model.predict(predictors)

    r2 = r2_score(target, pred)  # to compute R-squared
    adjr2 = adj_r2_score(predictors, target, pred)  # to compute adjusted R-squared
    rmse = np.sqrt(mean_squared_error(target, pred))  # to compute RMSE
    mae = mean_absolute_error(target, pred)  # to compute MAE
    mape = mape_score(target, pred)  # to compute MAPE

    # creating a dataframe of metrics
    df_perf = pd.DataFrame(
        {
            "RMSE": rmse,
            "MAE": mae,
            "R-squared": r2,
            "Adj. R-squared": adjr2,
            "MAPE": mape,
        },
        index=[0],
    )

    return df_perf


# In[151]:


print("Training Performance\n")
linearregression_train_perf = model_performance_regression(
    regression_model, X_train, y_train
)
linearregression_train_perf


# In[153]:


regression_model.score(X_train, y_train)


# In[152]:


regression_model.score(X_test, y_test)


# In[154]:


# we have to add the constant manually
X_train1 = sm.add_constant(X_train)
# adding constant to the test data
X_test1 = sm.add_constant(X_test)


# In[156]:


olsmodel = sm.OLS(y_train,X_train1) ## Complete the code to fit OLS model\n",
olsres=olsmodel.fit()
print(olsres.summary())


# Checking Linear Regression Assumptions
# We will be checking the following Linear Regression assumptions:
# 
# No Multicollinearity
# 
# Linearity of variables
# 
# Independence of error terms
# 
# Normality of error terms
# 
# No Heteroscedasticity

# ## TEST FOR MULTICOLLINEARITY
# 

# Variance Inflation Factor (VIF):
# 
# General Rule of thumb:
# 
# If VIF is between 1 and 5, then there is low multicollinearity. If VIF is between 5 and 10, we say there is moderate multicollinearity. If VIF is exceeding 10, it shows signs of high multicollinearity.

# In[145]:


# we will define a function to check VIF
def checking_vif(predictors):
    vif = pd.DataFrame()
    vif["feature"] = predictors.columns

    # calculating VIF for each feature
    vif["VIF"] = [
        round(variance_inflation_factor(predictors.values, i), 2)
        for i in range(len(predictors.columns))
    ]
    return vif


# In[146]:


checking_vif(X_train1)


# **REMOVING MULTICOLLINEARITY**

# Let's define a function that will help us do this

# In[147]:


# Let's define a function that will help us do this.

def treating_multicollinearity(predictors, target, high_vif_columns):
    """
    Checking the effect of dropping the columns showing high multicollinearity
    on model performance (adj. R-squared and RMSE)

    predictors: independent variables
    target: dependent variable
    high_vif_columns: columns having high VIF
    """
    # empty lists to store adj. R-squared and RMSE values
    adj_r2 = []
    rmse = []

    # build ols models by dropping one of the high VIF columns at a time
    # store the adjusted R-squared and RMSE in the lists defined previously
    for cols in high_vif_columns:
        # defining the new train set
        train = predictors.loc[:, ~predictors.columns.str.startswith(cols)]

        # create the model
        olsmodel = sm.OLS(target, train).fit()

        # adding adj. R-squared and RMSE to the lists
        adj_r2.append(olsmodel.rsquared_adj)
        rmse.append(np.sqrt(olsmodel.mse_resid))

    # creating a dataframe for the results
    temp = pd.DataFrame(
        {
            "col": high_vif_columns,
            "Adj. R-squared after_dropping col": adj_r2,
            "RMSE after dropping col": rmse,
            }
    ).sort_values(by="Adj. R-squared after_dropping col", ascending=False)
    temp.reset_index(drop=True, inplace=True)

    return temp


# In[157]:


col_list = [
    "screen_size",
    "brand_name_Apple",
    "os_iOS",
    "brand_name_Others",
    "brand_name_Samsung",
    "brand_name_Huawei",
    "brand_name_LG",
]

res = treating_multicollinearity(X_train1, y_train, col_list)
res


# **As per the summary, our Rsquared value is 0.845 whereas adjusted Raquared value is 0.842. We can drop all the above mentioned columns except the screen_size**

# In[158]:


# Drop brand_name_Huawei

col_to_drop = "brand_name_Huawei"
X_train2 = X_train1.loc[:, ~X_train1.columns.str.startswith(col_to_drop)]
X_test2 = X_test1.loc[:, ~X_test1.columns.str.startswith(col_to_drop)]

# Check VIF now
vif = checking_vif(X_train2)
print("VIF after dropping ", col_to_drop)
vif


# In[159]:


#Dropping brand name Apple

col_to_drop = "brand_name_Apple"
X_train3 = X_train2.loc[:, ~X_train2.columns.str.startswith(col_to_drop)]
X_test3 = X_test2.loc[:, ~X_test2.columns.str.startswith(col_to_drop)]

# Check VIF now
vif = checking_vif(X_train3)
print("VIF after dropping ", col_to_drop)
vif


# **We cannot remove the remaining two variable with high VIF i.e weight and screen_size as they influence the adj_rsquared value of the model.The above predictors have little to no multicollinearity and the assumption is satisfied.**
# 
# 

# In[160]:


olsmod1 = sm.OLS(y_train, X_train3).fit()
print(olsmod1.summary())
#Model 


# In[161]:


# initial list of columns
cols = X_train3.columns.tolist()

# setting an initial max p-value
max_p_value = 1

# Loop to check for p-values of the variables and drop the column with the highest p-value.
while len(cols) > 0:
    # defining the train set
    X_train_aux = X_train3[cols]

    # fitting the model
    model = sm.OLS(y_train, X_train_aux).fit()

    # getting the p-values and the maximum p-value
    p_values = model.pvalues
    max_p_value = max(p_values)

    # name of the variable with maximum p-value
    feature_with_p_max = p_values.idxmax()

    if max_p_value > 0.05:
        cols.remove(feature_with_p_max)
    else:
        break

selected_features = cols  # variables with p-values lesser than 0.05
print(selected_features)


# In[162]:


# Use only the variables with p-values less than 0.05 to train model

X_train4 = X_train3[selected_features]
X_test4 = X_test3[selected_features]


# In[163]:


olsmod2 = sm.OLS(y_train, X_train4).fit()
print(olsmod2.summary())


# In[164]:


# Create a dataframe with actual, fitted and residual values
df_pred = pd.DataFrame()

df_pred["Actual Values"] = y_train  # actual values
df_pred["Fitted Values"] = olsmod2.fittedvalues  # predicted values
df_pred["Residuals"] = olsmod2.resid  # residuals

df_pred.head()


# In[165]:


# let's plot the fitted values vs residuals

sns.residplot(
    data=df_pred, x="Fitted Values", y="Residuals", color="purple", lowess=True
)
plt.xlabel("Fitted Values")
plt.ylabel("Residuals")
plt.title("Fitted vs Residual plot")
plt.show()


# In[166]:


# histogram plot of the residual
sns.histplot(data=df_pred, x="Residuals", kde=True)
plt.title("Normality of residuals")
plt.show()


# In[167]:


import pylab
import scipy.stats as stats

stats.probplot(df_pred["Residuals"], dist="norm", plot=pylab)
plt.show()


# In[168]:


# Shipiro test for normality
stats.shapiro(df_pred["Residuals"])


# Since p-value < 0.05, the residuals are not normal as per the Shapiro-Wilk test. Strictly speaking, the residuals are not normal. However, as an approximation, we can accept this distribution as close to being normal. So, the assumption is satisfied.

# **Lets now check for the multicollinearity using Variation inflation factor** 

# In[169]:


# goldfeldquandt test for homoscedasticity

import statsmodels.stats.api as sms
from statsmodels.compat import lzip

name = ["F statistic", "p-value"]
test = sms.het_goldfeldquandt(df_pred["Residuals"], X_train4)
lzip(name, test)


# Since p-value > 0.05, we can say that the residuals are homoscedastic. So, this assumption is satisfied.

# ## Final Model Summary
# 

# In[171]:


# Let us write the equation of linear regression
Equation = "Used Phone Price ="
print(Equation, end=" ")
for i in range(len(X_train4.columns)):
    if i == 0:
        print(np.round(olsmod2.params[i], 4), "+", end=" ")
    elif i != len(X_train4.columns) - 1:
        print(
            "(",
            np.round(olsmod2.params[i], 4),
            ")*(",
            X_train4.columns[i],
            ")",
            "+",
            end="  ",
        )
    else:
        print("(", np.round(olsmod2.params[i], 4), ")*(", X_train4.columns[i], ")")


# In[172]:


# predictions on the test set
pred = olsmod2.predict(X_test4)

df_pred_test = pd.DataFrame({"Actual": y_test, "Predicted": pred})
df_pred_test.sample(10, random_state=1)


# In[173]:


df2 = df_pred_test.sample(25, random_state=1)
df2.plot(kind="bar", figsize=(15, 7))
plt.show()


# In[174]:


# checking model performance on train set (seen 70% data)
print("Training Performance\n")
olsmod2_train_perf = model_performance_regression(olsmod2, X_train4, y_train)
olsmod2_train_perf


# In[175]:


# checking model performance on test set (seen 30% data)
print("Test Performance\n")
olsmod2_test_perf = model_performance_regression(olsmod2, X_test4, y_test)
olsmod2_test_perf


# In[176]:


olsmodel_final = sm.OLS(y_train, X_train4).fit()
print(olsmodel_final.summary())


# ## INSIGHTS
# 

# Newly released phones have high used price, which makes sense because the newer the phone, the higher the new price hence used price would be affected and the older the phone, the lower the used price. since most customers want phones in demand.
# 
# release_year, days_used, new_price, brand_name_Gionee, whether 4g or 5g seem to be affect the used price. this is understandable the longer the phone is used, we cant determine its originality and its wholeness hence a decrease in used price which negatively impacts the amount it could be sold for used. 5g comes more with new phones, hence it would also shoot up in a high used price since its new.
# 
# Phones with 4g and Gionee brand phones have lower the used price. they seem to not be a demand for customers and should probably be discontinued.

# ## RECOMMENDATIONS

# We can use the model to make predictions of the price of used phone. Newly released phones should also be focused on as they have a high resale price. 5g network enabled phones have high resale price and should be focused on rather than those with less 4g phones an example is Gionee phones. i recommend discontinuation of Gionee phones.
# 
# Future data collections need to be done on the age of customers purchasing products, since age could be a major drive. millenial customers may tend to want a 5g or a newer version.
# 
# Future data collection on income could also be done to know what more high income customer want.

# In[ ]:




