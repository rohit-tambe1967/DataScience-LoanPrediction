# -*- coding: utf-8 -*-
"""
Created on Sun Jan 20 15:38:29 2019

@author: COMPAQ
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats.stats import pearsonr

path = "D:\\DataScience\\LoanPredictionIII\\"
trainFile = path + "train_u6lujuX_CVtuZ9i.csv"
dfTrain = pd.read_csv (trainFile)

testFile = path + "test_Y3wMUE5_7gLdaTN.csv"
dfTest = pd.read_csv (testFile)

#- Explore data for invalid data
dfTrain.head()
dfTrain.info()
dfTrain.describe(include='all').LoanAmount
type(dfTrain)
dfTrain.shape
dfTrain.isnull().LoanAmount
dfTrain.isnull().sum(axis=0)

#- Check distribution of Income for all applicants
dfTrain[:]["ApplicantIncome"].describe(include = 'all')
dfTrain.ApplicantIncome.isnull()

dfTrain[:]["CoapplicantIncome"].describe(include = 'all')
dfTrain[:]["LoanAmount"].describe(include = 'all')
dfTrain.LoanAmount.isnull()


#-- Make a copy of Train dataset to keep a backup
dfTrain2 = dfTrain

#- Check records with Null Loan Amount
dfTrain[np.isnan(dfTrain.LoanAmount)]   # - gives 22 rows
dfTrain2[np.isnan(dfTrain2.LoanAmount)]

#- Filter out the Null value rows. '~' will give opposite result
dfTrain2[~np.isnan(dfTrain2.LoanAmount)]
dfTrain2 = dfTrain2[~np.isnan(dfTrain2.LoanAmount)] 
dfTrain2.LoanAmount.describe()

#- Show records with income above/below 3rd Quantile Q3
dfTrain.ApplicantIncome.describe()['75%']
dfTrain[:][dfTrain.ApplicantIncome > dfTrain.ApplicantIncome.describe()['75%']].ApplicantIncome
dfTrain[:][dfTrain.ApplicantIncome <= dfTrain.ApplicantIncome.describe()['75%']]

#- Extract records upto Q3 
dfTrainQ3 = dfTrain[:][dfTrain.ApplicantIncome <= dfTrain.ApplicantIncome.describe()['75%']]

#- show 99 percentile records using np.percentile()  -- This is said tobe faster
dfTrain2.ApplicantIncome.quantile(0.99)
dfTrain.ApplicantIncome.max()
dfTrain2[:][dfTrain2.ApplicantIncome > np.percentile(dfTrain2.ApplicantIncome , 99)].ApplicantIncome
dfTrain2[:][dfTrain2.ApplicantIncome <= np.percentile(dfTrain2.ApplicantIncome , 99)].describe()

#-- Create a new dataset with 99 percentile of Applicant Income
#dfTrain2 = dfTrain2[:][dfTrain2.ApplicantIncome <= np.percentile(dfTrain2.ApplicantIncome , 99)]
dfTrain2.head()
#- 
#-- Filter ot Applicant income > 20000
dfTrain[:][dfTrain.ApplicantIncome <= 20000].ApplicantIncome
dfTrain2 = dfTrain2[:][dfTrain.ApplicantIncome <= 20000]
dfTrain2 = dfTrain2[:][dfTrain.CoapplicantIncome <= 20000]

#- Get count of nulls in each column
dfTrain2.isnull().sum(axis=0)

#- Check for blank Loan Term
dfTrain2[:]["Loan_Amount_Term"].describe(include = 'all')
dfTrain2.Loan_Amount_Term.isnull()
dfTrain2.Loan_Amount_Term.median()
dfTrain2.Loan_Amount_Term.quantile(0.75)
dfTrain2[:][dfTrain2.Loan_Amount_Term < 343].Loan_Amount_Term.sort_values()

#- Since 75% of the Loan terms is 360, set blank loan term to Median value 
dfTrain2.loc[np.isnan(dfTrain2['Loan_Amount_Term']), ['Loan_Amount_Term']]
dfTrain2.loc[np.isnan(dfTrain2['Loan_Amount_Term']), ['Loan_Amount_Term']] = dfTrain2['Loan_Amount_Term'].median()

# - Check No of dependents
dfTrain2.Dependents.unique()
dfTrain2.Dependents.describe()
dfTrain2[dfTrain2.Dependents.isnull()].Dependents
dfTrain2.Dependents.value_counts()

#- Replace NaN Values of Dep. with 0
dfTrain2.loc[pd.isnull(dfTrain2['Dependents']),['Dependents']] = 0

#- filter out dependents with Null value
#dfTrain2 = dfTrain2[~dfTrain2.Dependents.isnull()]

dfTrain2[dfTrain2.Dependents == '3+']

#-There are some non numeric values '3+'.  Replace '3+' values with 3
dfTrain2['Dependents'].apply(lambda x: 3 if x == '3+' else x)
dfTrain2['Dependents'] = dfTrain2['Dependents'].apply(
        lambda x: 3 if x == '3+' else x)

#- Due to '3+' the data type of Dependents col became char. 
#- Needs to be converted to Numeric
dfTrain2.Dependents.value_counts()
dfTrain2['Dependents'] = dfTrain2['Dependents'].astype(int)

dfTrCoappIncomeN.Dependents.value_counts()

#- get count of unique values in each col
[len(dfTrain2[i].unique()) for i in dfTrain2.columns]

uniqueValDict = {i : len(dfTrain2[i].unique()) 
                    for i in dfTrain2.columns 
                        if len(dfTrain2[i].unique()) < 10}

uniqueValDict

#- Check Null values in Coapplicant Income
np.isnan(dfTrain2.CoapplicantIncome)
np.isnan(dfTrain2.ApplicantIncome)
dfTrain2[np.isnan(dfTrain2.appTotalIncome)]
dfTrain2.appTotalIncome.describe(include='all')
dfTrain2.describe(include='all')
dfTrain2.CoapplicantIncome.describe()

#- Check the Credit History col for nulls and replace with 0
dfTrain2.Credit_History.isnull()
dfTrain2.loc[pd.isnull(dfTrain2['Credit_History']), 'Loan_ID']
dfTrain2.loc[~pd.isnull(dfTrain2['Credit_History']),'Loan_ID']

dfTrain2.loc[pd.isnull(dfTrain2['Credit_History']),'Credit_History'] = 0


#- Mark Coapplicant income as boolean Flag (Y/N)

# dfCoAppIncome0 = dfTrain2

# dfCoAppIncome0['IsCoappIncomeZero'] = dfCoAppIncome0['CoapplicantIncome'].apply(
#        lambda x: 'Y' if x <= 0 else 'N')
# dfCoAppIncome0['IsCoappIncomeZero']

dfTrain2['IsCoappIncomeZero'] = dfTrain2['CoapplicantIncome'].apply(
        lambda x: 'Y' if x <= 0 else 'N')
dfTrain2['IsCoappIncomeZero']

#- Add a new column for applicant total income
dfTrain2.ApplicantIncome + dfTrain2.CoapplicantIncome
dfTrain2['appTotalIncome'] = dfTrain2.ApplicantIncome + dfTrain2.CoapplicantIncome
dfTrain2.head()
dfTrain2.describe(include='all').appTotalIncome

#-- Plot the points on scatter dia to find  outliers
plt.figure(figsize =(16,8))
plt.subplot(2,4,1)
sns.boxplot(dfTrain2.ApplicantIncome)
plt.subplot(2,4,2)
sns.boxplot(dfTrain2.CoapplicantIncome )
plt.subplot(2,4,3)
sns.boxplot(dfTrain2.appTotalIncome)
plt.subplot(2,4,4)
sns.countplot(dfTrain2.Property_Area)
plt.subplot(2,4,5)
sns.countplot(dfTrain2.Self_Employed)
plt.subplot(2,4,6)
sns.countplot(x = dfTrain2.Loan_Status, hue = dfTrain2.Property_Area)
plt.subplot(2,4,7)
sns.countplot(x = dfTrain2.Loan_Status, hue = dfTrain.Self_Employed)
plt.subplot(2,4,8)
sns.countplot(x = (dfTrain2.Loan_Status), hue = dfTrain2.IsCoappIncomeZero)
plt.subplot(2,4,9)
sns.countplot( dfTrain2.Loan_Status, hue = dfTrain2.Dependents)
plt.show()

sns.countplot( dfTrain2.Loan_Status, hue = dfTrain2.Dependents)

plt.figure(figsize =(16,8))
plt.subplot(2,4,1)
sns.boxplot(dfTrain.ApplicantIncome)
plt.subplot(2,4,2)
sns.boxplot(dfTrain.CoapplicantIncome )
plt.subplot(2,4,3)
sns.boxplot(dfTrain2.appTotalIncome)
plt.subplot(2,4,4)
sns.countplot(dfTrain2.Property_Area)
plt.subplot(2,4,5)
sns.countplot(dfTrain2.Loan_Status)
plt.subplot(2,4,6)
sns.countplot(x = dfTrain2.Property_Area, hue = dfTrain2.Loan_Status)
plt.subplot(2,4,7)
sns.countplot(x = dfTrain2.LoanAmount, hue = dfTrain2.Property_Area)
plt.subplot(2,4,8)
sns.countplot(x = (dfTrain2.ApplicantIncome * 1000), hue = dfTrain2.Loan_Status)
plt.subplot(2,4,8)
sns.countplot( y=dfTrain2.IsCoappIncomeZero, hue = dfTrain2.Loan_Status)
plt.show()


sns.countplot( y=dfTrCoappIncomeN.Loan_Status, hue = dfTrCoappIncomeN.Dependents)
sns.countplot( y=dfTrain2.Self_Employed, hue = dfTrain2.Loan_Status)
sns.countplot( y=dfTrain2.Married, hue = dfTrain2.Loan_Status)
sns.countplot( y=dfTrain2.Loan_Amount_Term, hue = dfTrain2.Loan_Status)

sns.countplot(x = dfTrain2.Property_Area, col = dfTrain2.IsCoappIncomeZero, 
              hue = dfTrain2.Loan_Status, kind = 'count')

dfTrain2[:][dfTrain2.IsCoappIncomeZero == 'Y']
dfTrain2.IsCoappIncomeZero == 'Y'  #- Shows True for each record meeing the condition


sns.countplot(x = dfTrain2.appTotalIncome, hue = dfTrain2.Loan_Status,
              data = dfTrain2[:][dfTrain2.IsCoappIncomeZero == 'Y'])

sns.countplot(x = (dfTrain2.appTotalIncome), hue = dfTrain2.Loan_Status,
              data = dfTrain2[:][dfTrain2.IsCoappIncomeZero == 'N'])

#- Segregate CoApplicant income = 0 into new DF
dfTrCoappIncomeN = dfTrain2[:][dfTrain2.IsCoappIncomeZero == 'Y']
dfTrCoappIncomeN.head()

#- Segregate CoApplicant income > 0 into new DF
dfTrCoappIncomeY = dfTrain2[:][dfTrain2.IsCoappIncomeZero == 'N']
dfTrCoappIncomeY.head()

#- Converting Property Area values to numeric values 1 - Urban, 
#- 2 - SemiUrban, 3- Rural
dfTrCoappIncomeY['Property_Area'].apply(
        lambda x: 1 if x== 'Urban' else 2 if x == 'SemiUrban' else 3)


dfTrain2['nLocale'] = dfTrain2['Property_Area'].apply(
        lambda x: 1 if x== 'Urban' else 2 if x == 'SemiUrban' else 3)

dfTrCoappIncomeY['nLocale'] = dfTrCoappIncomeY['Property_Area'].apply(
        lambda x: 1 if x== 'Urban' else 2 if x == 'SemiUrban' else 3)
dfTrCoappIncomeY

dfTrCoappIncomeN['nLocale'] = dfTrCoappIncomeN['Property_Area'].apply(
        lambda x: 1 if x== 'Urban' else 2 if x == 'SemiUrban' else 3)


#- Convert Data type of Dependents, nLocale to int type
dfTrain2['Dependents'] = dfTrain2['Dependents'].astype(int)


dfTrCoappIncomeY['Dependents'] = dfTrCoappIncomeY['Dependents'].astype(int)
dfTrCoappIncomeN['Dependents'] = dfTrCoappIncomeN['Dependents'].astype(int)

dfTrain2['nLocale'] = dfTrain2['nLocale'].astype(int)

dfTrCoappIncomeY['nLocale'] = dfTrCoappIncomeY['nLocale'].astype(int)

dfTrCoappIncomeN['nLocale'] = dfTrCoappIncomeN['nLocale'].astype(int)


#- See if co-Applicant income has any impact on Loan Status

sns.countplot(x = (dfTrCoappIncomeY.Loan_Status), hue = dfTrCoappIncomeY.Property_Area)
sns.countplot(x = (dfTrCoappIncomeN.Loan_Status), hue = dfTrCoappIncomeN.Property_Area)

#- Find correlation between different factors
dfTrain2['nLoanStat'] = dfTrain2['Loan_Status'].apply(
        lambda x: 1 if x== 'Y' else 0)


dfTrCoappIncomeY['nLoanStat'] = dfTrCoappIncomeY['Loan_Status'].apply(
        lambda x: 1 if x== 'Y' else 0)
dfTrCoappIncomeY.head()

dfTrCoappIncomeN['nLoanStat'] = dfTrCoappIncomeN['Loan_Status'].apply(
        lambda x: 1 if x== 'Y' else 0)
dfTrCoappIncomeN.head()

#--- Convert Self Employed flag to Boolean
dfTrCoappIncomeY['nSelfEmp'] = dfTrCoappIncomeY['Self_Employed'].apply(
        lambda x: 1 if x== 'Yes' else 0 )
dfTrCoappIncomeN['nSelfEmp'] = dfTrCoappIncomeN['Self_Employed'].apply(
        lambda x: 1 if x== 'Yes' else 0 )
dfTrain2['nSelfEmp'] = dfTrain2['Self_Employed'].apply(
        lambda x: 1 if x== 'Yes' else 0 )


#- Correlation between various features
#- Total income , Loan Status
pearsonr(dfTrCoappIncomeY.appTotalIncome, dfTrCoappIncomeY.nLoanStat)
np.corrcoef(dfTrCoappIncomeY.appTotalIncome, dfTrCoappIncomeY.nLoanStat)
#-- corr. coeff. = -0.17634946

np.corrcoef(dfTrCoappIncomeN.appTotalIncome, dfTrCoappIncomeN.nLoanStat)
#-- corr. coeff. = 0.10593875
pearsonr(dfTrCoappIncomeN.appTotalIncome, dfTrCoappIncomeN.nLoanStat)

np.corrcoef(dfTrCoappIncomeY.ApplicantIncome, dfTrCoappIncomeY.nLoanStat)
#-- corr. coeff. =  -0.08087301  when CoApp income > 0
np.corrcoef(dfTrCoappIncomeN.ApplicantIncome, dfTrCoappIncomeN.nLoanStat)
#-- corr. coeff. =  0.10593875  when CoApp income <= 0

np.corrcoef(dfTrCoappIncomeY.CoapplicantIncome, dfTrCoappIncomeY.nLoanStat)
#-- corr. coeff. =  -0.16138968  when CoApp income > 0

#- Dependents , Loan Status
pearsonr(dfTrCoappIncomeY.Dependents, dfTrCoappIncomeY.nLoanStat)
np.corrcoef(dfTrCoappIncomeY.Dependents, dfTrCoappIncomeY.nLoanStat)
#-- corr. coeff. = -0.0198283

np.corrcoef(dfTrCoappIncomeN.Dependents, dfTrCoappIncomeN.nLoanStat)
#-- corr. coeff. = 0.06135438

#- Area and Loan Status
np.corrcoef(dfTrCoappIncomeY.nLocale, dfTrCoappIncomeY.nLoanStat)
#-- corr. coeff. = 0.02167013
np.corrcoef(dfTrCoappIncomeN.nLocale, dfTrCoappIncomeN.nLoanStat)
#-- corr. coeff. = 0.0360375
#--- 
#- Area and Loan Status
np.corrcoef(dfTrCoappIncomeY.nLocale, dfTrCoappIncomeY.nLoanStat)
#-- corr. coeff. = 0.02167013
np.corrcoef(dfTrCoappIncomeN.nLocale, dfTrCoappIncomeN.nLoanStat)
#-- corr. coeff. = 0.0360375
#--- 

#-- Credit_history and Loan Status
dfTrCoappIncomeY.loc[pd.isnull(dfTrCoappIncomeY['Credit_History']),'Credit_History'] = 0
dfTrCoappIncomeN.loc[pd.isnull(dfTrCoappIncomeN['Credit_History']),'Credit_History'] = 0

np.corrcoef(dfTrCoappIncomeY.Credit_History, dfTrCoappIncomeY.nLoanStat)
#-- corr. coeff. = 0.44859261
np.corrcoef(dfTrCoappIncomeN.Credit_History, dfTrCoappIncomeN.nLoanStat)
#-- corr. coeff. = 0.38571242
#---


#-- Self_Employed and Loan Status
#---- Since there are more Loan approvals for records with 
#--- Blank 'Self Employed' column, set it to 'Yes'
dfTrCoappIncomeY.loc[pd.isnull(dfTrCoappIncomeY['Self_Employed']),'Self_Employed'] = 'Yes'
dfTrCoappIncomeN.loc[pd.isnull(dfTrCoappIncomeN['Self_Employed']),'Self_Employed'] = 'Yes'

np.corrcoef(dfTrCoappIncomeY.nSelfEmp, dfTrCoappIncomeY.nLoanStat)
#-- corr. coeff. = -0.0126669
np.corrcoef(dfTrCoappIncomeN.nSelfEmp, dfTrCoappIncomeN.nLoanStat)
#-- corr. coeff. = 0.02508111
#---



pearsonr(dfTrCoappIncomeY.LoanAmount, dfTrCoappIncomeY.Loan_Amount_Term)
#-- Dependents, Area
pearsonr(dfTrCoappIncomeY.Dependents, dfTrCoappIncomeY.nLocale)
#- (-0.0357115448838013, 0.5224776553855053)
np.corrcoef(dfTrCoappIncomeY.Dependents, dfTrCoappIncomeY.nLocale)
#- coeff = -0.03571154  when CoApp income > 0

pearsonr(dfTrCoappIncomeN.Dependents, dfTrCoappIncomeN.nLocale)
#- coeff = (0.030550841283282563, 0.6313906282608055)
np.corrcoef(dfTrCoappIncomeN.Dependents, dfTrCoappIncomeN.nLocale)
#- coeff = 0.03055084 when CoApp income <= 0
#--


#---- As Pearsonr() gives a set of 2 values and corrcoef() gives 
#---- a 2x2 matrix with diagonal values equal. Lets use corrcoef() now on
#-- Applicant Income , Dependents
np.corrcoef(dfTrain2.ApplicantIncome, dfTrain2.Dependents)
#- coeff = 0.08109347
#-- CoApplicant Income , Dependents
np.corrcoef(dfTrain2.CoapplicantIncome, dfTrain2.Dependents)
#- coeff. = 0.04166559
#--
#-- Total Income, Dependents
#--- When coApplicant Income > 0
np.corrcoef(dfTrCoappIncomeY.appTotalIncome,dfTrCoappIncomeY.Dependents)
#--- coeff. = , 0.10277816 when CoApp income > 0
#--- When coApplicant Income <= 0
np.corrcoef(dfTrCoappIncomeN.appTotalIncome,dfTrCoappIncomeN.Dependents)
#--- coeff = 0.08234694 when CoApp income <= 0

#-- Applicant Income vs Dependents
#--- When coApplicant Income > 0
np.corrcoef(dfTrCoappIncomeY.ApplicantIncome,dfTrCoappIncomeY.Dependents)
#--- coeff. = 0.11678091 when CoApp income > 0
#--- When coApplicant Income <= 0
np.corrcoef(dfTrCoappIncomeN.ApplicantIncome,dfTrCoappIncomeN.Dependents)
#--- coeff. = 0.08234694 when CoApp income <= 0

#-- CoApplicant Income vs Dependents
#--- When coApplicant Income > 0
np.corrcoef(dfTrCoappIncomeY.CoapplicantIncome,dfTrCoappIncomeY.Dependents)
#--- coeff. = 0.04278019 when CoApp income > 0
#--- When coApplicant Income <= 0
np.corrcoef(dfTrCoappIncomeN.CoapplicantIncome,dfTrCoappIncomeN.Dependents)
#--- coeff. Can't compute as Coapplicant income is 0


#--- 
plt.scatter(dfTrCoappIncomeY.appTotalIncome, dfTrCoappIncomeY.LoanAmount, color = "Green", label="CoApplicant Income > 0")
plt.scatter(dfTrCoappIncomeN.appTotalIncome, dfTrCoappIncomeN.LoanAmount, color = "Red", label="CoApplicant Income = 0")
plt.xlabel('App Tot Income')
plt.ylabel('Loan Amt')
plt.legend(loc='upper right', frameon = False)
plt.show()
#---
plt.plot(dfTrCoappIncomeY.appTotalIncome, dfTrCoappIncomeY.LoanAmount, "g")
plt.plot(dfTrCoappIncomeN.appTotalIncome, dfTrCoappIncomeN.LoanAmount, 'r')
plt.show()

plt.plot(dfTrCoappIncomeY.ApplicantIncome, dfTrCoappIncomeY.CoapplicantIncome, 'g')
plt.scatter(dfTrCoappIncomeY.ApplicantIncome, dfTrCoappIncomeY.CoapplicantIncome, color = 'blue')

g = sns.catplot(x=dfTrain2.Property_Area, hue=dfTrain2.Loan_Status, 
                col= dfTrain2.IsCoapIncomeZero, 
                data = dfTrain2, kind="count",
                    height = 4, aspect=7)

#- Study the regression 
from sklearn import linear_model
import statsmodels.api as sm

#--- Regression for Coapplicant Income > 0
dfTrCoappIncomeY.isnull().sum(axis=0)

X1 = dfTrCoappIncomeY[['nSelfEmp','nLocale','Credit_History', 'appTotalIncome']]
X1.head()

y1 = dfTrCoappIncomeY[['nLoanStat']]
y1.head()

regr1 = linear_model.LinearRegression()
regr1.fit(X1, y1)

print('Intercept: \n', regr1.intercept_)
print('Coefficients: \n', regr1.coef_)

""" 
For
X1 => [['appTotalIncome','nSelfEmp','nLocale', 
                      'Dependents','Credit_History','Loan_Amount_Term']]

Intercept:  [0.50133756]
Coefficients:  [[-1.55212413e-05 -4.53260999e-02 -9.96154984e-03  5.24555738e-03
   4.69786947e-01 -2.03938784e-05]]
 """

#--- Predict the Loan Status
# regr.predict([[appIncome ,slfEmp, area, depnd, crHistry]])
""" Using multilinear eqn y = b0 + b1.X1 + b2. X2 +.....+ bm.xm
"""
b0 = regr1.intercept_
b = regr1.coef_ 

regr.predict([[ 0, 3, 0, 6091]])


regr.coef_

#--- Regression for Coapplicant Income = 0
X1 = dfTrCoappIncomeN[['nSelfEmp','nLocale',
                      'Credit_History']]
X1.head()

y1 = dfTrCoappIncomeN[['nLoanStat']]
y1.head()

regr1 = linear_model.LinearRegression()
regr1.fit(X1, y1)

print('Intercept: \n', regr1.intercept_)
print('Coefficients: \n', regr1.coef_)

""" Values - 
Intercept:  [0.38508252]
Coefficients: [[ 2.36090795e-06  6.18537459e-02  1.63626088e-02  1.68259671e-02
   4.39477946e-01 -4.36002311e-04]]
"""

#--- Predict the Loan Status
# regr.predict([[appIncome ,slfEmp, area, depnd, crHistry]])
""" Using multilinear eqn y = b0 + b1.X1 + b2. X2 +.....+ bm.xm
"""
b0 = regr1.intercept_
b = regr1.coef_ 

regr1.predict([[ 0, 1, 0]])

#-----

#--- Regression 
X2 = dfTrain2[['nSelfEmp','nLocale', 'Credit_History']]
X2.head()

y2 = dfTrain2[['nLoanStat']]
y2.head()

regr2 = linear_model.LinearRegression()
regr2.fit(X2, y2)

print('Intercept: \n', regr2.intercept_)
print('Coefficients: \n', regr2.coef_)

regr2.predict([[ 0, 1, 0]])

"""

#-----





dfTrain[:]["CoapplicantIncome"].describe(include = 'all')





