import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

"""MILESTONE 1"""

#reading csv file having missing values with real time dates
df1=pd.read_csv("visa_dataset_final_missing.csv", parse_dates=['Application_Date','Decision_Date']) 
print(df1.isnull().sum())

#reading preprocessed csv file 
df=pd.read_csv("visa_dataset.csv",parse_dates=['Application_Date','Decision_Date'])

#handling missing values
df['Decision_Date'].fillna(df['Decision_Date'].mode()[0], inplace=True)
df['Processing_Days'].fillna(df['Processing_Days'].mode()[0], inplace=True)
df['Visa_Class'].fillna('Unknown',inplace=True)
df['Gender'].fillna('Unknown',inplace=True)
df['Applicant_Age'].fillna(round(df['Applicant_Age'].median()),inplace=True)
df['Processing_Center'].fillna('Unknown',inplace=True)
df['Case_Status'].fillna('Unknown',inplace=True)

#checking whether the missing values are handled or not
print(df.isnull().sum())

#adding application month as feature 1
df['Application_Month']=df['Application_Date'].dt.month
df.to_csv('visa_dataset.csv',index=False)

#encoding the dataset 
df_encoded=pd.get_dummies(df, columns=['Visa_Type','Visa_Class','Gender','Applicant_Country','Visa_Country','Processing_Center','Case_Status'])
print(df_encoded.columns)


"""MILESTONE 2"""
"""Exploratory Data Analysis"""

#Numerical statistical analysis
print(df["Processing_Days"].describe()) 
#Observation: Mean processing time is 65 days (std 63), median 45 days which indicates right-skewness. 
#75% of applications are processed within 82 days, while a small fraction experience extreme delays up to 730 days.

#Distribution of Processing Days
sns.histplot(df["Processing_Days"],kde=True)
plt.title("Distribution of Processing Days")
plt.xlabel("Processing Days")
plt.ylabel("Count")
plt.show()
#Observation: Most visas are processed quickly, 
#but a small fraction take extremely long, heavily skewing the distribution.

#Outlier Detection
sns.boxplot(x=df["Processing_Days"])
plt.title("Outlier detection plot for Processing Days")
plt.show()
#Observation: Most visas are processed quickly, 
#but a non-trivial number take extraordinarily long, and those are flagged here as outliers.

#Calculation of  the correlation matrix for application month and processing days
corrMatrix=df[["Processing_Days","Application_Month"]].corr()
#plotting heatmap between application month and processing days
sns.heatmap(corrMatrix,annot=True,cmap="coolwarm")
plt.title("Correlation Heatmap for Application month and processing days")
plt.show()
#Observation: Visa processing time shows no meaningful 
#linear correlation with application month, indicating minimal seasonality effects.

#Plot between Application month and Processing Days
sns.scatterplot(x="Application_Month",y="Processing_Days",data=df)
plt.title("Processing Days v/s Application Month")
plt.show()
#Observation: Visa processing time shows no clear dependence on the application month. 
#Each month exhibits a similar distribution with substantial variability and persistent outliers, 
#indicating that processing delays are primarily influenced by non-seasonal factors.

#Plot between Processing Centre and Processing Days
df.groupby("Processing_Center")["Processing_Days"].mean().plot(kind="bar")
plt.title("Average Processing Days by Processing Center")
plt.show()
#Observation: The analysis of average processing days across processing centers reveals moderate variation,
#with most centers exhibiting mean processing times between 60 and 75 days.

#Correlation between Processing Days and Applicant Age
corrMatrix_1=df[["Processing_Days","Applicant_Age"]].corr()
#plotting heatmap between applicant age and processing days
sns.heatmap(corrMatrix_1,annot=True,cmap="coolwarm")
plt.title("Correlation Heatmap for Processing Days and Applicant Age")
plt.show()
#Observation: The correlation analysis between applicant age and visa processing time reveals a negligible linear relationship (r = 0.045). 
#This suggests that applicant age does not significantly impact processing duration, and processing delays are largely independent of demographic age factors.


"""Feature Engineering"""

#Feature 1: Application month
#Previously calculated as an attribute in MILESTONE 1

#Feature 2: Seasonal Index (Peak and Off-Peak seasons)
#Peak season months: May, June, July, August
df["Peak_Season"]=df['Application_Month'].apply(lambda x: "Peak" if x in [5,6,7,8] else "Off-Peak") 

#Feature 3: Country specific average processing time(days)
countryAvg=df.groupby("Visa_Country")["Processing_Days"].mean()
df["countryAvg"]=df["Visa_Country"].map(countryAvg)

#Feature 4: Average processing time(days) on the basis of Visa_Type
visa_type_Avg=df.groupby("Visa_Type")["Processing_Days"].mean()
df["visa_type_Avg"]=df["Visa_Type"].map(visa_type_Avg)

#Feature 5: Delay Status on the basis of processing days
df["Delay_Status"]=df["Processing_Days"].apply(
    lambda x:
        "Fast" if x<=30 else
        "Normal" if x<=50 else
        "Delayed" if x<=90 else
        "Severely Delayed"
)

#Feature 6: Average processing time(days) on the basis of Visa_Class
visa_class_Avg=df.groupby("Visa_Class")["Processing_Days"].mean()
df["visa_class_Avg"]=df["Visa_Class"].map(visa_class_Avg)