#!/usr/bin/env python
# coding: utf-8

# #                                                            
#                                             KINGSLEY JOLLY JACKSON 
#                            NEXUS BANK - CUSTOMER SEGMENTATION AND DEPOSIT DETECTION SYSTEM PROJECT 2023

# ## 1.  Import Libraries And Dataset For Exploration

# In[1]:


#Import Library

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import dendrogram,linkage
from sklearn.metrics import silhouette_score,homogeneity_score

import warnings
warnings.filterwarnings ('ignore')


# # 2. Exploratory Data Analysis

# In[2]:


#Load the dataset from csv file to a pandas dataframe

df=pd.read_csv ("C:/Users/CDL-KINGSLEY\Desktop/10alytics Class work/Data Captone Project/bank.csv")
df.head()


# In[3]:


#Data Shape
df.shape


# In[4]:


#Data Description

df.info()


# In[5]:


df.describe().T


# In[6]:


for col in df.select_dtypes(include='object').columns:
    print(col)
    print(df[col].unique())


# ##### Data Cleaning And Preprocessing

# In[7]:


#Check for Duplicate

print (df.duplicated().sum())


# In[8]:


#Checking missing values
print (df.isnull().sum())

#Visualizing the missing data


plt.figure(figsize=(6, 3))
sns.heatmap(df.isnull(), cbar=True, cmap='Blues_r');


# In[9]:


#Investigate Features with one value
for columns in df.columns:
    print (columns,df[columns].nunique())


# In[10]:


cat_features = [col for col in df.select_dtypes(include='object').columns if col != 'deposit']
cat_features


# In[11]:


for col in cat_features:
    print ('The col is {} and number of categories are {}'.format(col,len (df[col].unique())) )


# ######  Observation
# We have 9 categorical data with job and month dominating with the highest values 

# ### a. Univariate Analysis

# In[12]:


#find the categorical data distribution using count plot

plt.figure(figsize=(20, 80), facecolor='white')
plotnumber = 1

for cat_feature in cat_features:
    ax = plt.subplot(12, 3, plotnumber)
    sns.countplot(y=cat_feature, data=df)
    plt.xlabel(cat_feature)
    plt.title(cat_feature)
    plotnumber += 1

plt.tight_layout()
plt.show()


# ##### Observation
# 
# * Customers with job type as management records are high in the given dataset and house are less
# * Married customers are high and divorsed are less
# * From observation the dataset shows customers with secondary educational background are high
# * Default from the dataset does not play any significant role as it as shows number of 'no's' in high ratio comparared to the number of 'yes'
# * Monthly distribution in the data shows that May contributed high number while Dec shows lesser contribution 

# In[13]:


#exploring the relationship between the categorical data and label 'deposit' the dependant variable

for cat_feature in cat_features:
    sns.catplot(x='deposit',col= cat_feature,kind = 'count', data=df)
plt.show()


# In[14]:


#check the target label split over categorical data and find the count
for cat_feature in cat_features:
    print (df.groupby(['deposit',cat_feature]).size())


# ######  Observation
# 
# * From the dataset retired customers are interest in bank deposit
# * The month of May records high number with low ratio of interest in deposit 
# * Customers with a housing loan to be uninterested in making a bank deposit
# * Campaign outcome with poutcome = success have high tendency of making deposit
# * March , September, October, and December customers showed much interest on deposit

# ### b. Multivariate Analysis

# In[15]:


#Numerical features
num_features = [col for col in df.columns if ((df[col].dtypes != 'object') and (col != 'deposit'))]
print('Number of numerical variables:', len(num_features))
df[num_features].head()


# In[16]:


#Find the discrete feature

discrete_feature= [col for col in num_features if len (df[col].unique ())<25]
print ("Discrete variables count:{}".format(len (discrete_feature)))


# In[17]:


#Find continous nemerical features
continuous_features = [col for col in num_features if col not in discrete_feature+  ['deposit']]
print ("Continuous_features count:{}".format(len (discrete_feature)))


# ######  Observation
# From the dataset there are 7 continuous numerical features

# ##### Distribution of Continuous features

# In[18]:


#Plot univariate distribution of continuous features

plt.figure(figsize=(20, 80), facecolor='white')
plotnumber = 1

for continuous_feature in continuous_features:
    ax = plt.subplot(12, 3, plotnumber)
    sns.distplot(df[continuous_feature])
    plt.xlabel(continuous_feature)
    plotnumber += 1

plt.tight_layout()
plt.show()


# ######  Observation
# 
# * The dataset indicate that balance,campaign, pdays, and previous highly skewed towards the left hand and shows a number of outliers
# * Age and days are normally distributed 

# ##### Relationship between continuous numerical features and label

# In[19]:


#boxplot to show target 'deposit' distribution with respect to numerical features

plt.figure(figsize=(20, 60), facecolor='white')
plotnumber = 1

for col in continuous_features:
    ax = plt.subplot(12, 3, plotnumber)
    sns.boxplot(x="deposit", y=df[col], data=df)
    plt.xlabel(col)
    plotnumber += 1

plt.show()


# ######  Observation
# 
# * Long days engagement and communication shows that customers indicate interest on deposit

# ##### Find outliers in numerical features

# In[20]:


#boxplot on numerical features to find outliers
plt.figure(figsize=(20, 60), facecolor='white')
plotnumber = 1

for num_feature in num_features :
    ax = plt.subplot(12, 3, plotnumber)
    sns.boxplot(df[num_feature])
    plt.xlabel(num_feature)
    plotnumber += 1

plt.show()


# ######  Observation
# 
# * Previous, campaign, age, duration, pdays, and balance shows an outlier in the dataset

# ## 3. Explore the correlation between numerical features

# In[21]:


corr = df.corr()
fig = plt.figure(figsize=(10, 5))
sns.heatmap(corr, annot=True)
plt.show()


# ######  Observation
# 
# * There is no significant correlation observed among the variables in the dataset.
# 

# In[22]:


#check if the dataset is balanced or not based on target values in classification

sns.countplot(x='deposit',data=df)
plt.show()

df['deposit'].groupby(df['deposit']).count()


# ######  Observation
# 
# * The dataset shows to be balanced 

# In[23]:


sns.pairplot(df, hue='deposit', height=3);


# ## 4. Feature Engineering

# In[24]:


df2 = df.copy()
df2.head()


# In[25]:


df2.shape


# ##### Drop unwanted features

# In[26]:


#deposit against default

df2.groupby(['deposit','default']).size()


# In[27]:


#Drop default features because it does not play any significant role

df2.drop (['default'],axis=1,inplace =True)


# In[28]:


# deposit against pdays features 
df2.groupby(['deposit','pdays']).size()


# In[29]:


#drop pdays features as it has -1 value for around 40x

df2.drop(['pdays'],axis =1, inplace=True)


# ##### Remove Outliers

# In[30]:


#deposit against balance
df2.groupby(['deposit','balance'],sort=True)['balance'].count()


# ##### Observation
# From the dataset the balance feature should not be removed as balance goes high the customers shows interest on deposit

# In[31]:


#deposit against duration

df2.groupby (['deposit','duration'],sort=True) ['duration'].count()


# ##### Observation
# From the dataset the durationfeature should not be removed as duration goes high the customers shows interest on deposit

# In[32]:


#deposit against age
df2.groupby('age',sort=True) ['age'].count()


# ##### Observation
# The outlier can be ignored, the age of the customers lies between 18 and 95

# In[33]:


#deposit againt campaign
df2.groupby(['deposit','campaign'],sort=True)['campaign'].count()


# In[34]:


#remove the outlier in campaign

df3 = df2[df2['campaign'] < 33]
df3 = df3.groupby(['deposit', 'campaign'], sort=True)['campaign'].count()
df3


# ## 5. Model Building

# ##### Label encoding

# In[35]:


#Convert the object dataset to numeric

from sklearn.preprocessing import LabelEncoder
data = LabelEncoder()  # Method initiation

# Looping for columns
for r in df2.columns:
    if df2[r].dtype == 'object':
        df2[r] = data.fit_transform(df2[r])

df2.head()


# #### Scale the dataset

# In[36]:


from sklearn.preprocessing import MinMaxScaler

def min_max_scale_numeric (df2,columns):
    scaler = MinMaxScaler()
    df2[columns] =scaler.fit_transform (df2[columns])
    return df2
#choose columns to be scaled

scaled_columns = ['duration', 'balance']

#carryout min_max scaling 

scaled_1 = min_max_scale_numeric (df2, scaled_columns)

#print scaled_1

scaled_1.head()


# ##### Data training set and Test set

# In[37]:


X=df2[['balance','loan']]#idenpendent variables
y=df2['deposit']#depedent variables
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)#test set is 20%


# In[38]:


len (X_train)


# In[39]:


len (X_test)


# In[41]:


#Customer Segmentation with age, job, marital, education, balance, housing, loan, duration, deposit

df3=df2[['age', 'job', 'marital', 'education', 'balance', 'housing', 'loan',
         'duration', 'deposit']]


# ## 6. Finding the Optimal Number of Clusters with the Elbow Method

# In[42]:


sum_of_sqr_dist = {}

for k in range(1, 11):
    km = KMeans(n_clusters=k, init='k-means++', max_iter=1000)
    km = km.fit(df3)
    sum_of_sqr_dist[k] = km.inertia_


# In[43]:


sns.pointplot(x=list(sum_of_sqr_dist.keys()), y=list(sum_of_sqr_dist.values()))
plt.xlabel("Number of Clusters (K)")
plt.ylabel("Sum of Square Distances")
plt.title("Elbow Method for Optimal K")
plt.show()


# ## 7.  K-Means Clustering

# In[44]:


#model fitting

Model = KMeans(n_clusters=4,
             init='k-means++',
             max_iter=100)
Model.fit(df3)


# In[45]:


print("Labels", Model.labels_)


# In[46]:


#determine the centroids

print("Centroids", Model.cluster_centers_)


# In[47]:


centroids = Model.cluster_centers_


# In[48]:


#call up the clusters

df3_cluster = df3.copy()
df3_cluster["Cluster"] = Model.fit_predict(df3)
df3_cluster.head()


# ## 8.  Visualize the clusters

# In[49]:


plt.scatter (scaled_1['balance'], scaled_1['duration'], c=scaled_1 ['deposit'],cmap= 'plasma')
plt.xlabel("balance")
plt.ylabel("duration")
plt.title ('Customer Clusters')
plt.show()


# In[50]:


labels = Model.labels_


# ## 9.  Evaluation with Silhouette Score

# In[51]:


#calculate the silhoute score

silhouette_score(df3, labels)


# In[52]:


silhouette = {}

for k in range(2,8):
    km = KMeans(n_clusters=k, init='k-means++', max_iter=100)
    km.fit(df3)
    silhouette[k] = silhouette_score(df3, km.labels_)

#Visualize the silhoutte Scores

sns.pointplot(x=list(silhouette.keys()), y=list(silhouette.values()))
plt.xlabel("Number of Clusters (K)")
plt.ylabel("Silhouette Scores")
plt.title("Silhouette Scores for Each K")


# ##### Observation 
# 
# * From the silhouette score analysis it shows that the best and optimal cluster for the customer segmentation is k=4

# ## 10. Customer  Segmentation with 3 Features

# In[55]:


df4= df3.copy()


# In[56]:


#explore k=5

#Determine the optimal number of cluster

sum_of_sqr_dist = {}

for k in range(1, 10):
    km = KMeans(n_clusters=k, init='k-means++', max_iter=100)
    km = km.fit(df4)
    sum_of_sqr_dist[k] = km.inertia_


# In[57]:


#Visualize the elbow optimal cluster

sns.pointplot(x=list(sum_of_sqr_dist.keys()), y=list(sum_of_sqr_dist.values()))
plt.xlabel("Number of Clusters (K)")
plt.ylabel("Sum of Square Distances")
plt.title("Elbow Method for Optimal K")
plt.show()


# ##### Observation
# 
# * The inertia started decreasing significantly at K= 4

# In[58]:


Model2 = KMeans(n_clusters=5, init='k-means++', max_iter=100)
Model2.fit(df4)


# In[59]:


df4['Cluster'] = Model2.fit_predict(df4)
df4.head()


# #### 3D Customer cluster visualization

# In[60]:


labels = Model2.labels_
centroids = Model2.cluster_centers_


# In[61]:


import plotly.graph_objs as go
import plotly.offline as py

df4['labels'] = labels

trace = go.Scatter3d(
    x=df4['duration'],
    y=df4['balance'],
    z=df4['age'],
    mode='markers',
    marker=dict(
        color=df4['labels'],
        size=5,
        line=dict(color=df4['labels'], width=12),
        opacity=0.8
    )
)

data = [trace]
layout = go.Layout(
    title='3D Customer Clustering',
    scene=dict(
        xaxis=dict(title='duration'),
        yaxis=dict(title='balance'),
        zaxis=dict(title='age')
    )
)

fig = go.Figure(data=data, layout=layout)
py.iplot(fig)


# In[54]:


# Step 1: Calculate cluster-level statistics
cluster_stats = df3_cluster.groupby("Cluster").agg({
    "age": "mean",  
    "duration": "sum",    
    "balance": "mean",
    "deposit": "mean" 
    
}).reset_index()

# Step 2: Identify the most profitable cluster
profitable_cluster = cluster_stats.loc[cluster_stats["age"].idxmax()]
# Replace "column1" with the metric you want to use for profitability

# Step 3: Analyze cluster characteristics
selected_cluster = profitable_cluster["Cluster"]
cluster_data = df3_cluster[df3_cluster["Cluster"] == selected_cluster]
# Explore and analyze the data in the selected cluster

# Print the profitable cluster details and cluster-level statistics

print("Profitable Cluster:", selected_cluster)
print("Cluster Statistics:")
print(cluster_stats)


# ## SUMMARY:
# 
# ### Key Insights for Customer Segmentation and Deposit Marketing Strategy
# 
# ##### Target Audience: Retired Customers
# 
# * Retired customers have shown a strong interest in bank deposits.
# * Tailor marketing efforts specifically towards this customer segment.
# 
# ##### Month of May: Low Conversion Ratio
# 
# * May records a high number of interactions but a low ratio of interest in deposits.
# * Investigate reasons behind this trend and consider campaign adjustments for May.
# 
# ##### Impact of Housing Loans
# 
# * Customers with housing loans are generally uninterested in making bank deposits.
# * Develop targeted strategies to address concerns and promote deposit offerings to this customer segment.
# 
# ##### Successful Campaign Outcomes
# 
# * Campaign outcomes marked as "success" have a higher tendency to result in deposits.
# * Prioritize and replicate successful campaign strategies to increase overall deposit conversion rates.
# 
# ##### Seasonal Patterns of Interest
# 
# * Customers exhibit greater interest in deposits during March, September, October, and December.
# * Leverage these months for targeted marketing campaigns and promotional offers.
# 
# ##### Understanding Customer Data
# 
# * The dataset includes 7 continuous numerical features.
# * Certain features, such as balance, campaign, pdays, and previous, are skewed and contain outliers.
# 
# ##### Normal Distribution and Outliers
# 
# * Age and days follow a normal distribution, while other features have outliers.
# * Consider outliers in age, duration, pdays, campaign, previous, and balance for further analysis or data preprocessing.
# 
# ##### Dataset Balance
# 
# * The dataset exhibits balance between positive and negative responses for deposits.
# * Ensure equal representation of both deposit and non-deposit customers in future analysis and campaigns.
# 
# ##### Optimal Customer Segmentation
# 
# * Silhouette score analysis show the optimal number of clusters for customer segmentation is k=4 with silhouette score of 0.3444014756318102.
# * Utilize these segments to tailor marketing messages and strategies to specific customer groups.
# 
# ##### No Correlation Between Variables
# 
# * There is no significant correlation observed among the variables in the dataset.
# * Rely on other factors and insights to drive deposit marketing strategies rather than relying on inter-variable relationships.
# 
# These key insights provide actionable information to improve deposit conversion rates and develop targeted marketing strategies. By understanding customer behavior, seasonality, and campaign outcomes, the bank can enhance its marketing efforts and drive customer engagement and deposit growth.

# In[ ]:





# In[ ]:





# In[ ]:




