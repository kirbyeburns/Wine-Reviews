# Robert K Burns
# DSC550 Assignment 9.2

# August 7, 2019


import pandas as pd

#Step 1:  Load data into a dataframe
addr1 = "menu.csv"
data = pd.read_csv(addr1)

# Step 2:  check the dimension of the table
print("The dimension of the table is: ", data.shape)

#Step 3:  Look at the data
print(data.head(5))

#Step 5:  what type of variables are in the table
print("Describe Data")
print(data.describe())
print("Summarized Data")
print(data.describe(include=['O']))
print(data.describe)

#Step 6: import visulization packages
import matplotlib.pyplot as plt

# set up the figure size
plt.rcParams['figure.figsize'] = (20, 10)

# make subplots
fig, axes = plt.subplots(nrows = 2, ncols = 2)

# Specify the features of interest
num_features = ['Calories', 'Total Fat', 'Carbohydrates', 'Protein']
xaxes = num_features
yaxes = ['Counts', 'Counts', 'Counts', 'Counts']

# draw histograms
axes = axes.ravel()
for idx, ax in enumerate(axes):
    ax.hist(data[num_features[idx]].dropna(), bins=40)
    ax.set_xlabel(xaxes[idx], fontsize=10)
    ax.set_ylabel(yaxes[idx], fontsize=10)
    ax.tick_params(axis='both', labelsize=10)
plt.show()

# Remove some of the features of the Category variable and the Item variable
data= data[data.Calories != 0]
data= data[data.Item != 'Chicken McNuggets (40 piece)']
data= data[data.Item != 'Chicken McNuggets (20 piece)']
data= data[data.Category != 'Coffee & Tea']
print("The dimension of the table is: ", data.shape)

#7:  Barcharts: set up the figure size
#%matplotlib inline
plt.rcParams['figure.figsize'] = (20, 10)

# make subplots
fig, axes = plt.subplots(nrows = 1, ncols = 1)
plt.gcf().subplots_adjust(bottom= 0.4)

# make the data read to feed into the visulizer
X_Category = data.replace({'Category': {1: 'Breakfast', 2: 'Beef & Pork', 3: 'Chicken & Fish', 4: 'Salads',
                                    5: 'Snacks & Sides', 6: 'Desserts', 7: 'Beverages',
                                    8: 'Smoothies & Shakes'}}).groupby('Category').size().reset_index(name='Counts')['Category']
Y_Category = data.replace({'Category': {1: 'Breakfast', 2: 'Beef & Pork', 3: 'Chicken & Fish', 4: 'Salads',
                                    5: 'Snacks & Sides', 6: 'Desserts', 7: 'Beverages',
                                    8: 'Smoothies & Shakes'}}
                        ).groupby('Category').size().reset_index(name='Counts')['Counts']
# make the bar plot
axes.bar(X_Category, Y_Category)
axes.set_title('Category', fontsize=25)
axes.set_ylabel('Counts', fontsize=20)
axes.tick_params(axis='both', labelrotation=90, labelsize=15)
plt.show()

#Step 8: Pearson Ranking
#set up the figure size
#%matplotlib inline
plt.rcParams['figure.figsize'] = (15, 7)
plt.gcf().subplots_adjust(bottom= 0.25)

# import the package for visulization of the correlation
from yellowbrick.features import Rank2D

# extract the numpy arrays from the data frame
X = data[num_features].as_matrix()

# instantiate the visualizer with the Covariance ranking algorithm
visualizer = Rank2D(features=num_features, algorithm='pearson')
visualizer.fit(X)                # Fit the data to the visualizer
visualizer.transform(X)             # Transform the data
visualizer.poof(outpath="nutrients1.png") # Draw/show/poof the data

# Setting up smaller dataframes for the visualizer
data2= data[data.Category != 'Snacks & Sides']
data2= data2[data.Category != 'Smoothies & Shakes']
data2= data2[data.Category != 'Desserts']
data2= data2[data.Category != 'Beverages']

data3= data[data.Category != 'Beef & Pork']
data3= data3[data.Category != 'Breakfast']
data3= data3[data.Category != 'Chicken & Fish']
data3= data3[data.Category != 'Salads']


# Step 9:  Compare variables against Survived and Not Survived
#set up the figure size
plt.rcParams['figure.figsize'] = (15, 7)
plt.rcParams['font.size'] = 50

# setup the color for yellowbrick visulizer
from yellowbrick.style import set_palette
set_palette('sns_bright')

# import packages
from yellowbrick.features import ParallelCoordinates
# Specify the features of interest and the classes of the target
classes = ['Breakfast', 'Beef & Pork', 'Chicken & Fish', 'Salads']
num_features = ['Calories', 'Total Fat', 'Carbohydrates', 'Protein']

# copy data to a new dataframe
data2_norm = data2.copy()
# normalize data to 0-1 range
for feature in num_features:
    data2_norm[feature] = (data2[feature] - data2[feature].mean(skipna=True)) / (data2[feature].max(skipna=True) - data2[feature].min(skipna=True))

# Extract the numpy arrays from the data frame
X = data2_norm[num_features].as_matrix()
y = data2.Category.as_matrix()

# Instantiate the visualizer
visualizer = ParallelCoordinates(classes=classes, features=num_features)
visualizer.fit(X, y)      # Fit the data to the visualizer
visualizer.transform(X)   # Transform the data
visualizer.poof(outpath="nutrients2.png") # Draw/show/poof the data
plt.show()

# create a graph for the other dataframe
plt.rcParams['figure.figsize'] = (15, 7)
plt.rcParams['font.size'] = 50

# setup the color for yellowbrick visulizer
from yellowbrick.style import set_palette
set_palette('sns_bright')

# import packages
from yellowbrick.features import ParallelCoordinates
# Specify the features of interest and the classes of the target
classes = ['Snacks & Sides', 'Desserts', 'Beverages', 'Smoothies & Shakes']
num_features = ['Calories', 'Total Fat', 'Carbohydrates', 'Protein']

# copy data to a new dataframe
data3_norm = data3.copy()
# normalize data to 0-1 range
for feature in num_features:
    data3_norm[feature] = (data3[feature] - data3[feature].mean(skipna=True)) / (data3[feature].max(skipna=True) - data3[feature].min(skipna=True))

# Extract the numpy arrays from the data frame
X = data3_norm[num_features].as_matrix()
y = data3.Category.as_matrix()

# Instantiate the visualizer
# Instantiate the visualizer
visualizer = ParallelCoordinates(classes=classes, features=num_features)
visualizer.fit(X, y)      # Fit the data to the visualizer
visualizer.transform(X)   # Transform the data
visualizer.poof(outpath="nutrients4.png") # Draw/show/poof the data
plt.show()


# step 10. Normalize the data and see what we have
# import package
import numpy as np

# log-transformation of the main dataframe looking at sugars and calories from fat
def log_transformation(data):
    return data.apply(np.log1p)

data['Calories from Fat_log1p'] = log_transformation(data['Calories from Fat'])
data['Sugars_log1p'] = log_transformation(data['Sugars'])

# check the data
print(data.describe())

# log-transformation of the second dataframe looking at sugars and calories from fat
def log_transformation(data2):
    return data2.apply(np.log1p)

data2['Calories from Fat_log1p'] = log_transformation(data2['Calories from Fat'])
data2['Sugars_log1p'] = log_transformation(data2['Sugars'])

# check the data
print(data2.describe())

# log-transformation of the third dataframe looking at sugars and calories from fat
def log_transformation(data3):
    return data3.apply(np.log1p)

data3['Calories from Fat_log1p'] = log_transformation(data3['Calories from Fat'])
data3['Sugars_log1p'] = log_transformation(data3['Sugars'])

# check the data
print(data3.describe())

#Step 12 - adjust skewed data
#check the distribution using histogram
# set up the figure size
#%matplotlib inline
plt.rcParams['figure.figsize'] = (10, 5)

plt.hist(data2['Calories from Fat_log1p'], bins=40)
plt.xlabel('Calories from Fat_log1p', fontsize=20)
plt.ylabel('Counts', fontsize=20)
plt.tick_params(axis='both', labelsize=15)
plt.show()

plt.rcParams['figure.figsize'] = (10, 5)

plt.hist(data2['Sugars_log1p'], bins=40)
plt.xlabel('Sugars_log1p', fontsize=20)
plt.ylabel('Counts', fontsize=20)
plt.tick_params(axis='both', labelsize=15)
plt.show()


plt.rcParams['figure.figsize'] = (10, 5)

plt.hist(data3['Calories from Fat_log1p'], bins=40)
plt.xlabel('Calories from Fat_log1p', fontsize=20)
plt.ylabel('Counts', fontsize=20)
plt.tick_params(axis='both', labelsize=15)
plt.show()

plt.rcParams['figure.figsize'] = (10, 5)

plt.hist(data3['Sugars_log1p'], bins=40)
plt.xlabel('Sugars_log1p', fontsize=20)
plt.ylabel('Counts', fontsize=20)
plt.tick_params(axis='both', labelsize=15)
plt.show()

#Step 13 - convert categorical data to numbers
#get the categorical data
cat_features = ['Category', 'Serving Size']
data_cat = data[cat_features]
data_cat = data_cat.replace({'Category':{1: 'Breakfast', 2: 'Beef & Pork', 3: 'Chicken & Fish', 4: 'Salads',
                                    5: 'Snacks & Sides', 6: 'Desserts', 7: 'Beverages',
                                    8: 'Smoothies & Shakes'}})
# One Hot Encoding
data_cat_dummies = pd.get_dummies(data_cat)
# check the data
print(data_cat_dummies.head(8))


#Step 14 - create a whole features dataset that can be used for train and validation data splitting
# here we will combine the numerical features and the dummie features together
features_model = ['Calories', 'Total Fat', 'Carbohydrates', 'Protein']
data_model_X = pd.concat([data[features_model], data_cat_dummies], axis=1)

# create a whole target dataset that can be used for train and validation data splitting
data_model_y = data.replace({'Category':{1: 'Breakfast', 2: 'Beef & Pork', 3: 'Chicken & Fish', 4: 'Salads',
                                    5: 'Snacks & Sides', 6: 'Desserts', 7: 'Beverages',
                                    8: 'Smoothies & Shakes'}})['Category']
# separate data into training and validation and check the details of the datasets
# import packages
from sklearn.model_selection import train_test_split

# split the data
X_train, X_val, y_train, y_val = train_test_split(data_model_X, data_model_y, test_size =0.3, random_state=11)

# number of samples in each set
print("No. of samples in training set: ", X_train.shape[0])
print("No. of samples in validation set:", X_val.shape[0])

#
print('\n')
print('No. of each category in the training set:')
print(y_train.value_counts())

print('\n')
print('No. of each category in the validation set:')
print(y_val.value_counts())

# Step 15 - Eval Metrics
from sklearn.linear_model import LogisticRegression

from yellowbrick.classifier import ConfusionMatrix
from yellowbrick.classifier import ClassificationReport
from yellowbrick.classifier import ROCAUC

# Instantiate the classification model
model = LogisticRegression()

#The ConfusionMatrix visualizer taxes a model
classes = ['Breakfast', 'Beef & Pork', 'Chicken & Fish', 'Salads',
                                    'Snacks & Sides', 'Desserts', 'Beverages',
                                    'Smoothies & Shakes']
cm = ConfusionMatrix(model, classes=classes, percent=False)

#Fit fits the passed model. This is unnecessary if you pass the visualizer a pre-fitted model
cm.fit(X_train, y_train)

#To create the ConfusionMatrix, we need some test data. Score runs predict() on the data
#and then creates the confusion_matrix from scikit learn.
cm.score(X_val, y_val)

# change fontsize of the labels in the figure
for label in cm.ax.texts:
    label.set_size(20)

#How did we do?
cm.poof()

# Precision, Recall, and F1 Score
# set the size of the figure and the font size
#%matplotlib inline
plt.rcParams['figure.figsize'] = (15, 7)
plt.rcParams['font.size'] = 20

# Instantiate the visualizer
visualizer = ClassificationReport(model, classes=classes)

visualizer.fit(X_train, y_train)  # Fit the training data to the visualizer
visualizer.score(X_val, y_val)  # Evaluate the model on the test data
g = visualizer.poof()

# ROC and AUC
#Instantiate the visualizer
visualizer = ROCAUC(model)

visualizer.fit(X_train, y_train)  # Fit the training data to the visualizer
visualizer.score(X_val, y_val)  # Evaluate the model on the test data
g = visualizer.poof()









