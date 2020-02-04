# Load libraries
from pandas import read_csv 
from pandas.plotting import scatter_matrix
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

# Load dataset
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = read_csv(url, names=names)

#<----------------- Summarizing the Data Set-------------->
# shape
# .shape is used to get the idea of rows and columns from the data set
# we use .shape
# print(dataset.shape)

# head
# .head() is ued to peek at the actual data with names after 'read_csv'
# print(dataset.head(20))

# descriptions
# .describe is used to peek at count,mean,min,max value and %
# print(dataset.describe())

# class distribution
# print(dataset.groupby('class').size())

#<----------------- Data Visualization-------------->

#Plots:
#   1. Univariate Plots     #

# Box and Whisker plots
# dataset.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
# pyplot.show()

# Histograms
# dataset.hist()
# pyplot.show()

#   2. Mulrivariate Plots   #

# Scatter plot matrix
# scatter_matrix(dataset)
# pyplot.show()

#<----------------- Evaluating Algo's-------------->

# Split-out validation dataset
array = dataset.values
X = array[:,0:4]
#print(X) #80% data to train the model
Y = array[:,4]
#print(Y) 20% as validation dataset
X_train, X_validation, Y_train, Y_validation = train_test_split(X, Y, test_size=0.20, random_state=1)
