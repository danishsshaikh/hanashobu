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

#<----------------- Evaluating Algo's(Making Dataset)-------------->

# Split-out validation dataset
array = dataset.values
X = array[:,0:4]
#print(X) #80% data to train the model
Y = array[:,4]
#print(Y) 20% as validation dataset
X_train, X_validation, Y_train, Y_validation = train_test_split(X, Y, test_size=0.20, random_state=1)
# Training data in the X_train and Y_train for preparing models
# X_validation and Y_validation sets that we can use later

#<-----------------Testing with k-fold cross-validation-------------->
# Spot Check Algorithms with k-fold

models = []
models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC(gamma='auto')))
# evaluate each model in turn
results = []
names = []
for name, model in models:
	kfold = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)
	cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring='accuracy')
	results.append(cv_results)
	names.append(name)
	print('%s: %f (%f)' % (name, cv_results.mean(), cv_results.std()))

#<-----------------Comparing Algorithms with Box Plot-------------->

# Compare Algorithms
pyplot.boxplot(results, labels=names)
pyplot.title('Algorithm Comparison')
pyplot.show()

#<-----------------Making Prediction-------------->

# Make predictions on validation dataset
model = SVC(gamma='auto')
model.fit(X_train, Y_train)
predictions = model.predict(X_validation)

#<-----------------Checking Prediction-------------->

# Evaluate predictions
print(accuracy_score(Y_validation, predictions))
print(confusion_matrix(Y_validation, predictions))
print(classification_report(Y_validation, predictions))

# Comment out from 74th line to 96th line for prediction checking :)
 
