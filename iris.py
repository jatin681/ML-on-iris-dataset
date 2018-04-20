import numpy
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn import datasets
iris = datasets.load_iris()
iris = datasets.load_iris()
iris_data = iris.data
iris_data = pd.DataFrame(iris_data,columns= iris.feature_names)
iris_data['class'] = iris.target
iris_data.head()
iris.target_names
print(iris_data.shape)
ab =iris_data.describe()
import seaborn as sns
x=sns.boxplot(data = iris_data,width=0.5,fliersize=5)
sns.set(rc={'figure.figsize':(2,5)})

from sklearn.model_selection import train_test_split
X = iris_data.values[:,0:4]
Y = iris_data.values[:,4]
x_train, x_test, y_train,y_test = train_test_split(X,Y,test_size = 0.3,random_state=42)

model = KNeighborsClassifier()
model.fit(x_train,y_train)
predictions = model.predict(x_test)
print('KNC=',accuracy_score(y_test,predictions))

model = SVC()
model.fit(x_train,y_train)
predictions = model.predict(x_test)
print('SVC =',accuracy_score(y_test,predictions))


model = RandomForestClassifier()
model.fit(x_train,y_train)
predictions = model.predict(x_test)
print('RFC =',accuracy_score(y_test,predictions))

model = LogisticRegression()
model.fit(x_train,y_train)
predictions = model.predict(x_test)
print('LR =',accuracy_score(y_test,predictions))
