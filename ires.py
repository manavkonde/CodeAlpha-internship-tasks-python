from sklearn.datasets import load_iris
iris=load_iris()

X=iris.data
Y=iris.target
# print(Y)
# print(X)

feature_names = iris.feature_names
target_names = iris.target_names

print(target_names)

# print(type(X))

#splitting data 

from sklearn.model_selection import train_test_split
X_train, X_test,Y_train, Y_test = train_test_split(X,Y,test_size=0.5)

print(X_train.shape)
print(X_test.shape)

#created model

from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train,Y_train)
print(knn)

#using decision tree
# from sklearn.tree import DecisionTreeClassifier
# knn=DecisionTreeClassifier()
# knn.fit(X_train,Y_train)

#checking output
Y_pred=knn.predict(X_test)
print(Y_pred)

from sklearn import metrics
print(metrics.accuracy_score(Y_test,Y_pred))


#training and testing model with different data sets
sample =[[3,5,4,2,],[2,3,5,4]]
predictions=knn.predict(sample)
pred_species=[iris.target_names[p] for p in predictions]
print("predictions : ",pred_species)

#use joblib to direct load and use a model for model creation
from joblib import dump,load
model=dump(knn,"mlbrain.joblib")
# print(model)
model=load("mlbrain.joblib")
model.predict(X_test)
sample =[[3,5,4,2,],[2,3,5,4]]
predictions=model.predict(sample)
pred_species=[iris.target_names[p] for p in predictions]
print("predictions : ",pred_species)
