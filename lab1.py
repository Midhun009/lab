import numpy as np
from sklearn.datasets import load_iris
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
dataset=load_iris()
x_train,x_test,y_train,y_test=train_test_split(dataset["data"],dataset["target"],random_state=0)
kn=KNeighborsClassifier(n_neighbors=1,p=3,metric='euclidean')
kn.fit(x_train,y_train)
for i in range(len(x_test)):
    x=x_test[i]
    x_new=np.array([x])
    prediction=kn.predict(x_new)
    print("Target=",y_test[i],dataset["target_names"][y_test[i]],"PREDICTED=",prediction,dataset["target_names"][prediction])
y_pred=kn.predict(x_test)

cm=confusion_matrix(y_test,y_pred)

print('Confusion matrix is as follows\n',cm)
print('Accuracy Metrics')
print(classification_report(y_test,y_pred)) 
print(" correct predicition",accuracy_score(y_test,y_pred))
print(" wrong predicition",(1-accuracy_score(y_test,y_pred)))
