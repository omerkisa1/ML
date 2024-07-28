from sklearn.datasets import load_breast_cancer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import pandas as pd
cancer = load_breast_cancer()

df = pd.DataFrame(data = cancer.data, columns = cancer.feature_names) # dataset
df["target"] = cancer.target

#print(df) 

X = cancer.data # features
y = cancer.target # target

knn = KNeighborsClassifier()
knn.fit(X,y)


y_pred = knn.predict(X) # pred data

# We need reel data [y] , pred data[y_pred] for accuracy params
accuracy = accuracy_score(y, y_pred)
