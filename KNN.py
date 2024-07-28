from sklearn.datasets import load_breast_cancer
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
cancer = load_breast_cancer()

df = pd.DataFrame(data = cancer.data, columns = cancer.feature_names)
df["target"] = cancer.target

#print(df) 

X = cancer.data
y = cancer.target

knn = KNeighborsClassifier()
knn.fit(X,y)

y_pred = knn.predict(X)
print(y_pred)