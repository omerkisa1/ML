from sklearn.datasets import load_breast_cancer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
cancer = load_breast_cancer()

df = pd.DataFrame(data = cancer.data, columns = cancer.feature_names) # dataset
df["target"] = cancer.target

#print(df) 

X = cancer.data # features
y = cancer.target # target

#train test split
#We simply generated test data and calculated accuracy with it. compare with knn2
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


#Scaling
scaler = StandardScaler()
#train and implement
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

knn = KNeighborsClassifier()
knn.fit(X_train,y_train)


y_pred = knn.predict(X_test) # pred data

# We need reel data [y] , pred data[y_pred] for accuracy params
accuracy = accuracy_score(y_test, y_pred)
