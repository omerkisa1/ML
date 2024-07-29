from sklearn.datasets import load_breast_cancer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
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
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train,y_train)
    

y_pred = knn.predict(X_test) # pred data

# We need reel data [y] , pred data[y_pred] for accuracy params
accuracy = accuracy_score(y_test, y_pred)


conf_matrix = confusion_matrix(y_test, y_pred)


# hyperparameter tuning
#calculate accuracy for each k value
#for example K: 1, 2, 3, ...
#accuracy will be something like %A , %B, %C, ...

accuracy_values = []
k_values = []
for k in range(1,21):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    accuracy_values.append(accuracy)
    k_values.append(k)

plt.figure()
plt.plot(k_values, accuracy_values, marker = "o", linestyle = "-")
plt.title("Accuracy according to K value")
plt.xlabel(" K values")
plt.ylabel("Accuracy")
plt.xticks(k_values)
plt.grid(True)