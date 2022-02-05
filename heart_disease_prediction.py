import pandas as pd
import numpy as np

dir = "/content/drive/MyDrive/Pattern Lab/Project/heart.csv"

# Load dataset
df = pd.read_csv(dir, na_values=" ")
df.dropna(inplace=True)
df.reset_index(drop=True)

df.drop(labels="gender",axis=1,inplace=True)

data_x = df.loc[:, df.columns != "target"]
data_y = df["target"]

# Train-test split
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(data_x, data_y,
                                                    train_size = 0.8,
                                                    stratify = data_y,
                                                    random_state = 911)

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import  accuracy_score, precision_score, recall_score, f1_score
model_list = []
accuracy_list = []
precision_list = []
recall_list = []
f1_list = []

# Naive Bayes
from sklearn.naive_bayes import GaussianNB
model = GaussianNB()
model.fit(x_train, y_train)
y_pred = model.predict(x_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

accuracy_list.append(accuracy)
precision_list.append(precision)
recall_list.append(recall)
f1_list.append(f1)
print(accuracy, precision, recall, f1)

# Logistic Regression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
import seaborn as sns
model = make_pipeline(StandardScaler(), LogisticRegression())
model.fit(x_train, y_train)
y_pred = model.predict(x_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

accuracy_list.append(accuracy)
precision_list.append(precision)
recall_list.append(recall)
f1_list.append(f1)
print(accuracy, precision, recall, f1)

# K-Nearest Neigbor
from sklearn.neighbors import KNeighborsClassifier
scaler = StandardScaler()
scaler.fit(x_train)
x_train_2 = scaler.transform(x_train)
x_test_2 = scaler.transform(x_test)
model = KNeighborsClassifier(n_neighbors=5)
model.fit(x_train_2, y_train)
y_pred = model.predict(x_test_2)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

accuracy_list.append(accuracy)
precision_list.append(precision)
recall_list.append(recall)
f1_list.append(f1)
print(accuracy, precision, recall, f1)

# Decision Tree
from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier(criterion="entropy", max_depth=3)
model.fit(x_train, y_train)
y_pred = model.predict(x_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

accuracy_list.append(accuracy)
precision_list.append(precision)
recall_list.append(recall)
f1_list.append(f1)
print(accuracy, precision, recall, f1)

# Support Vector Machines
from sklearn.svm import SVC
model = SVC(kernel='linear')
model.fit(x_train, y_train)
y_pred = model.predict(x_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

accuracy_list.append(accuracy)
precision_list.append(precision)
recall_list.append(recall)
f1_list.append(f1)
print(accuracy, precision, recall, f1)

# Ensemble Learning
from sklearn.ensemble import VotingClassifier
estimators = []
model = make_pipeline(StandardScaler(), LogisticRegression())
estimators.append(('logistic', model))
model = DecisionTreeClassifier(criterion="entropy", max_depth=3)
estimators.append(('tree', model))
model = SVC(kernel='linear')
estimators.append(('svm', model))
model = GaussianNB()
estimators.append(('naive', model))

final_model = VotingClassifier(estimators=estimators, voting='hard')
final_model.fit(x_train.values, y_train)
y_pred = final_model.predict(x_test.values)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

accuracy_list.append(accuracy)
precision_list.append(precision)
recall_list.append(recall)
f1_list.append(f1)
conf = confusion_matrix(y_test, y_pred)
print(accuracy, precision, recall, f1)

sns.heatmap(conf, annot=True)

import matplotlib.pyplot as plt
import numpy as np
plt.style.use('seaborn')
y1 = np.array(accuracy_list)
y2 = np.array(precision_list)
y3 = np.array(recall_list)
y4 = np.array(f1_list)

x = np.arange(len(y1))
plt.bar(x + 0.2, y1, color ='blue', width = 0.2, label ='Accuracy')
plt.bar(x + 0.4, y2, color ='orange', width = 0.2, label ='Precision')
plt.bar(x + 0.6, y3, color ='grey', width = 0.2, label ='Recall')
plt.bar(x + 0.8, y4, color ='yellow', width = 0.2, label ='F1 Score')

plt.xticks([z + 0.5 for z in range(len(y1)+1)],
        ['Naive Bayes','LR','KNN', 'Decision Tree', 'SVM','Ensemble Learning'])

plt.xlabel('Models', fontsize=15)
plt.ylabel('Performance', fontsize=15)
plt.title('Performance Measurement', fontsize=20)
plt.show()

# Example_data = [63, 3, 145, 233, 1, 0, 150, 0, 2.3, 0, 0, 1]
input_data = []
input_1 = int(input("Age in years: "))
input_data.append(input_1)
input_2 = int(input("Chest pain type (0-3): "))
input_data.append(input_2)
input_3 = int(input("Blood pressure (in mm Hg): "))
input_data.append(input_3)
input_4 = int(input("Serum cholestoral in mg/dl: "))
input_data.append(input_4)
input_5 = int(input("Blood sugar &gt; 120 mg/dl (1 = true, 0 = false): "))
input_data.append(input_5)
input_6 = int(input("Electrocardiographic results (1 = yes, 0 = no): "))
input_data.append(input_6)
input_7 = int(input("Maximum heart rate achieved: "))
input_data.append(input_7)
input_8 = int(input("Exercise induced angina (1 = yes, 0 = no): "))
input_data.append(input_8)
input_9 = float(input("ST depression induced by exercise relative to rest: "))
input_data.append(input_9)
input_10 = int(input("Slope of the peak exercise ST segment (0-2): "))
input_data.append(input_10)
input_11 = int(input("Number of major vessels (0-3): "))
input_data.append(input_11)
input_12 = int(input("1 = normal, 2 = fixed defect, 3 = reversable defect: "))
input_data.append(input_12)
print()

input_data = np.array(input_data)
input_data = input_data.reshape(1, -1)
prediction = final_model.predict(input_data)
if prediction == 1:
  print(">> This person has heart disease.")
elif prediction == 0:
  print(">> This person doesn't have heart disease.")