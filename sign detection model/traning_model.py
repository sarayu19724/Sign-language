import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

# Load data from the pickle file
with open('data.pickle', 'rb') as f:
    data_dir = pickle.load(f)

# Extract data and labels
data = np.array(data_dir['data'])
labels = np.array(data_dir['labels'])

# Flatten the data and ensure landmarks are structured as arrays
data_flattened = []
for d in data:
    flattened_landmarks = np.concatenate([landmark.reshape(-1) for landmark in d])
    data_flattened.append(flattened_landmarks)

# Convert the flattened data to a numpy array
data_flattened = np.array(data_flattened)

# Split data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(data_flattened, labels, test_size=0.2, shuffle=True, stratify=labels)

# Initialize and train the RandomForestClassifier
model_rf = RandomForestClassifier()
model_rf.fit(x_train, y_train)
y_predict_rf = model_rf.predict(x_test)
score_rf = accuracy_score(y_predict_rf, y_test)
print('{}% of samples were classified correctly by Random Forest!'.format(score_rf * 100))

# Initialize and train the DecisionTreeClassifier
model_dt = DecisionTreeClassifier() 
model_dt.fit(x_train, y_train)
y_predict_dt = model_dt.predict(x_test) 
score_dt = accuracy_score(y_predict_dt, y_test) 
print('{}% of samples were classified correctly by Decision Tree!'.format(score_dt * 100))

# Initialize and train the LogisticRegression
model_lr = LogisticRegression(max_iter=1000)
model_lr.fit(x_train, y_train)
y_predict_lr = model_lr.predict(x_test)
score_lr = accuracy_score(y_predict_lr, y_test)
print('{}% of samples were classified correctly by Logistic Regression!'.format(score_lr * 100))

# Initialize and train the KNeighborsClassifier
model_knn = KNeighborsClassifier(n_neighbors=5)
model_knn.fit(x_train, y_train)
y_predict_knn = model_knn.predict(x_test)
score_knn = accuracy_score(y_predict_knn, y_test)
print('{}% of samples were classified correctly by K-Nearest Neighbors!'.format(score_knn * 100))

# Save the trained models
with open('model.p', 'wb') as f:
    pickle.dump({
        'random_forest': model_rf,
        'decision_tree': model_dt,
        'logistic_regression': model_lr,
        'knn': model_knn
    }, f)
 
