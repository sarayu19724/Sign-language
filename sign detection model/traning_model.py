import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier


with open('data.pickle', 'rb') as f:
    data_dir = pickle.load(f)


data = np.array(data_dir['data'])
labels = np.array(data_dir['labels'])


data_flattened = []
for d in data:
    flattened_landmarks = np.concatenate([landmark.reshape(-1) for landmark in d])
    data_flattened.append(flattened_landmarks)


data_flattened = np.array(data_flattened)


x_train, x_test, y_train, y_test = train_test_split(data_flattened, labels, test_size=0.2, shuffle=True, stratify=labels)
models = {
    'random_forest': RandomForestClassifier(),
    'decision_tree': DecisionTreeClassifier(),
    'logistic_regression': LogisticRegression(max_iter=1000, class_weight='balanced'),
    'knn': KNeighborsClassifier(n_neighbors=5)
}

results = {}
for name, model in models.items():
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    acc = accuracy_score(y_test, y_pred)
    results[name] = acc
    print(f"{name} accuracy: {acc*100:.2f}%")

# Save models
with open('model.p', 'wb') as f:
    pickle.dump(models, f)




 
