from sklearn import datasets
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import KFold

breast_cancer = datasets.load_breast_cancer()

# 569 entries
# 30 features
# Target: 0 (malign) or 1 (benign)

# Split the data into 5 folds
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Create a neural network
best_network = None
for train_index, test_index in kf.split(breast_cancer.data):
    entries_train, entries_test = breast_cancer.data[train_index], breast_cancer.data[test_index]
    targets_train, targets_test = breast_cancer.target[train_index], breast_cancer.target[test_index]
    
    neural_network = MLPClassifier(hidden_layer_sizes=(30,16), 
                                   max_iter=10000,
                                   activation='logistic',
                                   solver='adam',
                                   tol=0.000001,
                                   learning_rate_init=0.0005,
                                   random_state=42)
    neural_network.fit(entries_train, targets_train)
    
    if best_network is None or neural_network.score(entries_test, targets_test) > best_network.score(entries_test, targets_test):
        best_network = neural_network

print(f"Accuracy: {best_network.score(entries_test, targets_test)*100:.4f}%")
# Compare the prediction with the actual target and print the result
predictions = best_network.predict(entries_test)
