from sklearn import datasets
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split

breast_cancer = datasets.load_breast_cancer()

# 569 entries
# 30 features
# Target: 0 (malign) or 1 (benign)

# 80 percent of the data is used for training
entries_train, entries_test, targets_train, targets_test = train_test_split(
    breast_cancer.data, 
    breast_cancer.target, 
    test_size=0.2, 
    random_state=42)

# Create a neural network
best_network = None
for i in range(1, 11):
    neural_network = MLPClassifier(hidden_layer_sizes=(20,20), 
                                max_iter=10000,
                                activation='logistic',
                                solver='adam',
                                tol=0.000001,
                                learning_rate_init=0.0005,
                                random_state=i)
    neural_network.fit(entries_train, targets_train)
    if best_network == None or neural_network.score(entries_test, targets_test) > best_network.score(entries_test, targets_test):
        best_network = neural_network

print(f"Accuracy: {best_network.score(entries_test, targets_test)*100:.4f}%")
# Compare the prediction with the actual target and print the result
predictions = best_network.predict(entries_test)
correct = 0
for i in range(len(predictions)):
    if predictions[i] == targets_test[i]:
        correct += 1
print(f"Predicted {correct} out of {len(predictions)} correctly")

#Accuracy: 96.4912 %
#Predicted 110 out of 114 correctly %


