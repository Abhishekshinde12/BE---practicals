import numpy as np


# Generated Data
def generate_dummy_data(samples=100, features=10):
    # 100 eg with each 10 features.
    data = np.random.rand(samples, features)
    # Binary Labels for 100 length array
    labels = np.random.randint(0, 2, size=samples)
    return data, labels


# AIRS algorithm
# Detectors = set of solutions or antibodies that are used to recognize and bind to antigens (problems or patterns)
class AIRS:

    def __init__(self, num_detectors=10, hypermutation_rate=0.1):
        self.num_detectors = num_detectors
        self.hypermutation_rate = hypermutation_rate


    # takes features and labels
    # “Randomly choose num_detectors examples from my training data (without repeats) and keep them as the initial detectors.”
    def train(self, X, y):
        self.detectors = X[np.random.choice(len(X), self.num_detectors, replace=False)]


    # method to make predictions on new data
    def predict(self, X):
        # list to store predictions
        predictions = []
        # iterate on test samples
        for sample in X:
            # calculate euclidean distance between current sample and each detector
            distances = np.linalg.norm(self.detectors - sample, axis=1)
            # get prediction and store it
            prediction = int(np.argmin(distances))
            predictions.append(prediction)
        return predictions


# Generating data
data, labels = generate_dummy_data()


# Split data
# 80% data = train data, 20% = test data
split_ratio = 0.8
split_index = int(split_ratio * len(data))
train_data, test_data = data[:split_index], data[split_index:]
train_labels, test_labels = labels[:split_index], labels[split_index:]


# Initialize and train AIRS
airs = AIRS(num_detectors=10, hypermutation_rate=0.1)
airs.train(train_data, train_labels)


# Test AIRS on the test set
predictions = airs.predict(test_data)


# Evaluate accuracy
accuracy = np.mean(predictions == test_labels)
print(f"Accuracy: {accuracy}")