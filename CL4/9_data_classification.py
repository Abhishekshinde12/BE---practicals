from sklearn.datasets import lood_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Standardscaler 
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Load the tris dataset
iris = load_iris()

# Features

X = iris.data

# Target variable

y = iris.target

# Splitting the dataset into training and testing sets

X_train, X_test, y train, y_test =  train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# Initialize and train the logistic regression model
logreg = LogisticRegression()
logreg.fit(X_train_scaled, y_train)



# Predictions 
y_pred_train = logreg.predict(X_train_scaled)
y_pred_test = logreg.predict(X_test_scaled)

# Model evaluation
train_accuracy = accuracy_score(y_train, y_pred_train)
test_accuracy = accuracy_score(y_test, y_pred_test)

print("Training Accuracy:", train_accuracy)
print("Testing Accuracy:", test_accuracy)


# Additional Evaluation Metrics
print("Classification Report on Test Data :")
print(classification_report(y_test, y_pred_test))