import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
# Load dataset
data = pd.read_csv("survey.csv")
# Encode categorical variables
label_encoder = LabelEncoder()
data['campaign_type_encoded'] = label_encoder.fit_transform(data['campaign_type'])
# Define features and target variable
X = data[['campaign_type_encoded', 'audience_age', 'engagement_metrics']]
y = data['campaign_success']
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Create a decision tree classifier
clf = DecisionTreeClassifier(random_state=42)
# Train the model on the training data
clf.fit(X_train, y_train)
# Make predictions on the test data
y_pred = clf.predict(X_test)
# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)