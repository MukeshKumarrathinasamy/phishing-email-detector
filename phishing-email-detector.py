from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix

# Sample dataset
emails = [
    "Win money now click here",
    "Your account has been hacked",
    "Claim your free prize",
    "Meeting scheduled for tomorrow",
    "Project report attached",
    "Team lunch at 1 PM",
    "Verify your bank account immediately",
    "Congratulations you won lottery"
]

labels = [
    "Phishing",
    "Phishing",
    "Phishing",
    "Safe",
    "Safe",
    "Safe",
    "Phishing",
    "Phishing"
]

# Convert text to numbers
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(emails)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, labels, test_size=0.3, random_state=42
)

# Train model
model = MultinomialNB()
model.fit(X_train, y_train)

# Predictions
predictions = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, predictions)

print("Accuracy:", accuracy)

# Confusion Matrix
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, predictions))

# Test custom email
test_email = ["Click here to claim free reward"]

test_data = vectorizer.transform(test_email)

result = model.predict(test_data)

print("\nEmail:", test_email[0])
print("Prediction:", result[0])
