import pandas as pd
import re

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

data = pd.read_csv("mbti_1.csv", engine='python', on_bad_lines='skip')

print("Dataset shape:", data.shape)
print("\nPersonality types:\n")
print(data["type"].value_counts())

#clean the text
def clean_text(text):
    text = text.lower()

    #remove links
    text = re.sub(r"http\S+", " ", text)

    #remove punctuation and numbers
    text = re.sub(r"[^a-z\s]", "", text)

    #remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text

data["posts"] = data["posts"].apply(clean_text)

#features and labels
X = data["posts"]
y = data["type"]

# text->numbers
vectorizer = TfidfVectorizer(
    stop_words="english",
    max_features=5000
)

X = vectorizer.fit_transform(X)

#train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# we train the model
model = LogisticRegression(
    max_iter=3000
)

model.fit(X_train, y_train)

#predictions
predictions = model.predict(X_test)

print("\nAccuracy:", round(accuracy_score(y_test, predictions) * 100, 2), "%")

print("\nConfusion Matrix:\n")
print(confusion_matrix(y_test, predictions))

print("\nClassification Report:\n")
print(classification_report(y_test, predictions))

#example of prediction
example = """
I enjoy spending time alone reading books,
thinking about philosophy and programming.
Large parties make me tired.
"""

example = clean_text(example)

example_vector = vectorizer.transform([example])

prediction = model.predict(example_vector)[0]

print("\nExample personality prediction:", prediction)

#common words for one type
feature_names = vectorizer.get_feature_names_out()

chosen = "INTP"

if chosen in model.classes_:
    index = list(model.classes_).index(chosen)

    weights = model.coef_[index]

    top = weights.argsort()[-15:][::-1]

    print(f"\nTop words associated with {chosen}:\n")

    for i in top:
        print(f"{feature_names[i]:15} {weights[i]:.3f}")

# predicts your own text
while True:
    text = input("\nWrite something (or type quit): ")

    if text.lower() == "quit":
        break

    text = clean_text(text)

    result = model.predict(vectorizer.transform([text]))[0]

    print("Predicted type:", result)
