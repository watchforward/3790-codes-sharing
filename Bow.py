import pandas as pd
import re
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# path of the fule
file_path = 'E:\\3790testenvironment\\python\\nlp\\IMDB Dataset.csv'

# load the dataset
imdb_data = pd.read_csv(file_path)

# data cleaning
def clean_text(text):
    text = re.sub(r'<.*?>', '', text)  #
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    text = text.lower()  # lowercase
    text = text.split()
    text = ' '.join(text)  # rebuild
    return text

# clean
imdb_data['review'] = imdb_data['review'].apply(clean_text)


imdb_data['sentiment'] = imdb_data['sentiment'].apply(lambda x: 1 if x == 'positive' else 0)

# split dataset
X = imdb_data['review']
y = imdb_data['sentiment']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# creat the model
vectorizer = CountVectorizer(max_features=5000)
X_train_bow = vectorizer.fit_transform(X_train)
X_test_bow = vectorizer.transform(X_test)

# training
model_bow = MultinomialNB()
model_bow.fit(X_train_bow, y_train)

y_pred_bow = model_bow.predict(X_test_bow)
accuracy_bow = accuracy_score(y_test, y_pred_bow)
precision_bow = precision_score(y_test, y_pred_bow)
recall_bow = recall_score(y_test, y_pred_bow)
f1_bow = f1_score(y_test, y_pred_bow)

print(f'BoW模型准确率: {accuracy_bow:.4f}')
print(f'BoW模型精确率: {precision_bow:.4f}')
print(f'BoW模型召回率: {recall_bow:.4f}')
print(f'BoW模型F1分数: {f1_bow:.4f}')

# show the results
metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
scores = [accuracy_bow, precision_bow, recall_bow, f1_bow]

plt.figure(figsize=(10, 5))
plt.bar(metrics, scores, color=['blue', 'green', 'red', 'purple'])
plt.ylim(0, 1)
plt.title('BoW Model Performance Metrics')
plt.xlabel('Metrics')
plt.ylabel('Score')
plt.show()
