import pandas as pd
import re
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# path of the file
file_path = 'E:\\3790testenvironment\\python\\nlp\\IMDB Dataset.csv'

# load the dataset
imdb_data = pd.read_csv(file_path)

# data cleaning function
def clean_text(text):
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    text = text.lower()
    text = text.split()
    text = ' '.join(text)
    return text

# clean the data
imdb_data['review'] = imdb_data['review'].apply(clean_text)


imdb_data['sentiment'] = imdb_data['sentiment'].apply(lambda x: 1 if x == 'positive' else 0)

# split the dataset
X = imdb_data['review']
y = imdb_data['sentiment']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# create the TF-IDF mdoel
tfidf_vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)


model_tfidf = MultinomialNB()
model_tfidf.fit(X_train_tfidf, y_train)


y_pred_tfidf = model_tfidf.predict(X_test_tfidf)
accuracy_tfidf = accuracy_score(y_test, y_pred_tfidf)
precision_tfidf = precision_score(y_test, y_pred_tfidf)
recall_tfidf = recall_score(y_test, y_pred_tfidf)
f1_tfidf = f1_score(y_test, y_pred_tfidf)

print(f'TF-IDF模型准确率: {accuracy_tfidf:.4f}')
print(f'TF-IDF模型精确率: {precision_tfidf:.4f}')
print(f'TF-IDF模型召回率: {recall_tfidf:.4f}')
print(f'TF-IDF模型F1分数: {f1_tfidf:.4f}')

# show the results
metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
scores = [accuracy_tfidf, precision_tfidf, recall_tfidf, f1_tfidf]

plt.figure(figsize=(10, 5))
plt.bar(metrics, scores, color=['blue', 'green', 'red', 'purple'])
plt.ylim(0, 1)
plt.title('TF-IDF Model Performance Metrics')
plt.xlabel('Metrics')
plt.ylabel('Score')
plt.show()
