import pandas as pd
import re
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout

# path of the file
file_path = 'E:\\3790testenvironment\\python\\nlp\\IMDB Dataset.csv'

# load
imdb_data = pd.read_csv(file_path)

# data cleaning
def clean_text(text):
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    text = text.lower()
    text = text.split()
    text = ' '.join(text)
    return text

# clean word data
imdb_data['review'] = imdb_data['review'].apply(clean_text)


imdb_data['sentiment'] = imdb_data['sentiment'].apply(lambda x: 1 if x == 'positive' else 0)

# split dataset
X = imdb_data['review']
y = imdb_data['sentiment']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# setting
max_features = 5000
max_len = 500
embedding_dim = 128

# dta pre-doing
tokenizer = Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(X_train)
X_train_seq = tokenizer.texts_to_sequences(X_train)
X_test_seq = tokenizer.texts_to_sequences(X_test)
X_train_pad = pad_sequences(X_train_seq, maxlen=max_len)
X_test_pad = pad_sequences(X_test_seq, maxlen=max_len)

# create LSTM
model_lstm = Sequential([
    Embedding(max_features, embedding_dim, input_length=max_len),
    LSTM(128, dropout=0.2, recurrent_dropout=0.2),
    Dense(1, activation='sigmoid')
])

model_lstm.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# training
history = model_lstm.fit(X_train_pad, y_train, epochs=5, batch_size=64, validation_split=0.2)


loss, accuracy_lstm = model_lstm.evaluate(X_test_pad, y_test)
print(f'LSTM模型准确率: {accuracy_lstm:.4f}')

# show accuracy
plt.figure(figsize=(10, 5))
plt.plot(history.history['accuracy'], label='训练准确率')
plt.plot(history.history['val_accuracy'], label='验证准确率')
plt.title('LSTM模型训练和验证准确率')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# show loss
plt.figure(figsize=(10, 5))
plt.plot(history.history['loss'], label='训练损失')
plt.plot(history.history['val_loss'], label='验证损失')
plt.title('LSTM模型训练和验证损失')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
