import pandas as pd
import re
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense, SpatialDropout1D
from tensorflow.keras.callbacks import EarlyStopping

# file path
file_path = 'E:\\3790testenvironment\\python\\nlp\\IMDB Dataset.csv'

# load to the dataset
imdb_data = pd.read_csv(file_path)

# data cleaning
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

# pre-doing
tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(X_train)
X_train_tokens = tokenizer.texts_to_sequences(X_train)
X_test_tokens = tokenizer.texts_to_sequences(X_test)

# filling
max_len = 500
X_train_pad = pad_sequences(X_train_tokens, maxlen=max_len)
X_test_pad = pad_sequences(X_test_tokens, maxlen=max_len)

# create the RNN model
model = Sequential()
model.add(Embedding(input_dim=5000, output_dim=128, input_length=max_len))
model.add(SpatialDropout1D(0.2))
model.add(SimpleRNN(100, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# train the model
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

history = model.fit(
    X_train_pad, y_train,
    validation_split=0.2,
    epochs=10,
    batch_size=64,
    callbacks=[early_stopping]
)


loss, accuracy = model.evaluate(X_test_pad, y_test)
print(f'RNN模型准确率: {accuracy:.4f}')

# show the accuracy
plt.figure(figsize=(10, 5))
plt.plot(history.history['accuracy'], label='训练准确率')
plt.plot(history.history['val_accuracy'], label='验证准确率')
plt.title('RNN模型训练和验证准确率')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# show the loose, and the validation loss
plt.figure(figsize=(10, 5))
plt.plot(history.history['loss'], label='训练损失')
plt.plot(history.history['val_loss'], label='验证损失')
plt.title('RNN模型训练和验证损失')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
