import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.manifold import TSNE

# the path of dataset
dataset_dir = 'E:/3790testenvironment/python/face/2/archive'

# images size
img_height, img_width = 92, 112

# load data
def load_data(dataset_dir):
    X = []
    y = []
    label_map = {}
    current_label = 0

    for person in os.listdir(dataset_dir):
        person_dir = os.path.join(dataset_dir, person)
        if os.path.isdir(person_dir):
            if person not in label_map:
                label_map[person] = current_label
                current_label += 1
            for img_name in os.listdir(person_dir):
                if img_name.endswith('.pgm'):
                    img_path = os.path.join(person_dir, img_name)
                    img = Image.open(img_path)
                    img = img.resize((img_width, img_height))
                    img = img.convert('L')
                    img_array = np.array(img).flatten()
                    X.append(img_array)
                    y.append(label_map[person])

    X = np.array(X)
    y = np.array(y)
    return X, y, label_map

# load data
X, y, label_map = load_data(dataset_dir)

# show some figures
fig, axes = plt.subplots(3, 5, figsize=(10, 6))
axes = axes.flatten()
for i, ax in enumerate(axes):
    img = X[i].reshape(img_height, img_width)
    ax.imshow(img, cmap='gray')
    ax.axis('off')
plt.tight_layout()
plt.show()

# split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

pca = PCA(n_components=100, whiten=True, random_state=42)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

classifier = LogisticRegression(max_iter=1000, random_state=42)
classifier.fit(X_train_pca, y_train)
y_pred = classifier.predict(X_test_pca)

# find accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f'Validation Accuracy: {accuracy}')

# print the report
print(classification_report(y_test, y_pred, zero_division=1))

# show pca results
def plot_embedding(X, y, title=None):
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) / (x_max - x_min)

    plt.figure(figsize=(10, 10))
    for i in range(X.shape[0]):
        plt.text(X[i, 0], X[i, 1], str(y[i]),
                 color=plt.cm.Set1(y[i] / 10.),
                 fontdict={'weight': 'bold', 'size': 9})
    plt.xticks([])
    plt.yticks([])
    if title is not None:
        plt.title(title)

#
tsne = TSNE(n_components=2, init='pca', random_state=0)
X_train_tsne = tsne.fit_transform(X_train_pca)

plot_embedding(X_train_tsne, y_train, "t-SNE embedding of the training data (PCA + Logistic Regression)")
plt.show()
