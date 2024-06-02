import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score, classification_report

# path of dataset
dataset_dir = 'E:/3790testenvironment/python/face/2/archive'

# size of images
img_height, img_width = 92, 112

# load the data
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

# load the data
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

# split the dataset and the set for testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# doing PCA
pca = PCA(n_components=100, whiten=True, random_state=42)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

# use LDA
lda = LinearDiscriminantAnalysis()
lda.fit(X_train_pca, y_train)
y_pred = lda.predict(X_test_pca)

# calculate
accuracy = accuracy_score(y_test, y_pred)
print(f'Validation Accuracy: {accuracy}')

# print it out
print(classification_report(y_test, y_pred, zero_division=1))
