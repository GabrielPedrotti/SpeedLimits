import numpy as np
import cv2
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import seaborn as sns

path = 'Imagens'
imageDimensions = (32, 32, 3)
batch_size = 50
steps_per_epoch = 1000
epochs = 20

images = []
classNo = []
pastas = os.listdir(path)
print("Total Classes Detected:", len(pastas))
numClasses = len(pastas)

for x in range(0, numClasses):
    myPicList = os.listdir(path + "/" + str(x))
    for y in myPicList:
        curImg = cv2.imread(path + "/" + str(x) + "/" + y)
        images.append(curImg)
        classNo.append(x)
    print(x, end=" ")

images = np.array(images)
classNo = np.array(classNo)

def grayscale(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img

def equalize(img):
    img = cv2.equalizeHist(img)
    return img

def preprocessing(img):
    img = grayscale(img)
    img = equalize(img)
    img = img / 255
    return img

images = np.array(list(map(preprocessing, images)))
images = images.reshape(images.shape[0], images.shape[1], images.shape[2], 1)

classNo = to_categorical(classNo, numClasses)

# Carregar modelo base lenet
def create_model():
    model = Sequential([
        Conv2D(32, (5, 5), activation='relu', input_shape=(imageDimensions[0], imageDimensions[1], 1)),
        MaxPooling2D(2, 2),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Dropout(0.5),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(numClasses, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Cross-Validation
kf = KFold(n_splits=5, shuffle=True, random_state=1)
accuracy_scores = []

for train_index, test_index in kf.split(images):
    X_train, X_test = images[train_index], images[test_index]
    y_train, y_test = classNo[train_index], classNo[test_index]
    
    # Aumentar imagens com data generator

    dataGen = ImageDataGenerator(width_shift_range=0.1,
                                 height_shift_range=0.1,
                                 zoom_range=0.2,
                                 shear_range=0.1,
                                 rotation_range=10)
    dataGen.fit(X_train)
    
    model = create_model()
    history = model.fit(dataGen.flow(X_train, y_train, batch_size=batch_size), 
                        steps_per_epoch=steps_per_epoch, 
                        epochs=epochs, 
                        validation_data=(X_test, y_test), 
                        shuffle=1)
    
    score = model.evaluate(X_test, y_test, verbose=0)
    accuracy_scores.append(score[1])
    print('accuracy:', score[1])

print('Cross-validation acurácia:', accuracy_scores)
print('Acurácia Média:', np.mean(accuracy_scores))
print('Desvio Padrão:', np.std(accuracy_scores))

conf_matrix = confusion_matrix(y_test.argmax(axis=1), model.predict(X_test).argmax(axis=1))

print("Matriz de Confusão:")
print(conf_matrix)

plt.figure(figsize=(10,7))
sns.heatmap(conf_matrix, annot=True, fmt="d")
plt.xlabel('Valores Previstos')
plt.ylabel('Valores Verdadeiros')
plt.title('Matriz de Confusão')
plt.show()

# Salvando o modelo
model.save('test2.h5')
