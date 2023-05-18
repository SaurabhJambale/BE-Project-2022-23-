import pickle
X = pickle.load(open('features.pkl', 'rb'))
y = pickle.load(open('labels.pkl', 'rb'))

X = X/255
X = X.reshape(-1, 60, 60, 1)

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten

model = Sequential()

model.add(Conv2D(64, (3,3), activation = 'relu'))
model.add(MaxPooling2D((3,3)))

model.add(Conv2D(64, (3,3), activation = 'relu'))
model.add(MaxPooling2D((3,3)))

model.add(Flatten())

model.add(Dense(128, input_shape = X.shape[1:], activation = 'relu'))

model.add(Dense(3, activation = 'softmax'))


model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(X, y, epochs=3, validation_split=0.1,)

