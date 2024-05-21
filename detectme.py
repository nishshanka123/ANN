# Import the keras libraries and the packages
'''
import keras
from keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

print("Keras version is: --------------------: ", keras.__version__)

# Initialize the Convolution Neural Network
classifier = Sequential();
# Add the first convolution layers
# get the 64*64 pixel image, each pixel has 3 values and convert into 2D image data. this is the input layer
# also specify the activation function, relu is used in this case.
classifier.add(Conv2D(32, (3, 3), input_shape=(64,64,3), activation='relu'))
# Add second convolution layer, the hidden layers. Mapping and reducing
classifier.add(Conv2D(32,(3,3), activation = 'relu'))
# Add third layer, reducing
classifier.add(MaxPooling2D(pool_size=(2,2)))
# Add one for layer to flatten the data
classifier.add(Flatten())
# now we have 1 dimention data, add more layers
# specify the classifier and the activation.
classifier.add(Dense(units=128, activation='relu'))     # 128(64+64) pixel of input ??
classifier.add(Dense(units=1, activation='sigmoid'))    # single output

# add the optimization for reverse propagation
# compile the CNN
classifier.compile(optimizer='adam', loss='binary-crossentropy', metricy=['accuracy'])
'''

# compatible code for Intel-Irish graphic code.
import os
import keras
from keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# Force TensorFlow to use CPU
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

print("Keras version is: --------------------: ", keras.__version__)

# Initialize the Convolution Neural Network
classifier = Sequential()

# Add the first convolution layers
classifier.add(Conv2D(32, (3, 3), input_shape=(64, 64, 3), activation='relu'))

# Add second convolution layer, the hidden layers. Mapping and reducing
classifier.add(Conv2D(32, (3, 3), activation='relu'))

# Add third layer, reducing
classifier.add(MaxPooling2D(pool_size=(2, 2)))

# Add one more layer to flatten the data
classifier.add(Flatten())

# Now we have 1-dimensional data, add more layers
classifier.add(Dense(units=128, activation='relu'))  # 128 units

# Single output layer
classifier.add(Dense(units=1, activation='sigmoid'))

# Add the optimization for reverse propagation
# Compile the CNN with the correct 'metrics' argument
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Now you can train your model
# For example, assuming you have your training data ready:
# classifier.fit(X_train, y_train, epochs=10, batch_size=32)