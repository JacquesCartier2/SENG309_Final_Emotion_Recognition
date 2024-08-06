from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, BatchNormalization, Rescaling
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator

image_size = 48

normalization_layer = Rescaling(1./255)
early_stop = EarlyStopping(monitor='loss', patience=5, min_delta = 0.01)
datagen_train= ImageDataGenerator(rescale = 1.0/255.0, width_shift_range = 0.1, height_shift_range = 0.1, rotation_range = 20, horizontal_flip = True)
datagen_val = ImageDataGenerator(rescale= 1.0/255)

#train images are augmented and normalized as the are read
train = datagen_train.flow_from_directory("Images\\train",target_size=(image_size,image_size),color_mode='grayscale', batch_size=128, class_mode='categorical', shuffle=True)

#test images are unchanged
test = datagen_val.flow_from_directory("Images\\validation",target_size=(image_size,image_size),color_mode='grayscale', batch_size=128, class_mode='categorical', shuffle=True)

model = Sequential()
#convolutional layer 1
model.add(Conv2D(64, (3, 3), padding='same', input_shape=(image_size, image_size, 1), activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(BatchNormalization())
model.add(Dropout(0.25))

#convolutional layer 2
model.add(Conv2D(128, (3, 3), padding='same', activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(BatchNormalization())
model.add(Dropout(0.25))

#convolutional layer 3
model.add(Conv2D(256, (3, 3), padding='same', activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(BatchNormalization())
model.add(Dropout(0.25))

#convolutional layer 4
model.add(Conv2D(512, (3, 3), padding='same', activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(BatchNormalization())
model.add(Dropout(0.25))

#flattening layer
model.add(Flatten())

#fully connected layer 1
model.add(Dense(512, activation="relu"))
model.add(Dropout(0.25))

#fully connected layer 2
model.add(Dense(1024, activation="relu"))
model.add(Dropout(0.25))

#output layer
model.add(Dense(7, activation="softmax"))

model.compile(
    loss='categorical_crossentropy',
    optimizer="adam",
    metrics=['accuracy']
)

def Train_Model():
    model.fit(
        x=train,
        batch_size=64,
        epochs=200,
        validation_data=(test),
        shuffle=True,
        callbacks = [early_stop]
    )

Train_Model()
model.save("emotion_model_test.h5")
