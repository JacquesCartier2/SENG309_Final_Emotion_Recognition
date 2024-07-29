from tensorflow.keras.utils import image_dataset_from_directory

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, BatchNormalization

# Load data set
train, test = image_dataset_from_directory("Images\\train", labels='inferred', color_mode='grayscale', validation_split=0.25, subset='both', shuffle=True, seed=1234)

#reduced size for testing
train = train.take(5000)
test = test.take(1000)

# Create a model and add layers
# sequential means you will define the network layer by layer
model = Sequential()
# start with 1 standard convolutional layer
# 32 means create 32 convolutional filters each one is 3 X 3 (pretty standard)
# The padding parameter tells Keras what to do at the edges of the image.
# For weird historical reasons,
#"same" means that Keras will add padding at the edges of the images
# and "valid" means that Keras won’t add any padding (which is the default if you don’t specify anything).
# The first layer in Keras has a special input_shape parameter.
# This tells Keras what size the input data will be in the neural network.
# Since our training images are 32x32 pixels with three color channels (Red, Green, and Blue), the size is (32, 32, 3).
# activation function --> for image recognition use Relu
model.add(Conv2D(32, (3, 3), padding='same', input_shape=(256, 256, 1), activation="relu"))
model.add(Conv2D(32, (3, 3), activation="relu"))
# add maxpooling to downsample the data
# keep 1 value in each 2 X 2  -- mainly downsize the data to 25%
model.add(MaxPooling2D(pool_size=(2, 2)))
# add a BatchNormalization layer. This layer will continually normalize the data as it passes between layers. This just helps the model train a little faster.
model.add(BatchNormalization())
# common to add a Dropout after convolutional layers to help prevent overfitting
# a 25% Dropout means 25% of the data will be thrown away
# this saves on running time but more than 25% makes the network work harder to learn
model.add(Dropout(0.25))

# repeat same steps
model.add(Conv2D(64, (3, 3), padding='same', activation="relu"))
model.add(Conv2D(64, (3, 3), activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(BatchNormalization())
model.add(Dropout(0.25))

# There’s no right answer for how many filters to include in a convolutional layer. It requires guessing and checking.

# Now we’re ready to finish up the neural network definition by adding Dense layers that will map the convolutional features into either the “bird” class or “not bird” class.
# Flatten() call is required in Keras whenever you transition from convolutional layers to Dense layers.
model.add(Flatten())
model.add(Dense(512, activation="relu"))
model.add(Dropout(0.5))
# one output value ==> bird on not bird
# wait: what is sigmoid?
# The sigmoid function always produces a value between 0 and 1. That’s perfect for the output layer since we want to output a probability score.
model.add(Dense(1, activation="sigmoid"))

# Compile the model (build the NN)

# super important parameters:
# (1) lost/cost function to measure error, when you are predicting true or false always use binary_crossentropy
# there is a list you can review in keras documentation.
# (2) optimizer : optimization algorithm , our usual is Gradient descent. "adam" is a recent variant that works well for images
# (3) metrics:   metrics is a list of any other metrics we want to track during training. Since the loss value is just a number, it can be hard to interpret intuitively.
#                So I'll almost always track plain old accuracy, which is much easier to interpret than a loss function.

# TIP TIP TIP TIP TIP
#  (1) If you are classifying items into two categories, you’ll have one output node and use a sigmoid activation function on the output layer and a binary_crossentropy loss function.
#  (2) If you are classifying into more than two categories, you’ll have one output node for each possible category, you’ll use a softmax activation function and you’ll use a categorical_crossentropy loss function.
model.compile(
    loss='binary_crossentropy',
    optimizer="adam",
    metrics=['accuracy']
)

# Train the model

#First, we pass in the training data and the matching answers for each training example.
# (1) batch_size is how many images will be loaded into memory and considered at once during each step of the training process.
# If the batch size is too small, the neural network will never train since it won’t see enough data to get a good signal.
# If the batch size is too large, you’ll run out of memory.
# A batch size around 32 is usually a good tradeoff.
# (2) epochs is how many times we will loop through the entire training dataset before ending the training process.
# (3) validation_data lets us pass in a validation data set that Keras will automatically test after each full pass through the training data.
# (4) shuffle=True tells Keras to randomize the order of the input data it sees.
# This is super important. You always want to randomize the order of your training data unless you know for sure that you have it stored in random order already.

model.fit(
    x=train,
    batch_size=32,
    epochs=5,
    validation_data=(test),
    shuffle=True
)

# Save the trained model to a file so we can use it to make predictions later
model.save("emotion_model_test.h5")
