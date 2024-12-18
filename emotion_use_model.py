from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image #requires "pillow" module
from pathlib import Path
import numpy as np

categories = ["angry","disgust","fear","happy","neutral","sad","surprise"]
wrong = 0
right = 0

# Load the model we trained
model = load_model('emotion_model_j16.h5')
#Next, we’ll loop through all PNG image files in the current folder and load each one.
for f in sorted(Path("Images/demo").glob("*.jpg")):

    # Load an image file to test
    image_to_test = image.load_img(str(f), target_size=(48, 48), color_mode = "grayscale")

    # Convert the image data to a numpy array suitable for Keras
    image_to_test = image.img_to_array(image_to_test)

    # Normalize the image the same way we normalized the training data (divide all numbers by 255)
    image_to_test /= 255

    # Add a fourth dimension to the image since Keras expects a list of images
    list_of_images = np.expand_dims(image_to_test, axis=0)

    # Make a prediction using the bird model
    results = model.predict(list_of_images)

    print(f)

    predictions = ""
    highest_prediction = 0
    predicted_category = ""
    likelihood = 0


    #iterate through each category result
    for i in range(0,len(results[0])):
        likelihood = results[0][i]
        if likelihood > highest_prediction:
            highest_prediction = likelihood
            predicted_category = categories[i]

        predictions = predictions + f"{categories[i]} : {likelihood:.2f}. "

    if predicted_category in str(f):
        right = right + 1
    else:
        wrong = wrong + 1

    print(predicted_category)
    print(predictions)

print("correct: " + str(right))
print("incorrect: " + str(wrong))
print("accuracy: " + str(right/(right+wrong)))
