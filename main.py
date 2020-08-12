"""Main file to test the OSDN."""

from keras.models import load_model
import numpy as np
from utils.openmax import get_train_test, create_model
from utils.openmax import image_show, compute_activation, compute_openmax


def main():
    np.random.seed(12345)

    # Step 1: Train a CNN model for the dataset you choice

    # We skip this step as we already trained MNIST dataset

    # Step 2: Load the trained model
    model = load_model('data/MNIST_CNN.h5')

    # Step 3: Load the training data you trained the DNN model
    data = get_train_test()
    x_train, x_test, y_train, y_test = data

    # Step 4: Create a mean activation vector (MAV) and do weibull fit model
    # Send the trained model as argument and the test data you want to predict
    # using the OSDN
    create_model(model, data)

    # Step 5: Pass the sample to compute activation and openmax
    # Test on the openmax activation function to images from
    # the same distribution
    N = 5   # Testing N number of test images
    for i in range(N):
        random_char = np.random.randint(0, len(x_test))

        test_x1 = x_test[random_char]
        test_y1 = y_test[random_char]

        # Compute fc8 activation for the given image
        activation = compute_activation(model, test_x1)

        # Compute openmax activation
        softmax, openmax = compute_openmax(model, activation)

        print('Actual Label: ', np.argmax(test_y1))
        print('Prediction Softmax: ', softmax)
        if openmax == 10:
            openmax = 'Unknown'
        print('Prediction openmax: ', openmax)
        labels = (np.argmax(test_y1), softmax, openmax)

        # Show the image
        image_show(test_x1, labels)

    # Step 6: Test the trained openmax to images from different distribution
    # Opemax should return unknow to these types of images
    # We are testing Nepali numerals on the model trained on MNIST
    from utils.nepali_characters import split
    import keras

    train_x, train_y, test_x, text_y, valid_x, valid_y = split(0.9, 0.05, 0.05)
    train_y = keras.utils.to_categorical(train_y, 10)
    data = x_train, x_test, y_train, y_test

    for i in range(N):
        random_char = np.random.randint(0, len(train_y))

        test_x1 = train_x[random_char]
        test_x1 = np.reshape(test_x1, (32, 32)).T
        test_y1 = train_y[random_char]

        # Compute fc8 activation for the given image
        activation = compute_activation(model, test_x1)

        # Compute openmax activation
        softmax, openmax = compute_openmax(model, activation)

        print('Actual Label: ', np.argmax(test_y1))
        print('Prediction Softmax: ', softmax)
        if openmax == 10:
            openmax = 'Unknown'
        print('Prediction Openmax: ', openmax)
        labels = (np.argmax(test_y1), softmax, openmax)
        # Draw image
        image_show(test_x1, labels)


if __name__ == "__main__":
    main()