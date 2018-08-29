import numpy as np
import os
import six.moves.cPickle as pickle
from PIL import Image
from tensorflow.examples.tutorials.mnist import input_data


def resize_images(image_arrays, size=[32, 32]):
    image_arrays = (image_arrays * 255).astype('uint8')

    resized_image_arrays = np.zeros([image_arrays.shape[0]] + size)
    for i, image_array in enumerate(image_arrays):
        image = Image.fromarray(image_array)
        resized_image = image.resize(size=size, resample=Image.ANTIALIAS)

        resized_image_arrays[i] = np.asarray(resized_image)

    return np.expand_dims(resized_image_arrays, 3)


def download_and_process_mnist():

    if not os.path.exists('./data/mnist'):
        os.makedirs('./data/mnist')

    mnist = input_data.read_data_sets(train_dir='./data/mnist')

    train = {'X': resize_images(mnist.train.images.reshape(-1, 28, 28)),
             'y': mnist.train.labels}

    test = {'X': resize_images(mnist.test.images.reshape(-1, 28, 28)),
            'y': mnist.test.labels}

    with open('./data/mnist/train.pkl', 'wb') as f:
        pickle.dump(train, f, 3)

    with open('./data/mnist/test.pkl', 'wb') as f:
        pickle.dump(test, f, 3)


if __name__ == "__main__":
    download_and_process_mnist()
