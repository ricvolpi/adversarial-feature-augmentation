import numpy as np
import os
import cPickle
from PIL import Image
from tensorflow.examples.tutorials.mnist import input_data




def resize_images(image_arrays, size=[32,32]):
    image_arrays = (image_arrays * 255).astype('uint8')
    
    resized_image_arrays = np.zeros([image_arrays.shape[0]]+size)
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
        
    with open('./data/mnist/train.pkl','w') as f:
	cPickle.dump(train,f,cPickle.HIGHEST_PROTOCOL)
    
    with open('./data/mnist/test.pkl','w') as f:
	cPickle.dump(test,f,cPickle.HIGHEST_PROTOCOL)
	
if __name__ == "__main__":
    download_and_process_mnist()
    
