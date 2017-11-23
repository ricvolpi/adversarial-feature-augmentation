import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np

import sys
import os
import cPickle

import utils

class TrainOps(object):

    def __init__(self, model, batch_size=64, pretrain_epochs=100000, train_feature_generator_iter=40000, 
                 mnist_dir='./data/mnist', log_dir='./logs', model_save_path='./model', 
		 pretrained_feature_extractor='feature_extractor', pretrained_feature_generator='feature_generator'):
        
        self.model = model
        self.batch_size = batch_size
        
	# Number of iterations for Step 0, 1, 2.
	self.pretrain_epochs = pretrain_epochs
        self.train_feature_generator_iter = train_feature_generator_iter
        
	# Dataset directory
        self.mnist_dir = mnist_dir
        
	self.model_save_path = model_save_path
	 
	self.pretrained_feature_extractor = os.path.join(self.model_save_path,pretrained_feature_extractor)
	self.pretrained_feature_generator = os.path.join(self.model_save_path,pretrained_feature_generator)
	
	self.log_dir = log_dir
        
	self.config = tf.ConfigProto()
        self.config.gpu_options.allow_growth=True
	
    def load_mnist(self, image_dir, split='train'):
        print ('Loading MNIST dataset.')
	
	image_file = 'train.pkl' if split=='train' else 'test.pkl'
        image_dir = os.path.join(image_dir, image_file)
        with open(image_dir, 'rb') as f:
            mnist = cPickle.load(f)
        images = mnist['X'] / 127.5 - 1
        labels = mnist['y']
		
        return images, np.squeeze(labels).astype(int)

    def train_feature_extractor(self):
	
	print 'Pretraining feature extractor.'
	    
	images, labels = self.load_mnist(self.mnist_dir, split='train')
	test_images, test_labels = self.load_mnist(self.mnist_dir, split='test')
	 	
        # build a graph
        model = self.model
        model.build_model()
	
        with tf.Session(config=self.config) as sess:
            tf.global_variables_initializer().run()
            saver = tf.train.Saver()
	    
            summary_writer = tf.summary.FileWriter(logdir=self.log_dir, graph=tf.get_default_graph())

	    t = 0

	    for i in range(self.pretrain_epochs):
		
		print 'Epoch',str(i)
		
		for start, end in zip(range(0, len(images), self.batch_size), range(self.batch_size, len(images), self.batch_size)):
		    
		    t+=1
		       
		    feed_dict = {model.images: images[start:end], model.labels: labels[start:end]}  
		    
		    sess.run(model.train_op, feed_dict) 

		    if t%100==0:

			rand_idxs = np.random.permutation(images.shape[0])[:1000]
			
			summary, loss, accuracy = sess.run(fetches=[model.summary_op, model.loss, model.accuracy], feed_dict={model.images: images[rand_idxs], model.labels: labels[rand_idxs]})
			summary_writer.add_summary(summary, t)
			
			print 'Step: [%d/%d] loss: [%.4f] accuracy: [%.4f]'%(t+1, self.pretrain_epochs*len(images)/self.batch_size, loss, accuracy)
			
		print 'Saving'
		saver.save(sess, self.pretrained_feature_extractor)
	    
    def train_feature_generator(self):
	
	print 'Training sampler.'
        
	images, labels = self.load_mnist(self.mnist_dir, split='train')
	labels = utils.one_hot(labels, 10)
    
        # build a graph
        model = self.model
        model.build_model()

	batch_size = self.batch_size
	noise_dim = 100
	epochs = 5000

        with tf.Session(config=self.config) as sess:
	    
            # initialize variables
            tf.global_variables_initializer().run()
            
	    # restore feature extractor trained on Step 0
            print ('Loading pretrained feature extractor.')
            variables_to_restore = slim.get_model_variables(scope='feature_extractor')
            restorer = tf.train.Saver(variables_to_restore)
            restorer.restore(sess, self.pretrained_feature_extractor)
	    print 'Loaded'
            
            summary_writer = tf.summary.FileWriter(logdir=self.log_dir, graph=tf.get_default_graph())
            saver = tf.train.Saver()
	    
	    t = 0
	    
	    while(t <= self.train_feature_generator_iter):
		
		for start, end in zip(range(0, len(images), batch_size), range(batch_size, len(images), batch_size)):
		    
		    t += 1

		    Z_samples = utils.sample_Z(batch_size, noise_dim, 'uniform')

		    feed_dict = {model.noise: Z_samples, model.images: images[start:end], model.labels: labels[start:end]}
	    
		    avg_D_fake = sess.run(model.logits_fake, feed_dict)
		    avg_D_real = sess.run(model.logits_real, feed_dict)
		    
		    sess.run(model.d_train_op, feed_dict)
		    sess.run(model.g_train_op, feed_dict)
		    
		    if (t+1) % 50 == 0:
			summary, dl, gl = sess.run([model.summary_op, model.d_loss, model.g_loss], feed_dict)
			summary_writer.add_summary(summary, t)
			print ('Step: [%d/%d] d_loss: %.6f g_loss: %.6f avg_D_fake: %.2f avg_D_real: %.2f ' \
				   %(t+1, int(epochs*len(images) /batch_size), dl, gl, avg_D_fake.mean(), avg_D_real.mean()))
			
		    if (t+1) % 5000 == 0:  
			saver.save(sess, self.pretrained_feature_generator) 


if __name__=='__main__':

    from Model import Model
    model = Model(mode='train_feature_extractor', learning_rate=0.0003)
    op = TrainOps(model)
    op.train_feature_generator
