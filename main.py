import tensorflow as tf
from model import Model
from trainOps import TrainOps
import os
import numpy.random as npr

flags = tf.app.flags
flags.DEFINE_string('mode', 'train_feature_extractor', "'train_feature_extractor' or 'train_feature_generator'")
flags.DEFINE_string('gpu', '0', "'0', '1', '2' or '3'")
FLAGS = flags.FLAGS

def main(_):
    
    GPU_ID = FLAGS.gpu
    
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152 on stackoverflow
    os.environ["CUDA_VISIBLE_DEVICES"] = str(GPU_ID)
    
    if not os.path.exists('./model'):
	os.makedirs('./model')
    if not os.path.exists('./logs'):
	os.makedirs('./logs')
    
    if FLAGS.mode == 'train_feature_extractor':
	model = Model(mode=FLAGS.mode, learning_rate=0.0003)
        op = TrainOps(model)	
	op.train_feature_extractor()
	
    elif FLAGS.mode == 'train_feature_generator':
	model = Model(mode=FLAGS.mode, learning_rate=0.0001)
        op = TrainOps(model)	
	op.train_feature_generator()
	
    elif FLAGS.mode == 'train_DIFA':
	model = Model(mode=FLAGS.mode, learning_rate=0.00001)
        op = TrainOps(model)	
	op.train_DIFA()
	
    elif FLAGS.mode == 'train_decoder':
	model = Model(mode=FLAGS.mode, learning_rate=0.0001)
        op = TrainOps(model)	
	op.train_decoder()
	
    else:
	print 'Unrecognized mode.'
	
        
if __name__ == '__main__':
    tf.app.run()



    

