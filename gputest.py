from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())
 
"""
from keras import backend as K
K.tensorflow_backend._get_available_gpus()

import keras
config = tf.ConfigProto( device_count = {'GPU': 1 , 'CPU': 56} ) 
sess = tf.Session(config=config) 
keras.backend.set_session(sess)
"""