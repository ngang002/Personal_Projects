import os
import keras
from keras.layers import Dense, Bidirectional, Dropout, LSTM
from tensorflow.keras import Input, Model
import tensorflow as tf
tfkl = tf.keras.layers

import header as HD

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
tf.get_logger().setLevel('ERROR')

class LSTMNet(keras.Model):
    def __init__(self, pixels, features, learning_rate,
                input_time_steps, output_timesteps, \
                seed=12345, name="LSTM_Net", **kwargs):
        super(LSTMNet, self).__init__(name=name, **kwargs)

        self.lstmnet_layers = []
        self.pixels = pixels
        self.features = features
        self.learning_rate = learning_rate
        self.input_timesteps  = HD._previous_timesteps
        self.output_timesteps = HD._next_timesteps  
        self.num_layers = 0

    def add_initialLayer(self, dropoutFrac=0.2):
        self.lstmnet_layers.append(Bidirectional(LSTM(units=self.input_timesteps, activation='tanh', input_shape=(self.pixels, self.features), \
                                            return_sequences=True, name='LSTM'+'0')))
        self.lstmnet_layers.append(Dropout(dropoutFrac, name='Drop'+'0'))

    def add_layer(self, layerNum, dropoutFrac=0.2):

        self.lstmnet_layers.append(Bidirectional(LSTM(units=self.input_timesteps, activation='tanh', return_sequences=True, name='LSTM'+f'{layerNum}')))
        self.lstmnet_layers.append(Dropout(dropoutFrac, name = 'Drop'+f'{layerNum}'))
        self.num_layers += 1

    def add_finalLayer(self):
        self.lstmnet_layers.append(LSTM(units=self.output_timesteps+self.input_timesteps, activation='tanh', name='LSTM'+f'Final'))
        self.lstmnet_layers.append(Dense(units=self.output_timesteps+self.input_timesteps, activation='relu', name='Dense1'))
        self.lstmnet_layers.append(Dense(units=self.output_timesteps, name='Dense2'))


    def summary(self):
        x = Input(shape=(self.pixels, self.features))
        model = Model(inputs=x, outputs=self.call(x))
        return model.summary()


    def call(self, inputs):
        x = inputs
        for li, layer in enumerate(self.lstmnet_layers.layers):
            layer._name = 'LSTMLayer'+str(li+1)
            x = layer(x)
        
        # x = self.prob(self.dense(x))
        
        return x
