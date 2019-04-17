# disable GPU for parallel execution
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


# Import libraries
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
import cv2
import csv
import PIL
import re
import glob

# Import layers
from tensorflow.keras.layers import Input, Flatten, Dense, Activation, Reshape, Dropout, Lambda, Cropping2D
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import concatenate as Concat


# Other imports
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
from PIL import Image

data_dir = "../../../../../train/" # The directory of the training data

# Hyperparameters
dropout = 0.35
batch_size = 12
epochs = 64
learning_rate = 0.0003
learning_rate_decay = 0.035

class Classifier:

    # Initializes the class "Classifier", setting all the necessary variables.
    def __init__(self, dropout = 0.25, batch_size = 32, epochs = 16, learning_rate = 0.001, learning_rate_decay = 0.01):
        self.dropout = dropout
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.learning_rate_decay = learning_rate_decay

        self.model = self.define_model(dropout)

    # Loads the image and label data.
    def load_data(self):
        data = [] # For labels: slot 1 is none, 2 is green, 3 is yellow, 4 is red
    
        training_image_names = glob.glob(data_dir + "*")

        for image_name in training_image_names:

            #image = cv2.imread(image_name).astype(np.float32)
            image = image_name

            if re.search("green", image_name):
                data.append([image, [0, 1, 0, 0]])
            elif re.search("yellow", image_name):
                data.append([image, [0, 0, 1, 0]])
            elif re.search("red", image_name):
                data.append([image, [0, 0, 0, 1]])
            else:
                data.append([image, [1, 0, 0, 0]]) # If an image is not of a green, yellow, or red light, assume it is not a light
        return data


    # Converts the image data to the format used for training. Takes the requied parameter "samples" (the unmodified image and label data), and the unrequired parameter "batch_size", which defaults to 32.
    def get_images(self, samples, batch_size = 32):
        features = []
        targets = []
        
        _ = features.clear()
        _ = targets.clear()
        while True:
            _ = random.shuffle(samples)
            for line in samples:
                image, label = line
                image = cv2.imread(image).astype(np.float32)


                image = cv2.resize(image, dsize=(300, 400), interpolation=cv2.INTER_CUBIC)
                
                features.append(np.array(image).astype(np.float32))
                targets.append(label)
                flipped_image = cv2.flip(image.astype(np.float32), 1)
                features.append(np.array(flipped_image).astype(np.float32))
                targets.append(label)
                
                if len(features) >= batch_size:
                    output = (np.array(features), np.array(targets))
                    yield output
                    _ = features.clear()
                    _ = targets.clear()

        return None

    #def get_images(self, samples, generator, batch_size = 32):
        #generated_data = generator.flow(samples, batch_size = batch_size)
        #return generated_data


    # Defines the model and returns it. Takes the parameter "dropout", which defaults to 0.25 .
    def define_model(self, dropout = 0.25):
        inp = Input(shape = (400, 300, 3))
        normal = Lambda(lambda x: (x / 256.) - 0.5)(inp)
    
    
        x = Conv2D(16, 6, activation = "relu", padding = "same")(normal)
        x = MaxPooling2D(pool_size=(1, 2))(x)
        x = Dropout(dropout)(x)
        
        x = Conv2D(64, 5, activation = "relu", padding = "same")(x)
        x = Dropout(dropout)(x)
        
        x = Conv2D(32, 4, activation = "relu", padding = "same")(x)
        x = MaxPooling2D(pool_size = (1, 2))(x)
        x = Dropout(dropout)(x)
        
        x = Flatten()(x)
        
        x = Dense(32, activation = "relu")(x)
        x = Dropout(dropout)(x)
        
        x = Dense(4, activation = "softmax")(x)
        
        model = Model(inp, x)
        
        return model


    # The function which trains the model
    def train(self):

        training_data = self.load_data()

        batches_per_epoch = int(len(training_data) / self.batch_size)
        if (len(training_data) % self.batch_size) > 0: batches_per_epoch += 1
        
        random.shuffle(training_data)
        
        valid_start_index = int(len(training_data) * 0.9)
        train_data = training_data[0:valid_start_index]
        valid_data = training_data[valid_start_index:-1]
        test_data = valid_data[0:int(len(valid_data)/2)]
        valid_data = valid_data[int(len(valid_data)/2):len(valid_data)]

        optimizer = Adam(lr = learning_rate, decay = self.learning_rate_decay)
        _ = self.model.compile(optimizer, 'categorical_crossentropy', ['accuracy'])
        
        checkpointer = ModelCheckpoint(filepath = 'model.h5', verbose=1)
        _ = self.model.fit_generator(self.get_images(train_data, self.batch_size), steps_per_epoch = batches_per_epoch, epochs = self.epochs,
            validation_data = self.get_images(valid_data, self.batch_size), validation_steps=5,
            callbacks=[checkpointer])
        
        test_x = []
        test_y = []
        
        test_size = 10
        for i in range(test_size):
            batch_x, batch_y = next(self.get_images(test_data, batch_size))
            
            _ = test_x.extend(batch_x)
            _ = test_y.extend(batch_y)
            
        test_loss = self.model.evaluate(np.array(test_x), np.array(test_y))[0]
        print("Test loss: {}".format(test_loss))
        
        _ = self.model.save('model.h5')






classifier = Classifier(dropout = dropout, 
        batch_size = batch_size, 
        epochs = epochs, 
        learning_rate = learning_rate, 
        learning_rate_decay = learning_rate_decay)

classifier.train()
