'''
author: Karan Chauhan
github: @Karan-Chauhan19
organization: L.J University
'''
import os

class Config() :

    def __init__(self):
        
        #current working directory
        self.CWD = os.getcwd()  #get current working directory
        
        #training and validuation dataset path before storing dataset into .npy format  
        self.TRAIN_IMAGE_PATH = os.path.join(self.CWD,'docs/Images') #set training data directory
        self.VAL_IMAGE_PATH = os.path.join(self.CWD,'docs/VAL_data') #set validation data directory
        self.TRAIN_CAPTIONS_PATH = os.path.join(self.CWD,'docs/captions.txt') 
        self.VAL_CAPTIONS_PATH = os.path.join(self.CWD,'docs/val_captions.csv')
        self.INPUT = os.path.join(self.CWD, 'docs/') 

        #training and validation dataset path after storing dataset into .npy format
        self.TRAIN_IMAGES_DIR = '' # automatically set path for the directory contains stacked images in .npy format (training)
        self.TRAIN_CAPTIONS_DIR = '' # automatically set path for the directory contains mask images in .npy format (training)
        self.VAL_IMAGES_DIR = '' # set path for the directory contains stacked captions in .npy format (validation)
        self.VAL_CAPTIONS_DIR = '' # set path for the directory contains mask captionsn in .npy format (validation)
        self.PATH_TO_SAVE_TRAINED_MODEL = os.path.join(self.CWD,'saved_models/') # set path to save trained model

        #model training parameters
        self.BATCH_SIZE = 16 # set batch size for model training
        self.MAX_EPOCHS = 50 # set maximum number of epochs for model training
        self.LEARNING_RATE = 0.001 # set learning rate for model training
        self.TRANSFORM = True # set boolean value for applying augmentation techniques for training set and techniques are horizontal flip and vertical flip

    def printConfiguration(self):
        """
        This funtion is print all configuration related to paths and training parameters of model

        Parameters:
        - (None)

        Returns:
        - (None)
        """
        print("-"*20)
        print("Configurations:")
        print("-"*20)
        print(f"Current working directory: {self.CWD}, \nTraining image path: {self.TRAIN_IMAGE_PATH}, \n"
              f"Training captions path: {self.TRAIN_CAPTIONS_PATH}, \nValidation image path: {self.VAL_IMAGE_PATH}, \n "
              f"Validation caption path: {self.VAL_CAPTIONS_PATH}, \n"\
              f"Path_to_save_trained_model: {self.PATH_TO_SAVE_TRAINED_MODEL},\nBatch_size: {self.BATCH_SIZE},\nMax_epochs: {self.MAX_EPOCHS},\n"
              f"Learning_rate: {self.LEARNING_RATE},\nTransform/Data_augmentation: {self.TRANSFORM}")
        