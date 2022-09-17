from scipy.io import loadmat
import numpy as np

from tensorflow.keras import utils as np_utils
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split
import h5py
import pdb
import os
from keras.models import Model
from keras import backend as K

from sklearn.metrics import confusion_matrix
from keras.utils import plot_model
from Utils import PrintScore, ConfusionMatrix

os.environ["CUDA_VISIBLE_DEVICES"] = '1,2'
config = K.tf.ConfigProto()
config.gpu_options.allow_growth = True
session = K.tf.Session(config=config)
K.set_session(session)

dataIn = loadmat('/redhdd/changhongli/SSA_Feature_selected/testResults.mat', mdict=None)

Y_all = np.asarray(dataIn["Y_all"], dtype=np.float32)
Y_h = np.asarray(dataIn["Y_H"], dtype=np.float32)
Y_f = np.asarray(dataIn["Y_F"], dtype=np.float32)
Y_s = np.asarray(dataIn["Y_S"], dtype=np.float32)

Pre_all = np.asarray(dataIn["Pre_all"], dtype=np.float32)
Pre_h = np.asarray(dataIn["Pre_h"], dtype=np.float32)
Pre_f = np.asarray(dataIn["Pre_f"], dtype=np.float32)
Pre_s = np.asarray(dataIn["Pre_s"], dtype=np.float32)

Pre_S_all = np.asarray(dataIn["Pre_S_all"], dtype=np.float32)
Pre_S_h = np.asarray(dataIn["Pre_S_h"], dtype=np.float32)
Pre_S_f = np.asarray(dataIn["Pre_S_f"], dtype=np.float32)
Pre_S_s = np.asarray(dataIn["Pre_R_s"], dtype=np.float32)

Pre_R_all = np.asarray(dataIn["Pre_S_all"], dtype=np.float32)
Pre_R_h = np.asarray(dataIn["Pre_S_h"], dtype=np.float32)
Pre_R_f = np.asarray(dataIn["Pre_S_f"], dtype=np.float32)
Pre_R_s = np.asarray(dataIn["Pre_S_s"], dtype=np.float32)



WEIGHTS_PATH1 = r'/redhdd/changhongli/SSA_Feature_selected/BLDAall/'
PrintScore(Y_all, Pre_all, savePath=WEIGHTS_PATH1)
ConfusionMatrix(Y_all, Pre_all, classes=['Normal', 'Depression'], savePath=WEIGHTS_PATH1)

WEIGHTS_PATH2 = r'/redhdd/changhongli/SSA_Feature_selected/BLDAhcue/'
PrintScore(Y_h, Pre_h, savePath=WEIGHTS_PATH2)
ConfusionMatrix(Y_h, Pre_h, classes=['Normal', 'Depression'], savePath=WEIGHTS_PATH2)

WEIGHTS_PATH3 = r'/redhdd/changhongli/SSA_Feature_selected/BLDAfcue/'
PrintScore(Y_f, Pre_f, savePath=WEIGHTS_PATH3)
ConfusionMatrix(Y_f, Pre_f, classes=['Normal', 'Depression'], savePath=WEIGHTS_PATH3)

WEIGHTS_PATH4 = r'/redhdd/changhongli/SSA_Feature_selected/BLDAscue/'
PrintScore(Y_s, Pre_s, savePath=WEIGHTS_PATH4)
ConfusionMatrix(Y_s, Pre_s, classes=['Normal', 'Depression'], savePath=WEIGHTS_PATH4)

WEIGHTS_PATHH1 = r'/redhdd/changhongli/SSA_Feature_selected/BLDASSAall/'
PrintScore(Y_all, Pre_S_all, savePath=WEIGHTS_PATHH1)
ConfusionMatrix(Y_all, Pre_S_all, classes=['Normal', 'Depression'], savePath=WEIGHTS_PATHH1)

WEIGHTS_PATHH2 = r'/redhdd/changhongli/SSA_Feature_selected/BLDASSAhcue/'
PrintScore(Y_h, Pre_S_h, savePath=WEIGHTS_PATHH2)
ConfusionMatrix(Y_h, Pre_S_h, classes=['Normal', 'Depression'], savePath=WEIGHTS_PATHH2)

WEIGHTS_PATHH3 = r'/redhdd/changhongli/SSA_Feature_selected/BLDASSAfcue/'
PrintScore(Y_f, Pre_S_f, savePath=WEIGHTS_PATHH3)
ConfusionMatrix(Y_f, Pre_S_f, classes=['Normal', 'Depression'], savePath=WEIGHTS_PATHH3)

WEIGHTS_PATHH4 = r'/redhdd/changhongli/SSA_Feature_selected/BLDASSAscue/'
PrintScore(Y_s, Pre_S_s, savePath=WEIGHTS_PATHH4)
ConfusionMatrix(Y_s, Pre_S_s, classes=['Normal', 'Depression'], savePath=WEIGHTS_PATHH4)

WEIGHTS_PATHF1 = r'/redhdd/changhongli/SSA_Feature_selected/BLDARSSAall/'
PrintScore(Y_all, Pre_R_all, savePath=WEIGHTS_PATHF1)
ConfusionMatrix(Y_all, Pre_R_all, classes=['Normal', 'Depression'], savePath=WEIGHTS_PATHF1)

WEIGHTS_PATHF2 = r'/redhdd/changhongli/SSA_Feature_selected/BLDARSSAhcue/'
PrintScore(Y_h, Pre_R_h, savePath=WEIGHTS_PATHF2)
ConfusionMatrix(Y_h, Pre_R_h, classes=['Normal', 'Depression'], savePath=WEIGHTS_PATHF2)

WEIGHTS_PATHF3 = r'/redhdd/changhongli/SSA_Feature_selected/BLDARSSAfcue/'
PrintScore(Y_f, Pre_R_f, savePath=WEIGHTS_PATHF3)
ConfusionMatrix(Y_f, Pre_R_f, classes=['Normal', 'Depression'], savePath=WEIGHTS_PATHF3)

WEIGHTS_PATHF4 = r'/redhdd/changhongli/SSA_Feature_selected/BLDARSSAscue/'
PrintScore(Y_s, Pre_R_s, savePath=WEIGHTS_PATHF4)
ConfusionMatrix(Y_s, Pre_R_s, classes=['Normal', 'Depression'], savePath=WEIGHTS_PATHF4)

