from scipy.io import loadmat
import numpy as np
from EEGModels import EEGNet
from tensorflow.keras import utils as np_utils
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split
import h5py
import pdb
import os
from keras.models import Model
from keras import backend as K
from dataProcess import DataGenerate
import time
from sklearn.metrics import confusion_matrix
from keras.utils import plot_model
from Utils import PrintScore, ConfusionMatrix
os.environ["CUDA_VISIBLE_DEVICES"] = '1,2'
config = K.tf.ConfigProto()
config.gpu_options.allow_growth = True
session = K.tf.Session(config=config)
K.set_session(session)


chans, samples, kernels = 128, 125, 1

def readData(data_txt, SUB):
    data, label, subject = [], [], []
    with open(data_txt) as f:
        sub = 0
        lines = f.readlines()
        for l in lines:
            sub += 1
            if sub not in SUB:
                continue
            dataIn = loadmat(l.split()[0], mdict=None)
            #tempData, tempLabel = dataIn["EEGdata"], dataIn["Label"]
            tempData = np.asarray(dataIn["testS_X"], dtype=np.float32)
            tempLabel = np.asarray(dataIn["testS_Y"], dtype=np.float32)
            tempLabel = tempLabel[0,:]
            tempSub = sub * np.ones(tempLabel.shape[0])
            if not len(data):
                data, label, subject = tempData, tempLabel, tempSub
            else:
                data = np.concatenate((data, tempData), axis=0)
                label = np.concatenate((label, tempLabel), axis=0)
                subject = np.concatenate((subject, tempSub), axis=0)
        label = label.reshape(label.shape[0])
    return data, label, subject

all_sub_acc = []
all_c_matrix = []
start = time.time()
Test_Y=[]
Probs=[]
for n in range(53):
    SUB = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29,
           30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53]
    #
    print(SUB[n])
    dataDir = '/redhdd/changhongli/End2End_Depression_recognition/dataDirs.txt'
    #pdb.set_trace()
    data, label, subject = readData(dataDir, SUB)
    print('-----read data end-----')
    #pdb.set_trace()
    M = DataGenerate(data=data, label=label, subject=subject, testSub=SUB[n])
    #train_X, X_validate, train_Y, Y_validate, test_X, test_Y = DataGenerate(data=data, label=label, subject=subject, testSub=SUB[n])
    train_X, X_validate, train_Y, Y_validate, test_X, test_Y = M.train_X, M.X_validate, M.train_Y, M.Y_validate, M.test_data, M.test_label
    acc_max = 0
    print("--------------dataprocess end -----------------")
    model = EEGNet(nb_classes=2, Chans=128, Samples=samples,
                    dropoutRate=0.5, kernLength=100, F1=8, D=2, F2=16,
                    dropoutType='Dropout')
    #
    # # compile the model and set the optimizers
    #model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    #
    WEIGHTS_PATH = r'/redhdd/changhongli/End2End_Depression_recognition/arl-eegmodels-master/scue/%s.h5' % (SUB[n])
    #checkpointer = ModelCheckpoint(filepath=WEIGHTS_PATH, verbose=1, save_best_only=True)
    #
    #model.fit(train_X, train_Y, batch_size=64, epochs=50, verbose=2, validation_data=(X_validate, Y_validate),
     #          shuffle=True, callbacks=[checkpointer])
    model.summary()
    plot_model(model, to_file="/redhdd/changhongli/End2End_Depression_recognition/arl-eegmodels-master/model.png",
               show_shapes=True)
    model.load_weights(WEIGHTS_PATH)

    probs = model.predict(test_X)
    preds = probs.argmax(axis=-1)
    acc_max = np.mean(preds == test_Y)
    print("Classification accuracy: %f " % (acc_max))

    c_matrix = confusion_matrix(test_Y, preds)
    all_sub_acc.append(acc_max)
    all_c_matrix.append(c_matrix)
    Test_Y = np.concatenate((Test_Y, test_Y), axis=0)
    Probs = np.concatenate((Probs, preds), axis=0)
    #Test_Y.append(test_Y)
    #Probs.append(preds)
    # layer_model1 = Model(inputs=model.layers[0].input, outputs=model.get_layer('flatten').get_output_at(0))
    # layer_model1 = layer_model1.predict(test_X)
    pdb.set_trace()
end = time.time()
acc_mean = round(sum(all_sub_acc) / 53, 4) * 100
acc_std = round(np.std(all_sub_acc), 4) * 100
all_matrix = sum(all_c_matrix)
WEIGHTS_PATH1 = r'/redhdd/changhongli/End2End_Depression_recognition/arl-eegmodels-master/scue/'
acc = PrintScore(Test_Y, Probs, savePath=WEIGHTS_PATH1)
ConfusionMatrix(Test_Y, Probs, classes=['Normal', 'Depression'],
                                        savePath=WEIGHTS_PATH1)
print('***********************************************************')
print(all_sub_acc)
print('time is {}'.format(time.strftime('%Y%m%d_%H:%M:%S', time.localtime())))
print('mean/std = {}/{}, time is {}'.format(acc_mean, acc_std, time.strftime('%Y%m%d_%H:%M:%S', time.localtime())))
txtName =  '/redhdd/changhongli/End2End_Depression_recognition/arl-eegmodels-master/scue/'
txtName += time.strftime('%Y%m%d_%H:%M:%S', time.localtime()) + '.txt'
with open(txtName, 'a+') as t_f:
    t_f.write('\n\ntime is: ' + time.strftime('%Y%m%d_%H:%M:%S', time.localtime()))
    t_f.write('\n\nconfusion matrix is:\n' + str(all_matrix))
    t_f.write('\nmean/std acc = %.2f/%.2f' % (acc_mean, acc_std))
    t_f.write('\n\nall_sub_acc:\n' + str(all_sub_acc))