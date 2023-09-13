import numpy as np
import glob
import os

from sklearn.utils import compute_class_weight
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array

def get_data(train_dir, dataset_dir):

    void_label = -1. # non-ROI
      
    Y_list = glob.glob(os.path.join(train_dir, '*.png')) #ground_truth data
    X_list= glob.glob(os.path.join(dataset_dir, 'input','*.jpg'))
    
    if len(Y_list)<=0 or len(X_list)<=0:
        raise ValueError('System cannot find the dataset path or ground-truth path. Please give the correct path.')
    
    X_list_temp = []
    for i in range(len(Y_list)):
        Y_name = os.path.basename(Y_list[i])
        Y_name = Y_name.split('.')[0]
        Y_name = Y_name.split('gt')[1]
        for j in range(len(X_list)):
            X_name = os.path.basename(X_list[j])
            X_name = X_name.split('.')[0]
            X_name = X_name.split('in')[1]
            if (Y_name == X_name):
                X_list_temp.append(X_list[j])
                break
    X_list = X_list_temp
    
    if len(X_list)!=len(Y_list):
        raise ValueError('The number of X_list and Y_list must be equal.')
    
    #ensuring correspondence between ground truth and training sample
    X_list = sorted(X_list)
    Y_list = sorted(Y_list)
    
    # load training data
    X = []
    Y = []
    for i in range(len(X_list)):
        x = load_img(X_list[i])
        x = img_to_array(x)
        X.append(x)
        
        x = load_img(Y_list[i], grayscale = True)
        x = img_to_array(x)
        shape = x.shape
        x /= 255.0
        x = x.reshape(-1)
        idx = np.where(np.logical_and(x>0.25, x<0.8))[0] # find non-ROI
        if (len(idx)>0):
            x[idx] = void_label
        x = x.reshape(shape)
        x = np.floor(x)
        Y.append(x)
        
    X = np.asarray(X)
    Y = np.asarray(Y)
        
    #removing temporal connection/dependence
    idx = list(range(X.shape[0]))
    np.random.shuffle(idx)
    np.random.shuffle(idx)
    X = X[idx]
    Y = Y[idx]

    #calculating class weights to deal with imbalanced dataset
    pixel_weight = np.ones((Y.shape[0], (Y.shape[1] * Y.shape[2])))
    for i in range(Y.shape[0]):
        x = Y[i].reshape(-1)
        idx = np.where(x!=void_label)[0]
        if(len(idx)>0):
            z = x[idx]
        lb = np.unique(z) #  0., 1
        cls_weight = compute_class_weight(class_weight='balanced', classes=lb , y=z)
        class_0 = cls_weight[0]
        class_1 = cls_weight[1] if len(lb)>1 else 1.0

        pixel_weight[i][x==0] = class_0
        pixel_weight[i][x==1] = class_1

    Y = Y.reshape((Y.shape[0], (Y.shape[1] * Y.shape[2]) , Y.shape[3]))
    return [X, Y, pixel_weight]