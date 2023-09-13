import tensorflow as tf
import keras
import os
#import gc

from SegNet import SegNet
from get_data import get_data

physical_devices = tf.config.experimental.list_physical_devices('GPU')
print("Num GPUs Available: ", len(physical_devices))
tf.config.experimental.set_memory_growth(physical_devices[0], True)

def train(data, scene, mdl_path):
    
    #Defining Hyper-parameters
    lr = 1e-4
    val_split = 0.2
    max_epoch = 100
    batch_size = 1

    #initializing model
    img_shape = data[0][0].shape #(height, width, channels)
    model = SegNet(lr, img_shape)
    model = SegNet.init_model(model,0)
    #model.summary()

    #fitting model
    early = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=1e-4, patience=6, verbose=0, mode='auto')
    redu = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, verbose=1, mode='auto')
    model.fit(data[0], data[1], 
                      validation_split=val_split, 
                      epochs=max_epoch, batch_size=batch_size, 
                      callbacks=[redu, early], verbose=1, shuffle = True, sample_weight=data[2])
    
    model.save(mdl_path)
    #del model, data, early, redu


#Main Function
dataset = {
           'baseline':[
                   'highway', 
                   'pedestrians',
                   'office',
                   'PETS2006'
                   ],
           'cameraJitter':[
                   'badminton',
                   'traffic',
                   'boulevard', 
                   'sidewalk'
                   ],
           'badWeather':[
                   'skating', 
                   'blizzard',
                   'snowFall',
                   'wetSnow'
                   ],
            'dynamicBackground':[
                   'boats',
                   'canoe',
                   'fall',
                   'fountain01',
                   'fountain02',
                    'overpass'
                    ],
            'intermittentObjectMotion':[
                    'abandonedBox',
                    'parking',
                    'sofa',
                    'streetLight',
                    'tramstop',
                    'winterDriveway'
                    ],
            'lowFramerate':[
                    'port_0_17fps',
                    'tramCrossroad_1fps',
                    'tunnelExit_0_35fps',
                    'turnpike_0_5fps'
                    ],
            'nightVideos':[
                    'bridgeEntry',
                    'busyBoulvard',
                    'fluidHighway',
                    'streetCornerAtNight',
                    'tramStation',
                    'winterStreet'
                    ],
           'PTZ':[
                   'continuousPan',
                   'intermittentPan',
                   'twoPositionPTZCam',
                   'zoomInZoomOut'
                   ],
            'shadow':[
                    'backdoor',
                    'bungalows',
                    'busStation',
                    'copyMachine',
                    'cubicle',
                    'peopleInShade'
                    ],
           'thermal':[
                   'corridor',
                   'diningRoom',
                   'lakeSide',
                   'library',
                   'park'
                   ],
            'turbulence':[
                    'turbulence0',
                    'turbulence1',
                   'turbulence2',
                   'turbulence3'
                    ] 
}

num_frames = 25 # either 25 or 200 training frames
assert num_frames in [25,200], 'num_frames is incorrect.'

main_dir = 'VGG16_SegNet'
main_mdl_dir = os.path.join(main_dir, 'models' + str(num_frames))

for category, scene_list in dataset.items():
    mdl_dir = os.path.join(main_mdl_dir, category)
    if not os.path.exists(mdl_dir):
        os.makedirs(mdl_dir)

    for scene in scene_list:
        print ('Training ->>> ' + category + ' / ' + scene)
        
        train_dir = os.path.join('CDnet2014_train', category, scene + str(num_frames))
        dataset_dir = os.path.join('CDnet2014_dataset', category, scene)
        data = get_data(train_dir, dataset_dir)
        
        mdl_path = os.path.join(mdl_dir, 'mdl_' + scene + '.h5')
        train(data, scene, mdl_path)
        #del data
        
    #gc.collect()