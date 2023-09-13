import numpy as np
import tensorflow as tf
import glob, os

from keras.models import *
from PIL import Image

from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img

from model_utils import c_acc, c_loss

physical_devices = tf.config.experimental.list_physical_devices('GPU')
print("Num GPUs Available: ", len(physical_devices))
tf.config.experimental.set_memory_growth(physical_devices[0], True)

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

th = 0.7 #threshold for binary segmentation mask
num_frames = 25

raw_dataset_dir = 'CDnet2014_dataset'

main_mdl_dir = os.path.join('VGG16_SegNet', 'models' + str(num_frames))

results_dir = os.path.join('VGG16_SegNet', 'results' + str(num_frames))

for category, scene_list in dataset.items():
    for scene in scene_list:
        print ('\n->>> ' + category + ' / ' + scene)
        
        mask_dir = os.path.join(results_dir, category, scene)
        if not os.path.exists(mask_dir):
            os.makedirs(mask_dir)

        #Load path of files
        scene_path = os.path.join(raw_dataset_dir, category, scene)
        scene_input_path = os.path.join(scene_path, 'input')

        in_list = np.asarray(glob.glob(os.path.join(scene_input_path,'*.jpg'))) #input list

        if (in_list is None):
            raise ValueError('Input Path List Is Empty')
        
        #storing image dimension for later reshaping
        sample_img = load_img(in_list[0])
        img_t = img_to_array(sample_img)
        img_shape = img_t.shape

        # path of ROI to exclude non-ROI
        ROI_file = os.path.join(scene_path, 'ROI.bmp')
        
        roi = load_img(ROI_file, grayscale=True)
        roi = img_to_array(roi)
        roi = roi.reshape(-1) # to 1D
        idx = np.where(roi == 0.)[0] # get the non-ROI, black area

        #setting up data loader
        batches = tf.keras.preprocessing.image_dataset_from_directory(
        scene_path,
        label_mode=None,
        class_names=None,
        batch_size=4,
        image_size=(img_shape[0], img_shape[1]),
        shuffle=False
        ) #input batch

        #loading model
        mdl_path = os.path.join(main_mdl_dir, category , 'mdl_' + scene + '.h5')

        model = load_model(mdl_path, custom_objects={'c_loss': c_loss, 'c_acc': c_acc})

        counter = 0
        for batch in batches:
            #running predictions
            results = model.predict(batch, batch_size=4, verbose=1)
            results = np.squeeze(results, axis=2)
            #thresholding to create a binary mask
            results = np.where(results > th, 1, 0)

            for result in results:
                    #0 out non-ROI pixels
                    if (len(idx)>0):
                         result[idx] = 0

                    #reshaping back to image dimensions
                    img_arr = result.reshape(img_shape[0],img_shape[1])
                    
                    #saving images
                    fname = os.path.basename(in_list[counter]).replace('in','bin').replace('jpg','png')
                    img = Image.fromarray((img_arr*255).astype(np.uint8))
                    img = img.convert("L")
                    img.save(os.path.join(mask_dir, fname))
                    counter = counter + 1