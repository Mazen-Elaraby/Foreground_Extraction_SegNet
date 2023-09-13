import tensorflow as tf
import numpy as np
import os, glob

from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from PIL import Image


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

raw_dataset_dir = 'CDnet2014_dataset'

for category, scene_list in dataset.items():
    for scene in scene_list:
        scene_path = os.path.join(raw_dataset_dir, category, scene)
        scene_input_path = os.path.join(scene_path, 'input')
        
        in_list = np.asarray(glob.glob(os.path.join(scene_input_path,'*.jpg'))) #input list

        #check if video needs resizing
        sample_img = load_img(in_list[0])
        sample_img_t = img_to_array(sample_img)
        img_shape = (sample_img_t.shape[0], sample_img_t.shape[1])

        num_enc = 4 #number of encoders/decoders
        rows_flag = False if img_shape[0]%(2**num_enc)==0 else True
        cols_flag = False if img_shape[1]%(2**num_enc)==0 else True

        #resizing input images
        if (rows_flag or cols_flag):
            print('ofa7y7')
            #generating new dimensions
            new_height = int( (2**num_enc) * (np.ceil(img_shape[0] / (2**4))) )
            new_width = int((2**num_enc) * (np.ceil(img_shape[1] / (2**4))))
            new_dims = (new_height, new_width, img_shape[2])
            
            #setting up data loader
            batch_sz = 32
            batches = tf.keras.preprocessing.image_dataset_from_directory(
                scene_path,
                label_mode=None,
                class_names=None,
                batch_size=batch_sz,
                image_size=(img_shape[0], img_shape[1]),
                shuffle=False
            )

            #resizing video frames
            counter = 0
            resized_input_path = os.path.join(scene_path, 'resized_input')
            if not os.path.exists(resized_input_path):
                os.makedirs(resized_input_path)

            for batch in batches:

                resized_batch = tf.image.resize(
                    batch,
                    (new_dims[0], new_dims[1]),
                    method=tf.image.ResizeMethod.BILINEAR,
                    preserve_aspect_ratio=False,
                    antialias=False,
                    name=None
                )
                #saving batch of processed images
                for frame in resized_batch:
                    fname = os.path.basename(in_list[counter])
                    r_img = Image.fromarray(np.asarray(frame).astype('uint8'))
                    r_img.save(os.path.join(resized_input_path, fname))
                    counter = counter + 1
        
        #resizing corresponding ROI.bmp files
        ROI_file = os.path.join(scene_path, 'ROI.bmp')
        roi = load_img(ROI_file, grayscale=True)
        roi = img_to_array(roi)
        roi_shape = (roi.shape[0], roi.shape[1])
        if (not (roi_shape == img_shape)):
            
            resized_roi = tf.image.resize(
                roi,
                img_shape, #should be editted to new_dims if it is the first run of the script
                method=tf.image.ResizeMethod.BILINEAR,
                preserve_aspect_ratio=False,
                antialias=False,
                name=None
            )

            roi = np.squeeze(roi)
            resized_roi = np.squeeze(np.asarray(resized_roi))
            resized_roi = np.where(resized_roi > 128, 255, 0) #thresholding as ROI file is a binary mask

            roi_img = Image.fromarray(roi.astype('uint8'))
            resized_roi_img = Image.fromarray(resized_roi.astype('uint8'))

            os.remove(ROI_file)

            resized_roi_img.save(ROI_file)



#The following code is used to delete pre-resizing input directories & rename 'resized_input'
#directories to 'input' as consistent naming is required for down-stream tasks
#The code will be commented out and should be used only if your directory tree follows the above-mentioned structure

'''
paths = 'CDnet2014_dataset/*/*/resized_input'
in_list = glob.glob(paths)

fldr_paths = []
for path in in_list:
    fldr_paths.append(os.path.split(path)[0])

#deleting pre-resizing folders
import shutil

for path in fldr_paths:

    ip_fldr = os.path.join(path, 'input')
    
    if (os.path.isdir(ip_fldr)):
        shutil.rmtree(ip_fldr)

#renaming 'resized_input' folder to 'input' as consistent naming is required for down-stream tasks
for path in fldr_paths:
    src = os.path.join(path, 'resized_input')
    dst = os.path.join(path, 'input')
    
    os.rename(src,dst)

#checking
new_paths = 'CDnet2014_dataset/*/*/input'
new_list = glob.glob(new_paths)
if (len(new_list)==53):
    print('Deleting & Renaming Was Successful.')
else:
    print('Deleting & Renaming Failed.')
'''