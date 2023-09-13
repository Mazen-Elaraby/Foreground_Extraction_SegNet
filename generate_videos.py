import cv2
import os, glob

def generate_video(img_list, output_path):
    
    #setting up video properties
    sample_image = cv2.imread(img_list[0])
    height, width, _ = sample_image.shape
    frame_rate = 30 #there is not an inherent frame rate provided by the CDnet2014 datset

    #setting up video writer
    video_path = os.path.join(output_path, scene + '.mp4')
    out = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*"mp4v"), frame_rate, (width, height))

    #writing video to disk
    for frame in img_list:
        img = cv2.imread(frame)
        out.write(img)

    out.release()

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

num_frames = 25

raw_dataset_dir = 'CDnet2014_dataset'

results_dir = os.path.join('VGG16_SegNet', 'results' + str(num_frames))

vid_results_dir = os.path.join('VGG16_SegNet', 'video_results' + str(num_frames))

for category, scene_list in dataset.items():
    for scene in scene_list:
        print ('\n->>> ' + category + ' / ' + scene)

        vid_dir = os.path.join(vid_results_dir, category, scene)

        #load sequence of original images
        scene_path = os.path.join(raw_dataset_dir, category, scene, 'input')

        org_img_list = glob.glob(os.path.join(scene_path,'*.jpg')) #input list
        #setting up output path
        org_vid_dir = os.path.join(vid_dir, 'Original')
        if not os.path.exists(org_vid_dir):
            os.makedirs(org_vid_dir)

        #generating original video
        generate_video(org_img_list, org_vid_dir)

        #load sequence of binary masks
        masks_input_path = os.path.join(results_dir, category, scene)

        sg_img_list = glob.glob(os.path.join(masks_input_path,'*.png')) #input list
        #setting up output path
        sg_vid_dir = os.path.join(vid_dir, 'Segmented')
        if not os.path.exists(sg_vid_dir):
            os.makedirs(sg_vid_dir)

        #generating segmented video
        generate_video(sg_img_list, sg_vid_dir)
