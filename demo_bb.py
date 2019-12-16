#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 11 17:11:54 2019

@author: asabater
"""

# python /home/asabater/projects/bitbraindemo/CenterNet/src/demo.py ctdet --demo /home/asabater/projects/Egocentric_object_detection/test_images/video_1.mp4 --load_model /home/asabater/projects/bitbraindemo/CenterNet/models/ctdet_coco_dla_2x.pth


import sys

sys.path.append('/home/asabater/projects/bitbraindemo/SiamMask/')
sys.path.append('/home/asabater/projects/bitbraindemo/SiamMask/experiments/siammask_sharp/')
from SiamMask.experiments.siammask_sharp.custom import Custom
from SiamMask.utils.load_helper import load_pretrain
from SiamMask.tools.test import siamese_init, siamese_track

		
# %%

import sys
# sys.path.append('./CenterNet/')
# sys.path.append('/home/asabater/projects/bitbraindemo/CenterNet/')
# sys.path.append('/home/asabater/projects/bitbraindemo/CenterNet/src/lib/')
# sys.path.append('/home/asabater/projects/bitbraindemo/CenterNet/src/lib/models/')
import CenterNet.src._init_paths
from CenterNet.src.lib.detectors.ctdet import CtdetDetector
from CenterNet.src.lib.opts import opts
from CenterNet.src.lib.utils_center_net.debugger import coco_class_name
# sys.path.remove('./CenterNet/')
# sys.path.remove('./CenterNet/src/lib/')


import json
import cv2
import numpy as np
import time


# %%

import colorsys

def get_colors(num_classes):
	hsv_tuples = [(x / num_classes, 1., 1.)
				  for x in range(num_classes)]
	colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
	colors = list(
		map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
			colors))
	np.random.seed(10101)  # Fixed seed for consistent colors across runs.
	np.random.shuffle(colors)  # Shuffle colors to decorrelate adjacent classes.
	np.random.seed(None)  # Reset seed to default.
	
	return colors


def print_box(img, results, class_names, score_thr, colors):
	num_classes = len(class_names)
	for num_class in range(1, num_classes+1):
		for bbox in results[num_class]:
			if bbox[4] > score_thr:
			 
				cat = int(num_class)-1
				conf = bbox[4]
				bbox = np.array(bbox[:4], dtype=np.int32)
				c = colors[cat]
			 
				txt = '{}{:.2}'.format(coco_class_name[cat], conf)
				font = cv2.FONT_HERSHEY_SIMPLEX
				cat_size = cv2.getTextSize(txt, font, 0.5, 2)[0]
				cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), c, 2)
			 
				cv2.rectangle(img,
						(bbox[0], bbox[1] - cat_size[1] - 2),
						(bbox[0] + cat_size[0], bbox[1] - 2), c, -1)
				cv2.putText(img, txt, (bbox[0], bbox[1] - 2), 
					  font, 0.5, (0, 0, 0), thickness=1, lineType=cv2.LINE_AA)
		 


# %%

# Initialize CenterNet
model_path = '/home/asabater/projects/bitbraindemo/CenterNet/models/ctdet_coco_dla_2x.pth'
# model_path = '/home/asabater/projects/bitbraindemo/CenterNet/models/ctdet_coco_hg.pth'
opt = opts().init(args=['ctdet', '--load_model', model_path])
detector = CtdetDetector(opt)


# %%


cfg = json.load(open('./SiamMask/experiments/siammask_sharp/config_davis.json'))
model_path = './SiamMask/experiments/siammask_sharp/SiamMask_DAVIS.pth'
device = 'cuda' 								 # cuda / cpu

siammask = Custom(anchors=cfg['anchors'])
siammask = load_pretrain(siammask, model_path)
siammask.eval().to(device)


# %%

"""
Detect
Select ROI
	Cancel detect
	track roi
	cancel track
Detect

TODO: wait for fps
"""


frame_size = 608
skip_frames = 2 		# process one frame each skip_frames

class_names = coco_class_name
colors = get_colors(len(class_names))
# video_path = '/home/asabater/projects/Egocentric_object_detection/test_images/video_1.mp4'
video_path = '/mnt/hdd/datasets/home_videos/2019_06_04_censored.avi'

font = cv2.FONT_HERSHEY_SIMPLEX
cam = cv2.VideoCapture(video_path)
fps = cam.get(cv2.CAP_PROP_FPS)

DETECT, DET_THR = True, 0.3					 			 	 # min score to show detection
TRACK, MIN_SCORE_TRACK, track_state = True, 0.8, False 		 # min score to show track

assert DETECT or TRACK

if DETECT: mode = 'detect'
elif TRACK: mode = 'track'
else: raise ValueError('No mode selected')

ti = time.time()
num_frame = 1
while True:
	
	b, frame = cam.read()
	if not b: break
	num_frame += 1
	if num_frame % skip_frames != 0: 
		continue

	frame = cv2.resize(frame, (frame_size,frame.shape[0]*frame_size//frame.shape[1]))


	if DETECT and mode == 'detect':
		# Run detection
		ret = detector.run(frame)
		print_box(frame, ret['results'], class_names, DET_THR, colors)
		track_state = False
		
	if TRACK and mode == 'track':
		# Run tracking
		if track_state != False:
			track_state_new = siamese_track(track_state, frame, mask_enable=True, refine_enable=True, device=device)  # track
			
			if track_state['score'] > MIN_SCORE_TRACK:
				# Update and show tracking state if it's above threshold
				track_state = track_state_new
				location = track_state['ploygon'].flatten()
				mask = track_state['mask'] > track_state['p'].seg_thr
			
				frame[:, :, 2] = (mask > 0) * 255 + (mask == 0) * frame[:, :, 2]
				cv2.polylines(frame, [np.int0(location).reshape((-1, 1, 2))], True, (0, 255, 0), 3)		
		
		
	key = cv2.waitKey(1)
	if key == ord('q'):
		# Exist video
		cv2.destroyAllWindows()
		break  # esc to quit
	elif TRACK and key == ord('r'):
		# Select ROI
		if mode == 'track': 
			# Change mode to detect
			print('Detecting')
			mode = 'detect'
		else:
			# Change mode to track and init tracker
			init_rect = cv2.selectROI('input', frame, False, False)
			x, y, w, h = init_rect
			if x == y == w == h == 0: continue
			target_pos = np.array([x + w / 2, y + h / 2])
			target_sz = np.array([w, h])
			track_state = siamese_init(frame, target_pos, target_sz, siammask, cfg['hp'], device=device)
			print('Tracking')
			mode = 'track'
	
	
 	# Print FPS
	tf = time.time()
	t_frame = tf-ti; ti = time.time()
	txt = 'FPS: {:.1f}'.format(1/t_frame)
	fps_size = cv2.getTextSize(txt, font, 0.5, 2)[0]
	cv2.rectangle(frame,
 			(0, fps_size[1] + 4), (fps_size[0], 0), (255,0,0), -1)
	cv2.putText(frame, txt, (0, fps_size[1] + 2), font, 0.5, (0, 0, 0), thickness=1, lineType=cv2.LINE_AA)
 	
 	# Wait until next frame time
	time.sleep(max(0, (1/fps*skip_frames) - t_frame))
 	
	cv2.imshow('input', frame)

cv2.destroyAllWindows()


