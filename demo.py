#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 11 17:11:54 2019

@author: asabater
"""

import sys

sys.path.append('./SiamMask/')
sys.path.append('./SiamMask/experiments/siammask_sharp/')
from SiamMask.experiments.siammask_sharp.custom import Custom
from SiamMask.utils.load_helper import load_pretrain
from SiamMask.tools.test import siamese_init, siamese_track

import CenterNet.src._init_paths
from CenterNet.src.lib.detectors.ctdet import CtdetDetector
from CenterNet.src.lib.opts import opts
from CenterNet.src.lib.utils_center_net.debugger import coco_class_name

import json
import cv2
import numpy as np
import time
import colorsys
import argparse
import os


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
		 

def main():
	
	parser = argparse.ArgumentParser()
	parser.add_argument("video", help='input video')
	parser.add_argument("--weights_det", default='./CenterNet/models/ctdet_coco_dla_2x.pth', help='CenterNet weights')
	parser.add_argument("--weights_track", default='./SiamMask/experiments/siammask_sharp/SiamMask_DAVIS.pth', help='SiamMask weights')
	parser.add_argument("--gpu", default='0', help='gpu id to use')
	parser.add_argument("--skip_frames", default=1, type=int, help='process one frame for each skip_frames')
	parser.add_argument("--frame_size", default=608, type=int, help='size of the frame to show')
	parser.add_argument("--no_detect", action='store_true', help='True to perform Detection')
	parser.add_argument("--detect_thr", default=0.4, help='Detection threshold')
	parser.add_argument("--no_track", action='store_true', help='True to perform tracking')
	parser.add_argument("--track_thr", default=0.8, help='Tracking threshold')
	args = parser.parse_args()
	
	
	os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
	
	# Initialize CenterNet
	cn_opts = opts().init(args=['ctdet', '--load_model', args.weights_det])
	detector = CtdetDetector(cn_opts)
	
	
	device = 'cuda' 				# if args.gpu != '' else 'cpu'
	cfg = json.load(open('./SiamMask/experiments/siammask_sharp/config_davis.json'))
	siammask = Custom(anchors=cfg['anchors'])
	siammask = load_pretrain(siammask, args.weights_track)
	siammask.eval().to(device)

	class_names = coco_class_name
	colors = get_colors(len(class_names))
	
	font = cv2.FONT_HERSHEY_SIMPLEX
	cam = cv2.VideoCapture(args.video)
	fps = cam.get(cv2.CAP_PROP_FPS)
	
	track_state = False
	
	detect = not args.no_detect; track = not args.no_track
	assert detect or track
	
	if detect: mode = 'detect'
	elif track: mode = 'track'
	else: raise ValueError('No mode selected')

	
	ti = time.time()
	num_frame = 1
	while True:
		
		b, frame = cam.read()
		if not b: break
		num_frame += 1
		if num_frame % args.skip_frames != 0: 
			continue
	
		frame = cv2.resize(frame, (args.frame_size,frame.shape[0]*args.frame_size//frame.shape[1]))
	
	
		if detect and mode == 'detect':
			# Run detection
			ret = detector.run(frame)
			print_box(frame, ret['results'], class_names, args.detect_thr, colors)
			track_state = False
			
		if track and mode == 'track':
			# Run tracking
			if track_state != False:
				track_state_new = siamese_track(track_state, frame, mask_enable=True, refine_enable=True, device=device)  # track
				
				if track_state['score'] > args.track_thr:
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
		elif track and key == ord('r'):
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
		t_frame = tf-ti

	 	# Wait until next frame time
		sleep_sec = (1/fps*args.skip_frames) - t_frame
		time.sleep(max(0, sleep_sec))
		ti = time.time()

		# Show FPSs
		txt = 'FPS: {:.1f}'.format((1/(sleep_sec+t_frame)*args.skip_frames))
		fps_size = cv2.getTextSize(txt, font, 0.5, 2)[0]
		cv2.rectangle(frame,
	 			(0, fps_size[1] + 4), (fps_size[0], 0), (255,0,0), -1)
		cv2.putText(frame, txt, (0, fps_size[1] + 2), font, 0.5, (0, 0, 0), thickness=1, lineType=cv2.LINE_AA)
	 	
	 	# Show frame and predictions
		cv2.imshow('input', frame)
	
	cv2.destroyAllWindows()
	

if __name__ == '__main__': main()


