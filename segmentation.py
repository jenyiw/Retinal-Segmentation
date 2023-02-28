# -*- coding: utf-8 -*-
"""
Created on Wed Nov  9 11:24:31 2022

@author: Jenyi
"""
import cv2
import numpy as np
import os
import re
import time
import math
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd

from skimage.io import imread
from skimage.filters import hessian, threshold_multiotsu, rank, gaussian, laplace, threshold_local
from skimage.measure import regionprops_table
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.feature_selection import SequentialFeatureSelector as sfs
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score, jaccard_score, f1_score
from skimage.restoration import denoise_nl_means
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from skimage.morphology import medial_axis, disk, skeletonize, opening
from plantcv import plantcv as pcv
import matplotlib.backends.backend_pdf
from sklearn import tree
from scipy.ndimage import binary_dilation

dataset = 'DRIVE'
num_train = 5
num_features = 'auto'
clf_type = 'DT'

class ImageData(object):
	"""
	
	Class for retinal segmentation.
	
	PARAMETERS:
		num_train: int. 
			Number of training images
		num_features: int
			Number of features to used for segmentation. Default: 2
		clf_type: str
			Type of classifier to use. Default: KNN
			
	RETURN:
		None
		
	"""
	
	def __init__(self,
                dataset,
				num_train,
				num_features:int=2,
				clf_type:str='KNN',
                ) -> None:	
		
		self.num_features = num_features 
		self.clf_type = clf_type
		working_dir = os.getcwd()
		self.dataset = dataset
		print(f'Working on dataset: {self.dataset}')
		self.image_dir = os.path.join(working_dir, dataset, 'images')
		self.save_dir = os.path.join(working_dir, dataset, 'save')
		self.gt_dir = os.path.join(working_dir, dataset, 'ground_truth')	
		self.test_dir = os.path.join(working_dir, dataset, 'testing')
		self.train_dir = os.path.join(working_dir, dataset, 'training')

		#get filenames
		if self.dataset != 'dataset_folder':
			self.train_files = [os.path.join(self.image_dir, x) for x in os.listdir(self.image_dir)[0:num_train]]			
			self.test_files = [os.path.join(self.image_dir, x) for x in os.listdir(self.image_dir) if os.path.join(self.image_dir, x) not in self.train_files]		
		else:
			train_files = os.listdir(self.train_dir)
			self.train_files = [os.path.join(self.train_dir, x) for x in train_files if re.search('_image', x)]
			test_files = os.listdir(self.test_dir)			
			self.test_files = [os.path.join(self.test_dir, x) for x in test_files if re.search('_image', x)]
			self.image_dir = self.train_dir

		
		#initialize image plotting 
		self.fig, self.axs = plt.subplots(2, 5, figsize=(30, 15))
# 		self.axs=0
		#get image mask to cut out background 		
		self.mask = False
			
		
	def read_image(self, filename):
		
		"""
		
		Read image from data folders
		
		PARAMETERS:
			filename: str
				Name of file
		RETURN:
			img_0: np.array
				Image array
			
		"""

		if self.dataset != 'dataset_folder':
			img_0 = imread(filename)[...,1]
			
		else:
			img_0 = imread(filename)[...,0]
			
		self.scale = img_0.shape[0]/560
			
		img_0 = ((img_0-np.min(img_0))/(np.max(img_0)-np.min(img_0))*255).astype(np.uint8)

		
# 		print(f'Scale factor: {self.scale}')
		
		self.save_image(img_0, 'Original', self.axs[0,0])		
		
		if (self.dataset == 'DRIVE') and (self.mask == False):
			print(f'Detected image shape: {img_0.shape}')
			self.img_mask = img_0.copy()			
			kernel = np.ones((20, 20), np.uint8)
			self.img_mask = cv2.erode(self.img_mask, kernel=kernel).astype(np.uint8)
			self.img_mask[self.img_mask < 20] = 0
			self.img_mask[self.img_mask >= 20] = 1		
			
			self.mask = True
		
		elif (self.dataset == 'CHASE' or self.dataset == 'STARE') and (self.mask == False):
			print(f'Detected image shape: {img_0.shape}')					
			img_mask = img_0.copy()
			thresholds = threshold_multiotsu(img_mask)
		
			img_mask[img_mask < thresholds[0]-10] = 0
			img_mask[img_mask >= thresholds[0]-10] = 1
		
			kernel = np.ones((5, 5), np.uint8)
			img_mask = cv2.morphologyEx(img_mask, cv2.MORPH_OPEN, kernel)*255
			kernel = np.ones((45, 45), np.uint8)
			self.img_mask = cv2.erode(img_mask, kernel)
			self.mask = True

		elif (self.dataset == 'dataset_folder') and (self.mask == False):
			print(f'Detected image shape: {img_0.shape}')					
			img_mask = img_0.copy()
			thresholds = threshold_multiotsu(img_mask)
		
			img_mask[img_mask < thresholds[0]-10] = 0
			img_mask[img_mask >= thresholds[0]-10] = 1
		
			kernel = np.ones((5, 5), np.uint8)
			img_mask = cv2.morphologyEx(img_mask, cv2.MORPH_OPEN, kernel)*255
			kernel = np.ones((15, 15), np.uint8)
			self.img_mask = cv2.erode(img_mask, kernel)
			self.mask = True

	
		footprint = disk(30*self.scale)
		img_0 = rank.equalize(img_0, footprint=footprint)
	
		return img_0



	def get_segmented_image(self,
						image,
						   ):
		"""
		
		Process image and segment it
		
		PARAMETERS:
			image: np.array
				Image array as a (n, m) array 
		RETURN:
			output: np.array
				Segmented labeled mask of image as a (n, m) array
			
		"""
		
		if (self.dataset == 'DRIVE'):
			gaussian_sigma = 2
			sigmas = (1,1,2)
			h_value=0.8		
		elif self.dataset == 'CHASE':
			gaussian_sigma = 4
			sigmas = (1,5,2)
			h_value = 0.6			
		elif self.dataset == 'STARE':
			gaussian_sigma = 2
			sigmas = (1,1,2)
			h_value = 0.6		
		elif self.dataset == 'dataset_folder':
			gaussian_sigma = 1	
			sigmas = (1,1,2)
			h_value = 0.6	
			
		thresh_img = (gaussian(image, sigma=gaussian_sigma)*255).astype(np.uint8)
		
		open_img = hessian(thresh_img, sigmas=sigmas)*255
		lap_img = laplace(thresh_img,ksize=3)
		
		if self.dataset != 'dataset_folder':
			img = denoise_nl_means(lap_img, h=h_value*0.01, fast_mode=True,)
			img = ((img-np.min(img))/(np.max(img)-np.min(img))*255).astype(np.uint8)
			img = img + open_img/15
		else:
			img = denoise_nl_means(lap_img, h=0.01, fast_mode=True,)

		
		self.save_image(image, 'Equalized image', self.axs[0,1])
		self.save_image(open_img, 'Hessian', self.axs[0,2])		
		self.save_image(lap_img, 'Laplacian', self.axs[0,3])	

		
		img[img == 1] = 0
	
		img = ((img-np.min(img))/(np.max(img)-np.min(img))*255).astype(np.uint8)

		block_size = 75
		thresholds = threshold_local(img, block_size, offset=0)	
		
		img[img < thresholds-20] = 1
		img[img >= thresholds-20] = 0
		
		img[self.img_mask == 0] = 0

		self.save_image(img, 'Thresholding', self.axs[0,4])	  	   

		num_labels, output, stats, centroids = cv2.connectedComponentsWithStats(img, )
		
	
		for i in range(1, num_labels):
			   if stats[i, cv2.CC_STAT_AREA] < round(4*self.scale):
				   output[output == i] = 0				   
				   
		   
		output = output.astype(np.uint8)				   
		num_labels, output,= cv2.connectedComponents(output, )			   
		
		self.save_image(output, 'Connected regions', self.axs[1, 0])				     
	
		return output

	def get_features(self, 
				  image, 
				  labelled_image,
				  select_features:list=None):
		
	    """
		
		Calculate features used for classification
		
		PARAMETERS:
			image: np.array
				Image array as a (n, m) array 
			labelled_image: np.array
				Labeled mask as a (n, m) array 
			select_features: list
				Allows for user to pre-select features to use. 			
		RETURN:
			feature_arr: np.array
				Array of features for each segment
			feature_arr.keys: list
				Name of features used
			
		"""
	
	    
	    feature_names = ['area',
	 					  'eccentricity',
	 					  'orientation',
	 					  'axis_major_length',
						   'perimeter',   
						   'intensity_max',
						   'intensity_mean',
							'skel_length',
							'width',
							'circularity']
		
	    if np.all(select_features) != None:			
			   feature_names_selected = [b for a, b in zip(select_features, feature_names) if a]
			   feature_ind_selected = [n for n, a in enumerate(select_features) if a]
			   feature_names_props = []
			   for i in feature_ind_selected:
				   if i < 5:
					   feature_names_props.append(feature_names[i])
			   features_table = regionprops_table(labelled_image, properties=feature_names_props)

	    else:
			   features_table = regionprops_table(labelled_image, properties=feature_names[:5])			   
			   feature_names_selected = feature_names
			      
	    features = {}
			
	    for f in feature_names_selected:
			   if f in features_table.keys():
				      features[f] = features_table[f]
			   else:
				      features[f] = []		
			   if f == 'circularity':
				   perimeter = regionprops_table(labelled_image, properties=['perimeter',])	
				   area = regionprops_table(labelled_image, properties=['area',])					   
				   features[f] = np.divide(4*math.pi*area['area'], perimeter['perimeter']**2, where=perimeter['perimeter'] != 0)				   
				   
			   if f in ['skel_length', 'width']:
				   features[f] = []					   
				   skel_img = labelled_image.copy()
				   skel, distance = medial_axis(skel_img, return_distance=True)	
		
		   # 

	    num_labels = len(np.unique(labelled_image))
	
	    for i in range(1, num_labels):
			
			   if 'intensity_max' in features.keys():
				      features['intensity_max'].append(np.max(image[labelled_image == i]))
			   if 'intensity_mean' in features.keys():
				      features['intensity_mean'].append(np.mean(image[labelled_image == i]))
	
			   if 'skel_length' in features.keys() or 'width' in features.keys():
				      temp_img = skel[labelled_image == i]
				      temp_distance = distance[labelled_image == i]
				      length = np.sum(temp_img)
					  
				      if 'skel_length' in features.keys(): 
						       features['skel_length'].append(length)
					     
				      if 'width' in features.keys():				  
						       width = np.sum(temp_distance)/length
						       features['width'].append(width)		

	
	    feature_arr = np.zeros((num_labels-1, len(features.keys())))
		
	    for n, f in enumerate(features.keys()):	
		    feature_arr[:, n] = features[f]			
		
	    scaler = StandardScaler() 	
	    scaler.fit(feature_arr)	
	    feature_arr = scaler.transform(feature_arr)
	    feature_arr = np.nan_to_num(feature_arr, copy=True,)
	
	    return feature_arr, list(features.keys())

	def get_true_labels(self,
					       labelled_image,
						f,
						train:bool=False):
		"""
		
		Get ground truth labels from file and calculate accuracy
		
		PARAMETERS:
			labelled_image: np.array
				Image array as a (n, m) array 
			f: str
				filename of ground truth image
			train: bool
				whether this is for the training dataset or the test dataset
		
		RETURN:
			img_manual: np.array
				Ground truth image
			true_label: list
				True label of each segment in labelled_image
			
		"""

		if self.dataset == 'DRIVE':
			filename = f.split('\\')[-1]
			sample_num = re.search(r'\d+_', filename).group()[:-1]
			img_manual = imread(os.path.join(self.gt_dir, sample_num+'_manual1.gif'))

		elif self.dataset == 'CHASE':
			filename = f.split('\\')[-1]			
			sample_num = filename.split('_')[1].split('.')[0]
			img_manual = imread(os.path.join(self.gt_dir, 'Image_'+sample_num+'_1stHO.png'))

		elif self.dataset == 'STARE':
			filename = f.split('\\')[-1]				
			sample_num = filename.split('.')[0]
			img_manual = imread(os.path.join(self.gt_dir, sample_num+'.ah.ppm'))

		elif self.dataset == 'dataset_folder':
			filename = f.split('\\')[-1]				
			sample_num = filename.split('_')[0]
			if train:
				temp_dir = self.train_dir
			else:
				temp_dir = self.test_dir

			img_manual = imread(os.path.join(temp_dir, sample_num+'_label.tif'))[...,0].astype(np.uint8)
				
		img_manual[img_manual == 255] = 1
		img_manual[self.img_mask == 0] = 0		
		
		if train==False:
			self.save_image(img_manual, 'Ground truth', self.axs[1,3])
					
		else:
			if self.dataset == 'dataset_folder':
				train_save_dir = os.path.join(self.save_dir, 'save_training')
			else:
				train_save_dir = self.save_dir
				
			self.save_image(img_manual, 'Ground truth', self.axs[1,3], os.path.join(train_save_dir, filename))
			
		true_label = []
		num_labels = np.unique(labelled_image)
	
		for i in num_labels[1:]:		
		    label_value = np.mean(img_manual[labelled_image == i])
		    if label_value == np.NaN:
		        true_label.append(0)			
		        continue
		    if label_value > 0:
		        label_value = 1
		    true_label.append(label_value)
	
		return img_manual, true_label

	def save_image(self, 
				image, 
				   title,
				   axs,
				   save_img:str=None):
		"""
		
		Save images as subplots
		
		PARAMETERS:
			image: np.array
				Image array as a (n, m) array 
			title: str
				title of subplot
			axs: plt.axs object
				Axes of subplot			
			save_img: str
				whether to save the whole figure. Applied only after plotting the last subplot.
		
		RETURN:
			None.
			
		"""			
		plt.sca(axs)
		if image.ndim == 2:
			axs.imshow(image, cmap='gray')
		else:
			axs.imshow(image)			
		axs.set_title(title, {'fontsize': 16})
		axs.axis('off')
		
		if title == 'Connected regions':
			cmap = mpl.cm.get_cmap("OrRd").copy()
			cmap.set_bad(color='black')
			black_img = np.ma.masked_where(image < 1, image)
	
			axs.imshow(black_img, interpolation='none', cmap=cmap)
	
		if save_img != None:
			plt.gca()
			save_img_1 = save_img.split('.')[0]
			plt.savefig(save_img_1+'.jpg')


	def prune_image(self,
				   image):
		"""
		
		Prune small branches in the blood vessels
		
		PARAMETERS:
			image: np.array
				Image array as a (n, m) array 
		
		RETURN:
			labelled_image_mask: np.array
				Pruned image as a (n, m) array 
			
		"""			
		skel = skeletonize(image).astype(np.uint8)			   
		pruned_skeleton, segmented_img, segment_objects = pcv.morphology.prune(skel_img=skel, size=int(50*self.scale))	
		footprint = np.ones((int(5*self.scale),int(5*self.scale)))
		skel_mask = binary_dilation(pruned_skeleton, structure=footprint, mask=image)
		labelled_image_mask = np.clip(skel_mask + pruned_skeleton, a_min=None, a_max=1)
	
		return labelled_image_mask

	def train_model(self,
				   ):
		"""
		
		Train model
		
		PARAMETERS:
			None.
		RETURN:
			classifier: classifier object from sklearn
				Trained classifier
			
		"""	
		all_features = []
		all_labels = []	
		#train SVM classifier	
		for n, f in enumerate(self.train_files):	
			img = self.read_image(f)
			label_img = self.get_labelled_image(img)
			features, feature_names = self.get_features(img, label_img)
			_, labels = self.get_true_labels(label_img,
										  f,
										  train=True)

			all_features.append(features)
			all_labels.append(labels)
	
	
		feature_arr = np.concatenate(all_features)
		label_arr = np.concatenate(all_labels)
	
		print('Selecting features...')
		if self.clf_type == 'SVC':		
			classifier = SVC()
		elif self.clf_type == 'KNN':				
			classifier = KNN(n_neighbors=7)
		elif self.clf_type == 'DT':
			classifier = tree.DecisionTreeClassifier(max_depth=4)
		if (self.num_features == 'auto') or (self.clf_type == 'KNN') or (self.clf_type == 'DT'):	
			model = sfs(classifier,    
 	            scoring='roc_auc',
 	            )
		else:			
			model = sfs(classifier, 
 				 n_features_to_select=self.num_features,     
 	            scoring='roc_auc',
 	            )
 	
		model = model.fit(feature_arr, label_arr,)
		print(model.get_support())
		self.feature_list = model.get_support()
		feature_arr = model.transform(feature_arr)
	
		print('Fitting data...')
		if self.clf_type == 'SVC':		
			classifier = SVC()
		elif self.clf_type == 'KNN':				
			classifier = KNN(n_neighbors=7)
		elif self.clf_type == 'DT':
			classifier = tree.DecisionTreeClassifier(max_depth=4)
		classifier.fit(feature_arr, label_arr)

		return classifier

	def evaluate_model(self,
				   classifier,
				   ):
		"""
		
		Evaluate model on test data set
		
		PARAMETERS:
			classifier: str
				classifier used
		
		RETURN:
			None.
		"""			
		
		#evaluate the model
		print('Running test set...')	
		
		self.score_arr = np.zeros((len(self.test_files), 5))
		self.score_arr[:len(self.feature_list), -1] = self.feature_list
		
		if self.dataset == 'dataset_folder':
			self.image_dir = self.test_dir
					
		for m, f in enumerate(self.test_files):
			img = self.read_image(f)
			labelled_image = self.get_labelled_image(img)
			feature_arr, _,  = self.get_features(img, labelled_image, 
										select_features=self.feature_list
										)
# 			feature_arr = model.transform(feature_arr)	
			true_label_img, labels = self.get_true_labels(labelled_image, f)		
			pred_labels = classifier.predict(feature_arr)
					
	
			for n in range(len(pred_labels)):
			    labelled_image[labelled_image == n+1] = pred_labels[n]
				
			true_label_img[self.img_mask == 0] = 0	   		

			self.save_image(labelled_image, 'Before pruning', self.axs[1,1])	
			labelled_image_mask = self.prune_image(labelled_image)
# 			footprint = np.ones((int(2*self.scale), int(2*self.scale))) 
# 			labelled_image_mask = binary_dilation(labelled_image_mask, structure=footprint)
			combined_img = np.zeros(labelled_image_mask.shape+(3,))
			combined_img[...,0] = true_label_img
			combined_img[...,1] = labelled_image_mask
			
			self.save_image(combined_img, 'Overlay', self.axs[1, -1],)
			filename = f.split('\\')[-1]
			
			self.save_image(labelled_image_mask, 'Prediction', self.axs[1,2], save_img = os.path.join(self.save_dir, filename))			

		#print metrics
			self.score_arr[m, :4] = self.metrics_score(true_label_img.ravel(), labelled_image_mask.ravel())
			
			print(m, self.score_arr[m, :])

		df = pd.DataFrame({'Accuracy': self.score_arr[:, 0], 
					 'ROC_AUC': self.score_arr[:, 1],
					 'Jaccard': self.score_arr[:, 2],
					 'F1': self.score_arr[:, 3],
					 'Features': self.score_arr[:, 4]})

		df.to_csv(os.path.join(self.save_dir, f"scores_{self.num_features}_{self.clf_type}.csv"))		

		print(self.score_arr)


	def metrics_score(self, 
				   true_label,
						pred_label):
		"""
		
		Calculate the metrics score
		
		PARAMETERS:
			true_label: np.array
				Image array as a (n,) array 
			pred_label: str
				Image array as a (n,) array 
			train: bool
				whether this is for the training dataset or the test dataset
		
		RETURN:
			metrics_arr: np.array
				Calculated scores for the prediction
			
		"""	
		metrics_arr = np.zeros((1, 4))
		metrics_arr[0, 0] = accuracy_score(true_label, pred_label) 
		metrics_arr[0, 1] = roc_auc_score(true_label, pred_label)
		metrics_arr[0, 2] = jaccard_score(true_label, pred_label)
		metrics_arr[0, 3] = f1_score(true_label, pred_label)	
	
		return metrics_arr

	

if __name__ == '__main__':

# 	for s in ['CHASE']:
	for s in ['KNN', 'SVC', 'DT']:	
		imgD = ImageData(dataset, num_train, num_features, s)
		start_time = time.time()
		classifier = imgD.train_model()
		print(f'Time taken for training: {time.time()-start_time }')
		start_time = time.time()	   
		imgD.evaluate_model(classifier)	
		print(f'Time taken for testing: {time.time()-start_time}')	
		
