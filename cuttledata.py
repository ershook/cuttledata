# Standard library imports
import os
import json
import math
import random
import time
import warnings
from tempfile import TemporaryDirectory

# Third-party imports for data manipulation and numerical operations
import numpy as np
import pandas as pd
import scipy.ndimage
from scipy.stats import gaussian_kde

# Image processing and computer vision libraries
import cv2
from PIL import Image
import skimage.transform
from pycocotools import mask as mask_utils

# Machine learning and deep learning libraries
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision
from torchvision import datasets, models, transforms
from torchvision.io import read_image
import torch.backends.cudnn as cudnn
from tensorflow.keras.applications import vgg19
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from keract import get_activations

# Visualization libraries
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# Custom module imports
import largestinteriorrectangle as lir
from PortillaSimoncelliMinimalStats import *
import plenoptic as po

warnings.filterwarnings('ignore')


class Interval:
    def __init__(self, start, end):
        self.start = start
        self.end = end



class CuttleData:
    def __init__(self, images_folder):
        """
        Initializes the CuttleData object.

        Args:
            images_folder (str): Folder containing the behavior images..
        """
        # Determine what os system code is running on: 
        axel_server_path = ('/').join(os.getcwd().split('/')[:3])+'/engram/'

        if os.path.exists('/mnt/smb/locker/axel-locker/'):
            prefix = '/mnt/smb/locker/axel-locker/'
        elif os.path.exists('/Volumes/axel-locker/'):
            prefix = '/Volumes/axel-locker/'
        elif os.path.exists('Z:/cuttlefish'):
            prefix = 'Z:/'

        elif os.path.exists(axel_server_path):
            prefix = axel_server_path 
        else:
            raise Exception("Can't find path to data -- are you sure axel-locker is mounted?")
        
        self.mask_downsample_factor = 1 
        self.images_folder = images_folder
        self.images_path = prefix + 'cuttlefish/CUTTLEFISH_BEHAVIOR/2023_BEHAVIOR/E-ink_Tank/'+images_folder+'/Tifs_downsampled/'
        if not os.path.exists(self.images_path):
          
            self.images_path = prefix + 'cuttlefish/CUTTLEFISH_BEHAVIOR/2023_BEHAVIOR/E-ink_Tank/'+images_folder+'/0_5_pop_color_1/tifs/'
            self.mask_downsample_factor = .5
            
     
        self.path_to_masks = prefix + 'cuttlefish/CUTTLEFISH_BEHAVIOR/cuttle_data_storage/sam_data/' + images_folder + '/'
        knob_inds = np.load(prefix + 'cuttlefish/CUTTLEFISH_BEHAVIOR/cuttle_data_storage/knob_inds.npy')
        self.knob_inds = (knob_inds[0], knob_inds[1])
        self.storage_path = prefix + 'cuttlefish/CUTTLEFISH_BEHAVIOR/cuttle_data_storage/'
        self.vgg_weights_path = prefix + 'cuttlefish/CUTTLEFISH_BEHAVIOR/cuttle_data_storage/vgg_finetune.pt'
        self.vgg = True

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
    
    def _get_mask_inds_vgg(self, image_no):
        new_size = (256,256)
        

            
        if  not hasattr(self, "model_ft"):

            
            model_ft = models.resnet18()
            num_ftrs = model_ft.fc.in_features
            # Here the size of each output sample is set to 2.
            # Alternatively, it can be generalized to ``nn.Linear(num_ftrs, len(class_names))``.
            model_ft.fc = nn.Linear(num_ftrs, 3)

            model_ft = model_ft.to(self.device)

            criterion = nn.CrossEntropyLoss()

            # Observe that all parameters are being optimized
            optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)

            # Decay LR by a factor of 0.1 every 7 epochs
            exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

            self.model_ft = model_ft
            self.model_ft.load_state_dict(torch.load(self.vgg_weights_path))
            self.model_ft.eval()
            
        
        labels = ['mantle','cf','bg']

        masks = self.load_masks(image_no)
        results = []
        for mask in masks:
        

            # Define the transform
            transform = transforms.Compose([
               # transforms.ToTensor(),  # Convert the image to a tensor
                transforms.Resize((224, 224)),  # Resize the image to a specific size
            
            ])
            mask = self.decode_mask(mask['segmentation'])
            mask = cv2.resize(mask, new_size, interpolation=cv2.INTER_NEAREST)
            plt.imsave( 'temp_mask.png', mask)

            # Load and transform the image
            image = read_image('temp_mask.png')[:3,:,:]
            transformed_image = transform(image)

            # Add a batch dimension
            inputs = transformed_image.unsqueeze(0).to(torch.float32)
            inputs = inputs.to(self.device)

            outputs = self.model_ft(inputs)
            _, preds = torch.max(outputs, 1)
            preds = preds.cpu()
            inputs = inputs.cpu()    
            results.append(labels[int(np.array(preds)[0])])

        if 'cf' in results:
            inds = [int(np.where(np.array(results) == 'cf')[0])]
        else:
            inds = ['ERR']
        if 'mantle' in results:
            inds.append(int(np.where(np.array(results) == 'mantle')[0]))
        else:
            inds.append('ERR')
        return inds


    def num_frames(self):
        """
        Get the number of frames in the dataset.

        Returns:
            int: The total number of frames in the dataset.
        """
        if not hasattr(self, "n_frames"): # This can be a slow operation so only ever do it once.
            self.n_frames = len(os.listdir(self.images_path))
        return self.n_frames
    
    def num_masks(self): 
        """
        Get the number of masks in the dataset.

        Returns:
            int: The total number of masks in the dataset.
        """
        if not hasattr(self, "n_masks"): # This can be a slow operation so only ever do it once.
            self.n_masks = len(os.listdir(self.path_to_masks))
        return self.n_masks
        
    def _get_mask_inds(self, image):
        """
        Determines which masks correspond to cuttlefish parts. 
        Order: cuttlefish index, mantle index, head index.
        If mask for a given part doesn't exist defaults to index of 'ERR'

        Args:
            image_no (int): Image number.
        """
        # Load masks for the specified image
        masks = self.load_masks(image)

        # Get the areas of all masks in the image
        areas = [masks[ii]['area'] for ii in range(len(masks))]

        # Determine the index of the inverse mask, i.e., the mask with the cuttlefish 0 and background 1
        ind = np.argmax(areas)
        max_ind = ind

        # If there are more than two masks with an area greater than 1e6, set the area of the first mask to 0
        if np.sum(np.array(areas) > 1e6) > 1:
            areas[ind] = 0
            ind = np.argmax(areas)

        # Initialize lists to store good mask indices and their corresponding areas
        good_inds = []
        good_areas = []

        # Iterate through all masks and filter them based on various criteria
        for mask_ii in range(len(masks)):

            # Calculate the overlap with the inverse mask in a specific region
            overlap_w_inv = np.sum(mask_utils.decode(masks[mask_ii]["segmentation"])[180:1450,300:2000] * (1 - mask_utils.decode(masks[ind]["segmentation"])[180:1450,300:2000]))

            # Check if the mask meets various conditions, including size, non-maximum overlap, and knob location
            if areas[mask_ii] < 200000 and areas[mask_ii] > 10000 and mask_ii != max_ind and overlap_w_inv > 200 and np.sum(mask_utils.decode(masks[mask_ii]["segmentation"])[self.knob_inds]) < 500:
                good_inds.append(mask_ii)
                good_areas.append(areas[mask_ii])

        # Sort the good mask indices by area from largest to smallest
        
        good_inds = np.array(good_inds)[np.argsort(good_areas)][::-1]
        good_areas = np.sort(good_areas)[::-1]
        # Check for duplicate areas and remove duplicates
        if len(good_areas) > 1 and np.abs(good_areas[0] - good_areas[1]) < 1000:
            good_areas = np.delete(good_areas, 1)
            good_inds = np.delete(good_inds, 1)

        good_inds = list(good_inds)
        
        # Check the areas and insert 'ERR' at specific indices if needed
        if good_areas[0] < 100000:
            good_inds.insert(0, 'ERR')
        elif len(good_areas) > 1 and good_areas[1] < 60000:
            good_inds.insert(1, 'ERR')

        # Check if no good indices were found, and resort to an alternative approach
        if len(good_areas) == 0:

            # Initialize a list to store good mask indices
            good_inds = []

            # Initialize variables for maximum area and mask index with the maximum area
            max_area = 0
            max_area_index = 0

            # Iterate through all masks and check for conditions based on area, location, and aspect ratio
            for ii in range(len(masks)):
                area = masks[ii]['area']
                if area > 80000:
                    seg = mask_utils.decode(masks[ii]["segmentation"])
                    x_mar = np.mean(seg, axis=0)
                    y_mar = np.mean(seg, axis=1)

                    # Check if the mask meets specific conditions
                    if max(x_mar) < 0.5 and max(y_mar) < 0.5 and area < 2e6:
                        # Update the mask index if it has a larger area
                        if masks[ii]['area'] > max_area:
                            good_inds.insert(0, ii)
                            max_area = masks[ii]['area']
                        else:
                            good_inds.append(ii)

            try:
                # Try to decode the cuttlefish mask from the selected index
                full_cuttlefish_mask = mask_utils.decode(masks[good_inds[0]]["segmentation"])
            except:
                full_cuttlefish_mask = []

            # Check if the cuttlefish mask is empty
            if len(full_cuttlefish_mask) == 0:
                good_mask_inds.append(['ERR','ERR'])
            else:
                temp_areas = []
                temp_indices = []
                good_masks = []

                # Iterate through masks and filter them based on various criteria
                for ii in range(len(masks)):
                    mask_1 = mask_utils.decode(masks[ii]["segmentation"])
                    area = masks[ii]['area']

                    # Check conditions related to size and overlap with the cuttlefish mask
                    if area > 60000 and area < 1e6 and np.sum(np.multiply(mask_1, full_cuttlefish_mask)) > 300 and np.sum(np.multiply(mask_1, full_cuttlefish_mask)) - int(min(np.sum(mask_1), np.sum(full_cuttlefish_mask))) < 300:
                        temp_areas.append(area)
                        temp_indices.append(ii)
                        good_masks.append(mask_1)

                # Order the masks by area in ascending order
                order = np.argsort(temp_areas)

                # Construct a list of good mask indices with certain conditions
                good_inds = list(np.unique(list([good_inds[0]] + list(np.array((temp_indices))[order]))))

                # Check the number of good indices
                if len(good_inds) > 2:
                    final_inds = []

                    # Check for repeated masks
                    for ii in range(len(good_inds)):
                        mask_1 = mask_utils.decode(masks[good_inds[ii]]["segmentation"])
                        for jj in range(ii + 1, len(good_inds)):
                            mask_2 = mask_utils.decode(masks[good_inds[jj]]["segmentation"])

                            # Check if there is significant overlap between two masks
                            if int(np.sum(np.multiply(mask_1, mask_2) - mask_1)) / 3. > 50000:
                                final_inds += [good_inds[ii], good_inds[jj]]

                    # If no repeated masks are found, set the final indices to an empty list
                    if final_inds == []:
                        good_mask_inds.append([])
                    else:
                        good_mask_inds.append(list(np.unique(final_inds)))

                elif len(good_inds) == 1:
                    good_mask_inds.append([good_inds[0]])
                elif len(good_inds) == 0:
                    good_mask_inds.append([])
                else:
                    good_mask_inds.append(good_inds)

            # Sort and filter the final mask indices based on area and other conditions
            temp_inds = good_mask_inds[-1]
            temp_areas = []

            # Sort the mask indices by area in descending order
            if len(temp_inds) > 1:
                for ii_ in temp_inds:
                    temp_areas.append(masks[ii_]['area'])

                inds = np.argsort(temp_areas)
                final_inds = np.array(temp_inds)[inds][::-1]
            else:
                final_inds = temp_inds

            good_inds = final_inds

        # Return the list of good mask indices
        return good_inds

    def decode_mask(self, mask):
        """
        Decode a mask using mask utilities.

        Args:
            mask (dict): A mask represented as a dictionary.

        Returns:
            numpy.ndarray: The decoded mask as a NumPy array.
        """
        return mask_utils.decode(mask)
        


    def is_image_out_of_focus(self, image_no, threshold=100):
        """
        *** NOT TESTED -- USE AT YOUR OWN RISK (but please report back if you do) ***
        Determines whether an image is out of focus based on its variance of Laplacian.

        Args:
            image_no (int): Image number.
            threshold (int): Variance threshold for determining focus. Default is 100.
        """
        image_file = f"{self.images_path}{str(image_no).zfill(5)}.tif"
        image = cv2.imread(image_file)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        variance = laplacian.var()
        if variance < threshold:
            print(f"Image {image_no} is out of focus.")
        else:
            print(f"Image {image_no} is not out of focus.")

    def load_masks(self, image_no):
        """
        Loads the masks for a given image.

        Args:
            image_no (int): Image number.

        Returns:
            dict: Loaded masks in a dictionary format.
        """
        if image_no ==0:
            raise Exception("Remember tiff files start at 1 not 0 :/")
        
        if not os.path.exists(self.path_to_masks):
            raise Exception("Mask files don't exist")
        try: 
            
            mask_path = f"{self.path_to_masks}{str(image_no).zfill(5)}.json"
            with open(mask_path) as f:
                all_masks = json.load(f)
        except:
            mask_path = f"{self.path_to_masks}{str(image_no).zfill(6)}.json"
            with open(mask_path) as f:
                all_masks = json.load(f)
      
        return all_masks


    def load_image(self, image_no):
        """
        Loads and returns an image.

        Args:
            image_no (int): Image number.

        Returns:
            numpy.ndarray: Loaded image as a NumPy array.
        """
        if image_no ==0:
            raise Exception("Remember tiff files start at 1 not 0 :/")
        else:
            image_file = f"{self.images_path}{str(image_no).zfill(6)}.tif"
            image = cv2.imread(image_file)
            if self.mask_downsample_factor != 1:
                image = cv2.resize(image, (0, 0), fx=self.mask_downsample_factor, fy=self.mask_downsample_factor, interpolation=cv2.INTER_AREA)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            return image
        
        
    def load_highres_image(self, image_no):
        pass

    def _plot_image(self, image, title=False, grayscale=True):
        """
        Plots the image.

        Args:
            image (numpy.ndarray): Image to plot.
            grayscale (bool): Whether to display the image in grayscale. Default is True.
        """
        if grayscale:
            plt.imshow(cv2.cvtColor(image, cv2.COLOR_RGB2GRAY), cmap='gray')
        else:
            plt.imshow(image)
        if title != False:
            plt.title(title)
        plt.axis('off')
        plt.show()
        
    def plot_image(self, image_no, title=False,  grey = False):
        """
        Wrapper that calls load_image and plot_image methods 
        Args:
            image_no (int): Image number.
            
        Returns: 
            image (numpy.ndarray): Image to plot.
        
        """
        image = self.load_image(image_no)
        self._plot_image(image,title, grey)
        return image

    def _get_ellipse_angle(self, thresh_image):
        """
        Computes and returns the angle of the ellipse fitted to the contour.

        Args:
            thresh_image (numpy.ndarray): Thresholded image.

        Returns:
            float: Angle of the fitted ellipse.
        """
        contours, _ = cv2.findContours(thresh_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contour = contours[0]
        (_, _), (_, _), angle = cv2.fitEllipse(contour)
        return angle

    def get_cuttlefish(self, image_no, title = False, show_image=True, correct_flip = True, bg_color = 255):
        """
        Retrieves the rotated cuttlefish image and its angle.

        Args:
            image_no (int): Image number.
            show_image (bool): Whether to display the rotated cuttlefish image. Default is True.

        Returns:
            tuple: Tuple containing the rotated cuttlefish image and its angle.
        """
        # Get the full cuttlefish mask for the specified image
        full_cuttlefish_mask = self.get_cuttlefish_mask(image_no)

        # Load masks and the image for the specified image
        masks = self.load_masks(image_no)
        image = self.load_image(image_no)

        # Get the cuttlefish mask index for the image
        cf_mask_ind = self.get_cf_ind(image_no)

        # Find indices where the cuttlefish mask is equal to 0 (non-cuttlefish region)
        non_cf_inds = np.where(full_cuttlefish_mask == 0)

        # Iterate through all color channels and set pixel values to 255 (white) in non-cuttlefish regions
        for ii in range(3):
            image[:, :, ii][non_cf_inds] = bg_color

        # Get the cuttlefish image using the specified mask index
        cf_image = self._get_object(image, masks[cf_mask_ind])

        # Get the cuttlefish mask object using the specified mask index
        cf_mask = self._get_object(full_cuttlefish_mask, masks[cf_mask_ind])
        
        cf_mask = self._remove_extraneous_parts(cf_mask)

        # Attempt to calculate the angle of rotation for the cuttlefish mask, or set it to 0 if there's an error
        try:
            angle = self._get_ellipse_angle(cf_mask)
        except:
            angle = 0
        
        # Rotate the cuttlefish image by the calculated angle + 180 degrees and resize with padding as 255 (white)
        rotated_cuttlefish = skimage.transform.rotate(cf_image, angle + 180, resize=True, cval=bg_color/255.)

        # If 'correct_flip' is True and the mantle mask index is valid, check if the cuttlefish is upside down
        if correct_flip and self.get_mantle_ind(image_no) != 'ERR' and self._is_cuttlefish_upside_down(image_no, masks, angle, full_cuttlefish_mask, non_cf_inds):
            # If upside down, rotate the cuttlefish image by 180 degrees and resize with padding as 255 (white)
            rotated_cuttlefish = skimage.transform.rotate(rotated_cuttlefish, 180, resize=True, cval=bg_color/255.)

        # If 'show_image' is True, display the rotated cuttlefish image without axis labels and with an optional title
        if show_image:
            plt.imshow(rotated_cuttlefish)
            plt.axis('off')
            if title:
                plt.title(title)

        # Return the rotated cuttlefish image
        return rotated_cuttlefish

    
    def _remove_extraneous_parts(self, mask):
           
        # Apply connected component labeling to the rotated mask
        num_labels, labels = cv2.connectedComponents(mask)

        if num_labels > 2:
            # If there are more than 2 components, find the largest connected component

            # Initialize a dictionary to store the size of each component
            component_sizes = {}

            # Calculate the size of each connected component
            for label in range(1, num_labels):  # Skip label 0, which represents the background
                component_size = np.sum(labels == label)
                component_sizes[label] = component_size

            # Find the label of the largest connected component
            largest_component_label = max(component_sizes, key=component_sizes.get)

            # Access the largest component using the label
            mask = (labels == largest_component_label).astype(np.uint8)
        return mask


    
    
    def get_mantle(self, image_no, show_image = True, title = False):
        """
        Retrieves the rotated mantle image.

        Args:
            image_no (int): Image number.
            show_image (bool): Whether to display the rotated cuttlefish image. Default is True.

        Returns:
            tuple: Tuple containing the rotated cuttlefish image and its angle.
        """
        
        # Get the mantle mask index for the specified image
        mantle_mask_ind = self.get_mantle_ind(image_no)

        # Check if the mantle mask index is valid (not 'ERR')
        if mantle_mask_ind != 'ERR':
            # Get the full mantle mask for the image
            full_mantle_mask = self.get_mantle_mask(image_no)

            # Load masks and the image for the specified image
            masks = self.load_masks(image_no)
            image = self.load_image(image_no)
            
            # Find indices where the mantle mask is equal to 0 (non-mantle region)
            non_mantle_inds = np.where(full_mantle_mask == 0)

            # Iterate through all color channels and set pixel values to 255 (white) in non-mantle regions
            for ii in range(3):
                image[:, :, ii][non_mantle_inds] = 255

            # Get the mantle image using the specified mask index
            mantle_image = self._get_object(image, masks[mantle_mask_ind])

            # Get the mantle mask object using the specified mask index
            mantle_mask = self._get_object(full_mantle_mask, masks[mantle_mask_ind])
     
            # Calculate the angle of rotation for the mantle mask
            angle = self._get_ellipse_angle(mantle_mask)

            # Rotate the mantle image by the calculated angle and resize with padding as 255 (white)
            rotated_mantle = skimage.transform.rotate(mantle_image, angle, resize=True, cval=255)

            if show_image:
                # Display the rotated mantle image without axis labels and with an optional title
                plt.axis('off')
                plt.imshow(rotated_mantle)

                if title:
                    plt.title(title)

            # Return the rotated mantle image
            return rotated_mantle

        # Return 'ERR' if the mantle mask index is not valid
        return 'ERR'

    def get_highres_mantle(self, image_no):
        pass
    

    
    def get_highres_cuttlefish(self, image_no):
        pass


    def _get_object(self, image, mask):
        """
        Retrieves the object within the image based on the mask's bounding box.

        Args:
            image (numpy.ndarray): Image.
            mask (dict): Mask dictionary.

        Returns:
            numpy.ndarray: Extracted object from the image.
        """
        bbox = mask['bbox']
        return image[bbox[1]:bbox[1] + bbox[3], bbox[0]:bbox[0] + bbox[2]]

    def get_cf_ind(self, image_no):
        """
        Retrieves the cuttlefish index for a given image.

        Args:
            image_no (int): Image number.

        Returns:
            int: Cuttlefish index.
        """
        if self.vgg:
            return self._get_mask_inds_vgg(image_no)[0]
        else:
            return self._get_mask_inds(image_no)[0]

    def get_mantle_ind(self, image_no):
        """
        Retrieves the mantle index for a given image.

        Args:
            image_no (int): Image number.

        Returns:
            int or str: Mantle index or 'ERR' if not available.
        """
        if self.vgg:
           inds = self._get_mask_inds_vgg(image_no)
        else:
            inds = self._get_mask_inds(image_no)
        if len(inds)>1:
            return inds[1]
        else:
            return 'ERR'

    def get_mantle_mask(self, image_no, show_image = False):
        """
        Retrieves the mantle mask for a given image.

        Args:
            image_no (int): Image number.

        Returns:
            binary mask (numpy.ndarray): Mantle mask.
        """
        masks = self.load_masks(image_no)
        mantle_ind = self.get_mantle_ind(image_no)
        if mantle_ind == 'ERR': 
            raise Exception("Don't have a mantle mask")
            
        full_mantle_mask = mask_utils.decode(masks[mantle_ind]["segmentation"])
        full_mantle_mask = self._remove_extraneous_parts(full_mantle_mask)
       # if self.mask_downsample_factor != 1:
         #   full_mantle_mask = cv2.resize(full_mantle_mask, (0, 0), fx=self.mask_downsample_factor, fy=self.mask_downsample_factor, interpolation=cv2.INTER_AREA)
        if show_image:
            plt.imshow(full_mantle_mask)
            plt.show()
           
        return full_mantle_mask

    def get_cuttlefish_mask(self, image_no, show_image = False):
        """
        Retrieves the cuttlefish mask for a given image.

        Args:
            image_no (int): Image number.

        Returns:
            binary mask (numpy.ndarray): Cuttlefish mask.
        """
        masks = self.load_masks(image_no)
        cf_ind = self.get_cf_ind(image_no)
        
        if cf_ind == 'ERR': 
            raise Exception("Don't have a cuttlefish mask")
            
        full_cuttlefish_mask = mask_utils.decode(masks[cf_ind]["segmentation"])
        full_cuttlefish_mask = self._remove_extraneous_parts(full_cuttlefish_mask)
        #if self.mask_downsample_factor != 1:
        #    full_cuttlefish_mask = cv2.resize(full_cuttlefish_mask, (0, 0), fx=self.mask_downsample_factor, fy=self.mask_downsample_factor, interpolation=cv2.INTER_AREA)
        if show_image:
            plt.imshow(full_cuttlefish_mask)
            plt.show()
        
        return full_cuttlefish_mask

    def _is_cuttlefish_upside_down(self, image_no, masks, angle, full_cuttlefish_mask, non_cf_inds):
        """
        Checks if the cuttlefish in the image is upside down.

        Args:
            image_no (int): Image number.
            masks (dict): Loaded masks.
            angle (float): Angle of the cuttlefish.
            full_cuttlefish_mask (numpy.ndarray): Full cuttlefish mask.
            non_cf_inds (tuple): Indices of non-cuttlefish areas.

        Returns:
            bool: True if the cuttlefish is upside down, False otherwise.
        """
        # Load the image for the specified image number
        image = self.load_image(image_no)

        # Get the full mantle mask for the image
        full_mantle_mask = self.get_mantle_mask(image_no)

        # Find indices where the mantle mask is equal to 1 (mantle region)
        mantle_inds = np.where(full_mantle_mask == 1)

        # Iterate through two sets of indices: non-cuttlefish region and mantle region
        for inds in [non_cf_inds, mantle_inds]:
            for ii in range(3):
                # Set the pixel values to 255 (white) in the specified regions for all color channels
                image[:, :, ii][inds] = 255

        # Get the cuttlefish image based on the cuttlefish mask
        cf_image = self._get_object(image, masks[self.get_cf_ind(image_no)])

        # Rotate the cuttlefish image by the angle + 180 degrees
        rotated_cuttlefish = scipy.ndimage.rotate(cf_image, angle + 180)

        # Calculate the median y-coordinate of non-background pixels in the rotated cuttlefish image
        y_med = np.median(np.where(np.array(rotated_cuttlefish)[:, :, 0] != 255)[0])

        # Get the dimensions of the rotated cuttlefish image
        y_size = np.shape(rotated_cuttlefish)[0]
        x_size = np.shape(rotated_cuttlefish)[1]

        # Check if the median y-coordinate is greater than half of the image height
        if y_med > y_size / 2.:
            # If true, return True
            return True
        # If false, return False
        return False
    
    
    def angle(self, image_no, cf = True):
        """
        TODO:

        Args:
            image_no (int): Image number.
            cf (bool): True for cuttlefish body, false for mantle

        Returns:
            TODO
        """
        
        if cf: 
            full_mask = self.get_cuttlefish_mask(image_no)
            mask_ind = self.get_cf_ind(image_no)
        else:
            full_mask = self.get_mantle_mask(image_no)
            mask_ind = self.get_mantle_ind(image_no)
        
        masks = self.load_masks(image_no)
        mask = self._get_object(full_mask, masks[mask_ind])
      
        angle = self._get_ellipse_angle(mask)
        return angle
    
    
    def _get_inscribed_rectangle(self, image_no, cf = True):
        """
        Inscribes a rectangle into the mantle or cuttlefish body.

        Args:
            image_no (int): Image number.
            cf (bool): True for cuttlefish body, false for mantle

        Returns:
            list: max_rect the coordinates of the corner of the inscribed rectnagle
        """
        
        # Determine whether to use cuttlefish or mantle data
        if cf:
            # If 'cf' is True, get the cuttlefish mask and mask index
            full_mask = self.get_cuttlefish_mask(image_no)
            mask_ind = self.get_cf_ind(image_no)
        else:
            # If 'cf' is False, get the mantle mask and mask index
            full_mask = self.get_mantle_mask(image_no)
            mask_ind = self.get_mantle_ind(image_no)

        # Load the masks for the specified image
        masks = self.load_masks(image_no)

        # Extract the specific mask object associated with the mask index
        mask = self._get_object(full_mask, masks[mask_ind])
        
        mask = self._remove_extraneous_parts(mask)
        
        # Calculate the angle of rotation for the mask
        angle = self._get_ellipse_angle(mask)

        if cf:
            # If 'cf' is True, rotate the mask by the angle + 180 degrees
            rotated_mesh = scipy.ndimage.rotate(mask, angle + 180)
        else:
            # If 'cf' is False, rotate the mask by the angle
            rotated_mesh = scipy.ndimage.rotate(mask, angle)

        # Find the contours of the rotated mask
        rotated_contours, hierarchy = cv2.findContours(rotated_mesh.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Get the contour of the rotated mask
        rotated_contour = rotated_contours[0]

        # Calculate the maximum rectangle inscribed within the rotated mask
        max_rect = lir.lir(np.array(rotated_mesh, 'bool'), rotated_contour[:, 0, :])

        # Return the maximum inscribed rectangle
        return max_rect

    
    
    
    def get_cuttlefish_pattern(self, image_no, show_image = False):
        """
        Return a rectangle inscribed in the cuttlefish body.
        
        Args:
            image_no (int): Image number.
       

        Returns:
            np.ndarray: inscribed rectangle image
        """
        
        max_rect = self._get_inscribed_rectangle(image_no)
        pattern = self.get_cuttlefish(image_no,  False, False)[max_rect[1]:max_rect[3]+max_rect[1],max_rect[0]:max_rect[2]+max_rect[0]]
        
        if show_image:
            plt.imshow(pattern)
            plt.show()
        
        return pattern
    

    def get_mantle_pattern(self, image_no, show_image = False):
        """
        Return a rectangle inscribed in the mantle.
        
        Args:
            image_no (int): Image number.
       

        Returns:
            np.ndarray: inscribed rectangle image
        """
        max_rect = self._get_inscribed_rectangle(image_no, False)
        pattern = self.get_mantle(image_no, False)[max_rect[1]:max_rect[3]+max_rect[1],max_rect[0]:max_rect[2]+max_rect[0]]
        if show_image:
            plt.imshow(pattern)
            plt.show()
     
        return pattern
        
        
    def plot_inscribed_rectangle(self, image_no, cf = True):
        """
        Plot the inscribed rectangle on top of the original image.
        
        Args:
            image_no (int): Image number.
            cf (bool): True for cuttlefish body, false for mantle.
       

        Returns:
            np.ndarray: inscribed rectangle image
        """
        # Determine whether to use cuttlefish or mantle data
        if cf:
            # If 'cf' is True, get the cuttlefish image and its inscribed rectangle
            rotated = self.get_cuttlefish(image_no)
            max_rect = self._get_inscribed_rectangle(image_no)
        else:
            # If 'cf' is False, get the mantle image and its inscribed rectangle
            rotated = self.get_mantle(image_no)
            max_rect = self._get_inscribed_rectangle(image_no, cf)

        # Display the rotated image with the inscribed rectangle

        # Show the rotated image
        plt.imshow(rotated)

        # Create a red rectangle for the inscribed rectangle
        rect = Rectangle((max_rect[0], max_rect[1]), max_rect[2], max_rect[3], linewidth=1, edgecolor='r', facecolor='none')

        # Add the inscribed rectangle to the plot
        plt.gca().add_patch(rect)

        # Show the plot with the inscribed rectangle
        plt.show()

        # Display the portion of the rotated image inside the inscribed rectangle

        # Show the specified region of the rotated image
        plt.imshow(rotated[max_rect[1]:max_rect[3] + max_rect[1], max_rect[0]:max_rect[2] + max_rect[0]])

        # Show the specified region
        plt.show()

        # Return the specified region of the rotated image
        return rotated[max_rect[1]:max_rect[3] + max_rect[1], max_rect[0]:max_rect[2] + max_rect[0]]

    
    def plot_bounding_box(self, image_no, cf = True):
        """
        Plot an image with a bounding box and its center.

        Args:
            image_no (int): The image number.
            cf (bool): If True, plots the bounding box for the cytoplasmic region.
                       If False, plots the bounding box for the mantle region.

        This function loads an image, retrieves the bounding box and center coordinates for the specified region,
        and then plots the image with the bounding box drawn in red and the center marked with a point.
        """
        # Load the image for the specified image number
        img = self.load_image(image_no)

        # Get the bounding box for the specified region (cuttlefish or mantle)
        bbox = self.get_bounding_box(image_no, cf=cf)

        # Get the center coordinates of the bounding box
        bbox_center = self.get_bounding_box_center(image_no, cf=cf)

        # Display the image with the bounding box and its center

        # Show the image
        plt.imshow(img)

        # Create a red rectangle for the bounding box
        rect = Rectangle((bbox[0], bbox[1]), bbox[2], bbox[3], linewidth=1, edgecolor='r', facecolor='none')

        # Add the bounding box to the plot
        plt.gca().add_patch(rect)

        # Add a blue dot at the center of the bounding box
        plt.scatter(bbox_center[0], bbox_center[1])

        # Show the plot with the bounding box and center
        plt.show()

    
    def get_cuttlefish_and_mask(self, image_no, cf = True):
        """
        Get both the cuttlefish and the mask of the cuttlefish image 
        Helper function for computing RGB average. 
        
        Args:
            image_no (int): Image number.
            cf (bool): True for cuttlefish body, false for mantle.
       

        Returns:
            np.ndarray: inscribed rectangle image
        """
        
        # Load the masks for the specified image
        masks = self.load_masks(image_no)

        # Load the image for the specified image number
        image = self.load_image(image_no)

        if cf:
            # If 'cf' is True, get the cuttlefish mask index and decode the segmentation mask
            mask_ind = self.get_cf_ind(image_no)
            full_mask = mask_utils.decode(masks[mask_ind]["segmentation"])
        else:
            # If 'cf' is False, get the mantle mask index and decode the segmentation mask
            mask_ind = self.get_mantle_ind(image_no)
            full_mask = mask_utils.decode(masks[mask_ind]["segmentation"])

        # Find indices where the segmentation mask is zero (non-cuttlefish region)
        non_cf_inds = np.where(full_mask == 0)

        # Set the pixel values of the non-cuttlefish region to 255 (white) for all color channels
        for ii in range(3):
            image[:, :, ii][non_cf_inds] = 255

        # Extract the cuttlefish object and mask from the image and segmentation mask
        new_image = self._get_object(image, masks[mask_ind])
        mask = self._get_object(full_mask, masks[mask_ind])

        # Return the modified image and cuttlefish mask
        return new_image, mask
    
    def mean_rgb(self, image_no, cf = True):
        """
        Compute the mean RGB pixel value of cuttlefish body or mantle
        
        Args:
            image_no (int): Image number.
            cf (bool): True for cuttlefish body, false for mantle.
       

        Returns:
            np.ndarray: inscribed rectangle image
        """
       
        image, mask = self.get_cuttlefish_and_mask(image_no, cf)
        return np.mean(image[np.where(mask != 0)])
    
    def mean_bg(self, image_no, seed = 3000):
        """
        Calculate the mean value of the background pattern in an image.

        Args:
            image_no (int): The image number.
            seed (int): A seed for randomization.

        Returns:
            float: The mean value of the background pattern in the specified image.
        """
        
        return np.mean(self.get_background_pattern( image_no, seed, show_image = False))
            


    def distance(self, start_frame, end_frame = False, cf = True):
        """
        Calculates the distance between the center of the cuttlefish between 
        the start frame and end frame. 

        Args:
            image_no (int): Image number.
        """
        if not end_frame:
            end_frame = start_frame + 1
            
        # Getting bounding box center is equivalent to fitting an ellipse and taking its center
        x_start, y_start = self.get_bounding_box_center(start_frame, cf)
        x_end, y_end = self.get_bounding_box_center(end_frame, cf)
       
        # Calculate the Euclidean distance
        return math.sqrt((x_start - x_end)**2 + (y_start - y_end)**2)

    def get_bounding_box_center(self, image_no, cf= True):
        """
        Get the center coordinates of the bounding box.

        Args:
            image_no (int): The image number.
            cf (bool): If True, retrieves the bounding box for the cuttlefish. 
                       If False, retrieves the bounding box for the mantle.

        Returns:
            tuple: A tuple containing the x and y coordinates of the center of the bounding box.
        """
    
        bbox = self.get_bounding_box(image_no, cf)
        
        y_center, x_center = bbox[1] + bbox[3]/2, bbox[0] + bbox[2]/2
        
        return x_center, y_center
    
    def get_bounding_box(self, image_no, cf= True):
        """
        Get the bounding box for a specified region.

        Args:
            image_no (int): The image number.
            cf (bool): If True, retrieves the bounding box for the cytoplasmic region. 
                       If False, retrieves the bounding box for the mantle region.

        Returns:
            list: A list containing the coordinates and dimensions of the bounding box.
        """
    
        if cf:
            # If 'cf' is True, get the index for the cuttlefish mask
            mask_ind = self.get_cf_ind(image_no)
        else:
            # If 'cf' is False, get the index for the mantle mask
            mask_ind = self.get_mantle_ind(image_no)

        # Load the masks for the specified image
        masks = self.load_masks(image_no)

        # Retrieve the mask associated with the determined mask index
        mask = masks[mask_ind]

        # Get the bounding box coordinates and dimensions from the mask
        bbox = mask['bbox']

        # Return the bounding box information
        return bbox
    
    def background_pattern(self, image_no):
        """
        Determines the background pattern for a given image.

        Args:
            image_no (int): Image number.
        """
        # TODO: Determine background pattern
        pass
    
    def rgb2gray(self, rgb):
        """
        Convert an RGB image to grayscale using luminance conversion.

        Args:
            rgb (numpy.ndarray): An RGB image represented as a NumPy array.

        Returns:
            numpy.ndarray: Grayscale version of the input image.
        """
        r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
        gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

        return gray


    def texture_statistics(self, image_no, img_size = 224, cf = False, background = False, seed = 3000, spatial_corr_width = 7):
        """
        Gets Portilla and Simoncelli textures statistics via Plenoptic.
        Computes statistics on the output of either get_cuttlefish_pattern or 
        get_mantle_pattern. 

        Args:
            image_no (int): Image number.
            cf (bool): True for cuttlefish pattern, false for mantle pattern.
            spatial_corr_width: Width of the spatial window of texture statistics. 
                                Default in Portilla and Simoncelli model is 7.
        
        Returns:
            stats (np.array): Array of texture statistics
            
        """
        
        if background:
            # If 'background' is True, get the background pattern image
            img = self.get_background_pattern(image_no, seed=seed, show_image=False)
        elif cf:
            # If 'background' is False and 'cf' is True, get the cuttlefish pattern
            img = self.get_cuttlefish_pattern(image_no)
        else:
            # If neither 'background' nor 'cf' is True, get the mantle pattern
            img = self.get_mantle_pattern(image_no)

        # Convert the image to grayscale
        gray_img = self.rgb2gray(img)

        # Resize the grayscale image (texture model's size preference)
        resized_img = gray_img[:img_size, :img_size]

        # Prepare the image for PyTorch
        img = torch.from_numpy(np.array([[resized_img]])).float()

        # Initialize the minimal model with specific parameters
        model_min = PortillaSimoncelliMinimalStats([img_size, img_size], n_scales=4,
                                                  n_orientations=4,
                                                  spatial_corr_width=spatial_corr_width,
                                                  use_true_correlations=False)

        # Extract the dictionary indicating which statistics are in the minimal set
        mask_min_dict = model_min.statistics_mask

        # Convert the dictionary of statistics to a vector
        mask_min_vec = model_min.convert_to_vector(mask_min_dict)

        # Compute statistics from the image
        stats = model_min(img)

        # Extract minimal statistics
        po_stats = stats[mask_min_vec]

        # Create a dictionary to map statistics to labels
        dict_of_stat_labels = {}
        for key in mask_min_dict:
            dict_of_stat_labels[key] = np.where(mask_min_dict[key], key, 'False')

        # Flatten and extract labels
        all_label_vec = []
        for (_, val) in dict_of_stat_labels.items():
            all_label_vec += list(np.ndarray.flatten(np.squeeze(val)))
        label_vec = list(np.array(all_label_vec)[mask_min_vec.squeeze()])

        # Return the minimal statistics and their corresponding labels
        return np.array(po_stats), label_vec

    
    def plot_density_scatter(self, points):
        """
        Plot a density scatter plot of points.

        Args:
            points (list of tuples): List of (x, y) points.

        This function calculates the point density and creates a scatter plot with point density represented by color.
        """
        # Extract the x and y coordinates from the 'points' list
        x = np.array(points)[:, 0]
        y = np.array(points)[:, 1]

        # Calculate the point density by stacking the x and y coordinates
        xy = np.vstack([x, y])

        # Use a Gaussian Kernel Density Estimation (KDE) to estimate point density
        z = gaussian_kde(xy)(xy)

        # Create a new figure and axis for the scatter plot
        fig, ax = plt.subplots()

        # Create a scatter plot of the points with point density represented by color
        ax.scatter(x, 1582 - y, c=z, s=100)

        # Set the x and y axis limits
        plt.xlim(0, 2380)
        plt.ylim(0, 1582)

        # Display the scatter plot
        plt.show()
        
        
                
    def get_values_across_session(self, func):
        """
        Get values computed across multiple sessions and store/retrieve them.

        Args:
            func (callable): A function to compute values for each session.

        This function computes and stores values obtained from running the given function across multiple sessions.
        It caches the results in a file for future use.
        """
        
        # Define the path for storing or loading cached values
        path = self.storage_path + self.images_folder.replace('/', '') + '_' + func.__name__ + '.npy'

        # Check if the cached values file exists
        if os.path.exists(path):
            # If it exists, load and return the cached values
            return np.load(path)

        # If the file doesn't exist, the values need to be computed
        print('Values have not been computed before — this may take some time.')
        values = []

        # Iterate through frames and compute values using the provided function
        for ii in range(1, self.num_frames()):
            if ii % 100 == 0:
                print('Frame:', str(ii), '/', str(self.num_frames()))

            try:
                # Attempt to compute values using the provided function
                values.append(func(ii))

            except:
                # If there's an error, use the last computed value and note that it's missing
                values.append(values[-1])

        # Save the computed values to the specified path for future use
        np.save(path, values)

        # Return the computed values
        return values

    
    def mean_image(self, image_no):
        """
        Calculate the mean value of an image.

        Args:
            image_no (int): The image number.

        Returns:
            float: The mean value of the specified image.
        """
        v = self.load_image(image_no)
        return np.mean(v)
    
    def isOverlap(self, interval1, interval2):
        """
        Check if two intervals overlap.

        Args:
            interval1 (Interval): The first interval.
            interval2 (Interval): The second interval.

        Returns:
            bool: True if the intervals overlap, False otherwise.
        """
        return interval1.end > interval2.start and interval1.start < interval2.end


    
    def get_background_pattern_bbox(self, image_no, seed = 3000):
        """
        Get the bounding box for a background pattern.

        Args:
            image_no (int): The image number.
            seed (int): A seed for randomization.

        Returns:
            list: A list containing the coordinates and dimensions of the bounding box.
        """
        # Set the random seed for repeatability
        random.seed(seed)

        # Load the image for the specified image number
        img = self.load_image(image_no)

        # Get the dimensions of the image
        image_height, image_width, _ = np.shape(img)

        # Get the coordinates and dimensions of the bounding box for the foreground object
        point_x, point_y, bbox_width, bbox_height = self.get_bounding_box(image_no)

        # Initialize empty lists to store potential background pattern positions
        xs = []
        ys = []

        # Define intervals for the foreground object in both x and y directions
        prop_x_int = Interval(point_x, point_x + bbox_width)
        prop_y_int = Interval(point_y, point_y + bbox_height)

        # Loop through possible positions within the image
        for ii in range(250, image_width - bbox_width - 300):
            for jj in range(200, image_height - bbox_height - 150):
                # Exclude certain regions (ii < 420 and jj < 400) from consideration
                if ii < 420 and jj < 400:
                    pass
                else:
                    # Create intervals for the current position in x and y
                    x_interval = Interval(ii, ii + bbox_width)
                    y_interval = Interval(jj, jj + bbox_height)

                    # Check for overlap with the foreground object
                    if self.isOverlap(x_interval, prop_x_int) and self.isOverlap(y_interval, prop_y_int):
                        pass
                    else:
                        # If no overlap, store the position
                        xs.append(ii)
                        ys.append(jj)

        # Randomly select an index for the potential positions
        ind = random.randint(0, len(xs))

        # Return the coordinates and dimensions of the selected background pattern
        return [xs[ind], ys[ind], bbox_width, bbox_height]
    
    
    def get_background_pattern(self, image_no, seed = 3000, show_image = True):
        """
        Get the background pattern from an image.

        Args:
            image_no (int): The image number.
            seed (int): A seed for randomization.
            show_image (bool): Whether to display the background pattern.

        Returns:
            numpy.ndarray: The background pattern.
        """
        # Load the image for the specified image number
        img = self.load_image(image_no)

        # Get the bounding box for the background pattern with the specified seed
        bbox = self.get_background_pattern_bbox(image_no, seed=seed)

        # Extract the background pattern from the image using the bounding box coordinates
        pattern = img[bbox[1]:bbox[1]+bbox[3], bbox[0]:bbox[0]+bbox[2]]

        # If show_image is True, display the background pattern
        if show_image:
            plt.imshow(pattern)
            plt.show()

        # Return the extracted background pattern
        return pattern
    
    def plot_background_pattern(self, image_no, seed = 3000):
        """
        Plot the image with bounding boxes for the foreground and background patterns.

        Args:
            image_no (int): The image number.
            seed (int): A seed for randomization.

        This function plots the image with bounding boxes for both foreground and background patterns.
        """
        # Load the image for the specified image number
        img = self.load_image(image_no)

        # Display the image
        plt.imshow(img)

        # Get the bounding box for the specified image
        bbox = self.get_bounding_box(image_no)

        # Get the bounding box for the background pattern with a specified seed
        bg_sample = self.get_background_pattern_bbox(image_no, seed)

        # Create a red rectangle for the foreground bounding box
        rect = Rectangle((bbox[0], bbox[1]), bbox[2], bbox[3], linewidth=1, edgecolor='r', facecolor='none')

        # Add the foreground bounding box to the plot
        plt.gca().add_patch(rect)

        # Create a blue rectangle for the background pattern bounding box
        rect2 = Rectangle((bg_sample[0], bg_sample[1]), bg_sample[2], bg_sample[3], linewidth=1, edgecolor='b', facecolor='none')

        # Add the background pattern bounding box to the plot
        plt.gca().add_patch(rect2)

        # Display the plot with the bounding boxes
        plt.show()
        
    def swap_1_and_0(self, mask_array):
        """
        Swap 1s and 0s in a binary mask array.

        Args:
            mask_array (numpy.ndarray): A binary mask array with values 0 and 1.

        Returns:
            numpy.ndarray: A copy of the input array with 1s and 0s swapped.
        """
        # Create a copy of the mask array
        swapped_array = np.copy(mask_array)

        # Swap 1 and 0 in the copied array
        swapped_array[mask_array == 1] = 0
        swapped_array[mask_array == 0] = 1
        return swapped_array


    

    def preprocess_image(self, img,img_width=224,img_height=224):
        # From Laurent lab: https://gitlab.mpcdf.mpg.de/mpibr/laur/cuttlefish/texture-code/-/blob/main/featurespace.py?ref_type=heads


        # from keract import get_activations
        from tensorflow.keras.preprocessing.image import load_img, img_to_array
        img = img_to_array(img)
        img = cv2.resize(img, dsize=(img_width,img_height), interpolation=cv2.INTER_NEAREST)

        if len(img.shape)==2:
            img = img.reshape((img.shape[0], img.shape[1], 1))
            img=np.concatenate((img, img,img),axis=2)


        img = img.reshape((1, img.shape[0], img.shape[1], img.shape[2]))
        img = img.astype('float32')
        img = vgg19.preprocess_input(img)
        return img




    def gram_matrix(self, x):
        # From Laurent lab: https://gitlab.mpcdf.mpg.de/mpibr/laur/cuttlefish/texture-code/-/blob/main/featurespace.py?ref_type=heads

        x = np.squeeze(x)
        dims = x.shape
        features = np.reshape(np.transpose(x,(2,0,1)),(dims[2],dims[0]*dims[1]))
        gram = np.dot(features, np.transpose(features))
        return gram


    def style_rep(self, style):
        # From Laurent lab: https://gitlab.mpcdf.mpg.de/mpibr/laur/cuttlefish/texture-code/-/blob/main/featurespace.py?ref_type=heads

        style=np.squeeze(style)
        dims = style.shape
        S = self.gram_matrix(style)
        size=dims[0]*dims[1]*dims[2]
        return S / (size ** 2)  #think about this normalization some more. Now it's to sort of match the texture synthesis


    def get_vgg19_activations(self, image_no, model = 'DEFAULT', LAYER = 'block5_conv1', sz=224):
        # based on Laurent lab: https://gitlab.mpcdf.mpg.de/mpibr/laur/cuttlefish/texture-code/-/blob/main/featurespace.py?ref_type=heads
        img = self.get_cuttlefish(image_no, show_image = False, bg_color = 127)
        if model == 'DEFAULT':
            model = vgg19.VGG19(weights='imagenet', include_top=False)

        currImg = self.preprocess_image(img,sz,sz)
        model.compile(loss="categorical_crossentropy", optimizer="adam")
        activations = get_activations(model, currImg, layer_names=[LAYER], auto_compile=True)[LAYER]

        return activations

    def laurentlab_texrep(self, acts):
        # based on Laurent lab: https://gitlab.mpcdf.mpg.de/mpibr/laur/cuttlefish/texture-code/-/blob/main/featurespace.py?ref_type=heads
        fifth = acts.max(axis=(1,2))
        vggRep = fifth.ravel()
        return vggRep


    def get_gram_matrix(self, image_no):
        acts = self.get_vgg19_activations(image_no)
        return self.gram_matrix(acts)

    def get_ll_texrep(self, image_no):
        acts = self.get_vgg19_activations(image_no)
        return self.laurentlab_texrep(acts)

    def get_style_rep(self, image_no):
        acts = self.get_vgg19_activations(image_no)
        return self.style_rep(acts)




        
        
    