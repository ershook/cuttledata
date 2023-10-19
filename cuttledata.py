import os
import json
import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage
import skimage.transform
from pycocotools import mask as mask_utils
from matplotlib.patches import Rectangle
from scipy.stats import gaussian_kde


import largestinteriorrectangle as lir
from PortillaSimoncelliMinimalStats import *
import plenoptic as po

import cv2

import warnings
warnings.filterwarnings('ignore')




class CuttleData:
    def __init__(self, images_folder):
        """
        Initializes the CuttleData object.

        Args:
            images_folder (str): Folder containing the behavior images..
        """
        # Determine what os system code is running on: 
        
        if os.path.exists('/mnt/smb/locker/axel-locker/'):
            prefix = '/mnt/smb/locker/axel-locker/'
        elif os.path.exists('/Volumes/axel-locker/'):
            prefix = '/Volumes/axel-locker/'
        elif os.path.exists('Z:/cuttlefish'):
            prefix = 'Z:/'
        else:
            raise Exception("Can't find path to data -- are you sure axel-locker is mounted?")
        
        
        self.images_folder = images_folder
        self.images_path = prefix + 'cuttlefish/CUTTLEFISH_BEHAVIOR/2023_BEHAVIOR/E-ink_Tank/'+images_folder+'/Tifs_downsampled/'
        self.path_to_masks = prefix + 'es3773/data/sam_data/' + images_folder + '/'
        knob_inds = np.load(prefix + 'es3773/data/knob_inds.npy')
        self.knob_inds = (knob_inds[0], knob_inds[1])
        self.storage_path = prefix + 'axel-locker/cuttlefish/CUTTLEFISH_BEHAVIOR/cuttle_data_storage/'
        
    
    
    def num_frames(self):
        
        if not hasattr(self, "n_frames"): # This can be a slow operation so only ever do it once.
            self.n_frames = len(os.listdir(self.images_path))
        return self.n_frames
    
    def num_masks(self): 
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
        masks = self.load_masks(image)
        
        areas = [masks[ii]['area'] for ii in range(len(masks))]
        
        # Determine the index of the inverse mask ie. the mask with the cuttlefish 0 and bg 1 
        ind = np.argmax(areas)
        max_ind = ind
        if np.sum(np.array(areas)>1e6)>1:
            areas[ind]=0 # Want the second biggest mask
            ind=np.argmax(areas) # This is the inverse mask

        good_inds = []
        good_areas = []
        for mask_ii in range(len(masks)):
            
            overlap_w_inv = np.sum(mask_utils.decode(masks[mask_ii]["segmentation"])[180:1450,300:2000]*(1-mask_utils.decode(masks[ind]["segmentation"])[180:1450,300:2000]))
            
            # Determine which masks are within the inverse mask 
            # Knob inds are indices of the inflow valve -- if don't ignore these sometimes will get the knob as a mask
            if areas[mask_ii] < 200000 and areas[mask_ii]>10000 and mask_ii != max_ind and overlap_w_inv >  200 and np.sum(mask_utils.decode(masks[mask_ii]["segmentation"])[self.knob_inds]) < 500:     
                good_inds.append(mask_ii)
                good_areas.append(areas[mask_ii])
                
        # Sort size from biggest to smallest 
        good_inds = np.array(good_inds)[np.argsort(good_areas)][::-1]

        # Dleelete duplicates 
        if len(good_areas)>1 and np.abs(good_areas[0]-good_areas[1])<1000:
            good_areas = np.delete(good_areas,1)
            good_inds = np.delete(good_inds,1)

        good_inds = list(good_inds)
        
        
        if  good_areas[0]<100000: # Missing cuttlefish mask, fill index 0 with 'ERR'
            good_inds.insert(0, 'ERR')
        elif len(good_areas)>1 and good_areas[1]<60000: # Missing mantle mask, fill index 1 with 'ERR'
            good_inds.insert(1, 'ERR')
        
    
        ### On rare occasions an inverse mask doesn't exist in that case we resort to looking at the size of the masks
        if len(good_areas) == 0:

            good_inds = []
            max_area = 0
            for ii in range(len(masks)):
                area = masks[ii]['area']
                if area >80000:
                    seg = mask_utils.decode(masks[ii]["segmentation"])
                    x_mar = np.mean(seg, axis = 0)
                    y_mar = np.mean(seg, axis = 1)
                    if max(x_mar)<.5 and max(y_mar)<.5 and area<2e6:
                        if masks[ii]['area']> max_area: 
                            good_inds.insert(0, ii)
                            max_area = masks[ii]['area']
                        else:
                            good_inds.append(ii)

            try:
                full_cuttlefish_mask = mask_utils.decode(masks[good_inds[0]]["segmentation"])
            except:
                full_cuttlefish_mask=[]
            if len(full_cuttlefish_mask) == 0:
                good_mask_inds.append(['ERR','ERR'])
            else:
                temp_areas = []
                temp_indices = []
                good_masks = []
                for ii in range(len(masks)):

                    mask_1 =mask_utils.decode(masks[ii]["segmentation"])

                    area = masks[ii]['area']

                    if   area> 60000 and area<1e6 and np.sum(np.multiply(mask_1,full_cuttlefish_mask)) >300  and np.sum(np.multiply(mask_1,full_cuttlefish_mask)) - int(min(np.sum(mask_1),np.sum(full_cuttlefish_mask))) <300:

                        temp_areas.append(area)
                        temp_indices.append(ii)
                        good_masks.append(mask_1)

                order = np.argsort(temp_areas)

                good_inds = list(np.unique(list([good_inds[0]]+ list(np.array((temp_indices))[order]))))


                if len(good_inds) > 2:

                    ### Check for repeats
                    final_inds = []
                    for ii in range(len(good_inds)):
                        mask_1 = mask_utils.decode(masks[good_inds[ii]]["segmentation"])
                        for jj in range(ii+1, len(good_inds)):
                           

                            mask_2 = mask_utils.decode(masks[good_inds[jj]]["segmentation"])
                            if int(np.sum(np.multiply(mask_1, mask_2)-mask_1))/3.>50000:
                                final_inds += [good_inds[ii], good_inds[jj]]
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

            temp_inds = good_mask_inds[-1]
            
            temp_areas = []
            if len(temp_inds)>1:
                for ii_ in temp_inds:
                    temp_areas.append(masks[ii_]['area'])

                inds = np.argsort(temp_areas)
                final_inds = np.array(temp_inds)[inds][::-1]
            else:
                final_inds = temp_inds


            good_inds = final_inds
        return good_inds

    def decode_mask(self, mask):
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
    
        mask_path = f"{self.path_to_masks}{str(image_no).zfill(5)}.json"
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
            image_file = f"{self.images_path}{str(image_no).zfill(5)}.tif"
            image = cv2.imread(image_file)
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

    def get_cuttlefish(self, image_no, title = False, show_image=True, correct_flip = True):
        """
        Retrieves the rotated cuttlefish image and its angle.

        Args:
            image_no (int): Image number.
            show_image (bool): Whether to display the rotated cuttlefish image. Default is True.

        Returns:
            tuple: Tuple containing the rotated cuttlefish image and its angle.
        """
        full_cuttlefish_mask = self.get_cuttlefish_mask(image_no)
        masks = self.load_masks(image_no)
        image = self.load_image(image_no)
        cf_mask_ind = self.get_cf_ind(image_no)
        non_cf_inds = np.where(full_cuttlefish_mask == 0)

        for ii in range(3):
            image[:, :, ii][non_cf_inds] = 255

        cf_image = self._get_object(image, masks[cf_mask_ind])
        cf_mask = self._get_object(full_cuttlefish_mask, masks[cf_mask_ind])

        try:
            angle = self._get_ellipse_angle(cf_mask)
        except:
            angle = 0

        rotated_cuttlefish = skimage.transform.rotate(cf_image, angle + 180, resize=True, cval=255)

        if correct_flip and self.get_mantle_ind(image_no) != 'ERR' and self._is_cuttlefish_upside_down(image_no, masks, angle,
                                                                                      full_cuttlefish_mask,
                                                                                      non_cf_inds):
            rotated_cuttlefish = skimage.transform.rotate(rotated_cuttlefish, 180, resize=True, cval=255)

        if show_image:
            plt.imshow(rotated_cuttlefish)
            plt.axis('off')
            if title:
                plt.title(title)
                plt.savefig('/mnt/smb/locker/axel-locker/es3773/data/'+str(title)+'.png')
          
        return rotated_cuttlefish
    
    def get_mantle(self, image_no, show_image = True, title = False):
        """
        Retrieves the rotated mantle image.

        Args:
            image_no (int): Image number.
            show_image (bool): Whether to display the rotated cuttlefish image. Default is True.

        Returns:
            tuple: Tuple containing the rotated cuttlefish image and its angle.
        """
        
        mantle_mask_ind = self.get_mantle_ind(image_no)
        if mantle_mask_ind != 'ERR':
        
            full_mantle_mask = self.get_mantle_mask(image_no)
            masks = self.load_masks(image_no)
            image = self.load_image(image_no)

            non_mantle_inds = np.where(full_mantle_mask == 0)
            for ii in range(3):
                image[:, :, ii][non_mantle_inds] = 255
            mantle_image = self._get_object(image, masks[mantle_mask_ind])

            mantle_mask = self._get_object(full_mantle_mask, masks[mantle_mask_ind])
            angle = self._get_ellipse_angle(mantle_mask)

            rotated_mantle = skimage.transform.rotate(mantle_image, angle, resize=True, cval=255)
            if show_image:
                plt.axis('off')
                plt.imshow(rotated_mantle)
                
                if title:
                    plt.title(title)
                    plt.savefig('/mnt/smb/locker/axel-locker/es3773/data/'+str(title)+'.png')
            return rotated_mantle
    
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
        return self._get_mask_inds(image_no)[0]

    def get_mantle_ind(self, image_no):
        """
        Retrieves the mantle index for a given image.

        Args:
            image_no (int): Image number.

        Returns:
            int or str: Mantle index or 'ERR' if not available.
        """
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
        image = self.load_image(image_no)
        full_mantle_mask = self.get_mantle_mask(image_no)
        mantle_inds = np.where(full_mantle_mask == 1)

        for inds in [non_cf_inds, mantle_inds]:
            for ii in range(3):
                image[:, :, ii][inds] = 255

        cf_image = self._get_object(image, masks[self.get_cf_ind(image_no)])

        rotated_cuttlefish = scipy.ndimage.rotate(cf_image, angle + 180)
        y_med = np.median(np.where(np.array(rotated_cuttlefish)[:, :, 0] != 255)[0])
        y_size = np.shape(rotated_cuttlefish)[0]
        x_size = np.shape(rotated_cuttlefish)[1]

        if y_med > y_size / 2.:
            return True
        return False
    
    
    
    def _get_inscribed_rectangle(self, image_no, cf = True):
        """
        Inscribes a rectangle into the mantle or cuttlefish body.

        Args:
            image_no (int): Image number.
            cf (bool): True for cuttlefish body, false for mantle

        Returns:
            list: max_rect the coordinates of the corner of the inscribed rectnagle
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
        
        if cf:
            rotated_mesh = scipy.ndimage.rotate(mask, angle+180)
        else:
            rotated_mesh = scipy.ndimage.rotate(mask, angle)
            
        rotated_contours, hierarchy  = cv2.findContours(rotated_mesh.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        rotated_contour = rotated_contours[0]
        
        max_rect = lir.lir(np.array(rotated_mesh,'bool'), rotated_contour[:, 0, :])
        
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
        if cf: 
            rotated= self.get_cuttlefish(image_no)
            max_rect = self._get_inscribed_rectangle(image_no)
        else: 
            rotated = self.get_mantle(image_no)
            max_rect = self._get_inscribed_rectangle(image_no, cf)
        plt.imshow(rotated)
        rect = Rectangle((max_rect[0], max_rect[1]), max_rect[2], max_rect[3],linewidth=1,edgecolor='r',facecolor='none')
        plt.gca().add_patch(rect)
        plt.show()
        plt.imshow(rotated[max_rect[1]:max_rect[3]+max_rect[1],max_rect[0]:max_rect[2]+max_rect[0]])
        plt.show()
        return rotated[max_rect[1]:max_rect[3]+max_rect[1],max_rect[0]:max_rect[2]+max_rect[0]]
    
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
        
        masks = self.load_masks(image_no)
        image = self.load_image(image_no)
        
        if cf: 
            mask_ind = self.get_cf_ind(image_no)
            full_mask = mask_utils.decode(masks[mask_ind]["segmentation"])
        else:
            mask_ind = self.get_mantle_ind(image_no)
            full_mask = mask_utils.decode(masks[mask_ind]["segmentation"])
            
        non_cf_inds = np.where(full_mask == 0)

        for ii in range(3):
            image[:, :, ii][non_cf_inds] = 255

        new_image = self._get_object(image, masks[mask_ind])
        mask = self._get_object(full_mask, masks[mask_ind])
        
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
    
        
        if cf:
            mask_ind = self.get_cf_ind(image_no)
        else:
            mask_ind = self.get_mantle_ind(image_no)
        
        masks = self.load_masks(image_no)
        mask = masks[mask_ind]
        bbox = mask['bbox']
        y_center, x_center = bbox[1] + bbox[3]/2, bbox[0] + bbox[2]/2
        return x_center, y_center
    
    def background_pattern(self, image_no):
        """
        Determines the background pattern for a given image.

        Args:
            image_no (int): Image number.
        """
        # TODO: Determine background pattern
        pass
    
    def rgb2gray(self, rgb):
        

        r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
        gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

        return gray


    def compute_texture_statistics(self, image_no, cf = False, spatial_corr_width = 7):
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
        img_size = 224
        if cf: 
            img = self.get_cuttlefish_pattern(image_no) # Get the cuttlefish pattern
        else:
            img = self.get_mantle_pattern(image_no) # Get the mantle pattern
            
        gray_img = self.rgb2gray(img) # Convert image to gray scale
        resized_img = gray_img[:img_size,:img_size] # Texture model is finicky in terms of size
        img = torch.from_numpy(np.array([[resized_img]])).float() # Prepare image for torch
        
        
        # Initialize the minimal model. Use same params as paper
        model_min = PortillaSimoncelliMinimalStats([img_size,img_size], n_scales=4,
                                                  n_orientations=4,
                                                  spatial_corr_width=spatial_corr_width,
                                                  use_true_correlations=False)

        # Extract the dictionary indicating which statistics are the minimal set
        mask_min_dict = model_min.statistics_mask
        
        mask_min_vec = model_min.convert_to_vector(mask_min_dict)
        stats = model_min(img)
        po_stats = stats[mask_min_vec]
        
        dict_of_stat_labels = {}
        for key in mask_min_dict:

            dict_of_stat_labels[key] = np.where(mask_min_dict[key], key, 'False')
            
        # Below is the equivalent of using .convert_to_vector but for array of strings 
        # stat_label_vec = model_min.convert_to_vector(dict_of_stat_labels)
        all_label_vec = []
        for (_, val) in dict_of_stat_labels.items():
            all_label_vec += list(np.ndarray.flatten(np.squeeze(val)))
            
        label_vec= list(np.array(all_label_vec)[mask_min_vec.squeeze()])
        
        return np.array(po_stats), label_vec

    
    def plot_density_scatter(self, points):
        
        x = np.array(points)[:,0]
        y = np.array(points)[:,1]
        
        # Calculate the point density
        xy = np.vstack([x,y])
        z = gaussian_kde(xy)(xy)

        fig, ax = plt.subplots()
        ax.scatter(x, 1582 - y, c=z, s=100)
        plt.xlim(0, 2380)
        plt.ylim(0, 1582)
        plt.show()
        
        
                
    def get_values_across_session(self, func):
        
        path = self.storage_path + self.images_folder.replace('/','') + '_' + func.__name__ + '.npy'
        if os.path.exists(path):
            return np.load(path  )    
                           
                           
        print('values have not been computed before -- may take a bit of time.')
        values = []
    
        for ii in range(1,self.num_frames()): 
            
            if ii%100 == 0:
                print('frame: ', str(ii), '/', str(self.num_frames()))
            try:
                values.append(func(ii))
                
            except:
                values.append(values[-1]) 
                # If don't have mask just copy the last center point and note that its missing
                
        np.save(path, values)
        return values
        

    
    def mean_image(self, image_no):
        v = self.load_image(image_no)
        return np.mean(v)
              
        
        
    
