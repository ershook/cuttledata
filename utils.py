import numpy as np
import cv2
from PIL import Image
import os

def determine_server_path():
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
    return prefix


def available_sessions():
    prefix = determine_server_path()
    return os.listdir(prefix + 'cuttlefish/CUTTLEFISH_BEHAVIOR/cuttle_data_storage/sam_data')


def LAPV(img):
    """Implements the Variance of Laplacian (LAP4) focus measure
    operator. Measures the amount of edges present in the image.

    :param img: the image the measure is applied to
    :type img: numpy.ndarray
    :returns: numpy.float32 -- the degree of focus
    """
    return np.std(cv2.Laplacian(img, cv2.CV_64F)) ** 2


def LAPM(img):
    """Implements the Modified Laplacian (LAP2) focus measure
    operator. Measures the amount of edges present in the image.

    :param img: the image the measure is applied to
    :type img: numpy.ndarray
    :returns: numpy.float32 -- the degree of focus
    """
    kernel = np.array([-1, 2, -1])
    laplacianX = np.abs(cv2.filter2D(img, -1, kernel))
    laplacianY = np.abs(cv2.filter2D(img, -1, kernel.T))
    return np.mean(laplacianX + laplacianY)


def TENG(img):
    """Implements the Tenengrad (TENG) focus measure operator.
    Based on the gradient of the image.

    :param img: the image the measure is applied to
    :type img: numpy.ndarray
    :returns: numpy.float32 -- the degree of focus
    """
    gaussianX = cv2.Sobel(img, cv2.CV_64F, 1, 0)
    gaussianY = cv2.Sobel(img, cv2.CV_64F, 1, 0)
    return np.mean(gaussianX * gaussianX +
                      gaussianY * gaussianY)


def MLOG(img):
    """Implements the MLOG focus measure algorithm.

    :param img: the image the measure is applied to
    :type img: numpy.ndarray
    :returns: numpy.float32 -- the degree of focus
    """
    return np.max(cv2.convertScaleAbs(cv2.Laplacian(img, 3)))


def pad_images_in_folder(folder_path):
    max_width = 0
    max_height = 0
    image_paths = []

    # Step 1: Find the max dimensions
    for filename in os.listdir(folder_path):
        if filename.endswith(".tif"):
            image_path = os.path.join(folder_path, filename)
            image_paths.append(image_path)
            print("checking: " + str(filename))
            with Image.open(image_path) as img:
                width, height = img.size
                max_width = max(max_width, width)
                max_height = max(max_height, height)

    # Create new folder for padded images
    padded_folder_path = folder_path + "_padded"
    if not os.path.exists(padded_folder_path):
        os.makedirs(padded_folder_path)

    # Step 2: Pad each image
    for image_path in image_paths:
        print("Padding image:", image_path)
        with Image.open(image_path) as img:
            width, height = img.size
            # Calculate padding sizes
            left = (max_width - width) // 2
            right = max_width - width - left
            top = (max_height - height) // 2
            bottom = max_height - height - top

            # Create a new image with the desired size and black background
            new_img = Image.new('RGB', (max_width, max_height), (0, 0, 0))
            new_img.paste(img, (left, top))

            # Save the padded image in the new folder
            new_image_path = os.path.join(padded_folder_path, os.path.basename(image_path))
            new_img.save(new_image_path)




def resize(rgb, downsample_factor):
    return cv2.resize(rgb, (0, 0), fx=downsample_factor, fy=downsample_factor, interpolation=cv2.INTER_AREA)


def apply_color(resized, spectrum, grading):
    # first apply the appropriate spectrum (raw, ocean sim, daylight sim)

    if spectrum == "daylight":
        #print("Daylight_sim")
        resized=resized*256

        popt_red = np.array([2.49977028e-01, 1.12525005e+00, 5.18763052e+03])
        popt_green = np.array([5.76949661e-01, 1.04726996e+00, 1.47926918e+03])
        popt_blue = np.array([7.88687346e+00, 8.20434666e-01, - 9.25428880e+03])

        red_channel = resized[:, :, 0]
        green_channel = resized[:, :, 1]
        blue_channel = resized[:, :, 2]

        # Apply the fitted functions to each channel
        red_output = nonlinear_func(red_channel, *popt_red)
        green_output = nonlinear_func(green_channel, *popt_green)
        blue_output = nonlinear_func(blue_channel, *popt_blue)

        red_output_clipped = np.clip(red_output, 0, 65535)
        green_output_clipped = np.clip(green_output, 0, 65535)
        blue_output_clipped = np.clip(blue_output, 0, 65535)

        colored = np.stack((red_output_clipped, green_output_clipped, blue_output_clipped), axis=-1)
        colored = colored/256
    if spectrum == "ocean":
        #print("Ocean_sim")

        red_channel = resized[:, :, 0]
        green_channel = resized[:, :, 1]
        blue_channel = resized[:, :, 2]

        # Apply the fitted functions to each channel
        red_output = red_channel * (0.3 / 0.433333333) *2
        green_output = green_channel * (0.4 / 0.433333333) *2
        blue_output = blue_channel * (0.6 / 0.433333333) *2

        red_output_clipped = np.clip(red_output, 0, 65535)
        green_output_clipped = np.clip(green_output, 0, 65535)
        blue_output_clipped = np.clip(blue_output, 0, 65535)

        colored = np.stack((red_output_clipped, green_output_clipped, blue_output_clipped), axis=-1)

    if spectrum == "raw":
        # scale 12 bit to 16 bit
        #colored = resized * (16 / 12)
        colored = resized

    if spectrum == "pop":
        resized = resized*256

        popt_red = np.array([2.49977028e-01, 1.12525005e+00, 5.18763052e+03])
        popt_green = np.array([5.76949661e-01, 1.04726996e+00, 1.47926918e+03])
        popt_blue = np.array([7.88687346e+00, 8.20434666e-01, - 9.25428880e+03])

        red_channel = resized[:, :, 0]
        green_channel = resized[:, :, 1]
        blue_channel = resized[:, :, 2]

        # Apply the fitted functions to each channel
        red_output = nonlinear_func(red_channel, *popt_red)
        green_output = nonlinear_func(green_channel, *popt_green)
        blue_output = nonlinear_func(blue_channel, *popt_blue)

        red_output_clipped = np.clip(red_output, 0, 65535)
        green_output_clipped = np.clip(green_output, 0, 65535)
        blue_output_clipped = np.clip(blue_output, 0, 65535)

        colored = np.stack((red_output_clipped, green_output_clipped, blue_output_clipped), axis=-1)

        #colored = np.clip((2.1 * colored - 32000), 0, 65535)
        #colored = np.clip((1.8 * colored - 24000), 0, 65535)
        #colored = np.clip((1.7 * colored - 20000), 0, 65535)
        colored = 1.6 * colored - 14000

        red_channel = colored[:, :, 0]
        green_channel = colored[:, :, 1]
        blue_channel = colored[:, :, 2]

        # Apply the fitted functions to each channel
        red_output = red_channel
        green_output = green_channel - 2000
        blue_output = blue_channel - 2600

        red_output_clipped = np.clip(red_output, 0, 65535)
        green_output_clipped = np.clip(green_output, 0, 65535)
        blue_output_clipped = np.clip(blue_output, 0, 65535)

        colored = np.stack((red_output_clipped, green_output_clipped, blue_output_clipped), axis=-1)
        colored = np.float32(colored)
        colored = colored/256
        #print("pop")
    if spectrum == "rawpop":
        resized = resized*256

        popt_red = np.array([2.49977028e-01, 1.12525005e+00, 5.18763052e+03])
        popt_green = np.array([5.76949661e-01, 1.04726996e+00, 1.47926918e+03])
        popt_blue = np.array([7.88687346e+00, 8.20434666e-01, - 9.25428880e+03])

        red_channel = resized[:, :, 0]
        green_channel = resized[:, :, 1]
        blue_channel = resized[:, :, 2]

        # Apply the fitted functions to each channel
        red_output = nonlinear_func(red_channel, *popt_red)
        green_output = nonlinear_func(green_channel, *popt_green)
        blue_output = nonlinear_func(blue_channel, *popt_blue)

        red_output_clipped = np.clip(red_output, 0, 65535)
        green_output_clipped = np.clip(green_output, 0, 65535)
        blue_output_clipped = np.clip(blue_output, 0, 65535)

        colored = np.stack((red_output_clipped, green_output_clipped, blue_output_clipped), axis=-1)

        #colored = np.clip((2.1 * colored - 32000), 0, 65535)
        #colored = np.clip((1.8 * colored - 24000), 0, 65535)
        #colored = np.clip((1.7 * colored - 20000), 0, 65535)
        #colored = 1.6 * colored - 20000
        colored = 1.4 * colored - 25000
        red_channel = colored[:, :, 0]
        green_channel = colored[:, :, 1]
        blue_channel = colored[:, :, 2]

        # Apply the fitted functions to each channel
        red_output = red_channel
        green_output = green_channel + 4000
        blue_output = blue_channel + 5200

        red_output_clipped = np.clip(red_output, 0, 65535)
        green_output_clipped = np.clip(green_output, 0, 65535)
        blue_output_clipped = np.clip(blue_output, 0, 65535)

        colored = np.stack((red_output_clipped, green_output_clipped, blue_output_clipped), axis=-1)
        colored = np.float32(colored)
        colored = colored/256
        #print("pop")
    if grading == "color":
        graded = colored

    if grading == "bw":
        # graded = color.rgb2gray(colored)
        graded = np.float32((colored * (8 / 12)))
        graded = cv2.cvtColor(graded, cv2.COLOR_RGB2GRAY)

    return graded

def nonlinear_func(x, a, b, c):
    return a * np.power(x, b) + c

def crop_image(processed, crop_percent):
    # Calculate crop size for both dimensions
    crop_pixels_width = int(processed.shape[1] * crop_percent / 100)
    #crop_pixels_height = int(processed.shape[0] * crop_percent / 200)
    crop_pixels_height = int(processed.shape[0] * .01)
    # Crop the image
    cropped_image = processed[crop_pixels_height:-crop_pixels_height, crop_pixels_width:-crop_pixels_width]
    return cropped_image
        
    