import rawpy
import numpy as np
import cv2
import shutil
import os
import time
import tifffile
from moviepy.editor import VideoClip, VideoFileClip, concatenate_videoclips
from PIL import Image
from pathlib import Path


def process_files(file_paths, file_dict1, list_of_images, save_images, make_movie, downsampling_factor):
    if len(file_paths) > 0:
        results = []
        for file_path in file_paths:
            result = generate_files(file_path, file_dict1, list_of_images, save_images, make_movie, downsampling_factor)
            if result is not None:
                results.append(result)
            else:
                print(f"Skipping file due to read error: {file_path}")
        return results
    else:
        print("no files in batch")


def generate_files(image_source, file_dict3, list_of_images, save_images, make_movie, downsampling_factor):
    print("reading image file " + str(image_source))
    filename, extension = os.path.splitext(image_source)
    filetype = extension.lower()
    
    try:
        if filetype in [".dng", ".arw", ".tif", ".tiff"]:
            rgb = None  # Initialize rgb to None for scope reasons
            if filetype in [".dng", ".arw"]:
                #print("DNG found)")
                with rawpy.imread(image_source) as raw:
                    rgb = raw.postprocess(gamma=(50, 50), no_auto_bright=True, output_bps=8)
            elif filetype in [".tif", ".tiff"]:
                #print("TIF found")
                rgb = np.array(Image.open(image_source))

            if rgb is not None:
                
                #if the image is taller than it is wide, rotate by 90 degrees
                height, width, _ = rgb.shape
                if height > width:
                    rgb = np.rot90(rgb)

                # then look at what images we need to make. For each list of parameters . . .
                filename = os.path.basename(image_source)
                file_number = file_dict3.get(filename)

                #make every output that you need with this one file
                for i in range(len(list_of_images)):

                    #save the outputs in a folder with the corresponding name for the image (scaling factor, color grading, color vs black and white, cropped), up two levels from where the DNG image was found
                    output_path = (((os.path.dirname(os.path.dirname(os.path.dirname(image_source)))))) + "/" + str(list_of_images[i][0]).replace('.', '_') + "_" + list_of_images[i][
                            1] + "_" + list_of_images[i][2] + "_" + str(list_of_images[i][3]) + "_" + str(downsampling_factor) + "/tifs/" + str(file_number).zfill(6) + ".tif"
                    # apply the downsampling
                    resized_image = resize(rgb, list_of_images[i][0])
                    # apply the color correction, which involves the color crading and the color-> black and white conversion
                    processed = apply_color(resized_image, list_of_images[i][1], list_of_images[i][2])
                    # crop the image
                    cropped = crop_image(processed, list_of_images[i][3])
                    # save the image in the corresponding directory

                    if save_images:
                        tifffile.imwrite(output_path, cropped.astype('uint8'))

                    if make_movie:
                        frame = Image.fromarray(rgb)

                #return the last image in the list of images to use to make the video
                return cropped.astype('uint8')
            else:
                print(f"Unsupported file format for file: {image_source}")
                return None
        else:
            print(f"Unsupported file format for file: {image_source}")
            return None
    except Exception as e:
        print(f"ERROR READING: {filename} due to {e}")
        return None



"""
###

Functions for multiprocessing and video-making

###
"""

def bin_files(files, num_bins):
    # Handle edge cases
    if num_bins <= 0:
        raise ValueError("Number of bins must be greater than zero.")

    num_files = len(files)

    if num_files == 0:
        return []

    # Calculate bin size and remainder
    bin_size, remainder = divmod(num_files, num_bins)

    bins = []
    start_index = 0

    # Distribute files to bins
    for i in range(num_bins):
        end_index = start_index + bin_size + (1 if i < remainder else 0)
        bins.append(files[start_index:end_index])
        start_index = end_index

    return bins

def convert_to_clip(frames, fps):
    return [VideoClip(lambda t: frames[int(fps*t)], duration=len(frames)/fps)]

#when all of the short video clips are made, this function combines them into one long output.
#this prevents high memory use when making long videos, and allows the script to pick up where it left off if
#stopped prematurely
def combine_videos(videos, output_file, fps):
    final_clip = concatenate_videoclips(videos)
    final_clip.write_videofile(output_file, fps)
    final_clip.close()

"""
###

Functions for file-handling

###
"""

# useful function that waits until no new files are being added to a directory to proceed
def check_directory(directory_path):
    start_time = time.time()
    file_dict = {}  # Dictionary to store file counts for each directory

    for root, dirs, files in os.walk(directory_path):
        file_dict[root] = len(files)

    while True:
        current_time = time.time()
        elapsed_time = current_time - start_time

        # Check if two minutes have passed without new files
        if elapsed_time >= 120:
            print("No new files for two minutes. Exiting...")
            break

        for root, dirs, files in os.walk(directory_path):
            new_file_count = len(files)
            if new_file_count > file_dict[root]:
                print(f"New file(s) detected in directory: {root}")
                file_dict[root] = new_file_count

        time.sleep(30)  # Wait for ten seconds before checking again

def transfer_files(source_dir, dest_dir):
    # Make a new folder on the server for this experiment. Then update the 'dest_dir' to that new folder
    os.makedirs(dest_dir + "/" + (os.path.basename(source_dir)), exist_ok=True)
    dest_dir = dest_dir + "/" + (os.path.basename(source_dir))

    #print(dest_dir)

    # Loop over the root and all subdirectories in the source directory

    for root, dirs, files in os.walk(source_dir):
        # Create the corresponding subdirectory in the output directory
        rel_path = os.path.relpath(root, source_dir)
        output_dir = os.path.join(dest_dir, rel_path)
        os.makedirs(output_dir, exist_ok=True)
        # Loop over all files in the current directory and copy them to the output directory
        for filename in files:
            filepath = os.path.join(root, filename)
            shutil.copy(filepath, output_dir)


"""
###

functions for image manipulation

###
"""

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

    if spectrum == "pop2":
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
        colored = 2.3 * colored - 40000

        red_channel = colored[:, :, 0]
        green_channel = colored[:, :, 1]
        blue_channel = colored[:, :, 2]

        # Apply the fitted functions to each channel
        red_output = 0.93*red_channel + 5000
        green_output = green_channel - 2000
        blue_output = blue_channel - 4500

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

    if spectrum == "rawpop_bright_for_aggression":
        resized = resized * 256

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

        # colored = np.clip((2.1 * colored - 32000), 0, 65535)
        # colored = np.clip((1.8 * colored - 24000), 0, 65535)
        # colored = np.clip((1.7 * colored - 20000), 0, 65535)
        # colored = 1.6 * colored - 20000
        colored = 1.9 * colored - 15000
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
        colored = colored / 256
        #print("pop")
    if spectrum == "raw_for_yellow_aggression":
        resized = resized * 256

        popt_red = np.array([2.49977028e-01, 1.12525005e+00, 5.18763052e+03])
        popt_green = np.array([5.76949661e-01, 1.04726996e+00, 1.47926918e+03])
        popt_blue = np.array([7.88687346e+00, 8.20434666e-01, - 9.25428880e+03])

        red_channel = resized[:, :, 0]
        green_channel = resized[:, :, 1]
        blue_channel = resized[:, :, 2]

        # Apply the fitted functions to each channel
        red_output = red_channel
        green_output = green_channel
        blue_output = blue_channel

        red_output_clipped = np.clip(red_output, 0, 65535)
        green_output_clipped = np.clip(green_output, 0, 65535)
        blue_output_clipped = np.clip(blue_output, 0, 65535)

        colored = np.stack((red_output_clipped, green_output_clipped, blue_output_clipped), axis=-1)

        # colored = np.clip((2.1 * colored - 32000), 0, 65535)
        # colored = np.clip((1.8 * colored - 24000), 0, 65535)
        # colored = np.clip((1.7 * colored - 20000), 0, 65535)
        # colored = 1.6 * colored - 20000
        colored = 1.7 * colored - 15000
        red_channel = colored[:, :, 0]
        green_channel = colored[:, :, 1]
        blue_channel = colored[:, :, 2]

        # Apply the fitted functions to each channel
        red_output = red_channel - 5500
        green_output = green_channel - 7000
        blue_output = blue_channel + 5200

        red_output_clipped = np.clip(red_output, 0, 65535)
        green_output_clipped = np.clip(green_output, 0, 65535)
        blue_output_clipped = np.clip(blue_output, 0, 65535)

        colored = np.stack((red_output_clipped, green_output_clipped, blue_output_clipped), axis=-1)
        colored = np.float32(colored)
        colored = colored / 256
        #print("pop")

    if grading == "color":
        graded = colored

    if grading == "bw":
        # graded = color.rgb2gray(colored)
        graded = np.float32((colored * (8 / 12)))
        graded = cv2.cvtColor(graded, cv2.COLOR_RGB2GRAY)

    return graded

def crop_image(processed, crop_percent):
    # Calculate crop size for the x-axis (width)
    crop_pixels_width = int(processed.shape[1] * crop_percent / 100)
    
    # If crop_percent is zero, crop_pixels_width will also be zero,
    # so the original width will be preserved.

    # Crop the image on the x-axis only
    cropped_image = processed[:, crop_pixels_width:-crop_pixels_width] if crop_pixels_width > 0 else processed

    return cropped_image


# this is the nonlinear function that was fit to the colored data to perform the color correction
def nonlinear_func(x, a, b, c):
    return a * np.power(x, b) + c

def temporal_downsample(source_dir, downsample_factor):
    """
    Create a downsampled copy of a directory of images.

    :param source_dir: Path to the directory containing the original images.
    :param downsample_factor: The number of images to skip.
    """
    # Ensure the source directory exists
    if not os.path.exists(source_dir):
        raise ValueError("Source directory does not exist.")

    # Create the new directory name
    parent_dir = os.path.dirname(source_dir)
    new_dir_name = os.path.basename(source_dir) + "_downsampled"
    new_dir = os.path.join(parent_dir, new_dir_name)

    # Create the new directory if it doesn't exist
    if not os.path.exists(new_dir):
        os.makedirs(new_dir)

    # List all files in the source directory
    files = sorted(os.listdir(source_dir))

    # Copy every nth file to the new directory
    for i, file in enumerate(files):
        if i % downsample_factor == 0:
            shutil.copy(os.path.join(source_dir, file), os.path.join(new_dir, file))

    return new_dir

def convert_png_to_tif(directory):
    # Check if the directory exists
    if not os.path.exists(directory):
        print(f"Directory {directory} does not exist.")
        return

    # Iterate over files in the directory
    for filename in os.listdir(directory):
        if filename.endswith(".png"):
            file_path = os.path.join(directory, filename)
            # Open the image file
            with Image.open(file_path) as img:
                # Define the new filename with .tif extension
                new_filename = os.path.splitext(filename)[0] + ".tif"
                new_file_path = os.path.join(directory, new_filename)
                # Save the image in TIFF format
                img.save(new_file_path, format='TIFF')
                print(f"Converted {filename} to {new_filename}")