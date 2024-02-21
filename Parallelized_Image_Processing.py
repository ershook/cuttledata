##########################################################################################
###
###     The image processing script has three parts. The script below can not be run in Jupyter directly because Jupyter does not handle parallel processing well.
###         1. The Jupyter notebook, which saves a short script to a text file and then executes this python file.
###         2. This file, which contains two functions for parallel processing. Make_videos turns Tifs or DNGs into mp4 videos. Parallelize_image_processing takes DNG files and converts them to cropped, downsampled, and/or color graded Tifs
###         3. The image_processing_functions file, which contains the image manipulation functions called by make_videos and parallelize_image_processing
###
###
###
##########################################################################################

from image_processing_function_3 import convert_png_to_tif, bin_files, process_files, convert_to_clip, combine_videos, resize, temporal_downsample
import rawpy
from multiprocessing import Pool, freeze_support
import math
import os
from moviepy.editor import VideoFileClip, concatenate_videoclips
from functools import partial
from pathlib import Path


if __name__ == '__main__':
    freeze_support()

#main function to make output Tifs from raw DNG files
def Parallelize_image_processing(input_dir, list_of_images, save_images, make_movie, fps, downsampling_factor):
    if __name__ == '__main__':
        freeze_support()

        #Check to see if the necessary directories for each output file exist. If not, make them
        for i in range(len(list_of_images)):
            save_location = os.path.dirname((os.path.dirname(input_dir))) + "/" + str(list_of_images[i][0]).replace('.', '_') + "_" + list_of_images[i][
                1] + "_" + list_of_images[i][2] + "_" + str(list_of_images[i][3]) + "_" + str(downsampling_factor) + "/tifs"
            if not os.path.exists(save_location):
                print("Save location does not exist!" + str(save_location))
                os.makedirs(save_location)
            else:
                print(f"Directory already exists: {save_location}")

        # Get a sorted list of DNG files in the directory.
        # Sort at the folder level, then at the image level to make sure the saved images are referenced in the correct order.
        files = []
        for root, dirs, filenames in os.walk(input_dir):
            dirs.sort()
            for filename in sorted(filenames):
                #print(filename)
                if filename.lower().endswith(".dng") or filename.lower().endswith(".ARW") or filename.lower().endswith(".arw"):
                    #print("found raw")
                    files.append(os.path.join(root, filename))

        #make a dictionary matching each file name with its index in the list. The dictionary is used later for the file naming,
        file_dict = {}  # Define an empty dictionary

        # Populate the dictionary with mappings
        for i, file_path in enumerate(files, start=1):
            filename = str(os.path.basename(file_path))
            file_number = str(i)
            file_dict[filename] = i

        # Determine which files should be processed, according to the downsampling factor
        downsampled_files = [file for index, file in enumerate(files) if index % downsampling_factor == 0]

        #set the number of bins for parallel processing, as well as the batch size
        num_bins = 32
        batch_size = 300
        num_batches = math.ceil(len(downsampled_files) / batch_size)

        #for every batch....
        for i in range(num_batches):
            # check to see if the corresponding video output exists. If it does, skip that batch. 
            output_file = os.path.join(os.path.dirname((os.path.dirname(input_dir))) + "/" + str(list_of_images[-1][0]).replace('.', '_') + "_" + list_of_images[-1][1] + "_" + list_of_images[-1][2] + "_" + str(list_of_images[-1][3]) + "_" + str(downsampling_factor) + "/VideoChunks", "output" + str(i).zfill(4) + ".mp4")
            print(output_file)
            if not (os.path.exists(output_file)):
                # determine the first and last images in the batch
                stop_point = batch_size * (i + 1)
                if (batch_size * (i + 1)) >= len(downsampled_files):
                    stop_point = len(downsampled_files)
                files_batch = downsampled_files[(batch_size * i):stop_point]
                # Divide the files into bins
                try:
                    file_bins = bin_files(files_batch, num_bins)
                except ValueError as e:
                    print(f"Error dividing files into bins: {e}")
                    # Handle the error appropriately here
                    # For example, you might want to skip the rest of this iteration or stop the script altogether
                    continue  # or `break` or `return` depending on the desired behavior

                # Check if there are any bins to process
                if not file_bins or all(len(file_bin) == 0 for file_bin in file_bins):
                    print("No files to process in this batch.")
                    continue  # Skip the rest of this iteration as there's nothing to process

                # Process files in parallel
                pool = Pool()
                file_paths = [[os.path.join(input_dir, file) for file in file_bin] for file_bin in file_bins]

                #print(file_paths)
                #this line maps the file paths to the process_files function, which takes as parameters the file paths, dictionary of file names, and the list_of_images (types of output files to be created)
                results = pool.starmap(process_files, zip(file_paths, [file_dict] * len(file_paths), [list_of_images] * len(file_paths), [save_images] * len(file_paths), [make_movie] * len(file_paths), [downsampling_factor] * len(file_paths)))

                pool.close()
                pool.join()

                if make_movie:
                    if not (os.path.exists(os.path.dirname((os.path.dirname(input_dir))) + "/" + str(list_of_images[-1][0]).replace('.', '_') + "_" + list_of_images[-1][1] + "_" + list_of_images[-1][2] + "_" + str(list_of_images[-1][3]) + "_" + str(downsampling_factor) + "/VideoChunks")):
                        os.makedirs(os.path.dirname((os.path.dirname(input_dir))) + "/" + str(list_of_images[-1][0]).replace('.', '_') + "_" + list_of_images[-1][1] + "_" + list_of_images[-1][2] + "_" + str(list_of_images[-1][3]) + "_" + str(downsampling_factor) + "/VideoChunks")
                    #print(output_file)

                    # Convert frames to video clips
                    videos = [video for result in results for video in result]
                    video_clips = convert_to_clip(videos, fps)

                    # Combine the processed video clips
                    combine_videos(video_clips, output_file, fps)

        # at the end, assemble the concatenated file from the videochunks
        if make_movie:

            print("assembling concatenated file")

            # Get a list of all files in the folder
            files = os.listdir(os.path.dirname((os.path.dirname(input_dir))) + "/" + str(list_of_images[-1][0]).replace('.', '_') + "_" + list_of_images[-1][1] + "_" + list_of_images[-1][2] + "_" + str(list_of_images[-1][3]) + "_" + str(downsampling_factor) + "/VideoChunks")

            # Filter the files to keep only the MP4 files
            mp4_files = [file for file in files if file.lower().endswith(".mp4")]

            # Sort the MP4 files by name
            mp4_files.sort()
            #print(mp4_files)
            # Create a list to store the VideoFileClip objects
            video_clips = []

            # Iterate over the sorted MP4 files and concatenate them
            for mp4_file in mp4_files:
                file_path = os.path.join(os.path.dirname((os.path.dirname(input_dir))) + "/" + str(list_of_images[-1][0]).replace('.', '_') + "_" + list_of_images[-1][1] + "_" + list_of_images[-1][2] + "_" + str(list_of_images[-1][3]) + "_" + str(downsampling_factor) + "/VideoChunks", mp4_file)
                video_clip = VideoFileClip(file_path)
                video_clips.append(video_clip)

            # Concatenate all the video clips into a single clip
            final_clip = concatenate_videoclips(video_clips)

            # Specify the output file path
            output_path = (os.path.dirname((os.path.dirname(input_dir))) + "/" + str(list_of_images[-1][0]).replace('.', '_') + "_" + list_of_images[-1][1] + "_" + list_of_images[-1][2] + "_" + str(list_of_images[-1][3]) + "_" + str(downsampling_factor) + "_VideoClip_" + "fps" + str(fps) + ".mp4")

            # Write the final concatenated clip to the output file
            final_clip.write_videofile(output_path, codec="libx264")

            # Close the clips and free up resources
            final_clip.close()

    
#This is the script that communicates with Jupyter through the 'telephone' text document
"""filename = "C:/Users/Cuttlefish/Documents/Jupyter to Py Telephone.txt"

with open(filename, 'r') as file:
    lines = file.readlines()

for line in lines:
    exec(line)

"""

# Set source_path equal to the folder that contains the image files ("CINEMA")
source_path='/home/gtb2115/engram/cuttlefish/CUTTLEFISH_BEHAVIOR/2024_BEHAVIOR/Social-Behavior-Experiments/2024-02-17_Social-behavior_1-chamber-removable_Hugin-Munin-Merlin_HEAD-FIXED/CINEMA/'
# Select the downsampling factor as a fraction, the color grading, whether the output should be black and white or color, and the percentage of the frame to crop off of the sides
list_of_images = [[1, "raw_for_yellow_aggression", "color", 0],[0.1, "raw", "color", 0]]
# Start the image processing, setting the source path, list of images to generate, whether you want to save the output as Tifs, whether you want to save the output as an MP4, the frame rate of the mp4, and the temporal downsampling factor
# (for example, a downsampling factor of 24 will only render one out of every 24 images)
Parallelize_image_processing(source_path, list_of_images, True, True, 24, 1)
