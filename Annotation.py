import os
from tkinter import *
from PIL import Image, ImageTk
import sys 


# Function to preload images

def preload_images(image_folder):
    image_files = sorted(
        [img for img in os.listdir(image_folder) if img.split('.')[-1].lower() in ['png', 'jpg', 'jpeg', 'gif']],
        key=lambda x: x.split('.')[0]
    )
    #image_files = image_files[:100]
    # Load PIL Images and not PhotoImage objects yet
    images = []
    print("Loading image file references...")
    for idx, image_file in enumerate(image_files, start=1):
            img_path = os.path.join(image_folder, image_file)
            img = Image.open(img_path)
            img.thumbnail((800/4, 600/4), Image.Resampling.LANCZOS)
            images.append(img)
            print(f"Loaded {idx}/{len(image_files)}: {image_file}")
    print("Image file references loaded successfully!")
    return images, image_files


def create_photo_images(images):
    return [ImageTk.PhotoImage(image) for image in images]



# Functions to handle user input and image display
def update_label(value):
    global index, label_status
    labels[image_files[index]] = value
    label_status.config(text=f'Current Label: {value}')
    save_labels()

def save_labels():
    with open(output_file, 'w') as f:
        for img_file, label in labels.items():
            f.write(f'{img_file}: {label}\n')

def load_image(skip_forward=True):
    global index, label, label_text, label_status, images
    # When moving forward, skip images with label '0'
    if skip_forward:
       # while index < len(images): #and labels[image_files[index]] == '0':
       #     index += 1
       pass
    if index < len(images):
        photo = images[index]
        label.config(image=photo)
        label.image = photo  # Keep a reference!
        label_text.config(text=f'Image: {image_files[index]} [{index + 1}/{len(image_files)}]')
        label_status.config(text=f'Current Label: {labels[image_files[index]]}')
    else:
        # If no more images to show, you can either close the application or display a message
        label.config(image=None)
        label_text.config(text="No more images to label.")
        label_status.config(text="")

def next_image(event):
    global index
    # Move to the next image if not at the end of the list
    while index < len(images) - 1:
        index += 1
        # Stop at the first image with a label of '1'
        if labels[image_files[index]] == '1' or labels[image_files[index]] == '2' or labels[image_files[index]] == '0':
            break
    load_image()

def prev_image(event):
    global index
    # Move to the previous image if not at the start of the list
    while index > 0:
        index -= 1
        # Stop at the first image with a label of '1'
        if labels[image_files[index]] == '1' or labels[image_files[index]] == '2' or labels[image_files[index]] == '0':
            if labels[image_files[index]] != '1': 
                update_label('1')
            break
    load_image()


def up_arrow(event):
    global index
    # Set the current image label to '0' and immediately move to the next image
    update_label('0')
    #next_image(event)  # This will move to the next image after labeling

def down_arrow(event):
    global index
    update_label('2')
    #next_image(event)  # This will move to the next image after labeling

# Function to initialize and start the GUI
def start_gui():
    global label, label_text, label_status, images, index

    root = Tk()
    root.title('Image Labeler')

    # Now convert loaded PIL Images to PhotoImage objects after the Tk instance is created
    images = create_photo_images(image_refs)

    label = Label(root)
    label.pack()

    label_text = Label(root, text='', font=('Helvetica', 14))
    label_text.pack()

    label_status = Label(root, text='', fg='red', font=('Helvetica', 14))
    label_status.pack()

    root.bind('<Right>', next_image)
    root.bind('<Left>', prev_image)
    root.bind('<Up>', up_arrow)
    root.bind('<Down>', down_arrow)

    index = 0
    load_image()

    root.mainloop()


import shutil


def copy_labeled_images(image_folder, output_folder, labels_file):
    # Ensure the output directory exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Read the labels
    labels = {}
    with open(labels_file, 'r') as file:
        for line in file:
            image_file, label = line.strip().split(': ')
            labels[image_file] = label

    # Copy images with label '1'
    for image_file, label in labels.items():
        if label == '1':
            source_path = os.path.join(image_folder, image_file)
            destination_path = os.path.join(output_folder, image_file)

            # Check if the file exists to avoid overwriting
            if not os.path.exists(destination_path):
                shutil.copy2(source_path, destination_path)
                print(f"Copied: {image_file}")
            else:
                print(f"File already exists: {image_file}")


folder_ind = int(sys.argv[1])
# Set the directory where your images are located
base_path = '/Volumes/axel-locker/es3773/data/mask_classifier_data/'
folders = os.listdir(base_path)
for ii, folder in enumerate(folders):
    print(ii, folder)
image_folder = base_path + folders[folder_ind] + '/segments/'
output_file = base_path + folders[folder_ind] + '/labels.txt'

# Load image file references and initialize labels
image_refs, image_files = preload_images(image_folder)
labels = {img_file: '1' for img_file in image_files}

# Start the GUI
start_gui()




##################

"""

# Usage
original_image_folder = 'Z:/cuttlefish/CUTTLEFISH_BEHAVIOR/2023_BEHAVIOR/E-ink_Tank/2023-05-05_Rocky_Quilt_51_25q-120s_26b-120s_alt_c2023-04-21/Cuttlefish'
duplicate_image_folder = 'Z:/cuttlefish/CUTTLEFISH_BEHAVIOR/2023_BEHAVIOR/E-ink_Tank/2023-05-05_Rocky_Quilt_51_25q-120s_26b-120s_alt_c2023-04-21/Cuttlefish_clean4'
labels_txt_file = 'Z:/cuttlefish/CUTTLEFISH_BEHAVIOR/2023_BEHAVIOR/E-ink_Tank/2023-05-05_Rocky_Quilt_51_25q-120s_26b-120s_alt_c2023-04-21/Cuttlefish/labels.txt'

# Copy over the images
copy_labeled_images(original_image_folder, duplicate_image_folder, labels_txt_file)

"""
