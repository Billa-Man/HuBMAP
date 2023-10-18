import os
os.environ['WANDB_DISABLED'] = 'true'

import shutil
import json
import random
from tqdm import tqdm

from ultralytics import YOLO
from PIL import Image

from functions import tiff_to_png, vertices_to_txt

# Create working directory ---------------------------------------------------------------------------------------------------------------------

if os.path.exists("./kaggle/") and os.path.isdir("./kaggle/"):
    shutil.rmtree("./kaggle/")

if os.path.exists("./datasets/") and os.path.isdir("./datasets/"):
    shutil.rmtree("./datasets/")

parent_dirpath = "./datasets/"

os.makedirs(parent_dirpath)
os.makedirs("./kaggle/working/temp_images")
os.makedirs("./kaggle/working/temp_labels")

os.makedirs(os.path.join(parent_dirpath, "train"))
os.makedirs(os.path.join(parent_dirpath, "train", "images"))
os.makedirs(os.path.join(parent_dirpath, "train", "labels"))

os.makedirs(os.path.join(parent_dirpath, "test"))
os.makedirs(os.path.join(parent_dirpath, "test", "images"))
os.makedirs(os.path.join(parent_dirpath, "test", "labels"))

train_filepath = "./hubmap-hacking-the-human-vasculature/train"

all_images = os.listdir(train_filepath)
print("No. of images:", len(all_images))

json_filepath = "./hubmap-hacking-the-human-vasculature/polygons.jsonl"
file_ids = []

with open(json_filepath, 'r') as file:
    
    for line in file:
        data = json.loads(line)
        file_id = data['id']
        annotations = data['annotations']
        list_of_vertices = annotations[0]['coordinates'][0]
        tiff_to_png(file_id)
        vertices_to_txt(file_id, annotations, list_of_vertices)
        file_ids.append(file_id)

all_file_ids = []

for file_name in all_images:
    file_name = file_name.split('.')
    all_file_ids.append(file_name[0])

unlabelled_images = list(set(all_file_ids).difference(file_ids))

random.shuffle(file_ids)

# Create train and test directory --------------------------------------------------------------------------------------------------------------

# Train directory
for i in range(0, int(0.8*len(file_ids))):
    
    old_path_img = "./kaggle/working/temp_images/" + str(file_ids[i]) + ".png"
    new_path_img = "./datasets/train/images/" + str(file_ids[i]) + ".png"
    shutil.copy(old_path_img, new_path_img)
    
    old_path_txt = "./kaggle/working/temp_labels/" + str(file_ids[i]) + ".txt"
    new_path_txt = "./datasets/train/labels/" + str(file_ids[i]) + ".txt"
    shutil.copy(old_path_txt, new_path_txt)

# Test directory
for i in range(int(0.8*len(file_ids)), len(file_ids)):
    
    old_path_img = "./kaggle/working/temp_images/" + str(file_ids[i]) + ".png"
    new_path_img = "./datasets/test/images/" + str(file_ids[i]) + ".png"
    shutil.copy(old_path_img, new_path_img)
    
    old_path_txt = "./kaggle/working/temp_labels/" + str(file_ids[i]) + ".txt"
    new_path_txt = "./datasets/test/labels/" + str(file_ids[i]) + ".txt"
    shutil.copy(old_path_txt, new_path_txt)
    
shutil.rmtree("./kaggle/working/temp_images")
shutil.rmtree("./kaggle/working/temp_labels")

test_set_file_ids = []
test_set_filepath = "./datasets/test/images"
test_set_images_filepaths = os.listdir(test_set_filepath)

for file_name in test_set_images_filepaths:
    file_name = file_name.split('.')
    test_set_file_ids.append(file_name[0])

# Model training for 10 iterations -------------------------------------------------------------------------------------------------------------

model = YOLO('yolov8x-seg.pt')

for iteration in range(10):

    results = model.train(data='./custom_config.yaml',
                          epochs=50, imgsz=512, optimizer='Adam',
                          seed=42, close_mosaic=0, mask_ratio=1, val=True,
                          degrees=90, translate=0.1, scale=0.5, flipud=0.5, fliplr=0.5, verbose=True)

    images_added = 0
    
    # 1. Get all file_ids from train_set
    train_set_file_ids = []
    train_set_filepath = "./datasets/train/images"
    train_set_images_filepaths = os.listdir(train_set_filepath)

    for file_name in train_set_images_filepaths:
        file_name = file_name.split('.')
        train_set_file_ids.append(file_name[0])
        
    # 2. Add file_ids from test_set
    all_labelled_image_file_ids = train_set_file_ids + test_set_file_ids
    
    # 3. Find differences from inputs and combined list & create new unlabelled image file_id list.
    unlabelled_images = list(set(all_file_ids).difference(all_labelled_image_file_ids))

    for file_id in tqdm(unlabelled_images):

        tiff_image_path = "./kaggle/input/hubmap-hacking-the-human-vasculature/train/" + str(file_id) + ".tif"
        tiff_image = Image.open(tiff_image_path)
        destination_path = "./kaggle/working/temp_image.png"
        tiff_image.save(destination_path, 'PNG')

        results = model.predict(destination_path, verbose=False)

        flag = 1
        file_contents = []

        for result in results:
            boxes = result.boxes.conf
            if len(boxes) != 0:
                classes = result.boxes.cls
                masks = result.masks.xyn
            else:
                flag = 0

        if (flag):
            for i in range(len(boxes)):
                if boxes[i] < 0.4:
                    flag=0
                    break

        if(flag):
            des_img_filepath = os.path.join("./datasets/train/images/" + str(file_id) + ".png")
            shutil.copy(destination_path, des_img_filepath)

            for i in range(len(boxes)):

                yolo_format = []

                if classes[i] == 1:
                    yolo_format.append(str(1))
                else:
                    yolo_format.append(str(0))

                list_of_vertices = masks[i]
                for vertex in list_of_vertices:
                    yolo_format.append(str(vertex[0]))
                    yolo_format.append(str(vertex[1]))

                yolo_format = " ".join(yolo_format)

                file_contents.append(yolo_format)

            file_name = os.path.join("./datasets/train/labels/" + str(file_id) + ".txt")

            with open(file_name, "w") as file:
                if (len(file_contents) == 1):
                    file.write(str(file_contents[-1]))
                else:
                    for k in range(len(file_contents)-1):
                        file.write(str(file_contents[k]) + "\n")

                    file.write(str(file_contents[-1]))

            images_added += 1

        flag = 1

    print("Images added to training set:", images_added)

print("Final length of Train dataset:", len(list(os.listdir("./datasets/train/labels"))))

# Final model training with updated metrics

results = model.train(data='./custom_config.yaml',
                      epochs=50, imgsz=512, optimizer='Adam',
                      seed=42, close_mosaic=0, mask_ratio=1, val=True,
                      degrees=90, translate=0.1, scale=0.5, flipud=0.5, fliplr=0.5, verbose=True)
