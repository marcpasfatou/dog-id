from imageai.Detection import ObjectDetection
import os, shutil
from PIL import Image
import pandas as pd
"""
    Preprocessing for the dog dataset

    Arguments:
    image from the dataset

    Returns:
    a cropped image for each dog
    """

execution_path = os.getcwd()
dataset_path = os.path.join(execution_path, "train")
new_dataset_path = os.path.join(execution_path, "new_train")
img_names = os.listdir(dataset_path)
class_names = pd.read_csv(os.path.join(execution_path,"classes.csv"))
train_label = pd.read_csv(os.path.join(execution_path,"labels.csv"))

try:
    os.mkdir(os.path.join(execution_path, "new_train"))
except OSError:
    print ("Creation of the directory %s failed" % os.path.join(execution_path, "new_train"))
else:
    print ("Successfully created the directory %s " % os.path.join(execution_path, "new_train"))

for index, row in class_names.iterrows():
    #print(row.to_string(header=False))
    try:
        os.mkdir(os.path.join(new_dataset_path, row["breed"]))
    except OSError:
        print ("Creation of the directory %s failed" % os.path.join(new_dataset_path, row["breed"]))
    else:
        print("Successfully created the directory %s " % os.path.join(new_dataset_path, row["breed"]))

detector = ObjectDetection()
detector.setModelTypeAsRetinaNet()
detector.setModelPath(os.path.join(execution_path,"weights/resnet50_coco_best_v2.0.1.h5"))
detector.loadModel()

custom = detector.CustomObjects(dog=True)

for name in img_names:
    img = Image.open(os.path.join(dataset_path, name))
    exists = os.path.isfile(os.path.join(new_dataset_path , name))
    class_name = train_label[train_label["id"] == name[:-4]]
    class_folder = class_name["breed"].to_string(index=False)[1:]
    print(class_folder)
    if exists:
        shutil.move(os.path.join(new_dataset_path, name), os.path.join(new_dataset_path, class_folder))
        continue
    #print(os.path.join(dataset_path , name))
    #print(os.path.join(new_dataset_path , name))
    _,detections = detector.detectCustomObjectsFromImage( custom_objects=custom, input_image=os.path.join(dataset_path , name), output_type = "array", minimum_percentage_probability=60)
    #print(detections)
    if not detections:
        shutil.copy(os.path.join(dataset_path , name), os.path.join(new_dataset_path,class_folder))
    else:
        max_detection = max(detections, key=lambda x: x['percentage_probability'])
        crop = img.crop(max_detection["box_points"])
        crop.save(os.path.join(new_dataset_path , class_folder, name))

