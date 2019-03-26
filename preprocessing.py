from imageai.Detection import ObjectDetection
import os, shutil
from PIL import Image
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

try:
    os.mkdir(os.path.join(execution_path, "new_train"))
except OSError:
    print ("Creation of the directory %s failed" % os.path.join(execution_path, "new_train"))
else:
    print ("Successfully created the directory %s " % os.path.join(execution_path, "new_train"))

detector = ObjectDetection()
detector.setModelTypeAsRetinaNet()
detector.setModelPath(os.path.join(execution_path,"weights/resnet50_coco_best_v2.0.1.h5"))
detector.loadModel()

custom = detector.CustomObjects(dog=True)

for name in img_names:
    img = Image.open(os.path.join(dataset_path, name))
    exists = os.path.isfile(os.path.join(new_dataset_path , name))
    if exists:
        continue
    #print(os.path.join(dataset_path , name))
    #print(os.path.join(new_dataset_path , name))
    _,detections = detector.detectCustomObjectsFromImage( custom_objects=custom, input_image=os.path.join(dataset_path , name), output_type = "array", minimum_percentage_probability=60)
    #print(detections)
    if not detections:
        shutil.copy(os.path.join(dataset_path , name), new_dataset_path)
    else:
        max_detection = max(detections, key=lambda x: x['percentage_probability'])
        crop = img.crop(max_detection["box_points"])
        crop.save(os.path.join(new_dataset_path , name))

