from imagepreprocessing.darknet_functions import create_training_data_yolo, auto_annotation_by_random_points, yolo_annotation_tool
import os

main_dir = "30_class/train_30_class"
# main_dir = "10_class/train_10_class"
# main_dir = "5_class/train_5_class"
# main_dir = "3_class/train_3_class"
   
folders = sorted(os.listdir(main_dir))
for index, folder in enumerate(folders):
    auto_annotation_by_random_points(os.path.join(main_dir, folder), index, annotation_points=((0.5,0.5), (0.5,0.5), (1.0,1.0), (1.0,1.0)))


create_training_data_yolo(main_dir)

# yolo_annotation_tool("train_10_class\\acibademKurabiyesi","train_10_class\\obj.names")