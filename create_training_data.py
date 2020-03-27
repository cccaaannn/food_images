from imagepreprocessing import create_training_data_yolo, auto_annotation_by_random_points, yolo_annotation_tool
import os

main_dir = "10_class/train_10_class"
# main_dir = "5_class/train_5_class"
# main_dir = "3_class/train_3_class"

exclude = ["obj.data", "obj.names", "train.txt", "test.txt", "yolo-obj.cfg"]


create_training_data_yolo(main_dir)

folders = sorted(os.listdir(main_dir))
for index, folder in enumerate(folders):
    if(folder not in exclude):
        auto_annotation_by_random_points(os.path.join(main_dir, folder), index, annotation_points=(0.5,0.5,1,1))


# yolo_annotation_tool("train_10_class\\acibademKurabiyesi","train_10_class\\obj.names")