from imagepreprocessing import make_prediction_from_directory_yolo
import os

darknet_command = "./darknet detector test data/train_10_class/obj.data data/train_10_class/yolo-obj.cfg backup/yolo-obj_last.weights {0} -dont_show"
darknet_path = "../darknet"

main_folder = "train_10_class/test_10_class"
image_folders = sorted(os.listdir(main_folder))


for image_folder in image_folders:
    make_prediction_from_directory_yolo(
    os.pat.join(main_folder, image_folder), 
    darknet_path, 
    save_path = "/home/can_kurt_aa/test_results/{image_folder}_results", 
    darknet_command = darknet_command
    )