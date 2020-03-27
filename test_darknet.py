from imagepreprocessing import make_prediction_from_directory_yolo
import os

darknet_command = "./darknet detector test data/train_10_class/obj.data data/train_10_class/yolo-obj.cfg backup/yolo-obj_final.weights {0} -dont_show"
darknet_path = "../darknet"

main_folder = "/home/can_kurt_aa/food_images/10_class/test_10_class"
main_save_path = "/home/can_kurt_aa/test_results"

image_folders = sorted(os.listdir(main_folder))


for image_folder in image_folders:
    
    save_path = os.path.join(main_save_path,image_folder)
    if(not os.path.exists(save_path)):
        os.makedirs(save_path)

    make_prediction_from_directory_yolo(
    os.path.join(main_folder, image_folder), 
    darknet_path, 
    save_path = save_path, 
    darknet_command = darknet_command
    )
