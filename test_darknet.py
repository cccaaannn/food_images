from imagepreprocessing import make_prediction_from_directory_yolo

darknet_command = "./darknet detector test data/train_10_class/obj.data data/train_10_class/yolo-obj.cfg backup/yolo-obj_last.weights {0} -i 0 -thresh 0.2 -dont_show"

images = "train_10_class/test_10_class/acibademKurabiyesi"
# aciliEzme
# adanaKenap

make_prediction_from_directory_yolo(images, "../darknet", save_path = "detection_results", darknet_command=darknet_command, files_to_exclude = [".DS_Store",""])