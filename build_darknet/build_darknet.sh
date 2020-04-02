cd 
git clone https://github.com/AlexeyAB/darknet
sudo apt-get update
sudo apt-get -y upgrade
sudo apt-get install libopencv-dev
cd darknet
sed -i 's/OPENCV=0/OPENCV=1/g' Makefile
sed -i 's/CUDNN=0/CUDNN=1/g' Makefile
sed -i 's/GPU=0/GPU=1/g' Makefile
wget https://pjreddie.com/media/files/yolov3.weights
wget http://pjreddie.com/media/files/darknet53.conv.74
make
./darknet detector test cfg/coco.data cfg/yolov3.cfg yolov3.weights data/dog.jpg