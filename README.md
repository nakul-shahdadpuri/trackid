# Trackid
Using YOLOv3 and DEEPSORT, this project attempts to track multiple objects on a screen and assign them a unique id to reduce overcounting.

This implements a computer vision program for a traffic system, using YOLOv3 machine learning model. The program is implemented in python3 and produces class wise output of the vehicles detected.

The abstracted procedure in the development and functioning of this process is:

1. Run the model on a live stream analysing each frame.
2. The frame wise output is stored in results/dump.csv
3. Now the dump.csv is periodically analysed at regular intervals by process.py.
4. The identified classes are stored in the database.
5. The information regarding the which camera is to be analysed is stored in the config/ directory.

## Dependencies:
1. **Python --3.7.6**
2. **Conda --4.8.3**

## Setup:
```sh
conda env create -f setup.yml
conda activate trackid
pip install -r setup.txt
```

## Installation

```sh
git clone https://github.com/nakul-shahdadpuri/trackid.git
cd trackid/
```

## Running trackid

#### 1. For Videos [Output displayed on cv2 screen]
```sh
python run.py [VIDEO NAME/PATH]
```

####2. For Webcam [Output displayed on cv2 screen]
```sh
python run.py 0
```

## Resources
1. Non Max Suppression 'https://towardsdatascience.com/non-maximum-suppression-nms-93ce178e177c'
2. YOLOv3 model 'https://pjreddie.com/darknet/yolo/'
3. cv2.BlobFromImage 'https://www.pyimagesearch.com/2017/11/06/deep-learning-opencvs-blobfromimage-works/'
4. OpenCv Documentation 'https://docs.opencv.org/2.4/'
5. DeepSort Repo 'https://github.com/nwojke/deep_sort' 
6. SORT Paper 'https://arxiv.org/abs/1602.00763'
7. Deep Sort 'https://medium.com/analytics-vidhya/yolo-v3-real-time-object-tracking-with-deep-sort-4cb1294c127f'
