# Trackid
Using YOLOv3 and DEEPSORT, this project attempts to track multiple objects on a screen and assign them a unique id to reduce overcounting.
This project return a cv2 screen with the classifications and also prints out the FPS and classes detected.

## Dependencies:
1. **Python --3.7.6**
2. **Conda --4.8.3**

## Installation

```sh
git clone https://github.com/nakul-shahdadpuri/trackid.git
cd trackid/
```
## Setup:
```sh
conda env create -f setup.yml
conda activate trackid
pip install -r setup.txt
```

## Running trackid

#### 1. For Webcam [Default]
```sh
python run.py 
```

## Resources
1. Non Max Suppression 'https://towardsdatascience.com/non-maximum-suppression-nms-93ce178e177c'
2. YOLOv3 model 'https://pjreddie.com/darknet/yolo/'
3. cv2.BlobFromImage 'https://www.pyimagesearch.com/2017/11/06/deep-learning-opencvs-blobfromimage-works/'
4. OpenCv Documentation 'https://docs.opencv.org/2.4/'
5. DeepSort Repo 'https://github.com/nwojke/deep_sort' 
6. SORT Paper 'https://arxiv.org/abs/1602.00763'
7. Deep Sort 'https://medium.com/analytics-vidhya/yolo-v3-real-time-object-tracking-with-deep-sort-4cb1294c127f'
