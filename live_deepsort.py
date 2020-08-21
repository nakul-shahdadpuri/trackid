import time, random
import numpy as np
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
from absl import app
import signal


from yolov3_tf2.models import YoloV3
from yolov3_tf2.dataset import transform_images
from yolov3_tf2.utils import draw_outputs, convert_boxes

from deep_sort import preprocessing
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker

from tools import generate_detections as gdet
from PIL import Image

class MjpegReader():
    def __init__(self, url: str):
        self._url = url

    def iter_content(self):
        r = requests.get(self._url, stream=True)
        # parse boundary
        content_type = r.headers['content-type']
        index = content_type.rfind("boundary=")
        assert index != 1
        boundary = content_type[index+len("boundary="):] + "\r\n"
        boundary = boundary.encode('utf-8')

        rd = io.BufferedReader(r.raw)
        while True:
            self._skip_to_boundary(rd, boundary)
            length = self._parse_length(rd)
            yield rd.read(length)

    def _parse_length(self, rd) -> int:
        length = 0
        while True:
            line = rd.readline()
            if line == b'\r\n':
                return length
            try:
            	if line.startswith(b"Content-Length"):
                	length = int(line.decode('utf-8').split(": ")[1])
                	assert length > 0
            except:
            	print("Malformed frame")


    def _skip_to_boundary(self, rd, boundary: bytes):
        for _ in range(10):
            if boundary in rd.readline():
                break
        else:
            pass

def nayanam(_argv):
    # Definition of the parameters
    max_cosine_distance = 0.5
    nn_budget = None
    nms_max_overlap = 1.0
    
    #initialize deep sort
    model_filename = 'model_data/mars-small128.pb'
    encoder = gdet.create_box_encoder(model_filename, batch_size=1)
    metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
    tracker = Tracker(metric)

    yolo = YoloV3(classes=80)

    yolo.load_weights(PATH_TO_WEIGHTS)
    print('weights loaded')

    class_names = [c.strip() for c in open(PATH_TO_CLASSES).readlines()]
    print('classes loaded')

    
    out = None
    fps = 0.0
    count = 0


    vid = cv2.VideoCapture(RTSP_URL)
    while(vid.isOpened()):
    	try:
    		_,img = vid.read()

    	except:
    		print("Empty frame")
    		continue
    	img_in = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 
    	img_in = tf.expand_dims(img_in, 0)
    	img_in = transform_images(img_in, 416)

    	t1 = time.time()
    	boxes, scores, classes, nums = yolo.predict(img_in)
    	classes = classes[0]
    	names = []

    	for i in range(len(classes)):
    		names.append(class_names[int(classes[i])])

    	names = np.array(names)
    	converted_boxes = convert_boxes(img, boxes[0])
    	features = encoder(img, converted_boxes)
    	detections = [Detection(bbox, score, class_name, feature) for bbox, score, class_name, feature in zip(converted_boxes, scores[0], names, features)]
    	cmap = plt.get_cmap('tab20b')
    	colors = [cmap(i)[:3] for i in np.linspace(0, 1, 20)]
# running NMS
    	boxs = np.array([d.tlwh for d in detections])
    	scores = np.array([d.confidence for d in detections])
    	classes = np.array([d.class_name for d in detections])
    	indices = preprocessing.non_max_suppression(boxs, classes, nms_max_overlap, scores)
    	detections = [detections[i] for i in indices]
# Deepsort tracker called here
    	tracker.predict()
    	tracker.update(detections)
#dump file set here
    	# file = open(PATH_TO_RESULTS,'a+')
    	for track in tracker.tracks:
    		if not track.is_confirmed() or track.time_since_update > 1:
    			continue 
    		bbox = track.to_tlbr()
    		class_name = track.get_class()
    		color = colors[int(track.track_id) % len(colors)]
    		color = [i * 255 for i in color]
    		if VIDEO_DEBUG == 1:
    			cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 2)
    			cv2.rectangle(img, (int(bbox[0]), int(bbox[1]-30)), (int(bbox[0])+(len(class_name)+len(str(track.track_id)))*17, int(bbox[1])), color, -1)
    			cv2.putText(img, class_name + "-" + str(track.track_id),(int(bbox[0]), int(bbox[1]-10)),0, 0.75, (255,255,255),2)
    		s = str(track.track_id) + ',' + class_name + ',' + str(int(bbox[0])) + ',' + str(int(bbox[1])) + '\n'
    		# file.write(s)
    		print(s)
    	fps  = ( fps + (1./(time.time()-t1)) ) / 2
    	print("fps = ", fps)
    	# file.close()
    	if VIDEO_DEBUG == 1:
    		cv2.imshow('output', img)
    		if cv2.waitKey(1) == 27:
    			break
    	signal.signal(signal.SIGINT,user_exit)
    vid.release()
    if VIDEO_DEBUG == 1:
    	cv2.destroyAllWindows()


def user_exit(a,b):
	print('Exiting.....')
	exit(1)

def main(MODE):
	app.run(nayanam)

if __name__ == '__main__':

	PATH_TO_CLASSES = './dataset/coco.names'
	PATH_TO_WEIGHTS = './weights/yolov3.tf'
	DEBUG = 1
	VIDEO_DEBUG = 1
	PATH_TO_RESULTS = '../results/dump.csv'
	MODE = 'rtsp'
	RTSP_URL = 0
	
	main(RTSP_URL)