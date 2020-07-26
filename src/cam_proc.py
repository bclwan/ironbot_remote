#!/usr/bin/python
import rospy
from sensor_msgs.msg import Image, CompressedImage

import cv2 as cv

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import matplotlib.animation as animation

import torch
import torchvision
import torchvision.transforms as T


torch.cuda.set_device(0)
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
model.eval()


COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]



cam_img = None
dep_img = None

def raw_img_callback(msg):
  global cam_img
  
  bArray = bytearray(msg.data)
  cam_img = np.array(bArray).reshape(msg.height, msg.width, 3)


def dep_img_callback(msg):
  global dep_img
  
  dep_ary = np.frombuffer(msg.data, dtype=np.int16)
  #print(bArray)
  dep_img = np.array(dep_ary).reshape(msg.height, msg.width)


def get_prediction(img, threshold):
  transform = T.Compose([T.ToTensor()]) # Defing PyTorch Transform
  img = transform(img) # Apply the transform to the image
  pred = model([img]) # Pass the image to the model
  pred_class = [COCO_INSTANCE_CATEGORY_NAMES[i] for i in list(pred[0]['labels'].numpy())] # Get the Prediction Score
  pred_boxes = [[(i[0], i[1]), (i[2], i[3])] for i in list(pred[0]['boxes'].detach().numpy())] # Bounding boxes
  pred_score = list(pred[0]['scores'].detach().numpy())
  pred_t = [pred_score.index(x) for x in pred_score if x > threshold][-1] # Get list of index with score greater than threshold.
  pred_boxes = pred_boxes[:pred_t+1]
  pred_class = pred_class[:pred_t+1]
  return pred_boxes, pred_class


def object_detection_api(img, threshold=0.5, rect_th=3, text_size=3, text_th=3):
  boxes, pred_cls = get_prediction(img, threshold) # Get predictions
  
  for i in range(len(boxes)):
    cv.rectangle(img, boxes[i][0], boxes[i][1],color=(0, 255, 0), thickness=rect_th) # Draw Rectangle with the coordinates
    cv.putText(img,pred_cls[i], boxes[i][0],  cv.FONT_HERSHEY_SIMPLEX, text_size, (0,255,0),thickness=text_th) # Write the prediction class
  plt.figure(figsize=(20,30)) # display the output image
  plt.imshow(img)
  plt.xticks([])
  plt.yticks([])
  plt.show()



def display_cam_img(i):
  global cam_img
  global dep_img
  
  

  threshold=0.5
  rect_th=3
  text_size=3
  text_th=3
  
  if (cam_img is not None) and (dep_img is not None):
    img = cam_img.copy()

    plt.cla()
    boxes, pred_cls = get_prediction(img, threshold) # Get predictions
  
    for i in range(len(boxes)):
      cv.rectangle(img, boxes[i][0], boxes[i][1],color=(0, 255, 0), thickness=rect_th) # Draw Rectangle with the coordinates
      cv.putText(img,pred_cls[i], boxes[i][0],  cv.FONT_HERSHEY_SIMPLEX, text_size, (0,255,0),thickness=text_th) # Write the prediction class
    #plt.figure(figsize=(20,30)) # display the output image
    plt.imshow(img)


    #object_detection_api(cam_img)

    #plt.imshow(cam_img)
    #plt.imshow(dep_img)
    


def cam_img_node():
  rospy.init_node('cam_proc')
  rospy.wait_for_message("/camera/color/image_raw", Image)
  rospy.wait_for_message("/camera/depth/image_rect_raw", Image)
  img_sub = rospy.Subscriber("/camera/color/image_raw", Image, raw_img_callback)
  dep_sub = rospy.Subscriber("/camera/depth/image_rect_raw", Image, dep_img_callback)
  rate = rospy.Rate(1e3)

  while not rospy.is_shutdown():

    img_plot = animation.FuncAnimation(plt.gcf(), display_cam_img, 1)
    plt.tight_layout()
    plt.show()

    rate.sleep()




if __name__=="__main__":
  cam_img_node()