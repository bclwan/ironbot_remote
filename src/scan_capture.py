#!/usr/bin/env python2

import rospy
from nav_msgs.msg import OccupancyGrid
from sensor_msgs.msg import LaserScan, Image
#from tf.transformations import euler_from_quaternion
#from tf2_msgs.msg import TFMessage
from cv_bridge import CvBridge

import numpy as np

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import style
import collections
from copy import copy

import cv2 as cv

#from robot_tf_func import *


class scan_proc():

  def __init__(self, scaleup=40.0, pub_scan_map=False, auto_gen_map=False, invert=False):
    self.scan_sub = rospy.Subscriber("/scan", LaserScan, self.scan_callback)
    self.scan_pub = rospy.Publisher("/scan_diagram", Image, queue_size=1)
    self.pub_scan_map = pub_scan_map

    self.auto_gen_map = auto_gen_map
    self.invert_scan = invert

    self.gScanInit = False
    self.gScanData = LaserScan()

    self.gAngleIndex = None
    self.gXcoord = None
    self.gYcoord = None

    self.gFrontView = None
    self.gLeftView = None
    self.gRightView = None
    self.gRearView = None
    self.gViewLock = False

    self.gRanges = None

    self.full_points = None

    self.scaleup = scaleup
    
    self.base_dim = 0.0
    self.shift = 0

    self.coll_f = 0.0
    self.coll_b = 0.0
    self.coll_l = 0.0
    self.coll_r = 0.0

    self.sample_size = 5


  def scan_callback(self, msg):
    self.gScanData = msg
    if not self.gScanInit:
      
      print "Scan Angle Max: %f (rad)" % self.gScanData.angle_max
      print "Scan Angle Inc: %f (rad)" % self.gScanData.angle_increment
      print "Scan Range Min: %f (m)" % self.gScanData.range_min
      print "Scan Range Max: %f (m)" % self.gScanData.range_max
      print "Scan Data Length: %d" % len(self.gScanData.ranges)
      
      self.base_dim = self.gScanData.range_max*2 + 1
      self.shift = int(self.base_dim*self.scaleup)>>1
      self.map_dim = int(self.base_dim*self.scaleup)

      angleIndex = np.array([i for i in range(0,len(self.gScanData.ranges))]) * self.gScanData.angle_increment
      self.gXcoord = np.cos(angleIndex)
      self.gYcoord = np.sin(angleIndex)
    
      self.gScanInit = True

    self.gViewLock = True
    self.gRanges = np.array(self.gScanData.ranges)
    if self.invert_scan:
      roll_step = len(self.gRanges)>>1
      self.gRanges = np.roll(self.gRanges, roll_step)
    self.gRanges[self.gRanges==np.inf] = self.gScanData.range_max
    self.gViewLock = False

    if self.auto_gen_map:
      self.gen_local_map()
      if self.pub_scan_map:
        self.scan_publish()

      


  def scan_publish(self):
    bridge = CvBridge()
    imgMsg = bridge.cv2_to_imgmsg(np.fliplr(np.rot90(self.scan_map, 1)), encoding="bgr8")
    self.scan_pub.publish(imgMsg)


  def get_range_data(self):
    while not self.gScanInit:
      pass
    while self.gViewLock:
      pass
    return self.gRanges.copy()

  def next_state_collision_est(self, safe_dist, direction):
    collision = True

    while not self.gScanInit:
      pass

    while self.gViewLock:
      pass

    ranges = self.gRanges.copy()
    full_size = ranges.shape[0]
    center = int(full_size*direction/(2*np.pi))
    samples = ranges[center-self.sample_size:center+self.sample_size].copy()
    size = samples.shape[0]
    
    if size<=0:
      print "No scan data"
      return True
    
 
    print "Test for direction: %f travel %f" % (direction, safe_dist)

    """
    if direction<=(np.pi/4) or direction>(7*np.pi/4):
      view = self.gFrontView
    elif direction>(np.pi/4) or direction<=(3*np.pi/4):
      view = self.gLeftView
    elif direction>(3*np.pi/4) or direction<=(5*np.pi/4):
      view = self.gRearView
    elif direction>(5*np.pi/4) or direction<=(7*np.pi/4):
      view = self.gRightView
    else:
      view = self.gFrontView
    """
    
    closest = np.min(samples)
    avg = np.mean(samples)
    collision = closest<safe_dist and avg<safe_dist
    print "Closest Dist.: %f (avg %f)" % (closest, avg)

    return collision


  def gen_local_map(self):
    while not self.gScanInit:
      pass
    while self.gViewLock:
      pass
    # Get raw range values
    ranges = self.gRanges.copy()
    #print ranges
    L = ranges.shape[0]
    half = L>>1
    quad = L>>2
    octa = L>>3
    center = np.array([self.shift, self.shift])

    """
    self.gFrontView = np.insert(ranges[0:octa], octa, center, axis=0)
    self.gFrontView = np.concatenate([self.gFrontView, ranges[7*octa:L]])
    self.gRearView = np.insert(ranges[half-octa:half+octa], 0, center, axis=0)
    self.gLeftView = np.insert(ranges[half-octa:half+octa], 0, center, axis=0)
    self.gRightView = np.insert(ranges[half+octa:7*octa], 0, center, axis=0)
    """

    scaleupRanges = ranges * self.scaleup
    pt_x = self.gXcoord * scaleupRanges + self.shift
    pt_y = self.gYcoord * scaleupRanges + self.shift
    points = np.column_stack([pt_x, pt_y]).astype(np.int32)
    #print points

    front = np.insert(points[0:octa], octa, center, axis=0)
    front = np.concatenate([front, points[7*octa:L]])
    left = np.insert(points[octa:octa+quad], 0, center, axis=0)
    back = np.insert(points[half-octa:half+octa], 0, center, axis=0)
    right = np.insert(points[half+octa:7*octa], 0, center, axis=0)

    self.scan_map = np.zeros([self.map_dim,self.map_dim,3], dtype=np.uint8)
    #cv.drawContours(scan_map, [points],0,(255,255,255),2)
    cv.drawContours(self.scan_map, [front],-1,(255,0,0),2)
    cv.drawContours(self.scan_map, [left],-1,(0,255,0),2)
    cv.drawContours(self.scan_map, [right],-1,(0,0,255),2)
    cv.drawContours(self.scan_map, [back],-1,(255,255,255),2)

    #plt.imshow(np.fliplr(np.rot90(scan_map, 1)))

    rate = rospy.Rate(1e6)
    rate.sleep()



scan_diag = None

def scan_img_callback(msg):
  global scan_diag
  bridge = CvBridge()
  scan_diag = bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")


def display_scan_diagram(i):
  global scan_diag
  if scan_diag is not None:
    plt.cla()
    plt.imshow(scan_diag)



def scan_disp_server():
  rospy.init_node('scan_capture')
  #scan_sub = rospy.Subscriber("/scan", LaserScan, scan_callback)

  scan_processor = scan_proc(scaleup=40.0, pub_scan_map=True, auto_gen_map=True, invert=True)
  img_sub = rospy.Subscriber("/scan_diagram", Image, scan_img_callback)

  rate = rospy.Rate(100)

  while not scan_processor.gScanInit:
    rate.sleep()

  while not rospy.is_shutdown():

    scan_plot = animation.FuncAnimation(plt.gcf(), display_scan_diagram, 1000)
    plt.tight_layout()
    plt.show()

    rate.sleep()


def scan_disp_client():
  rospy.init_node('scan_capture')
  img_sub = rospy.Subscriber("/scan_diagram", Image, scan_img_callback)
  rate = rospy.Rate(100)

  while not rospy.is_shutdown():

    scan_plot = animation.FuncAnimation(plt.gcf(), display_scan_diagram, 1000)
    plt.tight_layout()
    plt.show()

    rate.sleep()


if __name__=="__main__":
  scan_disp_client()



