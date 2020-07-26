#!/usr/bin/env python2

import rospy
from std_msgs.msg import Int8, Int32MultiArray, Float32
from nav_msgs.msg import OccupancyGrid
from sensor_msgs.msg import LaserScan, Image
from cv_bridge import CvBridge

import numpy as np
from scipy.ndimage import rotate

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import style

from robot_tf_func import *
from utility import *





class map_creator():
  def __init__(self, scaleup=40.0, upd_threshold=0.05):
    self.scaleup = scaleup
    self.upd_thres = upd_threshold
    
    self.map_init = False

    self.local_occp_map = None
    self.glb_map = None
    self.glb_map_size = (0,0)

    self.odom = odom_listener()
    self.scan_map_sub = rospy.Subscriber("/free_space", Image, self.occp_img_callback)

    self.ego_ort = 0.0
    self.ego_pos = (0.0, 0.0)
    self.ego_map_loc = (0,0)
    self.ego_map_org = (0.0, 0.0)



  def occp_img_callback(self, imgMsg):
    bridge = CvBridge()
    map_img = bridge.imgmsg_to_cv2(imgMsg, desired_encoding="mono8")
    #TO ADD: reverse flip and rot
    self.local_occp_map = np.rot90(np.fliplr(map_img), -1)
    
    self.map_update()


  def map_update(self):
    new_local_occp = (self.local_occp_map.copy()/255).astype(np.float)
    local_half_len = new_local_occp.shape[0]>>1

    if self.map_init==False:
      self.glb_map = np.ones(new_local_occp.shape, dtype=np.float)
      self.glb_map = (new_local_occp.copy()/255).astype(np.float)
      half_len = new_local_occp.shape[0]>>1
      self.ego_map_loc = (half_len, half_len)
      self.ego_map_org = (half_len, half_len)
      self.ego_init_ort = self.odom.get_ort()[2]
      self.ego_ort = self.odom.get_ort()[2]
      self.ego_pos = self.odom.get_pos()
      self.map_init = True
      print("Map Init")
    else:
      new_ort = self.odom.get_ort()[2]
      new_pos = self.odom.get_pos()

      dx = new_pos[0] - self.ego_pos[0]
      dy = new_pos[1] - self.ego_pos[1]
      dth = world_dir_diff(self.ego_ort, new_ort)

      if abs(dx)<=self.upd_thres:
        dx = 0
      
      if abs(dy)<=self.upd_thres:
        dy = 0

      if abs(dth)<= self.upd_thres:
        dth = 0

      if dx==0 and dy==0 and dth==0:
        return

      curr_map_size_x = self.glb_map.shape[0]
      curr_map_size_y = self.glb_map.shape[1]

      nxt_ego_map_loc = (self.ego_map_loc[0]+dx*self.scaleup, self.ego_map_loc[1]+dy*self.scaleup)
      occ_map_rot = world_dir_chg(self.ego_init_ort, new_ort)

      occ_map_range_x = (nxt_ego_map_loc[0]-local_half_len, nxt_ego_map_loc[0]+local_half_len)
      occ_map_range_y = (nxt_ego_map_loc[1]-local_half_len, nxt_ego_map_loc[1]+local_half_len)

      map_x_inc = 0
      map_x_inc_dir = 1

      if occ_map_range_x[0]>=0:
        if occ_map_range_x[1]>=curr_map_size_x:
          map_x_inc = occ_map_range_x[1]-curr_map_size_x+1
      elif occ_map_range_x[0]<0:
        map_x_inc_dir = -1
        map_x_inc = 0-occ_map_range_x[0]+1

      map_y_inc = 0
      map_y_inc_dir = 1

      if occ_map_range_y[0]>=0:
        if occ_map_range_y[1]>=curr_map_size_y:
          map_y_inc = occ_map_range_y[1]-curr_map_size_y+1
      elif occ_map_range_y[0]<0:
        map_y_inc_dir = -1
        map_y_inc = 0-occ_map_range_y[0]+1

      new_map_shape = (int(curr_map_size_x+map_x_inc), int(curr_map_size_y+map_y_inc))
      new_map = np.ones(new_map_shape, dtype=np.float)*0.5
      new_map_old_tmp = np.ones(new_map_shape, dtype=np.float)*0.5

      old_map_range_x = (0,0)
      old_map_range_y = (0,0)

      if map_x_inc_dir==1:
        old_map_range_x = (0, curr_map_size_x)
      else:
        old_map_range_x = (new_map_shape[0]-curr_map_size_x, new_map_shape[0])

      if map_y_inc_dir==1:
        old_map_range_y = (0, curr_map_size_y)
      else:
        old_map_range_y = (new_map_shape[1]-curr_map_size_y, new_map_shape[1])

      #insert old map
      new_map[old_map_range_x[0]:old_map_range_x[1], old_map_range_y[0]:old_map_range_y[1]] = self.glb_map.copy()
      new_map_old_tmp[old_map_range_x[0]:old_map_range_x[1], old_map_range_y[0]:old_map_range_y[1]] = self.glb_map.copy()

      new_occ = rotate(new_local_occp, int(180*occ_map_rot/np.pi), reshape=False)
      new_occ_range_x = (0,0)
      new_occ_range_y = (0,0)

      if map_x_inc_dir==1:
        new_occ_range_x = (new_map_shape[0]-new_occ.shape[0], new_map_shape[0])
      else:
        new_occ_range_x = (0, new_occ.shape[0])

      if map_y_inc_dir==1:
        new_occ_range_y = (new_map_shape[1]-new_occ.shape[1], new_map_shape[1])
      else:
        new_occ_range_y = (0, new_occ.shape[1])

      overlap_range_x = (max(old_map_range_x[0], new_occ_range_x[0]), min(old_map_range_x[1], new_occ_range_x[1]))
      overlap_range_y = (max(old_map_range_y[0], new_occ_range_y[0]), min(old_map_range_y[1], new_occ_range_y[1]))

      #mark down old map data coord
      occp_set = new_map>=0.8
      free_set = new_map<=0.2

      #update with new local map
      a = new_occ_range_x[0]
      b = new_occ_range_x[1]
      c = new_occ_range_y[0]
      d = new_occ_range_y[1]
      new_map[a:b, c:d] = new_occ

      #fuse two set data
      new_map[occp_set] = 0.05*new_map[occp_set] + 0.95*new_map_old_tmp[occp_set]
      new_map[free_set] = 0.05*new_map[free_set] + 0.95*new_map_old_tmp[free_set]

      new_ego_map_loc = (new_occ_range_x[1]>>1, new_occ_range_y[1]>>1)

      self.ego_map_loc = new_ego_map_loc
      self.glb_map = new_map
      self.glb_map_size = new_map_shape
      self.ego_pos = new_pos
      self.ego_ort = new_ort
      


def mapping():
  rospy.init_node('mapping')

  map_tool = map_creator()

  map_pub = rospy.Publisher("/global_map", Image, queue_size=1)
  bridge = CvBridge()
  rate = rospy.Rate(10)

  while not map_tool.map_init:
    rate.sleep()

  while not rospy.is_shutdown():
    imgMsg = bridge.cv2_to_imgmsg(np.fliplr(np.rot90((map_tool.glb_map*255).astype(np.uint8), 1)), encoding="mono8")
    map_pub.publish(imgMsg)
    rate.sleep()



if __name__=="__main__":
  mapping()