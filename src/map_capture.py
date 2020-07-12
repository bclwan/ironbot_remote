#!/usr/bin/env python2

import rospy
from nav_msgs.msg import OccupancyGrid
from tf.transformations import euler_from_quaternion
from tf2_msgs.msg import TFMessage

import numpy as np
import skimage.measure
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import style
import collections
from copy import copy

from robot_tf_func import *


global map_size
global map_org
global map_ort
global map_plt
global map_geo


map_plt = None


plt.style.use('fivethirtyeight')


def map_callback(data):
  #rospy.loginfo(rospy.get_caller_id() + "I heard %s", data.data)
  global map_size
  global map_org
  global map_ort
  global map_plt
  global map_geo
  
  w = data.info.width
  h = data.info.height
  map_arr = np.array(data.data).reshape((h,w))
  map_plt = np.rot90(map_arr.T, 2)
  map_geo = np.fliplr(np.flipud(map_plt))
  
  org = data.info.origin
  ort_q = (org.orientation.x, org.orientation.y, org.orientation.z, org.orientation.w)
  ort_e = euler_from_quaternion(ort_q)
  
  map_size = map_arr.size
  map_org = (org.position.x, org.position.y)
  map_ort = ort_e
  
  
  #print "Map size: %d x %d" %(map_arr.shape[0], map_arr.shape[1])
  #print "Origin: (%f, %f)" % (org.position.x, org.position.y)
  #print "Orientation: (%f, %f, %f)" % (ort_e[0], ort_e[1], ort_e[2])
  
  #plt.imshow(map_geo)
  #plt.show()
  #plt.imshow(map_plt)
  #plt.show()




class map_server():
  def __init__(self):
    self.map_src = None
    self.map_dns = None
    self.dns_scale = (0,0)
    self.dns_size = (0,0)
    self.dns_method = ['max']
    self.map_org_coord = (0,0)
    self.map_ort_e = (0,0,0)
  
  
  def map_update(self, map, org, ort):
    self.map_src = map
    self.map_org_coord = org
    self.map_ort_e = ort


  def map_dnsample(self, windows=(3,3), method='max', raise_unknown=False):
    if raise_unknown:
      self.map_dns = skimage.measure.block_reduce(self.map_raise_unknown(self.map_src), windows, np.max)
    else:
      self.map_dns = skimage.measure.block_reduce(self.map_src, windows, np.max)
    self.dns_size = self.map_dns.shape
    self.dns_scale = windows


  def map_raise_unknown(self, map):
    map[map == -1] = 50
    return map
  
  
  def get_neigh(self, point):
    i_min = point[0]-1
    i_max = point[0]+1
    j_min = point[1]-1
    j_max = point[1]+1
    
    if i_min<0: i_min=0
    if i_max>self.dns_size[0]: i_max = self.dns_size[0]
    if j_min<0: j_min=0
    if j_max>self.dns_size[1]: j_max = self.dns_size[1]

    submap = self.map_dns[i_min:i_max+1, j_min:j_max+1]
    idx = np.where(submap<50)
    idx_i = idx[0]+point[0]-1
    idx_j = idx[1]+point[1]-1
    neigh_list = [(i,j) for (i,j) in zip(idx_i, idx_j)]
    return neigh_list
  
  
  def marker(self, pt_list):
    for p in pt_list:
      self.map_dns[p] = 150



def breath_first_graph_search(map, start, goal, DBG=False):
  begin = list()
  begin.append(start)
  frontier = collections.deque(begin)
  if DBG: print frontier
  explored = set()
  while frontier:
    path = frontier.popleft()
    if not isinstance(path, list): path = [path]
    if DBG: print path, path[-1]
    if path[-1]==goal:
      return path
    if path[-1] not in explored:
      explored.add(path[-1])
      for child in map.get_neigh(path[-1]):
        new_path = copy(path)
        new_path.append(child)
        frontier.append(new_path)
    if DBG: print frontier
      

def animate(i):
  global map_plt
  global map_org
  global map_ort
  
  map_srv = map_server()
  map_srv.map_update(map_plt, map_org, map_ort)
  map_srv.map_dnsample(windows=(4,4), raise_unknown=True)
  #print "Map down-sample size: %d x %d" % (map_srv.dns_size[0], map_srv.dns_size[1])
  
  plt.cla()
  plt.imshow(map_srv.map_dns)
  
  #child = map_srv.get_neigh((20,10))
  #print child

  rate = rospy.Rate(1000)
  rate.sleep()



def map_listener():
  global map_plt
  global map_org
  global map_ort



  rospy.init_node('map_listener')

  rospy.Subscriber("/map", OccupancyGrid, map_callback)
  tf_sub = rospy.Subscriber('tf', TFMessage, tf_callback)

  rate = rospy.Rate(100)
  
  while map_plt is None:
    pass
  
  
  ani = animation.FuncAnimation(plt.gcf(), animate, 1000)
  plt.tight_layout()
  plt.show()
  

  """
  map_srv = map_server()
  map_srv.map_update(map_plt, map_org, map_ort)
  map_srv.map_dnsample(windows=(4,4), raise_unknown=True)  
  
  plt.gcf()
  plt.tight_layout()
  plt.imshow(map_srv.map_dns)
  #plt.show()
  
  
  path = breath_first_graph_search(map_srv, (35,15), (10,15))
  print path
  map_srv.marker(path)
  plt.imshow(map_srv.map_dns)
  plt.show()
  """

  while True:
    rate.sleep()
    



map_listener()


  
  
  