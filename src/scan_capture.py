#!/usr/bin/env python2

import rospy
from std_msgs.msg import Int8, Int32MultiArray, Float32
from nav_msgs.msg import OccupancyGrid
from sensor_msgs.msg import LaserScan, Image
#from tf.transformations import euler_from_quaternion
#from tf2_msgs.msg import TFMessage
from cv_bridge import CvBridge

from ironbot_rmt_ctrl.srv import GetScanPoint, GetScanPointResponse

import time
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import style
import collections
from copy import copy

import cv2 as cv

from robot_tf_func import *
from utility import *



class scan_proc():

  def __init__(self, get_ort, get_pos, 
              scaleup=40.0, zoom=1, pub_scan_map=False, auto_gen_map=False, invert=False, skip_inf=True,
              bot_circumference=(0.5,0.5), circum_check=False, print_path=False, marker=False):
    
    self.pub_scan_map = pub_scan_map

    self.get_ort = get_ort
    self.get_pos = get_pos

    self.auto_gen_map = auto_gen_map
    self.invert_scan = invert
    self.skip_inf = skip_inf
    self.bot_circumference = bot_circumference
    self.circum_check = circum_check
    self.print_path = print_path
    self.marker = marker

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

    
    self.raw_points = None

    self.scaleup = scaleup
    self.zoom = zoom
    self.base_dim = 0.0
    self.map_dim = 0
    self.shift = 0
    self.scan_map = None
    self.free_space = None

    self.next_path = []
    #self.next_path_pose = {"pos":self.get_pos(), "ort":self.get_ort()[2], "time":time.time()}
    self.next_path_pose = {"pos":(0.0,0.0), "ort":0.0, "time":time.time()}
    self.ego_space = None
    self.ego_mass = 0
    self.ego_loc = (0,0)

    self.coll_f = 0.0
    self.coll_b = 0.0
    self.coll_l = 0.0
    self.coll_r = 0.0

    self.sample_size = 5

    self.create_ego_occupancy(bot_circumference[0], bot_circumference[1])
    self.scan_sub = rospy.Subscriber("/scan", LaserScan, self.scan_callback)
    print("Subscribe scan data")
    self.scan_pub = rospy.Publisher("/scan_diagram", Image, queue_size=1)
    self.occp_pub = rospy.Publisher("/free_space", Image, queue_size=1)
    self.service_get_pts = rospy.Service('/scan_points', GetScanPoint, self.get_scan_pts)

    self.coll_flag = 0
    self.coll_flag_pub = rospy.Publisher("/coll_flag", Int8, queue_size=1)
    self.coll_flag_pub.publish(self.coll_flag)

    self.coll_dirc = 0.0
    self.coll_dirc_pub = rospy.Publisher("/coll_dirc", Float32, queue_size=1)
    self.coll_dirc_pub.publish(self.coll_dirc)

    print("Scan Processor setup completed")



  def scan_callback(self, msg):
    self.gScanData = msg
    if not self.gScanInit:
      print("Scan Angle Max: %f (rad)" % self.gScanData.angle_max)
      print("Scan Angle Inc: %f (rad)" % self.gScanData.angle_increment)
      print("Scan Range Min: %f (m)" % self.gScanData.range_min)
      print("Scan Range Max: %f (m)" % self.gScanData.range_max)
      print("Scan Data Length: %d" % len(self.gScanData.ranges))
      
      self.base_dim = self.gScanData.range_max*2 + 1
      self.shift = int(self.base_dim*self.scaleup)>>1
      self.map_dim = int(self.base_dim*self.scaleup)
      self.ego_loc = (self.shift, self.shift)

      angleIndex = np.array([i for i in range(0,len(self.gScanData.ranges))]) * self.gScanData.angle_increment
      self.gXcoord = np.cos(angleIndex)
      self.gYcoord = np.sin(angleIndex)
    
      self.gScanInit = True
      print("Scan Init: ", self.gScanInit)

    self.gViewLock = True
    self.gRanges = np.array(self.gScanData.ranges)
    if self.invert_scan:
      roll_step = len(self.gRanges)>>1
      self.gRanges = np.roll(self.gRanges, roll_step)

    if self.skip_inf:
      self.gRanges[abs(self.gRanges)==np.inf] = -1.0
    else:
      self.gRanges[abs(self.gRanges)==np.inf] = self.gScanData.range_max

    self.gViewLock = False

    if self.auto_gen_map:
      self.gen_local_map()
      if self.pub_scan_map:
        self.scan_publish()



  def scan_publish(self):
    pub_map = self.scan_map.copy()
    occ_map = self.free_space.copy()

    if self.print_path and len(self.next_path)>0:
      try:
        path = self.next_path[:]
        curr_pos = self.get_pos()
        curr_ort = self.get_ort()[2]
        rot =  self.next_path_pose["ort"] - curr_ort
        #print(rot)
        trans = np.array([self.next_path_pose["pos"][1]-curr_pos[1], self.next_path_pose["pos"][0]-curr_pos[0]])*self.scaleup
        path_org = np.array(path) - np.array(self.ego_loc)
        path_on_map = world_2d_tf((0,0), -rot, path_org).astype(np.int)
        path_on_map = path_on_map + np.array(self.ego_loc)
        path_on_map = world_2d_tf(trans, 0, path_on_map).astype(np.int)
        #print("Path on Map: ", path_on_map)
        pub_map[path_on_map[:,1],path_on_map[:,0],:] = np.array([255,255,0])
      except:
        pass
    
    if self.zoom>1:
      map_len = int(self.map_dim/self.zoom)
      half_len = map_len>>1
      pub_map = pub_map[self.ego_loc[0]-half_len:self.ego_loc[0]+half_len,self.ego_loc[1]-half_len:self.ego_loc[1]+half_len]
      occ_map = occ_map[self.ego_loc[0]-half_len:self.ego_loc[0]+half_len,self.ego_loc[1]-half_len:self.ego_loc[1]+half_len]
    
    occ_map = (occ_map*255).astype(np.uint8)
      

    bridge = CvBridge()
    imgMsg = bridge.cv2_to_imgmsg(np.fliplr(np.rot90(pub_map, 1)), encoding="bgr8")
    imgMsg_occ = bridge.cv2_to_imgmsg(np.fliplr(np.rot90(occ_map, 1)), encoding="mono8")
    self.scan_pub.publish(imgMsg)
    self.occp_pub.publish(imgMsg_occ)

  def get_range_data(self):
    while not self.gScanInit:
      pass
    while self.gViewLock:
      pass
    return self.gRanges.copy()


  def get_scan_pts(self, rqt):
    points = self.raw_points.copy()
    threshold_dist = 0.6

    px_list = []
    py_list = []
    for p in points:
      if abs(p[0])<threshold_dist and abs(p[1])<threshold_dist:
        px_list.append(p[0])
        py_list.append(p[1])
        
    points_x = points[:,0]
    points_y = points[:,1]
    #print(points_x[1:3], points_y[1:3])

    #return GetScanPointResponse(points_x.tolist(), points_y.tolist())
    return GetScanPointResponse(px_list, py_list)



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
      print("No scan data")
      return True
    
 
    print("Test for direction: %f travel %f" % (direction, safe_dist))
    
    closest = np.min(samples)
    avg = np.mean(samples)
    collision = closest<safe_dist and avg<safe_dist
    print("Closest Dist.: %f (avg %f)" % (closest, avg))

    return collision

 



  def gen_local_map(self):
    while not self.gScanInit:
      pass
    
    #Image Axis
    # o---------->x
    # |
    # |
    # |
    # |
    # v
    # y

    # Get raw range values
    ranges = self.gRanges.copy()
    #print ranges
    L = ranges.shape[0]
    half = L>>1
    quad = L>>2
    octa = L>>3
    center = np.array([self.shift, self.shift], dtype=int)

    """
    self.gFrontView = np.insert(ranges[0:octa], octa, center, axis=0)
    self.gFrontView = np.concatenate([self.gFrontView, ranges[7*octa:L]])
    self.gRearView = np.insert(ranges[half-octa:half+octa], 0, center, axis=0)
    self.gLeftView = np.insert(ranges[half-octa:half+octa], 0, center, axis=0)
    self.gRightView = np.insert(ranges[half+octa:7*octa], 0, center, axis=0)
    """

    scaleupRanges = ranges * self.scaleup
    pt_x = self.gXcoord * ranges
    pt_y = self.gYcoord * ranges

    if self.skip_inf:
      pt_x = np.delete(pt_x, ranges!=-1.0)
      pt_y = np.delete(pt_y, ranges!=-1.0)
    
    points = np.column_stack([pt_x, pt_y]).astype(np.float)
    self.raw_points = points.copy() #no scaled up or shift
    
    points = (points*self.scaleup).astype(np.int32)
    points += self.shift


    #print points
    """
    front = np.insert(points[0:octa], octa, center, axis=0)
    front = np.concatenate([front, points[7*octa:L]])
    left = np.insert(points[octa:octa+quad], 0, center, axis=0)
    back = np.insert(points[half-octa:half+octa], 0, center, axis=0)
    right = np.insert(points[half+octa:7*octa], 0, center, axis=0)
    """

    self.scan_map = np.zeros([self.map_dim,self.map_dim,3], dtype=np.uint8)
    cv.drawContours(self.scan_map, [points],0,(191,191,191),-1)
    cv.drawContours(self.scan_map, [points],0,(255,0,0),1)
    self.free_space = self.scan_map[:,:,1].copy()
    self.free_space = self.free_space.astype(np.float)
    non_occupied = self.free_space>0
    occupied = self.free_space==0
    self.free_space[non_occupied] = 0
    self.free_space[occupied] = 0.5
    self.free_space[points[:,1], points[:,0]] = 1
    """
    plt.imshow(self.free_space)
    plt.show()
    a = raw_input()
    """
    #Ego coordinate
    eCenter = (center[0], center[1])

    if self.marker:
      axes = (int(self.bot_circumference[0]*self.scaleup), int(self.bot_circumference[1]*self.scaleup))
      angle = 0.0
      startAng = 0.0
      endAng = 360.0
      eColor = (0,0,255)
      thickness = -1
      cv.ellipse(self.scan_map, eCenter, axes, angle, startAng, endAng, eColor, thickness)

      arrow_head = np.array((center[0]+7, center[1]))
      arrow_left = np.array((center[0]-5, center[1]-6))
      arrow_right = np.array((center[0]-5, center[1]+6))
      center_arrow = np.array([arrow_head, arrow_left, arrow_right])
      cv.drawContours(self.scan_map, [center_arrow],0,(0,255,0),-1)
      self.scan_map[center[0],center[1],0] = 255
      self.scan_map[center[0],center[1],1] = 0 
      self.scan_map[center[0],center[1],2] = 0

    """
    cv.drawContours(self.scan_map, [front],-1,(255,0,0),1)
    cv.drawContours(self.scan_map, [left],-1,(0,255,0),1)
    cv.drawContours(self.scan_map, [right],-1,(0,0,255),1)
    cv.drawContours(self.scan_map, [back],-1,(255,255,255),1)
    """
    #plt.imshow(np.fliplr(np.rot90(scan_map, 1)))

    if self.circum_check:
      self.check_overlap_dir(eCenter, self.free_space, dbg=False)

    rate = rospy.Rate(1e6)
    rate.sleep()


  def create_ego_occupancy(self, a, b): #unit in meter
    dim = int(max(a,b)*self.scaleup)*2
    self.ego_space = np.zeros((dim, dim), dtype=np.uint8)
    center = (dim>>1, dim>>1)
    axes = (int(a*self.scaleup),int(b*self.scaleup))
    angle = 0.0
    startAng = 0.0
    endAng = 360.0
    eColor = 1
    thickness = -1
    cv.ellipse(self.ego_space, center, axes, angle, startAng, endAng, eColor, thickness)
    self.ego_mass = np.sum(self.ego_space)



  def sample_free_pos(self, N=1, R=1.0, show=False):
    map_radius = int(R*self.scaleup)
    ego_lim_x0 = int(self.ego_loc[0]-self.bot_circumference[0]*self.scaleup)
    ego_lim_y0 = int(self.ego_loc[1]-self.bot_circumference[1]*self.scaleup)
    ego_lim_x1 = int(self.ego_loc[0]+self.bot_circumference[0]*self.scaleup)
    ego_lim_y1 = int(self.ego_loc[1]+self.bot_circumference[1]*self.scaleup)

    free_pos = []
    while len(free_pos)<N:
      #x = np.random.randint(0+self.bot_circumference[0], self.map_dim-self.bot_circumference[0])
      #y = np.random.randint(0+self.bot_circumference[1], self.map_dim-self.bot_circumference[1])
      x = np.random.randint(self.ego_loc[0]-map_radius, self.ego_loc[0]+map_radius)
      y = np.random.randint(self.ego_loc[1]-map_radius, self.ego_loc[1]+map_radius)
      if not ((x>=ego_lim_x0) and (x<=ego_lim_x1)):
        if not ((y>=ego_lim_y0) and (y<=ego_lim_y1)):
          if self.check_space_occupancy((x,y), self.free_space, show=show)==0:
            free_pos.append((x,y))

    return free_pos




  def check_space_occupancy(self, pos, occ_map, dbg=False, show=False, arr_ret=False):
    if pos is None:
      pos = self.ego_loc
    if occ_map is None:
      occ_map = self.free_space

    ego_shape = self.ego_space.shape[0]
    map_range_max = occ_map.shape[0]
    sub_space = np.zeros((ego_shape,ego_shape))
    map_range_x0 = pos[0]-(ego_shape>>1)
    map_range_x1 = pos[0]+(ego_shape>>1)
    map_range_y0 = pos[1]-(ego_shape>>1)
    map_range_y1 = pos[1]+(ego_shape>>1)
    
    if map_range_x0<0 or map_range_y0<0:
      if dbg: print("Into the Unknown: ", map_range_x0, map_range_y0)
      if arr_ret:
        return -1, None
      else:
        return -1
      
    if map_range_x1>=map_range_max or map_range_y1>=map_range_max:
      if dbg: print("Into the Unknown", map_range_x1, map_range_y1)
      if arr_ret:
        return -1, None
      else:
        return -1

    sub_space = occ_map[map_range_y0:map_range_y1, map_range_x0:map_range_x1]
    if show:
      print("POS: ", pos)
      plt.imshow(sub_space)
      plt.show()

    overlap_arr = np.multiply(sub_space, self.ego_space)
    overlap_sum = np.sum(overlap_arr)
    
    if overlap_sum>0:
      if dbg: print("Obstacles Around: %d" % overlap_sum)
      self.coll_flag = 1
    else:
      self.coll_flag = 0
    
    if self.circum_check:
      self.coll_flag_pub.publish(self.coll_flag)

    if arr_ret:
      return overlap_sum, overlap_arr.copy()
    else:
      return overlap_sum


  def check_overlap_dir(self, pos, occ_map, dbg=False, show=False):
    if pos is None:
      pos = self.ego_loc
    if occ_map is None:
      occ_map = self.free_space

    overlap_sum, overlap_arr = self.check_space_occupancy(pos, occ_map, dbg, show, arr_ret=True)
    if overlap_sum==0:
      return False, 0.0

    ret, thresh = cv.threshold(overlap_arr, 0, 255, cv.THRESH_BINARY)
    M = cv.moments(thresh)
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])

    ego_mask_center = (self.ego_space.shape[0]>>1, self.ego_space.shape[0]>>1)
    direction = world_get_dir(ego_mask_center, (cX,cY))
    self.coll_dirc = direction

    if dbg: 
      print("Ego Mask Center: ", ego_mask_center)
      print("Overlap Center: ", cX, ',', cY)
      print("Overlap Direction: ", direction)

    if self.circum_check:
      self.coll_dirc_pub.publish(self.coll_dirc)

    return True, direction




    


  def AStar_hFunc(self, org, pos, goal, occ_map):
    score = 0.0

    dist_weight = 1.0

    
    dir_limit = np.pi*60.0/180.0
    main_dir = world_get_dir(org, goal)
    move_dir = world_get_dir(org, pos)
    diff = world_dir_diff(main_dir, move_dir)
    
    #Deviation Check
    if diff>dir_limit:
      score = -1.0
      return score
    

    #Collision Check
    overlap = self.check_space_occupancy(pos, occ_map)
    if overlap>0:
      score = -1.0
      return score

    dist = manhattan_dist(pos, goal)
    if dist==0:
      score = 0.0
    else:
      score += dist*dist_weight

    return score

    

  def get_free_loc_around(self, pos, occ_map, step=1):
    free_space = []
    candidates = [(pos[0]+step, pos[1]),
                  (pos[0]-step, pos[1]),
                  (pos[0], pos[1]+step),
                  (pos[0], pos[1]-step),
                  (pos[0]+step, pos[1]+step),
                  (pos[0]+step, pos[1]-step),
                  (pos[0]-step, pos[1]+step),
                  (pos[0]-step, pos[1]-step)
                  ]
    for c in candidates:
      if occ_map[c[0]][c[1]]==0:
        free_space.append(c)


    return free_space



  def local_path_AStar_search(self, dest, dbg=False):
    if self.free_space is None:#no map data
      return None
    startPos = self.ego_loc
    occ_map = self.free_space.copy()
    
    self.next_path_pose = {"pos":self.get_pos(), "ort":self.get_ort()[2], "time":time.time()}
    
    begin = []
    begin.append(startPos)
    if dbg:
      print("Begin: ", begin)
      print("End: ", dest)

    frontier = PriorityQueue()
    frontier.push(begin, 0)
    explored = set()

    #a=raw_input()

    while not frontier.isEmpty():
      statePath = frontier.pop()
      if dbg: print("Eval Path: ", statePath)
      currNode = statePath[-1]
      
      if (currNode==dest):
        if dbg: 
          print("Reached Goal State")
        
        #print("End: ", dest)
        #print("Path: ", statePath)
        #plt.imshow(occ_map)
        #plt.show()
        
        self.next_path = statePath
        return statePath

      if currNode not in explored:
        explored.add(currNode)
        if dbg: print("Eval Node: ", currNode)
        
        nextNode = self.get_free_loc_around(currNode, occ_map)
        if dbg: print("Next Node: ", nextNode)

        if len(nextNode)>0:
          for n in nextNode:
            stateVal = self.AStar_hFunc(currNode, n, dest, occ_map)
            if dbg: print(n, stateVal)
            if stateVal>=0:
              newStatePath = statePath[:]
              newStatePath.insert(len(newStatePath), n)
              frontier.push(newStatePath, stateVal)
              if dbg: print("Add Path: ", newStatePath)

      #a=raw_input()



############################################################################

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
  tf_sub = rospy.Subscriber('tf', TFMessage, tf_callback)
  print("Subscribe TF")
  #scan_sub = rospy.Subscriber("/scan", LaserScan, scan_callback)

  scan_processor = scan_proc(scaleup=21.0, zoom=2, get_pos=tf_get_pos, get_ort=tf_get_ort_e, 
                            pub_scan_map=True, auto_gen_map=True, invert=False, skip_inf=False, print_path=False,
                            bot_circumference=(0.2, 0.2), circum_check=True)

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
  rospy.wait_for_message("/scan_diagram", Image)
  img_sub = rospy.Subscriber("/scan_diagram", Image, scan_img_callback)
  rate = rospy.Rate(100)

  while not rospy.is_shutdown():

    scan_plot = animation.FuncAnimation(plt.gcf(), display_scan_diagram, 1000)
    plt.tight_layout()
    plt.show()

    rate.sleep()


if __name__=="__main__":
  import sys
  option = sys.argv[1]
  if option=="client":
    scan_disp_client()
  elif option=="server":
    scan_disp_server()
  else:
    print("Error: Option unavailable")



