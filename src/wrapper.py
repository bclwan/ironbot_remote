import rospy
from std_msgs.msg import Int8, Int32
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist


#from robot_tf_func import odom_listener
from ironbot_rmt_ctrl.srv import RstMapping, GetMapArea
from ironbot_rmt_ctrl.srv import GetScanPoint

import time

import numpy as np


class RobotSimAct():
  def __init__(self):
    self.max_speed = 0.1
    self.action_list = []

    self.action_list.append((0,0))
    self.action_list.append((self.max_speed, 0.2))
    self.action_list.append((self.max_speed, 0.4))
    self.action_list.append((self.max_speed, 0.6))
    self.action_list.append((self.max_speed, -0.2))
    self.action_list.append((self.max_speed, -0.4))
    self.action_list.append((self.max_speed, -0.6))
    self.action_list.append((-self.max_speed, 0.2))
    self.action_list.append((-self.max_speed, 0.4))
    self.action_list.append((-self.max_speed, 0.6))
    self.action_list.append((-self.max_speed, -0.2))
    self.action_list.append((-self.max_speed, -0.4))
    self.action_list.append((-self.max_speed, -0.6))

    self.n = len(self.action_list)

    self.invalid_act = 0

  def sample(self):
    import random
    idx = random.randint(0, self.n-1)
    return idx

  def get_action(self, idx):
    
    return self.action_list[idx]




class RobotSimObv():
  def __init__(self):
    self.shape = 0
    self.service_get_scan_point = rospy.ServiceProxy("scan_points", GetScanPoint)
    self.scan_sub = rospy.Subscriber("free_space", Image, self.scan_callback)

    self.scan = np.zeros(0)
    self.scan_prev = np.zeros(0)
    self.scan_stat = 0

    self.scan_img = np.zeros(0)
    self.scan_img_init = False

    #init scan data
    #rospy.wait_for_service("scan_points")
    #self.updateScanPoint()
    #self.shape = (self.scan.shape[0], self.scan.shape[1], 1)
    #print("Observation Shape: ", self.shape)
    
    while not self.scan_img_init:
      pass
    print("Got first scan img")

    self.shape = (1, self.scan_img.shape[0], self.scan_img.shape[1])
    print("Observation Shape: ", self.shape)


  def scan_callback(self, imgMsg):
    bArray = bytearray(imgMsg.data)
    scan_img = np.array(bArray).reshape(imgMsg.height, imgMsg.width)
    self.scan_img = np.rot90(np.fliplr(scan_img), -1)
    if not self.scan_img_init:
      self.scan_img_init = True


  def getScanImg(self):
    return self.scan_img.copy()

  def updateScanPoint(self):
    pt = self.service_get_scan_point(0)
    self.scan_prev = self.scan.copy()
    self.scan = np.column_stack([pt.points_x, pt.points_y])
    if self.scan_stat<2:
      self.scan_stat += 1




class RobotSimEnv():
  def __init__(self, dbg=True):
    rospy.init_node("robot_sim_wrapper")

    self.action_space = RobotSimAct()
    self.observation_space = RobotSimObv()

    self.T = 180
    self.t0 = time.time()
    self.clear_cmd = Twist()

    self.map_size = 0
    self.map_size_prev = 0

    self.coll_flag = 0
    self.coll_cnt = 0

    self.cmd_publisher = rospy.Publisher('cmd_vel', Twist, queue_size=1)
    self.coll_flag_sub = rospy.Subscriber('coll_flag', Int8, self.check_coll)
    self.service_rst_mapping = rospy.ServiceProxy('rst_mapping', RstMapping)
    self.service_get_map_area = rospy.ServiceProxy('get_map_area', GetMapArea)
    
    #self.odom = odom_listener()
    self.act_rate = rospy.Rate(1)

    self.dbg = dbg


  def check_coll(self, msg):
    self.coll_flag = msg.data
    if self.coll_flag:
      self.coll_cnt += 1



  def gen_reward(self):
    reward = 0
    self.map_size_prev = self.map_size
    self.map_size = int(self.service_get_map_area(0).area)
    if self.map_size>self.map_size_prev:
      reward += 1

    if self.coll_flag:
      reward -= 10

    if self.coll_cnt>0:
      reward -= 2*self.coll_cnt
      self.coll_cnt = 0 #reset coll_cnt

    return reward



  def reset(self):
    self.t0 = time.time()
    self.service_rst_mapping(0)
    self.cmd_publisher.publish(self.clear_cmd)
    self.observation_space.updateScanPoint()
    self.coll_flag = 0
    self.coll_cnt = 0
    return self.observation_space.scan


  def step(self, action):
    is_done = False
    
    act = self.action_space.get_action(action)
    new_cmd = Twist()
    new_cmd.linear.x = act[0]
    new_cmd.angular.z = act[1]

    self.cmd_publisher.publish(new_cmd)
    self.act_rate.sleep()
    self.cmd_publisher.publish(self.clear_cmd)

    #self.observation_space.updateScanPoint()
    reward = self.gen_reward()

    t = time.time()

    if self.dbg:
      print("[T=%f] Action: %d, Reward: %d" % (t-self.t0, action, reward))

    if t - self.t0 > self.T:
      is_done = True
      if self.dbg:
        print("<<Episode End>>")

    state = self.observation_space.scan_img.copy()
    state = state.reshape(84,84).astype(np.float)/255
    #print("State shape: ", state.shape)
    
    return state, reward, is_done, 0




def make_env(dummy):
  env = RobotSimEnv()
  return env

  
