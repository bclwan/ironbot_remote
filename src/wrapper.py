import rospy
from std_msgs.msg import Int8, Int32
from geometry_msgs.msg import Twist

from robot_tf_func import odom_listener
from ironbot_rmt_ctrl.srv import RstMapping, GetMapArea
from ironbot_rmt_ctrl.srv import GetScanPoint

import time

import numpy as np


class RobotSimAct():
  def __init__(self):
    self.n = 13
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

  def sample(self):
    import random
    idx = random.randint(0, self.n)
    return idx

  def get_action(self, idx):
    return self.action_list[idx]



class RobotSimEnv():
  def __init__(self):
    rospy.init_node("robot_sim_wrapper")

    self.action_space = RobotSimAct()

    self.T = 600
    self.t0 = time.time()

    self.scan = None
    self.scan_prev = None
    self.scan_stat = 0

    self.clear_cmd = Twist()

    self.map_size = 0
    self.map_size_prev = 0

    self.coll_flag = 0
    self.coll_cnt = 0

    self.cmd_publisher = rospy.Publisher('cmd_vel', Twist, queue_size=1)
    self.coll_flag_sub = rospy.Subscriber('coll_flag', Int8, self.check_coll)
    self.service_rst_mapping = rospy.ServiceProxy('rst_mapping', RstMapping)
    self.service_get_map_area = rospy.ServiceProxy('get_map_area', GetMapArea)
    self.service_get_scan_point = rospy.ServiceProxy("scan_points", GetScanPoint)
    self.odom = odom_listener()
    self.act_rate = rospy.Rate(1)


  def check_coll(self, msg):
    self.coll_flag = msg.data
    if self.coll_flag:
      self.coll_cnt += 1


  def updateScanPoint(self):
    pt = self.service_get_scan_point(0)
    self.scan_prev = self.scan.copy()
    self.scan = np.column_stack([pt.points_x, pt.points_y])
    if self.scan_stat<2:
      self.scan_stat += 1


  def gen_reward(self):
    reward = 0
    self.map_size_prev = self.map_size
    self.map_size = self.service_get_map_area(0)
    if self.map_size>self.map_size_prev:
      reward += 1

    if self.coll_flag:
      reward -= 10

    if self.coll_cnt>0:
      reward -= 2*self.coll_cnt
      self.coll_cnt = 0 #reset coll_cnt

    return reward


  def reset(self):
    self.service_rst_mapping(0)
    self.cmd_publisher.publish(self.clear_cmd)
    self.updateScanPoint()
    self.coll_flag = 0
    self.coll_cnt = 0
    return self.scan


  def step(self, action):
    is_done = False
    
    act = self.action_space.get_action(action)
    new_cmd = Twist()
    new_cmd.linear.x = act[0]
    new_cmd.angular.z = act[1]

    self.cmd_publisher.publish(new_cmd)
    self.act_rate.sleep()
    self.cmd_publisher.publish(self.clear_cmd)

    self.updateScanPoint()
    reward = self.gen_reward()

    if time.time() - self.t0 > self.T:
      is_done = True

    return self.scan, reward, is_done, 0




def make_env(dummy):
  env = RobotSimEnv()
  return env

  
