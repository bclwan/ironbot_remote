#!/usr/bin/python2
import __future__

import rospy
from std_msgs.msg import Int8, Float32
from sensor_msgs.msg import Imu
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Point, Quaternion, Twist, Vector3, TransformStamped
import tf
import tf2_ros

from ironbot_rmt_ctrl.srv import RstLocalOdom, RstLocalOdomResponse
from ironbot_rmt_ctrl.srv import GetScanPoint
from utility import theta_rot

import sys
import math

import numpy as np
from numpy.random import uniform, randn
from scipy.spatial.transform import Rotation as R
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import style

import cv2

from icp_func import *

plt.style.use('fivethirtyeight')

# Declare Global Variable
glbAX = 0.0
glbAY = 0.0
glbWZ = 0.0
glbRPS = 0.0
glbCmd = Twist()

glbBotLoc = []



def create_uniform_particles(x_range, y_range, hdg_range, N):
    particles = np.empty((N, 3))
    particles[:, 0] = uniform(x_range[0], x_range[1], size=N)
    particles[:, 1] = uniform(y_range[0], y_range[1], size=N)
    particles[:, 2] = uniform(hdg_range[0], hdg_range[1], size=N)
    particles[:, 2] %= 2 * np.pi
    return particles



def create_gaussian_particles(mean, std, N):
    particles = np.empty((N, 3))
    particles[:, 0] = mean[0] + (randn(N) * std[0])
    particles[:, 1] = mean[1] + (randn(N) * std[1])
    particles[:, 2] = mean[2] + (randn(N) * std[2])
    particles[:, 2] %= 2 * np.pi
    return particles




class robot_state():
  def __init__(self, start_time=0.0, tyre_r=0.0336, steer_max=2.84, steer_range=50.0, L=0.14, lr=0.02, N_particles=50, model="ACKM", use_scan=False):
    self.rstLocalOdom_service = rospy.Service('rst_local_odom', RstLocalOdom, self.resetLocalOdom)

    self.state = np.zeros(5, dtype=np.float)  #x, y, theta, vx, w
    self.local_state = np.zeros(5, dtype=np.float)
    self.last_time = start_time
    
    self.tyre_radius = tyre_r
    self.steer_max = steer_max
    self.steer_bound = np.pi*steer_range/180.0
    self.L = L
    self.lr = lr
    self.N_particles = N_particles
    self.model = model

    self.twoPiR = 2*np.pi*self.tyre_radius
    self.steerConv = self.steer_bound/self.steer_max

    self.record_size = 50
    self.record_pt = 0
    self.imu_record = np.zeros((self.record_size,4), dtype=np.float)
    self.imu_chop_lv = 0.03

    self.idx = 0

    self.theta_rot_vec = np.vectorize(theta_rot)
    if use_scan:
      rospy.wait_for_service("scan_points")
      self.service_get_scan_point = rospy.ServiceProxy("scan_points", GetScanPoint)
      self.scan = np.zeros(1)
      self.scan_prev = np.zeros(1)
      self.scan_stat = 0
    
    rospy.Subscriber("est_cmd", Int8, self.consoleCmd_callback)



  def consoleCmd_callback(self, msg):
    if msg.data==1: #reset odom state
      self.state = np.zeros(5, dtype=np.float)
      self.local_state = np.zeros(5, dtype=np.float)
      self.last_time = rospy.Time.now().to_sec()
      self.imu_record = np.zeros((self.record_size,4), dtype=np.float)


  def updateScanPoint(self):
    pt = self.service_get_scan_point(0)
    self.scan_prev = self.scan.copy()
    self.scan = np.column_stack([pt.points_x, pt.points_y])
    if self.scan_stat<2:
      self.scan_stat += 1



  def resetLocalOdom(self, rst):
    self.local_state = np.zeros(5, dtype=np.float)
    print("Reset Local Odometry")
    return RstLocalOdomResponse(0)
    

  def ackermann_model_pred(self, control, std, dt):
    dX = dY = dTheta = 0.0

    #lookup table for steer input to hardware steer angle
    #control[1] -> steer angle
    steer = self.steerConv*control[1] #inaccurate assumption

    steer += abs(np.sign(control[1]))*randn(1)[0]*std[0] #noise

    #lookup table for motor input to forward speed
    acc = control[0]
    acc += abs(np.sign(control[0]))*randn(1)[0]*std[1]
    fSpeed = self.state[3] + acc*dt

    slip = np.arctan(self.lr*np.tan(steer)/self.L)
    turn = (self.state[2] + slip) % (2*np.pi)

    dx = fSpeed*np.cos(turn)*dt
    dy = fSpeed*np.sin(turn)*dt

    pred_angular_v = fSpeed*np.cos(slip)*np.tan(steer)/self.L
    dTheta = (pred_angular_v*dt) % (2*np.pi)

    return dX, dY, dTheta



  def differential_model_pred(self, control, std, t):

    dX = dY = dTheta = 0.0

    return dX, dY, dTheta

    

  def predict_step(self, particles, control, std, t):
    dt = float(t - self.last_time)

    dX = dY = dTheta = 0.0
    if self.model=="ACKM":
      dX, dY, dTheta = self.ackermann_model_pred(control, std, dt)
    elif self.model=="DIFF":
      dX, dY, dTheta = self.differential_model_pred(control, std, dt)

    N = len(particles)

    particles[:,0] += dX
    particles[:,1] += dY
    particles[:,2] = self.theta_rot_vec(particles[:,2], dTheta)



  def update_step(self, particles):
    pass


  def direct_state_inc(self, dx, dy, dth):
    self.state[0] += dx
    self.state[1] += dy
    self.state[2] = theta_rot(self.state[2], dth)


  def state_pred_vanilla(self, wheel_rps, angular_v, acc_x, acc_y, control, t, dbg=False):
    dt = float(t - self.last_time)
    steer = self.steerConv*control[1]
    direction = np.sign(self.steerConv*control[0])
    if direction==0:
      direction = np.sign(acc_x)
    
    encoder_vel = direction*wheel_rps*self.twoPiR
    imu_vel = np.dot(self.imu_record[:,0], self.imu_record[:,1])
    if imu_vel<self.imu_chop_lv:
      imu_vel = 0.0
    
    #fSpeed = 0.5*encoder_vel + 0.5*imu_vel
    fSpeed = encoder_vel

    slip = np.arctan(self.lr*np.tan(steer)/self.L)
    turn = (self.state[2] + slip) % (2*np.pi)

    dx = fSpeed*np.cos(turn)*dt
    dy = fSpeed*np.sin(turn)*dt

    if dbg: print("[%d] dt:%f, dx:%f, EncVel:%f, Turn:%f, IMUVel:%f"% (self.idx, dt, dx, encoder_vel, turn, imu_vel))
    
    pred_angular_v = fSpeed*np.cos(slip)*np.tan(steer)/self.L
    mean_angular_v = 0.0
    if fSpeed!=0:
      mean_angular_v = 0.5*(angular_v+pred_angular_v)
    theta = theta_rot(self.state[2], mean_angular_v*dt)

    self.state[0] += dx
    self.state[1] += dy
    self.state[2] = theta
    self.state[3] = fSpeed
    self.state[4] = mean_angular_v

    self.local_state[0] += dx
    self.local_state[1] += dy
    self.local_state[2] = theta_rot(self.local_state[2], mean_angular_v*dt)
    self.local_state[3] = fSpeed
    self.local_state[4] = mean_angular_v

    self.last_time = t

    self.imu_record[self.record_pt] = np.array([dt, acc_x, acc_y, angular_v])
    self.record_pt = (self.record_pt+1)%self.record_size

    self.idx+=1
    """
    if self.record_pt>5:
      print self.imu_record[self.record_pt-5:self.record_pt]
      print imu_derived_v
      print
    print self.state[0], self.state[1]
    """

  
    

def plt_animate(i):
  global glbBotLoc

  plt.cla()
  plt.plot()



def imu_sub_callback(imuData):
  global glbAX
  global glbAY
  global glbWZ

  glbAX = imuData.linear_acceleration.x
  glbAY = imuData.linear_acceleration.y
  glbWZ = imuData.angular_velocity.z
  #print "AX:%f, AY:%f, WZ:%f" % (glbAX,glbAY,glbWZ)

  
  
def rps_sub_callback(rpsData):
  global glbRPS
  glbRPS = rpsData.data
  #print "New RPS: %f" % glbRPS



def cmd_sub_callback(cmdData):
  global glbCmd
  glbCmd = cmdData
  #print glbCmd.linear.x



def odometer():
  PUB_ODOM = True
  USE_SCAN = True

  if len(sys.argv)>1:
    if sys.argv[1]=="sim":
      PUB_ODOM = False

  # In ROS, nodes are uniquely named. If two nodes with the same
  # name are launched, the previous one is kicked off. The
  # anonymous=True flag means that rospy will choose a unique
  # name for our 'listener' node so that multiple listeners can
  # run simultaneously.
  rospy.init_node('odometer', anonymous=False)
  print("Setup Odometer Node")

  
  rospy.Subscriber("imu", Imu, imu_sub_callback)
  rospy.Subscriber("rps", Float32, rps_sub_callback)
  rospy.Subscriber("cmd_vel", Twist, cmd_sub_callback)

  odom_broc = tf.TransformBroadcaster()
  twist_pub = rospy.Publisher("speedometer", Twist, queue_size=1)
  localOdom_pub = rospy.Publisher("local_odom", Odometry, queue_size=1)
  
  msgLocalOdom = Odometry()
  localOdom_pub.publish(msgLocalOdom)

  vel_msg = Twist()

  rate = rospy.Rate(2)

  bot = robot_state(start_time=rospy.Time.now().to_sec(), use_scan=USE_SCAN)


  # Setup static transformint

  # spin() simply keeps python from exiting until this node is stopped
  
  while not rospy.is_shutdown():
    global glbAX
    global glbAY
    global glbWZ
    global glbRPS
    global glbCmd
    current_time = rospy.Time.now()

    """
    bot.state_pred_vanilla( wheel_rps=glbRPS, 
                            angular_v=glbWZ, 
                            acc_x=glbAX,
                            acc_y=glbAY,
                            control=(glbCmd.linear.x, glbCmd.angular.z), 
                            t=current_time.to_sec(), dbg=True)
    """

    if USE_SCAN:
      bot.updateScanPoint()

      if bot.scan_stat >= 2:
        T_hist, pts = icp(bot.scan_prev, bot.scan, distance_threshold=0.05, verbose=True)
        print(len(T_hist))
        print

        #a = raw_input()
      

    #print("WZ:", glbWZ, "RPS:", glbRPS, "CMD:", glbCmd)
    #print("X=", bot.state[0], "Y=", bot.state[1], "VX=", bot.state[3])

    if PUB_ODOM:

      odom_tran = (float(bot.state[0]), float(bot.state[1]), 0.0)
      odom_quat = tf.transformations.quaternion_from_euler(0.0,0.0,float(bot.state[2]))
      odom_broc.sendTransform(odom_tran, odom_quat, current_time, "base_link", "odom")

      msgLocalOdom.pose.pose.position.x = bot.local_state[0]
      msgLocalOdom.pose.pose.position.y = bot.local_state[1]
      local_ort = tf.transformations.quaternion_from_euler(0.0,0.0,bot.local_state[2])
      msgLocalOdom.pose.pose.orientation.x = local_ort[0]
      msgLocalOdom.pose.pose.orientation.y = local_ort[1]
      msgLocalOdom.pose.pose.orientation.z = local_ort[2]
      msgLocalOdom.pose.pose.orientation.w = local_ort[3]
      localOdom_pub.publish(msgLocalOdom)

    vel_msg.linear.x = bot.state[3]
    vel_msg.angular.z = bot.state[4]
    twist_pub.publish(vel_msg)
    
    rate.sleep()




if __name__=='__main__':
  odometer()

