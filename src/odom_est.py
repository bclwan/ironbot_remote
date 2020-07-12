#!/usr/bin/python2

import rospy
from std_msgs.msg import Int8, Float32
from sensor_msgs.msg import Imu
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Point, Quaternion, Twist, Vector3, TransformStamped
import tf
import tf2_ros

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import style

plt.style.use('fivethirtyeight')

# Declare Global Variable
glbAX = 0.0
glbAY = 0.0
glbWZ = 0.0
glbRPS = 0.0
glbCmd = Twist()

glbBotLoc = []

class robot_state():
  def __init__(self, start_time=0.0, tyre_r=0.0336, steer_max=2.84, steer_range=50.0, L=0.14, lr=0.02):
    self.state = np.zeros(5, dtype=np.float)  #x, y, theta, vx, w
    self.last_time = start_time
    
    self.tyre_radius = tyre_r
    self.steer_max = steer_max
    self.steer_bound = np.pi*steer_range/180.0
    self.L = L
    self.lr = lr

    self.twoPiR = 2*np.pi*self.tyre_radius
    self.steerConv = self.steer_bound/self.steer_max

    self.record_size = 50
    self.record_pt = 0
    self.imu_record = np.zeros((self.record_size,4), dtype=np.float)
    self.imu_chop_lv = 0.03

    self.idx = 0
    
    rospy.Subscriber("est_cmd", Int8, self.consoleCmd_callback)

  def consoleCmd_callback(self, msg):
    if msg.data==1:
      self.state = np.zeros(5, dtype=np.float)
      self.last_time = rospy.Time.now().to_sec()
      self.imu_record = np.zeros((self.record_size,4), dtype=np.float)

  def state_pred(self, wheel_rps, angular_v, acc_x, acc_y, control, t):
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

    print "[%d] dt:%f, dx:%f, EncVel:%f, Turn:%f, IMUVel:%f"% (self.idx, dt, dx, encoder_vel, turn, imu_vel)
    

    pred_angular_v = fSpeed*np.cos(slip)*np.tan(steer)/self.L
    mean_angular_v = 0.0
    if fSpeed!=0:
      mean_angular_v = 0.5*(angular_v+pred_angular_v)
    theta = (self.state[2] + mean_angular_v*dt) % (2*np.pi)

    self.state[0] += dx
    self.state[1] += dy
    self.state[2] = theta
    self.state[3] = fSpeed
    self.state[4] = mean_angular_v
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
  # In ROS, nodes are uniquely named. If two nodes with the same
  # name are launched, the previous one is kicked off. The
  # anonymous=True flag means that rospy will choose a unique
  # name for our 'listener' node so that multiple listeners can
  # run simultaneously.
  rospy.init_node('odometer', anonymous=False)

  
  rospy.Subscriber("imu", Imu, imu_sub_callback)
  rospy.Subscriber("rps", Float32, rps_sub_callback)
  rospy.Subscriber("cmd_vel", Twist, cmd_sub_callback)

  odom_broc = tf.TransformBroadcaster()
  twist_pub = rospy.Publisher("speedometer", Twist, queue_size=1)

  vel_msg = Twist()

  rate = rospy.Rate(100)

  bot = robot_state(start_time=rospy.Time.now().to_sec())


  # Setup static transform

  # spin() simply keeps python from exiting until this node is stopped
  
  while not rospy.is_shutdown():
    global glbAX
    global glbAY
    global glbWZ
    global glbRPS
    global glbCmd
    current_time = rospy.Time.now()

    bot.state_pred( wheel_rps=glbRPS, 
                    angular_v=glbWZ, 
                    acc_x=glbAX,
                    acc_y=glbAY,
                    control=(glbCmd.linear.x, glbCmd.angular.z), 
                    t=current_time.to_sec())
    
    #print("WZ:", glbWZ, "RPS:", glbRPS, "CMD:", glbCmd)
    #print("X=", bot.state[0], "Y=", bot.state[1], "VX=", bot.state[3])

    odom_tran = (float(bot.state[0]), float(bot.state[1]), 0.0)
    odom_quat = tf.transformations.quaternion_from_euler(0.0,0.0,float(bot.state[2]))
    odom_broc.sendTransform(odom_tran, odom_quat, current_time, "base_link", "odom")

    vel_msg.linear.x = bot.state[3]
    vel_msg.angular.z = bot.state[4]
    twist_pub.publish(vel_msg)
    
    rate.sleep()




if __name__=='__main__':
  odometer()

