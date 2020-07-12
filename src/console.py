#!/usr/bin/python2
import time
import Tkinter as tk

import rospy
from std_msgs.msg import UInt8, Int8, Float32, String
from sensor_msgs.msg import Imu
from nav_msgs.msg import Odometry, OccupancyGrid
from geometry_msgs.msg import Point, Quaternion, Twist, Vector3, TransformStamped
from tf.transformations import euler_from_quaternion
from tf2_msgs.msg import TFMessage


from robot_tf_func import *



class control_panel():
  def __init__(self):
    global map_plt
    global map_org
    global map_ort

    self.state = 0

    self.panel = tk.Tk()
    self.panel.title("Ironbot Auto Control Panel")
    self.panel.geometry("320x480")
    self.panel.resizable(0,0)

    self.tf_sub = rospy.Subscriber('tf', TFMessage, tf_callback)
    self.state_sub = rospy.Subscriber('state', String, self.st_callback)
    self.state_pub = rospy.Publisher('state_cmd', String, queue_size=1, latch=True)
    self.odomest_cmd = rospy.Publisher('est_cmd', Int8, queue_size=1, latch=True)
    self.drive_lock = rospy.Publisher('DRV_LOCK', UInt8, queue_size=1, latch=True)

    #Initialize Cmd
    self.state_pub.publish("NONE")
    self.odomest_cmd.publish(0)
    self.drive_lock.publish(0)

    intro_msg = "IRONBOT"
    msg = tk.Label(self.panel, text=intro_msg, width=10)
    msg.pack()

    buttonCommand_n = tk.Button(self.panel, text="NONE", command=lambda: self.set_robot_state(-1))
    buttonCommand_n.pack()

    buttonCommand_0 = tk.Button(self.panel, text="IDLE", command=lambda: self.set_robot_state(0))
    buttonCommand_0.pack()

    buttonCommand_1 = tk.Button(self.panel, text="WANDER", command=lambda: self.set_robot_state(1))
    buttonCommand_1.pack()

    buttonCommand_2 = tk.Button(self.panel, text="TRAVEL", command=lambda: self.set_robot_state(2))
    buttonCommand_2.pack()

    buttonCommand_3 = tk.Button(self.panel, text="RESET ODOM", command=lambda: self.reset_odom())
    buttonCommand_3.pack()

    buttonCommand_4 = tk.Button(self.panel, text="EMERGENCY STOP", command=lambda: self.stop_motor())
    buttonCommand_4.pack()

    buttonCommand_5 = tk.Button(self.panel, text="REBOOT MOTOR", command=lambda: self.reboot_motor())
    buttonCommand_5.pack()

    buttonClose = tk.Button(self.panel, text="!==Close==!", command=self.panel.destroy)
    buttonClose.pack()

    self.robot_state = tk.Label(self.panel, text=self.get_robot_state(), anchor=tk.W, width=100)
    self.robot_state.pack()

    self.robot_pose = tk.Label(self.panel, text=self.get_robot_tf(), anchor=tk.W, width=100)
    self.robot_pose.pack()

    self.panel.after(1, self.update_info)
    self.panel.mainloop()


  def update_info(self):
    self.robot_pose["text"] = self.get_robot_tf()
    self.robot_state["text"] = "Ironbot State: " + state_var[self.state]
    rate = rospy.Rate(1000)
    rate.sleep()
    self.panel.after(1, self.update_info)
    


  def st_callback(self, msg):
    cur_state = msg.data
    if cur_state=="IDLE":
      self.state = 0
    elif cur_state=="WANDER":
      self.state = 1
    elif cur_state=="TRAVEL":
      self.state = 2


  def get_robot_tf(self):
    x, y = tf_get_pos()
    o = tf_get_ort_e()
    x_msg = "%.3f" % x
    y_msg = "%.3f" % y
    o_msg = "%.3f" % o[2]

    return 'Pos: ('+x_msg+','+y_msg+')'+" Ort: "+o_msg


  def set_robot_state(self, cmd):
    self.state_pub.publish(self.get_cmd_name(cmd))

  
  def reset_odom(self):
    self.odomest_cmd.publish(1)
    time.sleep(0.5)
    self.odomest_cmd.publish(0)


  def stop_motor(self):
    self.drive_lock.publish(1)
    time.sleep(0.5)

  def reboot_motor(self):
    self.drive_lock.publish(0)
    time.sleep(0.5)

  def get_cmd_name(self, cmd):
    if cmd==-1:
      return "NONE"
    else:
      return state_var[cmd]


  def get_robot_state(self):
    return state_var[self.state]






def main_console():
  global robot_tf

  # ROS setup
  rospy.init_node('ironbot_console', anonymous=True)

  tf_sub = rospy.Subscriber('tf', TFMessage, tf_callback)
  

  print "ROS Node setup done"

  # Console setup
  ib_panel = control_panel()


  
  






if __name__=="__main__":
  main_console()