#!/usr/bin/env python
import rospy
import tf
from std_msgs.msg import Int32
from sensor_msgs.msg import Imu
from ack_steer.srv import ack_steer

from ack_ctrl import AckermannCtrl

import numpy as np


class IronbotControl:
    def __init__(self):
        rospy.init_node('ironbot_ctrl')
        rospy.wait_for_service('steer')
        
        self.rad2deg = 180.0/np.pi

        #init coordinates
        self.current_x = 0.0
        self.current_y = 0.0
        self.current_yaw = 0.0
        self.current_speed = 0.0

        self.set_steer = rospy.ServiceProxy('steer', ack_steer)
        self.steer = 90
        self.steer_l = 30
        self.steer_r = 150

        self.drv_pub = rospy.Publisher('drive', Int32, queue_size=1)

        self.tf_listener = tf.TransformListener()
        #self.imu_sub = rospy.Subscriber('imu', Imu, self.imu_cb)
        self.vel_sub = rospy.Subscriber('vel_x', Int32, self.vel_cb)

        self.timestamp = 0
        self.frame = 1

        waypoints = self.get_waypoints_from_file()
        controller = AckermannCtrl(waypoints)

        rate = rospy.Rate(50)
        while not rospy.is_shutdown():
            self.get_data()
            controller.update_values(self.current_x, self.current_y, self.current_yaw, 
                                     self.current_speed, self.timestamp, self.frame)
            controller.update_controls()
            cmd_throttle, cmd_steer, cmd_brake = controller.get_commands()
            self.publish_cmd(cmd_throttle, cmd_steer, cmd_brake)
            rate.sleep()
    
    def get_waypoints_from_file(self):
        waypoints = [[0.1, 0.0, 0.2],
                     [0.2, 0.0, 0.5],
                     [0.25, 0.1, 0.5], 
                     [0.3, 0.3, 0.5],
                     [0.35, 0.4, 0.5]]
        return waypoints

    def publish_cmd(self, cmd_throttle, cmd_steer, cmd_brake):
        steer = 90 + cmd_steer*self.rad2deg
        self.drv_pub.publish(max((cmd_throttle-cmd_brake), 0)*100)
        self.set_steer(steer)

    def get_data(self):
        try:
            (trans,rot) = self.tf_listener.lookupTransform('/odom', '/base_link', rospy.Time(0))
            self.current_x = trans[0]
            self.current_y = trans[1]
            euler = tf.transformations.euler_from_quaternion(rot)
            self.current_yaw = euler[2]
            
            print(self.current_x, self.current_y, self.current_yaw, self.current_speed)
            
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            return

    def vel_cb(self, msg):
        self.current_speed = msg.data
        
        
