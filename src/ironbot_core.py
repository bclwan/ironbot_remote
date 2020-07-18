#!/usr/bin/python2

import rospy
import tf
import tf2_ros
from std_msgs.msg import Float32, String
from sensor_msgs.msg import Imu
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Pose, Point, Quaternion, Twist, Vector3, TransformStamped
from tf.transformations import euler_from_quaternion
from tf2_msgs.msg import TFMessage

import sys

import numpy as np
from scipy.spatial.transform import Rotation as R

from robot_tf_func import *
from motion_ctrl import motion_control, path_planner
from scan_capture import scan_proc
from utility import *

robot_state = state_var[0]



def cmd_callback(msg):
  global robot_state
  for i in range(len(state_var)):
    if msg.data==state_var[i]:
      robot_state = state_var[i]


def get_next_ort(rot_ang):
  curr_ort = R.from_euler('xyz', tf_get_ort_e())
  rotat = R.from_rotvec([0,0,rot_ang])
  next_ort = rotat*curr_ort
  return next_ort.as_euler('xyz')




def get_next_pos(direction, distance):
  curr_pos = tf_get_pos()
  next_pos = (curr_pos[0]+np.cos(direction)*distance, curr_pos[1]+np.sin(direction)*distance)
  return next_pos



def main():
  #Pre-set
  SIM = False
  if len(sys.argv)>1:
    if sys.argv[1]=="sim":
      SIM = True

  # ROS setup
  rospy.init_node('ironbot', anonymous=True)

  tf_sub = rospy.Subscriber('tf', TFMessage, tf_callback)
  state_pub = rospy.Publisher('state', String, queue_size=1, latch=True)
  cmd_sub = rospy.Subscriber('state_cmd', String, cmd_callback)
  cmd_vel_pub = rospy.Publisher('cmd_vel', Twist, queue_size=1)
  
  empty_twist = Twist()

  rate = rospy.Rate(100)

  # Tools setup
  scanMapScale = 40.0
  scanInvert = True
  if SIM:
    scanInvert = False

  scan_processor = scan_proc(get_ort=tf_get_ort_e, get_pos=tf_get_pos, 
                            scaleup=scanMapScale, pub_scan_map=True, auto_gen_map=True, 
                            invert=scanInvert, bot_circumference=(0.2, 0.2), print_path=True)
  while not scan_processor.gScanInit:
    rate.sleep()


  robot_path_planner = path_planner(cmd_pub=cmd_vel_pub, 
                                    get_ort=tf_get_ort_e, 
                                    get_pos=tf_get_pos, 
                                    get_range=scan_processor.get_range_data, 
                                    step_size=0.15, Ps=2.0, Pa=4.0)
  
  standalone_motion_ctrl = motion_control(cmd_pub=cmd_vel_pub, 
                                          get_ort=tf_get_ort_e, 
                                          get_pos=tf_get_pos,
                                          get_range=scan_processor.get_range_data,
                                          target_thres=0.2, track_bound=0.5,
                                          view_angle=150, safe_dist=0.25,
                                          DBG=True)

  print("===Ironbot Core Ready===")
  print("Robot Local Pos: ", scan_processor.ego_loc)
  world_pos = tf_get_pos()
  trans = (world_pos[0]-scan_processor.ego_loc[0]/scanMapScale, world_pos[1]-scan_processor.ego_loc[1]/scanMapScale)
  pt = np.zeros((1,2))
  pt[0] = scan_processor.ego_loc
  print("Robot World Pos: ", world_2d_tf(trans, -tf_get_ort_e()[2], pt/scanMapScale))

  while not rospy.is_shutdown():
    global robot_state

    robot_state = "NAV"

    state_pub.publish(robot_state)

    if robot_state=="IDLE":
      rate.sleep()
    
    elif robot_state=="NAV":
      nav_rate = rospy.Rate(1)

      path = None
      while path==None:
        rand_goal = scan_processor.sample_free_pos(1, show=False)
        path = scan_processor.local_path_AStar_search(rand_goal[0], dbg=False)

      print("GO Next: ", rand_goal)  
      #print("Path: ", path)
      world_pos = tf_get_pos()
      trans = (world_pos[0]-scan_processor.ego_loc[0]/scanMapScale, world_pos[1]-scan_processor.ego_loc[1]/scanMapScale)
      path_in_world = world_2d_tf(trans, -tf_get_ort_e()[2], np.array(path)/scanMapScale)
      #print("Path(World): ", path_in_world)

      a = raw_input()
      nav_rate.sleep()


    elif robot_state=="WANDER":
      min_dist = 0.3
      max_dist = 0.5
      collision = True
      trapped = False

      distance = 0.0
      rot_angle = 0.0      

      trial = 0
      while collision and trial<20:
        distance = np.random.uniform(low=min_dist, high=max_dist, size=1)[0]
        safe_dist = distance + min_dist/2.0
        backward = np.random.rand()>0.8
        rot_angle = np.random.uniform(low=-np.pi/2, high=np.pi/2, size=1)[0]
        if backward:
          print("<WANDER> Backward Direction")
          rot_angle = rot_angle + np.random.uniform(low=-np.pi/2, high=0, size=1)[0]
        else:
          print("<WANDER> Forward Direction")
        abs_dir = rot_angle
        if abs_dir<0.0:
          abs_dir = abs(abs_dir)+np.pi
        collision = scan_processor.next_state_collision_est(safe_dist, abs_dir)
        if collision:
          print("Obstacle found")
        #rate.sleep()
        trial += 1
        if trial==20:
          trapped = True
      
      next_ort = get_next_ort(rot_angle)
      next_pos = get_next_pos(next_ort[2], distance)
      pose_msg = Pose()
      pose_msg.position.x = next_pos[0]
      pose_msg.position.y = next_pos[1]

      print "WANDER MODE..."
      print next_pos
      
      if not trapped:
        try:
          standalone_motion_ctrl.move(pose_msg, timeout=10)
        except KeyboardInterrupt:
          standalone_motion_ctrl.clear_motion()
      else:
        print("!!!WARNING!!!")
        print("ROBOT MAY BE TRAPPED")
    
    elif robot_state=="TRAVEL":
      pass

    rate.sleep()


  



if __name__=="__main__":
  main()
