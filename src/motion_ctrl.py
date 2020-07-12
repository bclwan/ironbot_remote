#!/usr/bin/env python2
# BEGIN ALL
import rospy
import tf
from tf.transformations import euler_from_quaternion
from geometry_msgs.msg import Twist, Pose, TransformStamped
from sensor_msgs.msg import LaserScan
from tf2_msgs.msg import TFMessage
import tf2_ros

import numpy as np

from ack_ctrl import AckermannCtrl

robot_tf = None

def scan_callback(msg):
  global g_range_ahead
  g_range_ahead = min(msg.ranges)


def tf_callback(msg):
  global robot_tf
  if msg.transforms[0].header.frame_id=='odom':
    robot_tf = msg.transforms[0].transform
  #print robot_tf.translation.x, robot_tf.translation.y

def tf_get_ort_q():
  global robot_tf
  while not robot_tf:
    pass
  return (robot_tf.rotation.x, robot_tf.rotation.y, robot_tf.rotation.z, robot_tf.rotation.w)

def tf_get_ort_e():
  global robot_tf
  while not robot_tf:
    pass
  return euler_from_quaternion(tf_get_ort_q())


def tf_get_pos():
  global robot_tf
  while not robot_tf:
    pass
    #print "Waiting current pose"
  return (robot_tf.translation.x, robot_tf.translation.y)



class motion_control():

  def __init__(self, cmd_pub, get_ort, get_pos, get_range, 
              Ps=2.5, Pa=4.0, 
              target_thres=0.2, track_bound=0.5,
              safe_dist=0.2, safe_avg=1.5, view_angle=180, 
              DBG=False):
    self.cmd = cmd_pub
    self.get_ort = get_ort
    self.get_pos = get_pos
    self.get_range = get_range
    self.Ps = Ps
    self.Pa = Pa
    self.ang_eps = 0.1
    self.lin_eps = 0.05
    self.lin_err = 0.5
    self.base_v = 0.4
    self.base_w = 0.1
    self.max_ops = 20

    self.target_thres = target_thres
    self.track_bound = track_bound

    self.safe_dist = safe_dist
    self.view_angle = view_angle
    self.safe_avg = safe_avg
    self.DBG = DBG

    self.USE_DIFFDRV = False
    self.USE_ACKERMANN = True
    self.ack_drv = AckermannCtrl(waypoints=None)
    self.vel_sub = rospy.Subscriber("speedometer", Twist, self.vel_callback)
    self.vel_msg = Twist()

    if self.USE_ACKERMANN:
      print "Using Ackermann System"

  def vel_callback(self, msg):
    self.vel_msg = msg


  def close_zone_2d(self, pos_a, pos_b, epsilon=0.1, err=0.5, up_bound=True):
    diff = np.linalg.norm(np.array(pos_a)-np.array(pos_b))
    det = diff<=epsilon or (diff>=err and up_bound)
    if self.DBG and det:
      print ("Hit Target")

    return det
  
  def check_off_track(self, init, goal, curr, B=0.5):
    A = np.sqrt((init[0]-goal[0])**2 + (init[1]-goal[1])**2)
    #find alpha direction 
    alp = 0.0
    if init[0]==goal[0]:
      alp = 0.0
    elif init[1]==goal[1]:
      if init[0]>goal[0]:
        alp = np.pi
      else:
        alp = 0.0
    else:
      alp = np.arctan((goal[1]-init[1])/(goal[0]-init[0]))

    ell_bound = (((curr[0]-goal[0])*np.cos(alp) + (curr[1]-goal[1])*np.sin(alp))**2)/A**2 + (((curr[0]-goal[0])*np.sin(alp) - (curr[1]-goal[1])*np.cos(alp))**2)/B**2

    if ell_bound > 1:
      print "Off Track"
      return True
    else:
      return False


  def check_target_zone(self, goal, curr, R=0.5):
    hit = (goal[0]-curr[0])**2 + (goal[1]-curr[1])**2
    if hit<=(R**2):
      print "Hit Target"
      return True
    else:
      return False
    

  def close_range(self, a, b, epsilon=0.1):
    return np.abs(a-b)<=epsilon


  def angle_opt_move(self, a, b):
    mov = 0
    rot = 1
    dist = b-a
    if abs(dist)>np.pi:
      mov = 2*np.pi-abs(dist)
      rot = -1*np.sign(dist)
    else:
      mov = abs(dist)
      rot = np.sign(dist)
      
    return rot, mov


  def clear_motion(self):
    twist = Twist()
    twist.linear.x=0
    twist.angular.z=0
    self.cmd.publish(twist)

  
  def slight_move(self, vel=0.5, timeout=3.0):
    import time
    start_time = time.time()
    timeup = False
    
    linear_vel = vel
    rate = rospy.Rate(1000)
    twist = Twist()
    twist.linear.x = linear_vel

    if vel<0:
      print "Trying slight backward move"
    else:
      print "Trying slight forward move"

    while not timeup:
      self.cmd.publish(twist)
      rate.sleep()
      timeup = (time.time()-start_time) > timeout

    self.clear_motion()



  def check_obstacle(self, EmergenceStop=True, StopTime=5):
    safe_dist = self.safe_dist
    half_view_angle = self.view_angle>>1
    inv_mov = not self.ack_drv.mov_forward


    ranges = self.get_range()
    full_size = ranges.shape[0]
    idx_midpt = full_size>>1
    half_sample_size = int(half_view_angle*full_size/360)
    samples = None
    if inv_mov:
      samples = ranges[idx_midpt-half_sample_size:idx_midpt+half_sample_size]
    else:
      samples = np.concatenate([ranges[0:half_sample_size], ranges[full_size-half_sample_size:full_size]])
    min_dist = np.min(samples)
    avg_dist = np.mean(samples)
    collision = (min_dist<safe_dist) or (avg_dist<self.safe_avg)
    
    
    if collision:
      print ""
      print "===================="
      print "WARNING: Approaching Obstacles!"
      print "===================="
      print("Front Scan: %f, %f") % (min_dist, avg_dist)
      print""
      

      if EmergenceStop:
        import time
        self.clear_motion()
        time.sleep(StopTime)


    return collision

  def move(self, target_pose, lin_eps=0.15, lin_err=1.0, timeout=None, chk_obst=True):
    action_result = 0
    if self.USE_ACKERMANN:
      action_result = self.ack_line_move(target_pose, timeout=timeout, chk_obst=chk_obst)
    elif self.USE_DIFFDRV:
      action_result = self.diff_line_move(target_pose, lin_eps=lin_eps, lin_err=lin_err, timeout=timeout)

    return action_result


  def ack_line_move(self, target_pose, timeout=None, chk_obst=True):
    import time
    start_time = time.time()
    runover = False
    obstacle = False
    init_trans = self.get_pos()
    target_trans = (target_pose.position.x, target_pose.position.y)

    waypoints = np.zeros((2,2))
    waypoints[0][0] = init_trans[0]
    waypoints[0][1] = init_trans[1]
    waypoints[1][0] = target_trans[0]
    waypoints[1][1] = target_trans[1]
    print waypoints
    
    self.ack_drv.update_waypoints(waypoints)
    
    twist = Twist()

    rate = rospy.Rate(10) #each command run for 0.1 sec

    hit_target = False
    off_track = False
    target_thres = self.target_thres
    track_bound = self.track_bound

    while (not hit_target) and (not off_track):
      curr_x, curr_y = self.get_pos()
      curr_trans = (curr_x, curr_y)
      curr_d = self.get_ort()[2]
      curr_v = self.vel_msg.linear.x
      curr_t = rospy.Time.now().to_sec()
      
      self.ack_drv.update_values(curr_x, curr_y, curr_d, curr_v, curr_t)
      self.ack_drv.update_controls(DBG=self.DBG)

      cmd_throttle, cmd_steer, cmd_brake = self.ack_drv.get_commands()
      twist.linear.x = cmd_throttle
      twist.angular.z = cmd_steer
      
      if self.DBG: print "T=%d: Throttle: %f \t\t Steer: %f" % (time.time(), twist.linear.x, twist.angular.z)
      #a = raw_input()

      self.cmd.publish(twist)      

      if chk_obst:
        obstacle = self.check_obstacle()

      if obstacle:
        break

      if timeout:
        if (time.time()-start_time) > timeout:
          #print "TIMEOUT"
          runover = True
          break

      rate.sleep()

      hit_target = self.check_target_zone(target_trans, curr_trans, R=target_thres)
      off_track = self.check_off_track(init_trans, target_trans, curr_trans, B=track_bound)

      if hit_target:
        self.clear_motion()

      if off_track:
        break
      
    
    self.clear_motion()

    if runover:
      print "End with Timeout"
      return -1
    elif obstacle:
      print "End with Obstruction"
      return -2
    elif off_track:
      print "End with Off Track"
      return -3
    else:
      print "Motion Controller: End with Goal"
      return 0


  def diff_line_move(self, target_pose, lin_eps=0.15, lin_err=1.0, timeout=None):
    import time
    start_time = time.time()
    runover = False
    obstacle = False

    #Proportional Control
    Ps = self.Ps
    Pa = self.Pa
    
    linear_vel = self.base_v
    angular_vel=self.base_w
    ang_eps=self.ang_eps
    max_ops=self.max_ops

    rate = rospy.Rate(1000)
    twist = Twist()
    
    target_trans = (target_pose.position.x, target_pose.position.y)
    #current_trans = robot_tf.translation
    current_trans = self.get_pos()
    current_rot = self.get_ort()
    
    ops = 0
    print ""
    #Rotate
    
    if(self.DBG): print "R"
    #find the trajetory gradient 
    dy = target_trans[1]-current_trans[1]
    dx = target_trans[0]-current_trans[0]
    x_dir = np.sign(dx)
    y_dir = np.sign(dy)
    theta = 0.0
    if abs(dx)!=0.0:
      theta = np.arctan(abs(dy)/abs(dx))
    if x_dir==-1 and y_dir==1:
      theta = np.pi-theta
    elif x_dir==1 and y_dir==-1:
      theta = -theta
    elif x_dir==-1 and y_dir==-1:
      theta = -np.pi+theta
      
    if(self.DBG): print "Current Pos: (%f, %f)" % (current_trans[0], current_trans[1])
    if(self.DBG): print "Target Pos: (%f, %f)" % (target_trans[0], target_trans[1])
    if(self.DBG): print "Direction: %f" % theta
    
    ang_dir, ang_mov = self.angle_opt_move(current_rot[2], theta)
    if(self.DBG): print "Turn: %d, %f" % (ang_dir, ang_mov)

    twist.angular.z = angular_vel*ang_dir*(Pa*ang_mov)
    while not self.close_range(current_rot[2], theta, ang_eps):
      self.cmd.publish(twist)
      current_rot = self.get_ort()
      #print "Orientation: %f" % current_rot[2]
      rate.sleep()
      if timeout:
        if (time.time()-start_time) > timeout:
          #print "TIMEOUT"
          runover = True
          break
    
    twist.angular.z = 0
    self.cmd.publish(twist)
    
    
    #Translate
    if(self.DBG): print "T"
    twist.angular.z = 0
    abs_dist_diff = np.sqrt(dx**2 + dy**2)
    if(self.DBG): print "Abs. dist.: %f" % abs_dist_diff
    twist.linear.x = linear_vel*(Ps*abs_dist_diff)
    #twist.linear.x = linear_vel*np.cos(theta)*x_dir*(Ps*abs_dist_diff)
    #twist.linear.y = linear_vel*np.sin(theta)*y_dir*(Ps*abs_dist_diff)
    current_trans = self.get_pos()
    while (not self.close_zone_2d(current_trans, target_trans, lin_eps, lin_err)):
      obstacle = self.check_obstacle()
      if obstacle:
        break
      self.cmd.publish(twist)
      ops += 1
      current_trans = self.get_pos()
      #print "Position: %f, %f" %(robot_tf.translation.x, robot_tf.translation.y)
      rate.sleep()
      if timeout:
        if (time.time()-start_time) > timeout:
          #print "TIMEOUT"
          runover = True
          break

    #Clear cmd
    twist.linear.x = 0
    twist.linear.y = 0
    self.cmd.publish(twist)

    if runover:
      return -1
    elif obstacle:
      return -2
    else:
      return 0
    



class path_planner():
  def __init__(self, cmd_pub, get_ort, get_pos, get_range, 
              start=(0,0), goal=(0,0), step_size=0.3, lin_eps=0.05, map_size=(10,10), 
              max_path_node=10, max_mission_node=10, Ps=2.0, Pa=4.0):
    self.cmd = cmd_pub
    self.get_ort = get_ort
    self.get_pos = get_pos
    self.get_range = get_range
    self.max_path_node = max_path_node
    self.max_mission_node = max_mission_node
    self.map = np.zeros(map_size)

    self.step_size = step_size
    self.lin_eps = lin_eps

    self.Ps = Ps
    self.Pa = Pa

    self.full_path = []
    self.current_mission = []#list of 2-D nodes

  def update_map(self):
    #Get local map
    pass

  def path_search(self):
    #Search min. cost path from local map
    #Path length constrainted by max_node_ahead
    pass

  def path_monitor(self):
    #Main control of path travel
    pass

  def set_mission(self, mission):
    self.current_mission = mission


  def mission_exec(self):
    for next_pose in self.current_mission:
      self.next_point_planner(None, next_pose, self.step_size, self.lin_eps)


  def mission_expl(self, timeout):
    for next_pose in self.current_mission:
      self.next_point_planner(None, next_pose, self.step_size, self.lin_eps, timeout=timeout)


  def next_point_planner(self, init_pose, goal_pose, step_size=0.25, lin_eps=0.2, timeout=None):
    rate = rospy.Rate(1000)
    motion = motion_control(self.cmd, self.get_ort, self.get_pos, self.get_range, Ps=self.Ps, Pa=self.Pa, DBG=True)
    init = None
    if not init_pose:
      print "No assigned init pose, seeking tf..."
      init = self.get_pos()
    else:
      init = (init_pose[0], init_pose[1])
    
    init_vec = np.array([init[0], init[1]])
    goal = (goal_pose[0], goal_pose[1])
    curr = init
    
    print "Initial Pos: (%f , %f)" %(init[0], init[1])
    print "Goal Pos: (%f , %f)" %(goal[0], goal[1])

    next_pose = Pose()
    
    #direction vector
    dir_i = goal[0]-init[0]
    dir_j = goal[1]-init[1]
    dir_vec = np.array([dir_i, dir_j])
    dir_mag = np.dot(dir_vec, dir_vec)
    dir_unit = dir_vec/np.sqrt(dir_mag)
    
    org_vec = np.array([0,0])

    move_res = 0
    
    print "Direction Vector: %f, %f" %(dir_vec[0], dir_vec[1])
    
    while not motion.check_target_zone(goal=goal, curr=curr, R=0.2):
      curr_vec = np.array([curr[0], curr[1]])
      corr_pt = ((np.dot(curr_vec-init_vec, dir_vec)/dir_mag)*dir_vec) + init_vec
      print "==Corr: %f, %f" %(corr_pt[0], corr_pt[1])
      forward = corr_pt + dir_unit*step_size
      exceed = np.sign(goal[0]-forward[0])!=np.sign(dir_unit[0]) and np.sign(goal[1]-forward[1])!=np.sign(dir_unit[1])
      if exceed:
        print "==Exceed"
        next_pose.position.x = goal[0]
        next_pose.position.y = goal[1]
      else:
        next_pose.position.x = forward[0]
        next_pose.position.y = forward[1]
      
      print "==Next: %f, %f" %(next_pose.position.x, next_pose.position.y)
      if exceed:
        move_res = motion.move(next_pose, lin_eps=lin_eps, lin_err=1.5*step_size, timeout=timeout)
      else:
        move_res = motion.move(next_pose, lin_eps=motion.lin_eps, lin_err=1.5*step_size, timeout=timeout)
        
      curr = self.get_pos()
      rate.sleep()

      if move_res<0:
        break
    
    if move_res==-1:
      print "End with Timeout"
    elif move_res==-2:
      print "End with Obstruction"
      #motion.slight_move(vel=-0.1, timeout=1.0)
    elif move_res==-3:
      print "End with Off Track"
    else:
      print "Path Planner: Goal"
  

g_range_ahead = 1 # anything to start

def main():
  from scan_capture import scan_proc

  global g_range_ahead

  print "==Robot Controller=="

  tf_name = ['base_footprint', 'odom']
  #scan_sub = rospy.Subscriber('scan', LaserScan, scan_callback)
  cmd_vel_pub = rospy.Publisher('cmd_vel', Twist, queue_size=1)
  scan_processor = scan_proc(scaleup=40.0, pub_scan_map=False, auto_gen_map=False, invert=True)
  
  rospy.init_node('wander')
  state_change_time = rospy.Time.now()
  driving_forward = True
  empty_twist = Twist()
  rate = rospy.Rate(10)


  robot_path_planner = path_planner(cmd_pub=cmd_vel_pub, 
                                    get_ort=tf_get_ort_e, get_pos=tf_get_pos, 
                                    get_range=scan_processor.get_range_data, 
                                    step_size=0.15, Ps=2.0, Pa=4.0)
  
  standalone_motion_ctrl = motion_control(cmd_pub=cmd_vel_pub, 
                                          get_ort=tf_get_ort_e, 
                                          get_pos=tf_get_pos, 
                                          get_range=scan_processor.get_range_data, 
                                          target_thres=0.2, track_bound=0.5,
                                          DBG=True)


  #robot_path_planner.set_mission([(0.5, 0.0)]) #vanilla test
  #robot_path_planner.set_mission([(2.0,1.0), (-2.0,5.0), (-5.0,4.0), (-4.0,1.0)]) #for house
  #robot_path_planner.set_mission([(2.0,2.0), (4.0,2.0), (4.0,5.0), (1.0,1.0)]) #for empty world

  #tf_sub = tf.TransformListener()
  tf_sub = rospy.Subscriber('tf', TFMessage, tf_callback)
  
  new_pose = Pose()
  new_pose.position.x = -1.0
  new_pose.position.y = 0.0
  try:
    #next_point_planner(None, new_pose, cmd_vel_pub, step_size=0.25, lin_eps=0.05)
    #robot_path_planner.mission_expl(timeout=10)
    standalone_motion_ctrl.move(new_pose, timeout=10, chk_obst=True)
  except KeyboardInterrupt:
    cmd_vel_pub.publish(empty_twist)
    rospy.signal_shutdown(KeyboardInterrupt)
  
  """
  while not rospy.is_shutdown():    
    rate.sleep()
  """
  # END ALL
  rate.sleep()
  rospy.signal_shutdown(0)
  print "WANDER NODE CLOSE"


if __name__=="__main__":
  main()
  