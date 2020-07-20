from tf.transformations import euler_from_quaternion
from tf2_msgs.msg import TFMessage
from nav_msgs.msg import Odometry

global robot_tf
robot_tf = None

global robot_local_pose
robot_local_pose = Odometry()

state_var = ["IDLE", "WANDER", "NAV", "TRAVEL"]

def tf_callback(msg):
  global robot_tf
  if msg.transforms[0].header.frame_id=='odom':
    robot_tf = msg.transforms[0].transform
    #print robot_tf
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



# Local Pose
def local_pose_callback(msg):
  global robot_local_pose
  robot_local_pose = msg


def local_pose_get_pos():
  global robot_local_pose
  return (robot_local_pose.pose.pose.position.x, robot_local_pose.pose.pose.position.y)


def local_pose_get_ort_q():
  global robot_local_pose
  ort = robot_local_pose.pose.pose.orientation
  return (ort.x, ort.y, ort.z, ort.w)


def local_pose_get_ort_e():
  return euler_from_quaternion(local_pose_get_ort_q())





