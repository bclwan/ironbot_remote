from tf.transformations import euler_from_quaternion
from tf2_msgs.msg import TFMessage


global robot_tf
robot_tf = None

state_var = ["IDLE", "WANDER", "TRAVEL"]

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

