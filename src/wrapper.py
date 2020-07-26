import rospy

from ironbot_rmt_ctrl.srv import RstMapping
from ironbot_rmt_ctrl.srv import GetScanPoint

import numpy as np


class RobotSimEnv():
  def __init__(self):
    rospy.init_node("robot_sim_wrapper")
    self.service_rst_mapping = rospy.ServiceProxy('rst_mapping', RstMapping)
    self.service_get_scan_point = rospy.ServiceProxy("scan_points", GetScanPoint)
    self.odom = odom_listener()

    self.scan = None
    self.scan_prev = None
    self.scan_stat = 0


  def updateScanPoint(self):
    pt = self.service_get_scan_point(0)
    self.scan_prev = self.scan.copy()
    self.scan = np.column_stack([pt.points_x, pt.points_y])
    if self.scan_stat<2:
      self.scan_stat += 1