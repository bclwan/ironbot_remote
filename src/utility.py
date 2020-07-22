import collections
import heapq

import numpy as np

from scipy.spatial.transform import Rotation as R



"""
The following PriorityQueue class is captured from the assignment skeleton code
"""
class PriorityQueue:
  """
    Implements a priority queue data structure. Each inserted item
    has a priority associated with it and the client is usually interested
    in quick retrieval of the lowest-priority item in the queue. This
    data structure allows O(1) access to the lowest-priority item.
  """
  def  __init__(self):
    self.heap = []
    self.count = 0

  def push(self, item, priority):
    entry = (priority, self.count, item)
    heapq.heappush(self.heap, entry)
    self.count += 1

  def pop(self):
    (_, _, item) = heapq.heappop(self.heap)
    return item

  def isEmpty(self):
    return len(self.heap) == 0

  def update(self, item, priority):
    # If item already in priority queue with higher priority, update its priority and rebuild the heap.
    # If item already in priority queue with equal or lower priority, do nothing.
    # If item not in priority queue, do the same thing as self.push.
    for index, (p, c, i) in enumerate(self.heap):
      if i == item:
        if p <= priority:
          break
        del self.heap[index]
        self.heap.append((priority, c, item))
        heapq.heapify(self.heap)
        break
    else:
      self.push(item, priority)




def world_2d_tf(world_coord, world_ort, points):
  points_3d = np.array(points)

  if points_3d.shape[1]==2:
    points_3d = np.column_stack([points_3d, np.zeros(points_3d.shape[0])])

  rot = R.from_euler('z', world_ort, degrees=False)
  tra = np.array([world_coord[0], world_coord[1], 0])    

  new_points = rot.apply(points_3d) + tra

  return np.delete(new_points, 2, axis=1)



def world_get_dir(src, dst):
  x0, y0 = src[0], src[1]
  x1, y1 = dst[0], dst[1]
  dy = y1-y0
  dx = x1-x0

  if dx==0:
    return np.sign(dy)*np.pi/2

  theta = np.arctan(dy/abs(dx))
  
  if dx<0:
    if dy==0:
      theta = np.pi
    else:
      theta += np.sign(dy)*np.pi/2
  
  return theta



def world_dir_diff(a0, a1):
  diff = abs(a0-a1)
  if diff>np.pi:
    diff = 2*np.pi - diff

  return diff



def manhattan_dist(p0, p1):
  return abs(p0[0]-p1[0]) + abs(p0[1]-p1[1])



def theta_rot(src_ort, theta):
  rot = R.from_euler('z', theta, degrees=False)
  ego = np.array([np.cos(src_ort), np.sin(src_ort), 0.0])
  new_ego = rot.apply(ego)
  new_ort = 0.0

  if new_ego[1]==0.0 and np.sign(new_ego[0])==-1:
    new_ort = np.pi
  elif new_ego[0]==0.0:
    new_ort = np.sign*np.pi/2
  else:
    new_ort = np.arctan(new_ego[1]/abs(new_ego[0]))

  return new_ort


