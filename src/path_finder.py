#!/usr/bin/env python2

import time
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import style
import collections
from copy import copy

import cv2 as cv

from utility import *


class PathFinder():
  def __init__(self, scaleup=40.0, bot_circumference=(0.5,0.5)):
    self.scaleup = scaleup
    self.occp_map = np.zeros(0)
    self.ego_loc = (0,0)
    self.ego_space = np.zeros(0)
    self.ego_mass = 0
    self.bot_circumference = bot_circumference

    self.create_ego_occupancy(bot_circumference[0], bot_circumference[1])

  

  def upd_map(self, new_map):
    self.occp_map = new_map
    self.ego_loc = (self.occp_map.shape[0]>>1, self.occp_map.shape[1]>>1)



  def create_ego_occupancy(self, a, b): #unit in meter
    dim = int(max(a,b)*self.scaleup)*2
    self.ego_space = np.zeros((dim, dim), dtype=np.uint8)
    center = (dim>>1, dim>>1)
    axes = (int(a*self.scaleup),int(b*self.scaleup))
    angle = 0.0
    startAng = 0.0
    endAng = 360.0
    eColor = 1
    thickness = -1
    cv.ellipse(self.ego_space, center, axes, angle, startAng, endAng, eColor, thickness)
    self.ego_mass = np.sum(self.ego_space)


  def sample_free_pos(self, N=1, R=1.0, show=False):
    map_radius = int(R*self.scaleup)
    ego_lim_x0 = int(self.ego_loc[0]-self.bot_circumference[0]*self.scaleup)
    ego_lim_y0 = int(self.ego_loc[1]-self.bot_circumference[1]*self.scaleup)
    ego_lim_x1 = int(self.ego_loc[0]+self.bot_circumference[0]*self.scaleup)
    ego_lim_y1 = int(self.ego_loc[1]+self.bot_circumference[1]*self.scaleup)

    free_pos = []
    while len(free_pos)<N:
      #x = np.random.randint(0+self.bot_circumference[0], self.map_dim-self.bot_circumference[0])
      #y = np.random.randint(0+self.bot_circumference[1], self.map_dim-self.bot_circumference[1])
      x = np.random.randint(self.ego_loc[0]-map_radius, self.ego_loc[0]+map_radius)
      y = np.random.randint(self.ego_loc[1]-map_radius, self.ego_loc[1]+map_radius)
      if not ((x>=ego_lim_x0) and (x<=ego_lim_x1)):
        if not ((y>=ego_lim_y0) and (y<=ego_lim_y1)):
          if self.check_space_occupancy((x,y), self.occp_map, show=show)==0:
            free_pos.append((x,y))

    return free_pos


  def check_space_occupancy(self, pos, occ_map, dbg=False, show=False, arr_ret=False):
    if pos is None:
      pos = self.ego_loc
    if occ_map is None:
      occ_map = self.occp_map

    ego_shape = self.ego_space.shape[0]
    map_range_max = occ_map.shape[0]
    sub_space = np.zeros((ego_shape,ego_shape))
    map_range_x0 = pos[0]-(ego_shape>>1)
    map_range_x1 = pos[0]+(ego_shape>>1)
    map_range_y0 = pos[1]-(ego_shape>>1)
    map_range_y1 = pos[1]+(ego_shape>>1)
    
    if map_range_x0<0 or map_range_y0<0:
      if dbg: print("Into the Unknown: ", map_range_x0, map_range_y0)
      if arr_ret:
        return -1, None
      else:
        return -1
      
    if map_range_x1>=map_range_max or map_range_y1>=map_range_max:
      if dbg: print("Into the Unknown", map_range_x1, map_range_y1)
      if arr_ret:
        return -1, None
      else:
        return -1

    sub_space = occ_map[map_range_y0:map_range_y1, map_range_x0:map_range_x1]
    if show:
      print("POS: ", pos)
      plt.imshow(sub_space)
      plt.show()

    overlap_arr = np.multiply(sub_space, self.ego_space)
    overlap_sum = np.sum(overlap_arr)
    
    if overlap_sum>0:
      if dbg: print("Obstacles Around: %d" % overlap_sum)
      self.coll_flag = 1
    else:
      self.coll_flag = 0

    if arr_ret:
      return overlap_sum, overlap_arr.copy()
    else:
      return overlap_sum



  def AStar_hFunc(self, org, pos, goal, occ_map):
    score = 0.0

    dist_weight = 1.0

    
    dir_limit = np.pi*60.0/180.0
    main_dir = world_get_dir(org, goal)
    move_dir = world_get_dir(org, pos)
    diff = world_dir_diff(main_dir, move_dir)
    
    #Deviation Check
    if diff>dir_limit:
      score = -1.0
      return score
    

    #Collision Check
    overlap = self.check_space_occupancy(pos, occ_map)
    if overlap>0:
      score = -1.0
      return score

    dist = manhattan_dist(pos, goal)
    if dist==0:
      score = 0.0
    else:
      score += dist*dist_weight

    return score

    

  def get_free_loc_around(self, pos, occ_map, step=1):
    free_space = []
    candidates = [(pos[0]+step, pos[1]),
                  (pos[0]-step, pos[1]),
                  (pos[0], pos[1]+step),
                  (pos[0], pos[1]-step),
                  (pos[0]+step, pos[1]+step),
                  (pos[0]+step, pos[1]-step),
                  (pos[0]-step, pos[1]+step),
                  (pos[0]-step, pos[1]-step)
                  ]
    for c in candidates:
      if occ_map[c[0]][c[1]]==0:
        free_space.append(c)


    return free_space



  def local_path_AStar_search(self, dest, dbg=False):
    if self.occp_map is None:#no map data
      return None
    startPos = self.ego_loc
    occ_map = self.occp_map.copy()
    
    begin = []
    begin.append(startPos)
    if dbg:
      print("Begin: ", begin)
      print("End: ", dest)

    frontier = PriorityQueue()
    frontier.push(begin, 0)
    explored = set()

    #a=raw_input()

    while not frontier.isEmpty():
      statePath = frontier.pop()
      if dbg: print("Eval Path: ", statePath)
      currNode = statePath[-1]
      
      if (currNode==dest):
        if dbg: 
          print("Reached Goal State")
        
        #print("End: ", dest)
        #print("Path: ", statePath)
        #plt.imshow(occ_map)
        #plt.show()
        
        self.next_path = statePath
        return statePath

      if currNode not in explored:
        explored.add(currNode)
        if dbg: print("Eval Node: ", currNode)
        
        nextNode = self.get_free_loc_around(currNode, occ_map)
        if dbg: print("Next Node: ", nextNode)

        if len(nextNode)>0:
          for n in nextNode:
            stateVal = self.AStar_hFunc(currNode, n, dest, occ_map)
            if dbg: print(n, stateVal)
            if stateVal>=0:
              newStatePath = statePath[:]
              newStatePath.insert(len(newStatePath), n)
              frontier.push(newStatePath, stateVal)
              if dbg: print("Add Path: ", newStatePath)
