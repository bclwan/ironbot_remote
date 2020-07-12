import numpy as np

class AckermannCtrl():
    def __init__(self, waypoints, init_throttle=0.5):
        self._current_x          = 0
        self._current_y          = 0
        self._current_yaw        = 0
        self._current_speed      = 0
        self._desired_speed      = 0.2
        self._current_frame      = 0
        self._current_timestamp  = 0
        self._start_control_loop = False
        self._set_throttle       = 0
        self._set_brake          = 0
        self._set_steer          = 0
        self._waypoints          = waypoints
        self._conv_rad_to_steer  = 180.0 / 70.0 / np.pi
        self._pi                 = np.pi
        self._2pi                = 2.0 * np.pi

        self.init_throttle       = init_throttle
        self.throttle_previous   = 0.0

        self.max_acc             = 1.5
        self.acc_error           = 0.0
        self.max_steer           = 0.785

        self.mov_forward         = True

        self.v_hist_len          = 20
        self.v_hist              = np.zeros(self.v_hist_len)
        self.v_hist_des          = np.zeros(self.v_hist_len)
        self.v_hist_ptr          = 0
        self.v_previous          = 0.0

        self.Kp                  = 1.7
        self.Ki                  = 0.15
        self.Kd                  = 0.32
        self.K_lat               = 0.5


    def update_values(self, x, y, yaw, speed, timestamp, frame=True):
        self._current_x         = x
        self._current_y         = y
        self._current_yaw       = yaw
        self._current_speed     = speed
        self._current_timestamp = timestamp
        self._current_frame     = frame
        if self._current_frame:
            self._start_control_loop = True


    def update_desired_speed(self):
        min_idx       = 0
        min_dist      = float("inf")
        desired_speed = 0
        for i in range(len(self._waypoints)):
            dist = np.linalg.norm(np.array([
                    self._waypoints[i][0] - self._current_x,
                    self._waypoints[i][1] - self._current_y]))
            if dist < min_dist:
                min_dist = dist
                min_idx = i
        if min_idx < len(self._waypoints)-1:
            desired_speed = self._waypoints[min_idx][2]
        else:
            desired_speed = self._waypoints[-1][2]
        self._desired_speed = desired_speed


    def update_waypoints(self, new_waypoints):
        self._waypoints = new_waypoints


    def get_commands(self):
        return self._set_throttle, self._set_steer, self._set_brake


    def set_throttle(self, input_throttle):
        # Clamp the throttle command to valid bounds
        #throttle           = np.fmax(np.fmin(input_throttle, 1.0), 0.0)
        self._set_throttle = input_throttle


    def set_steer(self, input_steer_in_rad):
        # Covnert radians to [-1, 1]
        input_steer = self._conv_rad_to_steer * input_steer_in_rad

        # Clamp the steering comm
        steer           = np.fmax(np.fmin(input_steer, 1.0), -1.0)
        self._set_steer = steer


    def set_brake(self, input_brake):
        # Clamp the steering command to valid bounds
        brake           = np.fmax(np.fmin(input_brake, 1.0), 0.0)
        self._set_brake = brake

    def update_controls(self, DBG=False):
        ######################################################
        # RETRIEVE SIMULATOR FEEDBACK
        ######################################################
        x               = self._current_x
        y               = self._current_y
        yaw             = self._current_yaw
        v               = self._current_speed
        #self.update_desired_speed()
        v_desired       = self._desired_speed
        t               = self._current_timestamp
        waypoints       = np.asarray(self._waypoints)
        throttle_output = 0
        steer_output    = 0
        brake_output    = 0

        ######################################################
        ######################################################
        # MODULE 7: DECLARE USAGE VARIABLES HERE
        ######################################################
        ######################################################

        # Skip the first frame to store previous v alues properly
        if self._start_control_loop:
            """
                Controller iteration code block.

                Controller Feedback Variables:
                    x               : Current X position (meters)
                    y               : Current Y position (meters)
                    yaw             : Current yaw pose (radians)
                    v               : Current forward speed (meters per second)
                    t               : Current time (seconds)
                    v_desired       : Current desired speed (meters per second)
                                      (Computed as the speed to track at the
                                      closest waypoint to the vehicle.)
                    waypoints       : Current waypoints to track
                                      (Includes speed to track at each x,y
                                      location.)
                                      Format: [[x0, y0, v0],
                                               [x1, y1, v1],
                                               ...
                                               [xn, yn, vn]]
                                      Example:
                                          waypoints[2][1]: 
                                          Returns the 3rd waypoint's y position

                                          waypoints[5]:
                                          Returns [x5, y5, v5] (6th waypoint)
                
                Controller Output Variables:
                    throttle_output : Throttle output (0 to 1)
                    steer_output    : Steer output (-1.22 rad to 1.22 rad)
                    brake_output    : Brake output (0 to 1)
            """

            ######################################################
            ######################################################
            # MODULE 7: IMPLEMENTATION OF LONGITUDINAL CONTROLLER HERE
            ######################################################
            ######################################################
            """
                Implement a longitudinal controller here. Remember that you can
                access the persistent variables declared above here. For
                example, can treat self.vars.v_previous like a "global variable".
            """
            
            # Change these outputs with the longitudinal controller. Note that
            # brake_output is optional and is not required to pass the
            # assignment, as the car will naturally slow down over time.

            #Check if movement valid
            #Movement Direction
            inv_mov = False
            mov_dir = 0.0
            dy = waypoints[1][1]-waypoints[0][1]
            dx = waypoints[1][0]-waypoints[0][0]

            if (waypoints[1][0] != waypoints[0][0]) and (waypoints[1][1]== waypoints[0][1]):#x0!==x1 and dy==0
                trans_dir = np.sign(waypoints[1][0]-waypoints[0][0])
                if abs(yaw)<=(np.pi/2): #Pointing in x pos direction
                    if trans_dir>=0:
                        mov_dir = 0.0
                    else:
                        mov_dir = np.pi
                else:
                    if trans_dir>=0:
                        mov_dir = np.pi
                    else:
                        mov_dir = 0.0
            else:
                mov_dir = np.arctan( dy / abs(dx) )
                if dx<0:
                    mov_dir = np.sign(mov_dir)*np.pi - mov_dir
            
            if DBG: print("Motion Direction: %f" % (mov_dir))    


            dir_diff = abs(mov_dir-yaw)
            if DBG: print("Diff of Motion/Yaw: %f" %(dir_diff))

            if dir_diff>(np.pi/2):
                inv_mov = True
                self.mov_forward = False
                if DBG: print("Reverse Motion")
            else:
                self.mov_forward = True
                
            #PID
            error = self.v_hist_des - self.v_hist
            curr_error = v_desired - v
            #print "Current V: %f  Desired V: %f" % (v, v_desired)

            P = self.Kp * curr_error
            I = self.Ki * sum(error)
            D = self.Kd * sum(error)/self.v_hist_len
            des_acc = P + I + D
            if inv_mov: des_acc = -des_acc
            if DBG: print "Desired Acc: %f" % des_acc

            if des_acc >= 0:
                if self.v_previous == 0:       
                    throttle_output = self.init_throttle
                else:
                    throttle_output = min(des_acc, self.max_acc)/self.max_acc
            elif inv_mov:
                if self.v_previous == 0:       
                    throttle_output = -self.init_throttle
                else:
                    throttle_output = max(des_acc, -self.max_acc)/self.max_acc
            else:
                throttle_output = self.throttle_previous*(1.0-min(abs(des_acc), self.max_acc)/self.max_acc)

            if DBG: print("ACK| Throttle OP: %f" % throttle_output)
            
            #No brake equipped
            brake_output = 0
            

            ######################################################
            ######################################################
            # MODULE 7: IMPLEMENTATION OF LATERAL CONTROLLER HERE
            ######################################################
            ######################################################
            """
                Implement a lateral controller here. Remember that you can
                access the persistent variables declared above here. For
                example, can treat self.vars.v_previous like a "global variable".
            """
            #print(yaw, waypoints.shape, waypoints[0][0], waypoints[0][1], x, y)

            # Change the steer output with the lateral controller. 
            """
            des_yaw = 0.0

            if waypoints[1][0] == waypoints[0][0]:#x0==x1
                des_yaw = np.pi/2 * np.sign(waypoints[1][1]-waypoints[0][1])
                if DBG: print("Laternal Motion")
            else:
                des_yaw = mov_dir
                if inv_mov:
                    des_yaw = des_yaw - np.pi*np.sign(mov_dir)

                if des_yaw < 0:
                    if dx<0:
                        des_yaw = -np.pi - des_yaw
                else:
                    if dx<0:
                        des_yaw = np.pi - des_yaw
                
            if DBG: print "Desired Yaw: %f" % des_yaw
            """
            #phi_angle = des_yaw - yaw
            
            phi_angle = mov_dir - yaw
            if dy==0:
                phi_angle = 0

            crosstrack_err = 0
            #Find Path equation Ax + By + C = 0
            A = waypoints[0][1] - waypoints[1][1]
            B = waypoints[1][0] - waypoints[0][0]
            C = waypoints[0][0]*waypoints[1][1] - waypoints[1][0]*waypoints[0][1]

            #Find distance of vehicle ref. pt. to Path
            crosstrack_err = np.abs(A*x + B*y + C)/np.sqrt(A**2 + B**2)

            crosstrack_ctrl = 0
            if v != 0:
                crosstrack_ctrl = np.arctan(self.K_lat*crosstrack_err/v)
                side = (A*x + B*y + C) > 0
                if side:
                    crosstrack_ctrl = -crosstrack_ctrl

            delta = phi_angle + crosstrack_ctrl
            steer_output    = min(max(-self.max_steer, delta), self.max_steer)
            #steer_output = 0.0

            if DBG: print "ACK| Steer OP: %f" % (steer_output)

            #print(yaw, des_yaw, phi_angle, delta, steer_output)
            #print(crosstrack_err, crosstrack_ctrl, phi_angle, steer_output)
            #print(waypoints[1][0]>waypoints[0][0], waypoints[1][1]>waypoints[0][1])
            #print('Path: (', waypoints[0][0], ',', waypoints[0][1], ') => (', waypoints[1][0], ',', waypoints[1][1], ')')

            #DEBUG ONLY########
            #throttle_output = 0.5
            #steer_output = 0.1
            #print(yaw)
            ###################

            ######################################################
            # SET CONTROLS OUTPUT
            ######################################################
            self.set_throttle(throttle_output)  # in percent (0 to 1)
            self.set_steer(steer_output)        # in rad (-1.22 to 1.22)
            self.set_brake(brake_output)        # in percent (0 to 1)

        ######################################################
        ######################################################
        # MODULE 7: STORE OLD VALUES HERE (ADD MORE IF NECESSARY)
        ######################################################
        ######################################################
        """
            Use this block to store old values (for example, we can store the
            current x, y, and yaw values here using persistent variables for use
            in the next iteration)
        """
        self.v_previous = v  # Store forward speed to be used in next step
        self.v_hist[self.v_hist_ptr] = v
        self.v_hist_des[self.v_hist_ptr] = v_desired
        if self.v_hist_ptr == (self.v_hist_len-1):
            self.v_hist_ptr = 0
        else:
            self.v_hist_ptr += 1

