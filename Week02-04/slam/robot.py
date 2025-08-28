import numpy as np

class Robot:
    EPS = 1*np.exp(-6)
    def __init__(self, wheels_width, wheels_scale, camera_matrix, camera_dist):

        self.cam_offset = np.zeros((2,1))  #EDIT MADE
        self.cam_yaw = 0.0  #EDIT MADE

        # State is a vector of [x,y,theta]'
        self.state = np.zeros((3,1))
        
        
        # Wheel parameters
        self.wheels_width = wheels_width  # The distance between the left and right wheels
        self.wheels_scale = wheels_scale  # The scaling factor converting ticks/s to m/s

        # Camera parameters
        self.camera_matrix = camera_matrix  # Matrix of the focal lengths and camera centre
        self.camera_dist = camera_dist  # Distortion coefficients
    
    def drive(self, drive_meas):
        # left_speed and right_speed are the speeds in ticks/s of the left and right wheels.
        # dt is the length of time to drive for

        # Compute the linear and angular velocity
        linear_velocity, angular_velocity = self.convert_wheel_speeds(drive_meas.left_speed, drive_meas.right_speed)

        # Apply the velocities
        dt = drive_meas.dt
        if angular_velocity == 0:
            self.state[0] += np.cos(self.state[2]) * linear_velocity * dt
            self.state[1] += np.sin(self.state[2]) * linear_velocity * dt
        else:
            th = self.state[2]
            self.state[0] += linear_velocity / angular_velocity * (np.sin(th+dt*angular_velocity) - np.sin(th))
            self.state[1] += -linear_velocity / angular_velocity * (np.cos(th+dt*angular_velocity) - np.cos(th))
            self.state[2] += dt*angular_velocity

    def measure(self, markers, idx_list):
        # Markers are 2d landmarks in a 2xn structure where there are n landmarks.
        # The index list tells the function which landmarks to measure in order.
        
        # Construct a 2x2 rotation matrix from the robot angle
        th = self.state[2]
        Rot_theta = np.block([[np.cos(th), -np.sin(th)],[np.sin(th), np.cos(th)]])
        #robot_xy = self.state[0:2,:] #EDIT MADE (commented out)
        R_bc = np.array([[np.cos(self.cam_yaw), -np.sin(self.cam_yaw)],[np.sin(self.cam_yaw), np.cos(self.cam_yaw)]])  #EDIT MADE (Added)
        t_bc = self.cam_offset.reshape(2,1)  #EDIT MADE (Added)
        robot_xy = self.state[0:2,:] + Rot_theta @ t_bc  #EDIT MADE (Changed)

        # measurements = [] #EDIT MADE (commented out)
        # for idx in idx_list:
        #     marker = markers[:,idx:idx+1]
        #     marker_bff = Rot_theta.T @ (marker - robot_xy)
        #     measurements.append(marker_bff)

        measurements = [] #EDIT MADE (Changed)
        for idx in idx_list:
            p_w = markers[:, idx:idx+1]                             # landmark in world
            # p_c = R_bc^T * ( R_theta^T (p_w - t) - t_bc )
            p_c = R_bc.T @ (Rot_theta.T @ (p_w - robot_xy) - t_bc)
            measurements.append(p_c)


        # Stack the measurements in a 2xm structure.
        markers_bff = np.concatenate(measurements, axis=1)
        return markers_bff
    
    def convert_wheel_speeds(self, left_speed, right_speed):
        # Convert to m/s
        left_speed_m = left_speed * self.wheels_scale
        right_speed_m = right_speed * self.wheels_scale

        # Compute the linear and angular velocity
        linear_velocity = (left_speed_m + right_speed_m) / 2.0
        angular_velocity = (right_speed_m - left_speed_m) / self.wheels_width
        
        return linear_velocity, angular_velocity

    # Derivatives and Covariance
    # --------------------------

    def derivative_drive(self, drive_meas):
        # Compute the differential of drive w.r.t. the robot state
        DFx = np.zeros((3,3))
        DFx[0,0] = 1
        DFx[1,1] = 1
        DFx[2,2] = 1

        lin_vel, ang_vel = self.convert_wheel_speeds(drive_meas.left_speed, drive_meas.right_speed)

        dt = drive_meas.dt
        th = self.state[2]

        #edits: 
        if abs(ang_vel) < 1e-12:  # Straight line
            #DFx[0, 2] = -lin_vel * np.sin(th) * dt
            #DFx[1, 2] =  lin_vel * np.cos(th) * dt
            DFx[0,2]=lin_vel*np.cos(th)*dt
            DFx[1,2]=lin_vel*np.sin(th)*dt
        else: #turning
            th2 = th + ang_vel * dt
            
            #DFx[0, 2] = (lin_vel/ang_vel) * (np.cos(th2) - np.cos(th))
            #DFx[1, 2] = (lin_vel/ang_vel) * (np.sin(th2) - np.sin(th))
            DFx[0, 2]=(lin_vel/ang_vel) *(-np.sin(th)+np.sin(th2))
            DFx[1,2]=(lin_vel/ang_vel) *(np.cos(th)-np.cos(th2))

        return DFx

    def derivative_measure(self, markers, idx_list):
        # Compute the derivative of the markers in the order given by idx_list w.r.t. robot and markers
        n = 2*len(idx_list)
        m = 3 + 2*markers.shape[1]

        DH = np.zeros((n,m))

        robot_xy = self.state[0:2,:]
        th = self.state[2]        
        Rot_theta = np.block([[np.cos(th), -np.sin(th)],[np.sin(th), np.cos(th)]])
        DRot_theta = np.block([[-np.sin(th), -np.cos(th)],[np.cos(th), -np.sin(th)]])

        R_bc = np.array([[np.cos(self.cam_yaw), -np.sin(self.cam_yaw)], #EDIT MADE
                     [np.sin(self.cam_yaw),  np.cos(self.cam_yaw)]])
        A_factor = R_bc.T @ Rot_theta.T #EDIT MADE

        for i in range(n//2):
            j = idx_list[i]
            # i identifies which measurement to differentiate.
            # j identifies the marker that i corresponds to.
            lmj_w = markers[:,j:j+1]

            lmj_inertial = markers[:,j:j+1]
            # lmj_bff = Rot_theta.T @ (lmj_inertial - robot_xy)

            # robot xy DH
            DH[2*i:2*i+2,0:2] = -A_factor  #- Rot_theta.T #EDIT MADE (changed)
            # robot theta DH
            # DH[2*i:2*i+2, 2:3] = DRot_theta.T @ (lmj_inertial - robot_xy)
            DH[2*i:2*i+2, 2:3] = R_bc.T @ (DRot_theta.T @ (lmj_w - robot_xy)) #EDIT MADE (changed)

            
            # lm xy DH
            DH[2*i:2*i+2, 3+2*j:3+2*j+2] = A_factor #Rot_theta.T #EDIT MADE (changed)

            # print(DH[i:i+2,:])

        return DH
    
    def covariance_drive(self, drive_meas):
        # Derivative of lin_vel, ang_vel w.r.t. left_speed, right_speed
        Jac1 = np.array([[self.wheels_scale/2, self.wheels_scale/2],
                [-self.wheels_scale/self.wheels_width, self.wheels_scale/self.wheels_width]])
        
        lin_vel, ang_vel = self.convert_wheel_speeds(drive_meas.left_speed, drive_meas.right_speed)
        th = self.state[2]
        dt = drive_meas.dt
        th2 = th + dt*ang_vel

        # Derivative of x,y,theta w.r.t. lin_vel, ang_vel
        Jac2 = np.zeros((3,2))
        
        # TODO: add your codes here to compute Jac2 using lin_vel, ang_vel, dt, th, and th2
        eps = 1e-6
        if abs(ang_vel) < eps: #if robot is not turning
            Jac2[0, 0] = dt * np.cos(th)                 
            Jac2[1, 0] = dt * np.sin(th)                 

            Jac2[0, 1] = -0.5 * lin_vel * dt**2 * np.sin(th)  
            Jac2[1, 1] =  0.5 * lin_vel * dt**2 * np.cos(th)   
            Jac2[2, 1] = dt                              
        else:
            sin_d = np.sin(th2) - np.sin(th)
            cos_d = -np.cos(th2) + np.cos(th)

            Jac2[0, 0] = sin_d / ang_vel                       
            Jac2[1, 0] = cos_d /ang_vel                       
            Jac2[2, 0] = 0.0                            

            Jac2[0, 1] = lin_vel * ((dt * ang_vel * np.cos(th2) - sin_d) / (ang_vel**2))  
            Jac2[1, 1] = lin_vel * ((dt * ang_vel * np.sin(th2) - cos_d) / (ang_vel**2)) 
            Jac2[2, 1] = dt                                            

        # Derivative of x,y,theta w.r.t. left_speed, right_speed
        Jac = Jac2 @ Jac1

        # Compute covariance
        cov = np.diag((drive_meas.left_cov, drive_meas.right_cov))
        cov = Jac @ cov @ Jac.T
        
        return cov

# import numpy as np

# class Robot:

#     EPS = 1e-6

#     def _init_(self, wheels_width, wheels_scale, camera_matrix, camera_dist):
#         # State is a vector of [x,y,theta]'
#         self.state = np.zeros((3,1))
        
#         # Wheel parameters
#         self.wheels_width = wheels_width  # The distance between the left and right wheels
#         self.wheels_scale = wheels_scale  # The scaling factor converting ticks/s to m/s

#         # Camera parameters
#         self.camera_matrix = camera_matrix  # Matrix of the focal lengths and camera centre
#         self.camera_dist = camera_dist  # Distortion coefficients
    
#     def drive(self, drive_meas):
#         # left_speed and right_speed are the speeds in ticks/s of the left and right wheels.
#         # dt is the length of time to drive for

#         v, w = self.convert_wheel_speeds(drive_meas.left_speed, drive_meas.right_speed)
#         dt = drive_meas.dt
#         th = float(self.state[2])

#         if abs(w) < self.EPS:
#             # straight-line limit
#             self.state[0] += v * dt * np.cos(th)
#             self.state[1] += v * dt * np.sin(th)
#             # theta unchanged
#         else:
#             th2 = th + w*dt
#             self.state[0] += (v/w) * (np.sin(th2) - np.sin(th))
#             self.state[1] += (-v/w) * (np.cos(th2) - np.cos(th))
#             self.state[2] += w*dt

#     def measure(self, markers, idx_list):
#         # Markers are 2d landmarks in a 2xn structure where there are n landmarks.
#         # The index list tells the function which landmarks to measure in order.
        
#         # Construct a 2x2 rotation matrix from the robot angle
#         th = self.state[2]
#         Rot_theta = np.block([[np.cos(th), -np.sin(th)],[np.sin(th), np.cos(th)]])
#         robot_xy = self.state[0:2,:]

#         measurements = []
#         for idx in idx_list:
#             marker = markers[:,idx:idx+1]
#             marker_bff = Rot_theta.T @ (marker - robot_xy)
#             measurements.append(marker_bff)

#         # Stack the measurements in a 2xm structure.
#         markers_bff = np.concatenate(measurements, axis=1)
#         return markers_bff
    
#     def convert_wheel_speeds(self, left_speed, right_speed):
#         # Convert to m/s
#         left_speed_m = left_speed * self.wheels_scale
#         right_speed_m = right_speed * self.wheels_scale

#         # Compute the linear and angular velocity
#         linear_velocity = (left_speed_m + right_speed_m) / 2.0
#         angular_velocity = (right_speed_m - left_speed_m) / self.wheels_width
        
#         return linear_velocity, angular_velocity

#     # Derivatives and Covariance
#     # --------------------------

#     def derivative_drive(self, drive_meas):
#         # Compute the differential of drive w.r.t. the robot state
#         DFx = np.zeros((3,3))
#         DFx[0,0] = 1
#         DFx[1,1] = 1
#         DFx[2,2] = 1

#         lin_vel, ang_vel = self.convert_wheel_speeds(drive_meas.left_speed, drive_meas.right_speed)

#         dt = drive_meas.dt
#         th = float(self.state[2])

#         #edits: 
#         if abs(ang_vel) < 1e-12:  # Straight line
#             #DFx[0, 2] = -lin_vel * np.sin(th) * dt
#             #DFx[1, 2] =  lin_vel * np.cos(th) * dt
#             DFx[0,2]=lin_vel*np.cos(th)*dt
#             DFx[1,2]=lin_vel*np.sin(th)*dt
#         else: #turning
#             th2 = th + ang_vel * dt
            
#             #DFx[0, 2] = (lin_vel/ang_vel) * (np.cos(th2) - np.cos(th))
#             #DFx[1, 2] = (lin_vel/ang_vel) * (np.sin(th2) - np.sin(th))
#             DFx[0, 2]=(lin_vel/ang_vel) *(-np.sin(th)+np.sin(th2))
#             DFx[1,2]=(lin_vel/ang_vel) *(np.cos(th)-np.cos(th2))

#         return DFx

#     def derivative_measure(self, markers, idx_list):
#         # Compute the derivative of the markers in the order given by idx_list w.r.t. robot and markers
#         n = 2*len(idx_list)
#         m = 3 + 2*markers.shape[1]

#         DH = np.zeros((n,m))

#         robot_xy = self.state[0:2,:]
#         th = self.state[2]        
#         Rot_theta = np.block([[np.cos(th), -np.sin(th)],[np.sin(th), np.cos(th)]])
#         DRot_theta = np.block([[-np.sin(th), -np.cos(th)],[np.cos(th), -np.sin(th)]])

#         for i in range(n//2):
#             j = idx_list[i]
#             # i identifies which measurement to differentiate.
#             # j identifies the marker that i corresponds to.

#             lmj_inertial = markers[:,j:j+1]
#             # lmj_bff = Rot_theta.T @ (lmj_inertial - robot_xy)

#             # robot xy DH
#             DH[2*i:2*i+2,0:2] = - Rot_theta.T
#             # robot theta DH
#             DH[2*i:2*i+2, 2:3] = DRot_theta.T @ (lmj_inertial - robot_xy)
#             # lm xy DH
#             DH[2*i:2*i+2, 3+2*j:3+2*j+2] = Rot_theta.T

#             # print(DH[i:i+2,:])

#         return DH
    
    # def covariance_drive(self, drive_meas):
    #     """
    #     Propagate wheel-speed noise -> [x,y,theta].
    #     Jac1 = ∂[v, w]/∂[l, r]
    #     Jac2 = ∂f/∂[v, w]
    #     """
    #     # ∂[v,w]/∂[l,r]
    #     Jac1 = np.array([
    #         [ self.wheels_scale/2.0,  self.wheels_scale/2.0],
    #         [-self.wheels_scale/self.wheels_width, self.wheels_scale/self.wheels_width]
    #     ])

    #     v, w = self.convert_wheel_speeds(drive_meas.left_speed, drive_meas.right_speed)
    #     dt = drive_meas.dt
    #     th = float(self.state[2])
    #     th2 = th + w*dt

    #     # ∂[x,y,th]/∂[v,w]
    #     Jac2 = np.zeros((3, 2))
    #     if abs(w) < self.EPS:
    #         # series expansion around w=0
    #         Jac2[0, 0] = dt * np.cos(th)
    #         Jac2[1, 0] = dt * np.sin(th)
    #         Jac2[0, 1] = -0.5 * v * dt * dt * np.sin(th)
    #         Jac2[1, 1] =  0.5 * v * dt * dt * np.cos(th)
    #         Jac2[2, 1] = dt
    #     else:
    #         A = np.sin(th2) - np.sin(th)
    #         B = np.cos(th2) - np.cos(th)

    #         Jac2[0, 0] =  (1.0/w) * A
    #         Jac2[0, 1] =  v * ( -A/(w*w) + (dt*np.cos(th2))/w )
    #         Jac2[1, 0] = -(1.0/w) * B
            
    #         Jac2[1, 1] =  v * (  B/(w*w) + (dt*np.sin(th2))/w )
    #         Jac2[2, 0] = 0.0
    #         Jac2[2, 1] = dt

    #     Jac = Jac2 @ Jac1

    #     # Wheel speed covariance (left/right); add a small floor to avoid degeneracy
    #     cov_lr = np.diag([drive_meas.left_cov, drive_meas.right_cov])
    #     cov = Jac @ cov_lr @ Jac.T

    #     cov += np.diag([1e-8, 1e-8, 1e-10])  # tiny process floor
    #     return cov