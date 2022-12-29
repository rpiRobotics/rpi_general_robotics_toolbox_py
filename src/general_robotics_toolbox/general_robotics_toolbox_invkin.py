# Copyright (c) 2018, Rensselaer Polytechnic Institute, Wason Technology LLC
# All rights reserved.
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
# 
#     * Redistributions of source code must retain the above copyright
#       notice, this list of conditions and the following disclaimer.
#     * Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in the
#       documentation and/or other materials provided with the distribution.
#     * Neither the name of the Rensselaer Polytechnic Institute, nor Wason 
#       Technology LLC, nor the names of its contributors may be used to 
#       endorse or promote products derived from this software without 
#       specific prior written permission.
# 
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

from __future__ import absolute_import

import numpy as np
from . import general_robotics_toolbox as rox
import warnings

ex = np.array([1,0,0])
ey = np.array([0,1,0])
ez = np.array([0,0,1])

class normalize_joints(object):
    
    """
    Internal use to help with joint limits, multiple revolutions,
    and last joint angles
    """
    
    def __init__(self, robot, last_joints, _ignore_limits = False):
        self._lower_limit = robot.joint_lower_limit
        self._upper_limit = robot.joint_upper_limit
        self._check_limits = self._lower_limit is not None \
            and self._upper_limit is not None and not _ignore_limits
                  
        self._last_joints = last_joints
        self._use_last_joints = last_joints is not None
            
    def normalize(self, joint, theta):
        if self._check_limits:
            l = self._lower_limit[joint]
            u = self._upper_limit[joint]
            
            if not (l < theta and theta < u ):
                a = 2*np.pi*np.array([-1,1])
                b = a + theta
                c = np.argwhere(np.logical_and(l < b,  b < u))
                if len(c) == 0:
                    return None                
                theta += (a[c[0]]).item()
        
        if self._use_last_joints:
            diff = self._last_joints[joint] - theta                               
            n_diff = np.floor_divide(diff, 2*np.pi)
            r_diff = np.remainder(diff, 2*np.pi)
            if (r_diff > np.pi): 
                n_diff+=1
            if np.abs(n_diff) > 0:
                if not self._check_limits:
                    theta += 2*np.pi*n_diff
                else:                
                    theta_v = theta + 2*np.pi*np.arange(n_diff, -np.sign(n_diff), - np.sign(n_diff))
                    theta_ind = np.argwhere(np.logical_and(l < theta_v, theta_v < u))
                    theta = theta_v[theta_ind[0]].item()
                
        return theta
                                                        
                
    
    def __call__(self, joint, theta):
                        
        theta_normed=[]
        if len(np.shape(joint)) == 0:
            for t1 in theta:
                t3=self.normalize(joint, t1)
                if t3 is not None:
                    theta_normed.append(t3)
        else:
            for t1 in theta:
                t3 = tuple([self.normalize(j2,t2) for j2, t2 in zip(joint, t1)])
                if not None in t3:
                    theta_normed.append(t3)                 
        
        if (not self._use_last_joints) or len(theta_normed) < 2:
            return theta_normed
        
        theta_last = np.array(np.take(self._last_joints, joint))
        if len(theta_last.shape) == 0:
            theta_dist = np.abs(np.subtract(theta_normed,theta_last))            
        else:
            theta_dist = np.linalg.norm(np.subtract(theta_normed,theta_last), axis=1)
        
        #Heuristic pruning of last_joints
        theta_ret1 = [t for t in theta_normed if np.all(np.less(np.abs(t - theta_last), np.pi/2.0))]        
        if len(theta_ret1) == 1:
            return theta_ret1
        if len(theta_ret1) == 0:
            theta_ret1 = theta_normed
        
        return [theta_normed[i] for i in list(np.argsort(theta_dist))]
        
  
def robot6_sphericalwrist_invkin(robot, desired_pose, last_joints = None):
    """
    Inverse kinematics for six axis articulated industrial robots
    with spherical wrists. Also referred to as "OPW" robots. Examples include Puma 260, ABB IRB6640, 
    Staubli TX40, etc. Note that this is not for Universal Robot 
    wrist configurations.
    
    :type    robot: general_robotics_toolbox.Robot
    :param   robot: The robot object representing the geometry of the robot
    :type    desired_pose: general_robotics_toolbox.Transform
    :param   desired_pose: The desired pose of the robot
    :type    last_joints: list, tuple, or numpy.array
    :param   last_joints: The joints of the robot at the last timestep. The returned 
             first returned joint configuration will be the closets to last_joints. Optional
    :rtype:  list of numpy.array
    :return: A list of zero or more joint angles that match the desired pose. An
             empty list means that the desired pose cannot be reached. If last_joints
             is specified, the first entry is the closest configuration to last_joints.    
    """
    
    
    desired_pose2 = rox.unapply_robot_aux_transforms(robot, desired_pose)

    R06 = desired_pose2.R
    p0T = desired_pose2.p
    
    H = robot.H
    P = robot.P
    
    theta_v = []
    
    #Correct for spherical joint position vectors
    if not np.all(P[:,4] == 0):
        P4_d = P[:,4].dot(H[:,3])
        assert np.all(P[:,4] - P4_d*H[:,3] == 0)
        P[:,3] += P[:,4]
        P[:,4] = np.zeros(3)
            
    if not np.all(P[:,5] == 0):
        P5_d = P[:,5].dot(H[:,5])
        assert np.all(P[:,5] - P5_d*H[:,5] == 0)
        P[:,6] += P[:,5]
        P[:,5] = np.zeros(3)       
    
    d1 = np.dot(ey, P[:,1] + P[:,2] + P[:,3])
    v1 = p0T - R06.dot(P[:,6])    
    p1 = ey
    
    Q1 = rox.subproblem4(p1, v1, -H[:,0], d1)
    
    normalize = normalize_joints(robot, last_joints)
        
    for q1 in normalize(0, Q1):
                
        R01=rox.rot(H[:,0], q1)
        
        p26_q1 = R01.T.dot(p0T - R06.dot(P[:,6])) - (P[:,0] + P[:,1])
        
        d3 = np.linalg.norm(p26_q1)
        v3 = P[:,2]        
        p3 = P[:,3]
        Q3 = rox.subproblem3(p3, v3, H[:,2], d3)
        
        for q3 in normalize(2,Q3):
                    
            R23=rox.rot(H[:,2],q3)
        
            v2 = p26_q1            
            p2 = P[:,2] + R23.dot(P[:,3])
            q2 = rox.subproblem1(p2, v2, H[:,1])
            
            q2 = normalize(1, [q2])
            if len(q2) == 0:
                continue
            q2 = q2[0]        
            
            R12 = rox.rot(H[:,1], q2)
            
            R03 = R01.dot(R12).dot(R23)
            
            R36 = R03.T.dot(R06)
            
            v4 = R36.dot(H[:,5])            
            p4 = H[:,5]
            
            Q4_Q5 = rox.subproblem2(p4, v4, H[:,3], H[:,4])
            
            for q4, q5 in normalize((3,4), Q4_Q5):
                                
                R35 = rox.rot(H[:,3], q4).dot(rox.rot(H[:,4], q5))
                R05 = R03.dot(R35)
                R56 = R05.T.dot(R06)
                
                p6 = H[:,4]
                v6 = R56.dot(H[:,4])
                
                q6 = rox.subproblem1(p6, v6, H[:,5])
                
                q6 = normalize(5, [q6])
                if len(q6) == 0:
                    continue
                q6 = q6[0]
                
                theta_v.append(np.array([q1, q2, q3, q4, q5, q6]))                        
    if last_joints is not None:
        theta_dist = np.linalg.norm(np.subtract(theta_v,last_joints), axis=1)
        return [theta_v[i] for i in list(np.argsort(theta_dist))]
    else:
        return theta_v

def ur_invkin(robot, desired_pose, last_joints = None):
    """
    Inverse kinematics for Universal Robot articulated industrial 
    robots. Examples include UR3, UR5, UR10
    
    :type    robot: general_robotics_toolbox.Robot
    :param   robot: The robot object representing the geometry of the robot
    :type    desired_pose: general_robotics_toolbox.Transform
    :param   desired_pose: The desired pose of the robot
    :type    last_joints: list, tuple, or numpy.array
    :param   last_joints: The joints of the robot at the last timestep. The returned 
             first returned joint configuration will be the closets to last_joints. Optional
    :rtype:  list of numpy.array
    :return: A list of zero or more joint angles that match the desired pose. An
             empty list means that the desired pose cannot be reached. If last_joints
             is specified, the first entry is the closest configuration to last_joints.    
    """
    
    desired_pose2 = rox.unapply_robot_aux_transforms(robot, desired_pose)

    R06 = desired_pose2.R
    p0T = desired_pose2.p
    
    H = robot.H
    P = robot.P

    assert np.allclose((np.array([ez,-ey,-ey,-ey,-ez,-ey]).T), H) or \
           np.allclose((np.array([ez,ey,ey,ey,-ez,ey]).T), H), \
           "Invalid UR robot format"
    
    theta_v = []

    d1 = np.dot(ey, P[:,1] + P[:,2] + P[:,3] + P[:,4])
    p04 = p0T - np.matmul(R06,P[:,6] + P[:,5])
    v1 = p04
    p1 = ey
    
    Q1 = rox.subproblem4(p1, v1, -H[:,0], d1)

    normalize = normalize_joints(robot, last_joints)

    for q1 in normalize(0, Q1):
                
        R01=rox.rot(H[:,0], q1)
        R16 = np.matmul(R01.transpose(), R06)

        v4 = np.matmul(R16,H[:,5])
        p4 = H[:,5]

        Q24_Q5 = rox.subproblem2(p4, v4, H[:,3], H[:,4])

        for q24, q5 in Q24_Q5:
            R04 = np.matmul(rox.rot(H[:,0],q1),rox.rot(H[:,3],q24))
            
            p23p = p04 - P[:,0] - np.matmul(R01,P[:,1]) - np.matmul(R04,P[:,4])

            d3 = np.linalg.norm(p23p)
            v3 = P[:,2]
            p3 = P[:,3]
            Q3 = rox.subproblem3(p3, v3, H[:,2], d3)

            for q3 in normalize(2,Q3):
                R23 = rox.rot(H[:,2],q3)
                v2 = np.matmul(R01.T,p23p)
                p2 = P[:,2] + np.matmul(R23, P[:,3])
                q2 = rox.subproblem1(p2, v2, H[:,1])
                
                q2 = normalize(1, [q2])
                if len(q2) == 0:
                    continue
                q2 = q2[0]        
                
                q4 = q24 - q2 - q3
                q4 = normalize(3, [q4])
                if len(q4) == 0:
                    continue
                q4 = q4[0]
                R45 = rox.rot(H[:,4], q5)
                R05 = np.matmul(R04,R45)
                R56 = np.matmul(R05.transpose(),R06)

                p6 = H[:,4]
                v6 = np.matmul(R56,H[:,4])
                
                q6 = rox.subproblem1(p6, v6, H[:,5])
                
                q6 = normalize(5, [q6])
                if len(q6) == 0:
                    continue
                q6 = q6[0]
                
                theta_v.append(np.array([q1, q2, q3, q4, q5, q6]))   

    if last_joints is not None:
        theta_dist = np.linalg.norm(np.subtract(theta_v,last_joints), axis=1)
        return [theta_v[i] for i in list(np.argsort(theta_dist))]
    else:
        return theta_v



def equivalent_configurations(robot, theta_v, last_joints = None):
    """
    Return equivalent robot configurations with joints +-360 degree joint offset within
    joint limits
    
    :type    robot: general_robotics_toolbox.Robot
    :param   robot: The robot object representing the geometry of the robot
    :type    theta_v: list
    :param   theta_v: The desired pose of the robot
    :type    last_joints: list, tuple, or numpy.array
    :param   last_joints: The joints of the robot at the last timestep. The returned 
             first returned joint configuration will be the closets to last_joints. Optional
    :rtype:  list of numpy.array
    :return: A list of zero or more additional configurations that are equivalent with +-360 degree
             joint offsets within joint limits.    
    """
    
    equiv_theta = []
    for theta1 in theta_v:
        equiv_theta1 = []
        upper_div, _ = np.divmod(robot.joint_upper_limit - theta1, 2*np.pi)
        lower_div, _ = np.divmod(theta1 - robot.joint_lower_limit, 2*np.pi)

        div_where = np.logical_or(upper_div > 0, lower_div > 0).nonzero()[0]

        for ind in div_where.flatten():
            if robot.joint_type[ind] != 0:
                # Only do this on revolute joints!
                continue 
            equiv_theta1_i = []
            for j in range(int(upper_div[ind])):
                for equiv_theta1_j in equiv_theta1:
                    theta2 = equiv_theta1_j.copy()
                    theta2[ind] += 2.0 * np.pi * (j+1)
                    equiv_theta1_i.append(theta2)    
                theta2 = theta1.copy()
                theta2[ind] += 2.0 * np.pi * (j+1)
                equiv_theta1_i.append(theta2)                        
            for j in range(int(lower_div[ind])):
                for equiv_theta1_j in equiv_theta1:
                    theta2 = equiv_theta1_j.copy()
                    theta2[ind] -= 2.0 * np.pi * (j+1)
                    equiv_theta1_i.append(theta2)
                theta2 = theta1.copy()
                theta2[ind] -= 2.0 * np.pi * (j+1)
                equiv_theta1_i.append(theta2)
            equiv_theta1.extend(equiv_theta1_i)
        equiv_theta.extend(equiv_theta1)

    theta_ret =[]

    for eq_theta in equiv_theta:
        for theta1 in theta_v:
            if np.allclose(theta1, eq_theta):
                continue
        theta_ret.append(eq_theta)
    
    if last_joints is not None:
        theta_dist = np.linalg.norm(np.subtract(theta_ret,last_joints), axis=1)
        return [theta_ret[i] for i in list(np.argsort(theta_dist))]
    else:
        return theta_ret

def iterative_invkin(robot, desired_pose, q_current, max_steps = 200, Kp = 0.3*np.eye(3), KR = 0.3*np.eye(3), tol = 1e-4): # inverse kinematics that uses Least Square solver
    
    """
    Inverse kinematics using gradient descent. Convex solution based on q_current input
    joint positions. Return may be out of range of robot since only produced a local solution!
    
    
    :type    robot: general_robotics_toolbox.Robot
    :param   robot: The robot object representing the geometry of the robot
    :type    desired_pose: general_robotics_toolbox.Transform
    :param   desired_pose: The desired pose of the robot
    :type    q_current: list, tuple, or numpy.array
    :param   q_current: Seed joint position for gradient descent
    :type    max_steps: int
    :param   max_steps: The maximum iterations before failing. Default 200
    :type    Kp: np.array
    :param   Kp: Gain array for position error. Default 0.3
    :type    KR: np.array
    :param   KP: Gain array for rotation error. Default 0.3
    :type    tol: float
    :param   tol: Convergence tolerance
    :rtype:  tuple converged, list of numpy.array
    :return: A tuple with a boolean of convergence, and a list of joint angles. This list will
             typically only contain one entry.    
    """

    # from scipy.optimize import lsq_linear

    # R_d, p_d: Desired orientation and position
    R_d = desired_pose.R
    p_d = desired_pose.p
    
    d_q = q_current

    num_joints = len(robot.joint_type)

    normalize = normalize_joints(robot, q_current, _ignore_limits = True)
    
    q_cur = d_q # initial guess on the current joint angles
    q_cur = q_cur.reshape((num_joints,1)) 
    
    # hist_b = []
    
    itr = 0 # Iterations
    converged = False
    while itr < max_steps and not converged:
        
        pose = rox.fwdkin(robot,q_cur.flatten(),_ignore_limits = True)
        R_cur = pose.R
        p_cur = pose.p
        
        #calculate current Jacobian
        J0T = rox.robotjacobian(robot,q_cur.flatten(),_ignore_limits = True)
        
        # Transform Jacobian to End effector frame from the base frame
        Tr = np.zeros((6,6))
        Tr[:3,:3] = R_cur.T 
        Tr[3:,3:] = R_cur.T
        J0T = np.matmul(Tr, J0T)
        
        ER = np.matmul(np.transpose(R_d),R_cur)
        
        EP = np.matmul(R_cur.T,(p_cur - p_d))
        
        #decompose ER to (k,theta) pair
        k, theta = rox.R2rot(ER)                  
        
        ## set up s for different norm for ER
        # s=2*np.dot(k,np.sin(theta)) #eR1
        # s = np.dot(k,np.sin(theta/2))         #eR2
        s = np.sin(theta/2) * np.array(k)         #eR2
        # s=2*theta*k              #eR3

        vd = np.matmul(Kp, EP)
        wd = np.matmul(KR, s)
        
        b = np.concatenate([wd,vd])
        # np.nan_to_num(b, copy=False, nan=0.0, posinf=None, neginf=None)
        # print(b)
        # print(J0T)
        
        # DEBUG --------------
        # hist_b.append(b)
        # if itr > 0:
        #     error_cur = np.linalg.norm(hist_b[itr-1]) - np.linalg.norm(hist_b[itr])
            #print("Error= " + str(error_cur))
        # DEBUG --------------

        # res = lsq_linear(J0T,b)
        delta = np.matmul(np.linalg.pinv(J0T),b)
                    
        # Convergence Check
        converged = (np.abs(np.hstack((s,EP))) < tol).all()

        if not converged:
            # Update for next iteration
            q_cur = q_cur - delta.reshape((num_joints,1))

        itr += 1 # Increase the iteration


    q_normed = normalize(np.arange(num_joints), [np.squeeze(q_cur)])
    
    return converged, q_normed