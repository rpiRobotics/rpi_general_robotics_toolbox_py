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

import math
import numpy as np
import warnings
import sys

if (sys.version_info > (3, 0)):
    xrange = range

def hat(k):
    """
    Returns a 3 x 3 cross product matrix for a 3 x 1 vector
    
             [  0 -k3  k2]
     khat =  [ k3   0 -k1]
             [-k2  k1   0]
    
    :type    k: numpy.array
    :param   k: 3 x 1 vector
    :rtype:  numpy.array
    :return: the 3 x 3 cross product matrix    
    """
    
    khat=np.zeros((3,3))
    khat[0,1]=-k[2]
    khat[0,2]=k[1]
    khat[1,0]=k[2]
    khat[1,2]=-k[0]
    khat[2,0]=-k[1]
    khat[2,1]=k[0]    
    return khat

def invhat(khat):
    return np.array([(-khat[1,2] + khat[2,1]),(khat[0,2] - khat[2,0]),(-khat[0,1]+khat[1,0])])/2
    

def rot(k, theta):
    """
    Generates a 3 x 3 rotation matrix from a unit 3 x 1 unit vector axis
    and an angle in radians using the Euler-Rodrigues formula
    
        R = I + sin(theta)*hat(k) + (1 - cos(theta))*hat(k)^2
        
    :type    k: numpy.array
    :param   k: 3 x 1 unit vector axis
    :type    theta: number
    :param   theta: rotation about k in radians
    :rtype:  numpy.array
    :return: the 3 x 3 rotation matrix 
        
    """
    I = np.identity(3)
    khat = hat(k)
    khat2 = khat.dot(khat)
    return I + math.sin(theta)*khat + (1.0 - math.cos(theta))*khat2

def R2rot(R):
    """
    Recover k and theta from a 3 x 3 rotation matrix
    
        sin(theta) = | R-R^T |/2
        cos(theta) = (tr(R)-1)/2
        k = invhat(R-R^T)/(2*sin(theta))
        theta = atan2(sin(theta),cos(theta)
        
    :type    R: numpy.array
    :param   R: 3 x 3 rotation matrix    
    :rtype:  (numpy.array, number)
    :return: ( 3 x 1 k unit vector, rotation about k in radians)   
    
    """
    
    R1 = R-R.transpose()
    
    sin_theta = np.linalg.norm(R1)/np.sqrt(8)
    
    cos_theta = (np.trace(R) - 1.0)/2.0
    theta = np.arctan2(sin_theta, cos_theta)
    
    #Avoid numerical singularity
    if sin_theta < 1e-6:
               
        if (cos_theta > 0):
            return [0,0,1], 0
        else:
            B = (1.0/2.0) *(R + np.eye(3))
            k = np.sqrt([B[0,0], B[1,1], B[2,2]])
            if np.abs(k[0]) > 1e-6:
                k[1] = k[1] * np.sign(B[0,1] / k[0])
                k[2] = k[2] * np.sign(B[0,2] / k[0])
            elif np.abs(k[1]) > 1e-6:
                k[2] = k[2] * np.sign(B[0,2] / k[1])
            return k, np.pi
    
    k = invhat(R1)/(2.0*sin_theta)    
    return k, theta

def screw_matrix(r):
    """
    Returns the matrix G representing the 6 x 6 transformation between
    screws in a rigid body, i.e.
 
      (twist)_2 = transpose(G)*(twist)_1
         (twist) = [angular velocity, linear velocity]
      (wrench)_1 = G*(wrench)_2   
         (wrench) = [torque, force]

    :type    r: numpy.array
    :param   r: the 3 x 1 displacement vector from point 1 to point 2
    :rtype:  numpy.array
    :return: the 6 x 6 screw matrix
    """
    
    G = np.identity(6)
    G[0:3,3:6] = hat(r)
    return G

def q2R(q):
    """
    Converts a quaternion into a 3 x 3 rotation matrix according to the
    Euler-Rodrigues formula.
    
    :type    q: numpy.array
    :param   q: 4 x 1 vector representation of a quaternion q = [q0;qv]
    :rtype:  numpy.array
    :return: the 3x3 rotation matrix    
    """
    
    I = np.identity(3)
    qhat = hat(q[1:4])
    qhat2 = qhat.dot(qhat)
    return I + 2*q[0]*qhat + 2*qhat2;

def R2q(R):
    """
    Converts a 3 x 3 rotation matrix into a quaternion.  Quaternion is
    returned in the form q = [q0;qv].
    
    :type    R: numpy.array
    :param   R: 3 x 3 rotation matrix
    :rtype:  numpy.array
    :return: the quaternion as a 4 x 1 vector q = [q0;qv] 
      
    """
    
    tr = np.trace(R)
    if tr > 0:
        S = 2*math.sqrt(tr + 1)
        q = np.array([(0.25*S), \
                      ((R[2,1] - R[1,2]) / S), \
                      ((R[0,2] - R[2,0]) / S), \
                      ((R[1,0] - R[0,1]) / S)])
                      
    elif (R[0,0] > R[1,1] and R[0,0] > R[2,2]):
        S = 2*math.sqrt(1 + R[0,0] - R[1,1] - R[2,2])
        q = np.array([((R[2,1] - R[1,2]) / S), \
                      (0.25*S), \
                      ((R[0,1] + R[1,0]) / S), \
                      ((R[0,2] + R[2,0]) / S)])
    elif (R[1,1] > R[2,2]):
        S = 2*math.sqrt(1 - R[0,0] + R[1,1] - R[2,2])
        q = np.array([((R[0,2] - R[2,0]) / S), \
                      ((R[0,1] + R[1,0]) / S), \
                      (0.25*S), \
                      ((R[1,2] + R[2,1]) / S)])
    else:
        S = 2*math.sqrt(1 - R[0,0] - R[1,1] + R[2,2])
        q = np.array([((R[1,0] - R[0,1]) / S), \
                      ((R[0,2] + R[2,0]) / S), \
                      ((R[1,2] + R[2,1]) / S), \
                      (0.25*S)])
    return q

def q2rot(q):
    """
    Converts a quaternion into k and theta    
    
    :type    q: numpy.array
    :param   q: 4 x 1 vector representation of a quaternion q = [q0;qv]
    :rtype:  (numpy.array, number)
    :return: the 3x1 rotation vector and theta    
    """
    theta = 2 * np.arccos(q[0])
    if np.abs(theta) < 1e-6:
        return np.array([0,0,1]), 0.0
    k = q[1:4]/np.sin(theta/2.0)
    return k, theta

def rot2q(k, theta):
    """
    Converts a 3 x 1 rotation vector and theta into a quaternion.  Quaternion is
    returned in the form q = [q0;qv].
    
    :type    k: numpy.array
    :param   k: 3 x 1 unit vector axis
    :type    theta: number
    :param   theta: rotation about k in radians
    :rtype:  numpy.array
    :return: the quaternion as a 4 x 1 vector q = [q0;qv] 
      
    """
    return np.concatenate(([np.cos(theta/2.0)], k.reshape((3,))*np.sin(theta/2.0)))

def quatcomplement(q):
    """
    Generates the quaternion complement

    in:  q  = [q0;qv];
    out: qc = [q0;-qv];
    
    :type     q: numpy.array
    :param    q: 4 x 1 vector representation of a quaternion q = [q0;qv]
    :rtype:   numpy.array
    :returns: the quaternion complement as a 4 x 1 vector q = [q0;-qv]
    
    """
    return np.array([q[0],-1*q[1],-1*q[2],-1*q[3]])


def quatproduct(q):
    """
    generates matrix representation of a Hamilton quaternion product
    operator
    
    in:  q = [q0;qv];
    out: Q = [q0 -qv'; qv q0*eye(3)+ cross(qv)]
    
    :type     q: numpy.array
    :param    q: 4 x 1 vector representation of a quaternion q = [q0;qv]
    :rtype:   numpy.array
    :returns: the 4 x 4 product matrix
    """
        
    I = np.identity(3)
    Q = np.zeros((4,4))
    Q[0,0] = q[0]
    Q[0,1:4] = -q[1:4]
    Q[1:4,0] = q[1:4]
    Q[1:4,1:4] = q[0]*I+hat(q[1:4])
            
    return Q

    
def quatjacobian(q):
    """
    Returns the 4 x 3 Jacobian matrix relating an angular velocity to the 
    quarternion rate of change
    
    :type     q: numpy.array
    :param    q: 4 x 1 vector representation of a quaternion q = [q0;qv]
    :rtype:   numpy.array
    :returns: the 4 x 3 Jacobian matrix
    """
    
    I = np.identity(3)
    J = np.zeros((4,3))
    J[0,:] = 0.5 * -q[1:4]
    J[1:4,:] = 0.5 * (q[0]*I - hat(q[1:4]))
        
    return J

def rpy2R(rpy):
    return np.matmul(np.matmul(rot([0,0,1],rpy[2]),rot([0,1,0],rpy[1])),rot([1,0,0],rpy[0]))

def R2rpy(R):
    assert np.linalg.norm(R[0:2,0]) > np.finfo(float).eps * 10.0, "Singular rpy requested"
    
    r=np.arctan2(R[2,1],R[2,2])
    y=np.arctan2(R[1,0], R[0,0])
    p=np.arctan2(-R[2,0], np.linalg.norm(R[2,1:3]))
        
    return (r,p,y)    

# Add slerp function using [w,x,y,z] quaternion representation (github copilot)
def slerp(q0, q1, t):
    """
    Spherical linear interpolation between two quaternions
    
    :type     q0: numpy.array
    :param    q0: 4 x 1 vector representation of a quaternion q = [q0;qv]
    :type     q1: numpy.array
    :param    q1: 4 x 1 vector representation of a quaternion q = [q0;qv]
    :type     t: number
    :param    t: interpolation parameter in the range [0,1]
    :rtype:   numpy.array
    :returns: the 4 x 1 interpolated quaternion
    """
    
    assert (t >= 0 and t <= 1), "t must be in the range [0,1]"
    
    q0 = q0/np.linalg.norm(q0)
    q1 = q1/np.linalg.norm(q1)
    
    if (np.dot(q0,q1) < 0):
        q0 = -q0
    
    theta = np.arccos(np.dot(q0,q1))
    
    if (np.abs(theta) < 1e-6):
        return q0
    
    q = (np.sin((1-t)*theta)*q0 + np.sin(t*theta)*q1)/np.sin(theta)
    
    return q/np.linalg.norm(q)

class Robot(object):
    """
    Holds the kinematic information for a single chain robot
    
    :attribute H: A 3 x N matrix containing the direction the joints as unit vectors, one joint per column
    :attribute P: A 3 x (N + 1) matrix containing the distance vector from i to i+1, one vector per column
    :attribute joint_type: A list of N numbers containing the joint type. 0 for rotary, 1 for prismatic, 2 and 3 for mobile
    :attribute joint_lower_limit: A list of N numbers containing the joint lower limits. Optional
    :attribute joint_upper_limit: A list of N numbers containing the joint upper limits. Optional
    :attribute joint_vel_limit: A list of N numbers containing the joint velocity limits. Optional
    :attribute joint_acc_limit: A list of N numbers containing the joint acceleration limits. Optional
    :attribute M: A list of N, 6 x 6 spatial inertia matrices for the links. Optional
    :attribute R_tool: A 3 x 3 rotation matrix for the tool frame. Optional
    :attribute p_tool: A 3 x 1 vector for the tool frame. Optional
    :attribute joint_names: A list of N strings containing the names of the joints if loaded from URDF. Optional
    :attribute root_link_name: A string containing the name of the kinematic chain root link if loaded from URDF. Optional
    :attribute tip_link_name: A string containing the name of the kinematic chain tip link if loaded from URDF. Optional
    :attribute T_flange: Optional transform between end of kinematic chain and the tool frame. This is for compatibility
                        with ROS tool formats.
    :attribute T_base: Optional transform of base of robot in world frame.
    """
    
    
    def __init__(self, H, P, joint_type, joint_lower_limit = None, joint_upper_limit = None, joint_vel_limit = None, joint_acc_limit = None, M = None, \
                 R_tool=None, p_tool=None, joint_names = None, root_link_name = None, tip_link_name = None, T_flange = None, T_base = None):
        
        """
        Construct a Robot object holding the kinematic information for a single chain robot
    
        :type  H: numpy.array
        :param H: A 3 x N matrix containing the direction the joints as unit vectors, one joint per column
        :type  H: numpy. array
        :param P: A 3 x (N + 1) matrix containing the distance vector from i to i+1, one vector per column
        :type  joint_type: list or numpy.array
        :param joint_type: A list or array of N numbers containing the joint type. 0 for rotary, 1 for prismatic, 2 and 3 for mobile
        :type  joint_lower_limit: list or numpy.array
        :param joint_lower_limit: A list or array of N numbers containing the joint type minimums. Optional
        :type  joint_upper_limit: list or numpy.array
        :param joint_upper_limit: A list or array of N numbers containing the joint type maximums. Optional
        :type  joint_vel_limit: list or numpy.array
        :param joint_vel_limit: A list of N numbers containing the joint velocity limits. Optional
        :type  joint_acc_limit: list or numpy.array
        :param joint_acc_limit: A list of N numbers containing the joint acceleration limits. Optional
        :type  M: list of numpy.array
        :param M: A list of N, 6 x 6 spatial inertia matrices for the links. Optional
        :type  R_tool: numpy.array
        :param R_tool: A 3 x 3 rotation matrix for the tool frame. Optional
        :type  p_tool: numpy.array
        :param p_tool: A 3 x 1 vector for the tool frame. Optional
        :type  joint_names: list of string
        :param joint_names: A list of N strings containing the names of the joints if loaded from URDF. Optional
        :type  root_link_name: string
        :param root_link_name: A string containing the name of the kinematic chain root link if loaded from URDF. Optional
        :type  tip_link_name: string
        :param tip_link_name: A string containing the name of the kinematic chain tip link if loaded from URDF. Optional
        :type  T_flange: Transform
        :param T_flange: Optional transform between end of kinematic chain and the tool frame. This is for compatibility
                         with ROS tool formats.
        :type  T_base: Transform
        :param T_base: Optional transform of base of robot in world frame.
    
        """
        
        
        for i in xrange(H.shape[1]):
            assert (np.isclose(np.linalg.norm(H[:,i]), 1))        
        
        for j in joint_type:            
            assert (j in [0,1,2,3])                
        
        assert (H.shape[0] == 3 and P.shape[0] == 3)
        assert (H.shape[1] + 1 == P.shape[1] and H.shape[1] == len(joint_type))
        
        if (joint_lower_limit is not None and joint_upper_limit is not None):
            assert (len(joint_lower_limit) == len(joint_type))
            assert (len(joint_upper_limit) == len(joint_type))
            self.joint_lower_limit=joint_lower_limit
            self.joint_upper_limit=joint_upper_limit
        else:
            self.joint_lower_limit=None
            self.joint_upper_limit=None
            
        if (joint_vel_limit is not None):
            assert (len(joint_vel_limit) == len(joint_type))
            self.joint_vel_limit=joint_vel_limit
        else:
            self.joint_vel_limit=None
            
        if (joint_acc_limit is not None):
            assert (len(joint_acc_limit) == len(joint_type))
            self.joint_acc_limit=joint_acc_limit
        else:
            self.joint_acc_limit=None
               
        if M is not None:
            assert (len(M) == P.shape[1])
            for m in M:
                assert (m.shape == (6,6))
            self.M = M
        else:
            self.M=None
        
        if R_tool is not None and p_tool is not None:
            self.R_tool = R_tool
            self.p_tool = p_tool
        else:
            self.R_tool = None
            self.p_tool = None        
        
        self.H = H
        self.P = P
        self.joint_type = joint_type
        
        if joint_names is not None:
            assert len(joint_names) == len(joint_type)
        self.joint_names = joint_names
        self.root_link_name = root_link_name
        self.tip_link_name = tip_link_name

        self.T_flange = T_flange
        self.T_base = T_base
        
    
            
class Transform(object):
    """
    Holds a transform consisting of a rotation matrix and a vector
    
    :attribute R: The 3 x 3 rotation matrix
    :attribute p: The 3 x 1 position vector
    
    Note: Transform objects are also used to represent the pose of links 
    """
    
    
    def __init__(self, R, p, parent_frame_id=None, child_frame_id=None):
        """
        Construct a Transform object consisting of a rotation matrix and a vector
    
        :type  R: numpy.array
        :param R: The 3 x 3 rotation matrix
        :type  p: numpy.array
        :param p: The 3 x 1 position vector
        """    
                
        assert (np.shape(R) == (3,3))
        assert (np.shape(p) == (3,) or np.shape(p) ==(3,1))
        
        self.R=np.array(R)
        self.p=np.reshape(p,(3,))
        self.parent_frame_id=parent_frame_id
        self.child_frame_id=child_frame_id
        
    def __mul__(self, other):
        R = np.matmul(self.R, other.R)
        p = self.p + np.matmul(self.R, other.p)
        return Transform(R,p,self.parent_frame_id, other.child_frame_id)
    
    def __eq__(self, other):
        #Use "np.isclose" because of numerical accuracy issues
        return np.all(np.isclose(self.R, other.R, 1e-6)) \
            and np.all(np.isclose(self.p, other.p, 1e-6))
            
    def __neq__(self, other):
        return not self.__eq__(other)
    
    def inv(self):
        R=np.transpose(self.R)
        p=-np.matmul(R,self.p)
        return Transform(R,p,self.child_frame_id, self.parent_frame_id)
    
    def __repr__(self):
        r = ["Transform(", \
            "    R = " + np.array_repr(self.R, precision=4, suppress_small=True).replace('\n', '\n' + ' '*8), \
            "    p = " + np.array_repr(self.p, precision=4, suppress_small=True)]
        if self.parent_frame_id is not None:
            r.append("    parent_frame_id = \"" + self.parent_frame_id + "\"")
        if self.child_frame_id is not None:
            r.append("    child_frame_id = \"" + self.child_frame_id + "\"")
        r.append(")\n")        
        return "\n".join(r)
    
    def __str__(self):
        r = ["R = " + np.array_str(self.R, precision=4, suppress_small=True).replace('\n', '\n' + ' '*4), \
          "p = " + np.array_str(self.p, precision=4, suppress_small=True)]
        if self.parent_frame_id is not None:
            r.append("parent_frame_id = \"" + self.parent_frame_id + "\"")
        if self.child_frame_id is not None:
            r.append("child_frame_id = \"" + self.child_frame_id + "\"")
        r.append("\n")        
        return "\n".join(r)

    def isclose(self, other, tol=1e-6):
        #Use "np.isclose" because of numerical accuracy issues
        return np.all(np.isclose(self.R, other.R, atol=tol)) \
            and np.all(np.isclose(self.p, other.p, atol=tol))
    
def fwdkin(robot, theta, _ignore_limits = False):
    """
    Computes the pose of the robot tool flange based on a Robot object
    and the joint angles.
    
    :type    robot: Robot
    :param   robot: The robot object containing kinematic information
    :type    theta: numpy.array
    :param   theta: N x 1 array of joint angles. Must have same number of joints as Robot object
    :rtype:  Transform
    :return: The Pose of the robot tool flange    
    """    
    
    if not _ignore_limits:

        if (robot.joint_lower_limit is not None and robot.joint_upper_limit is not None):
            assert np.greater_equal(theta, robot.joint_lower_limit).all(), "Specified joints out of range"
            assert np.less_equal(theta, robot.joint_upper_limit).all(), "Specified joints out of range"
    
    p = robot.P[:,[0]]
    R = np.identity(3)
    for i in xrange(0,len(robot.joint_type)):
        if (robot.joint_type[i] == 0 or robot.joint_type[i] == 2):
            R = np.matmul(R,rot(robot.H[:,[i]],theta[i]))
        elif (robot.joint_type[i] == 1 or robot.joint_type[i] == 3):
            p = p + theta[i] * np.matmul(R,robot.H[:,[i]])
        p = p + np.matmul(R,robot.P[:,[i+1]])
        
    p=np.reshape(p,(3,))
        
    return apply_robot_aux_transforms(robot,Transform(R, p))

    
def robotjacobian(robot, theta, _ignore_limits = False):
    """
    Computes the Jacobian matrix for the robot tool flange based on a Robot object
    and the joint angles.
    
    :type     robot: Robot
    :param    robot: The robot object containing kinematic information
    :type     theta: numpy.array
    :param    theta: N x 1 array of joint angles in radians or meters as appropriate. Must have same number of joints as Robot object.
    :rtype:   numpy.array
    :returns: The 6 x N Jacobian matrix    
    """
    
    if not _ignore_limits:
        if (robot.joint_lower_limit is not None and robot.joint_upper_limit is not None):
            assert np.greater_equal(theta, robot.joint_lower_limit).all(), "Specified joints out of range"
            assert np.less_equal(theta, robot.joint_upper_limit).all(), "Specified joints out of range"
    
    
    hi = np.zeros(robot.H.shape)
    pOi = np.zeros(robot.P.shape)
    
    p = robot.P[:,[0]]
    R = np.identity(3)
    
    pOi[:,[0]] = p
    
    H = robot.H
    P = robot.P
    joint_type = robot.joint_type
    
    for i in xrange(0, len(joint_type)):
        if (joint_type[i] == 0 or joint_type[i] == 2):
            R = np.matmul(R,rot(H[:,[i]],theta[i]))
        elif (joint_type[i] == 1 or joint_type[i] == 3):
            p = p + theta[i] * np.matmul(R,H[:,[i]])
        p = p + np.matmul(R,P[:,[i+1]])
        pOi[:,[i+1]] = p
        hi[:,[i]] = np.matmul(R,H[:,[i]])
    
    pOT = pOi[:,[len(joint_type)]]
    
    R_flange = robot.T_flange.R if (robot.T_flange is not None) else np.eye(3)

    if robot.T_flange is not None:
        pOT += np.matmul(R,np.reshape(robot.T_flange.p,(3,1)))
    if robot.p_tool is not None:
        pOT += np.matmul(np.matmul(R,R_flange),np.reshape(robot.p_tool,(3,1)))
    
    J = np.zeros([6,len(joint_type)])
    i = 0
    j = 0
    while (i < len(joint_type)):
        if (joint_type[i] == 0):
            J[0:3,[j]] = hi[:,[i]]
            J[3:6,[j]] = np.matmul(hat(hi[:,[i]]),(pOT - pOi[:,[i]]))
        elif (joint_type[i] == 1):
            J[3:6,[j]] = hi[:,[i]]
        elif (joint_type[i] == 3):
            J[3:6,[j]] = np.matmul(rot(hi[:,[i+2]], theta[i+2]),(hi[:,[i]]))
            J[0:3,[j+1]] = hi[:,[i+2]]
            J[3:6,[j+1]] = np.matmul(hat(hi[:,[i+2]]),(pOT - pOi[:,[i+2]]))
            J = J[:,0:-1]
            i = i + 2
            j = j + 1
        
        i = i + 1
        j = j + 1

    if not robot.T_base:
        return J
    else:
        R_J = np.block([[robot.T_base.R, np.zeros((3,3))],[np.zeros((3,3)), robot.T_base.R]])
        return np.matmul(R_J,J)


def subproblem0(p, q, k):
    """
    Solves canonical geometric subproblem 0, theta subtended between p and q according to
    
        q = rot(k, theta)*p
           ** assumes k'*p = 0 and k'*q = 0
           
    Requires that p and q are perpendicular to k. Use subproblem 1 if this is not
    guaranteed.

    :type    p: numpy.array
    :param   p: 3 x 1 vector before rotation
    :type    q: numpy.array
    :param   q: 3 x 1 vector after rotation
    :type    k: numpy.array
    :param   k: 3 x 1 rotation axis unit vector
    :rtype:  number
    :return: theta angle as scalar in radians
    """
    
    eps = np.finfo(np.float64).eps    
    assert (np.dot(k,p) < eps) and (np.dot(k,q) < eps), \
           "k must be perpendicular to p and q"
    
    norm = np.linalg.norm
    
    ep = p / norm(p)
    eq = q / norm(q)
    
    theta = 2 * np.arctan2( norm(ep - eq), norm (ep + eq))
    
    if (np.dot(k,np.cross(p , q)) < 0):
        return -theta
        
    return theta

def subproblem1(p, q, k):
    """
    Solves canonical geometric subproblem 1, theta subtended between p and q according to
    
        q = rot(k, theta)*p
    
    :type    p: numpy.array
    :param   p: 3 x 1 vector before rotation
    :type    q: numpy.array
    :param   q: 3 x 1 vector after rotation
    :type    k: numpy.array
    :param   k: 3 x 1 rotation axis unit vector
    :rtype:  number
    :return: theta angle as scalar in radians
    """
    
    eps = np.finfo(np.float64).eps
    norm = np.linalg.norm
    
    if norm (np.subtract(p, q)) < np.sqrt(eps):
        return 0.0
    
    
    k = np.divide(k,norm(k))
    
    pp = np.subtract(p,np.dot(p, k)*k)
    qp = np.subtract(q,np.dot(q, k)*k)
    
    epp = np.divide(pp, norm(pp))    
    eqp = np.divide(qp, norm(qp))
    
    theta = subproblem0(epp, eqp, k)
    
    if (np.abs(norm(p) - norm(q)) > norm(p)*1e-2):
        warnings.warn("||p|| and ||q|| must be the same!!!")
    
    return theta


def subproblem2(p, q, k1, k2):
    """
    Solves canonical geometric subproblem 2, solve for two coincident, nonparallel
    axes rotation a link according to
    
        q = rot(k1, theta1) * rot(k2, theta2) * p
    
    solves by looking for the intersection between cones of
    
        rot(k1,-theta1)q = rot(k2, theta2) * p
        
    may have 0, 1, or 2 solutions
       
    
    :type    p: numpy.array
    :param   p: 3 x 1 vector before rotations
    :type    q: numpy.array
    :param   q: 3 x 1 vector after rotations
    :type    k1: numpy.array
    :param   k1: 3 x 1 rotation axis 1 unit vector
    :type    k2: numpy.array
    :param   k2: 3 x 1 rotation axis 2 unit vector
    :rtype:  list of number pairs
    :return: theta angles as list of number pairs in radians
    """
    
    eps = np.finfo(np.float64).eps
    norm = np.linalg.norm
    
    k12 = np.dot(k1, k2)
    pk = np.dot(p, k2)
    qk = np.dot(q, k1)
    
    # check if solution exists
    if (np.abs( 1 - k12**2) < eps):
        warnings.warn("No solution - k1 != k2")
        return []
    
    a = np.matmul([[k12, -1], [-1, k12]],[pk, qk]) / (k12**2 - 1)
    
    bb = (np.dot(p,p) - np.dot(a,a) - 2*a[0]*a[1]*k12)
    if (np.abs(bb) < eps): bb=0
    
    if (bb < 0):
        warnings.warn("No solution - no intersection found between cones")
        return []
    
    gamma = np.sqrt(bb) / norm(np.cross(k1,k2))
    if (np.abs(gamma) < eps):
        cm=np.array([k1, k2, np.cross(k1,k2)]).T
        c1 = np.dot(cm, np.hstack((a, gamma)))
        theta2 = subproblem1(k2, p, c1)
        theta1 = -subproblem1(k1, q, c1)
        return [(theta1, theta2)]
    
    cm=np.array([k1, k2, np.cross(k1,k2)]).T
    c1 = np.dot(cm, np.hstack((a, gamma)))
    c2 = np.dot(cm, np.hstack((a, -gamma)))
    theta1_1 = -subproblem1(q, c1, k1)
    theta1_2 = -subproblem1(q, c2, k1)
    theta2_1 =  subproblem1(p, c1, k2)
    theta2_2 =  subproblem1(p, c2, k2)
    return [(theta1_1, theta2_1), (theta1_2, theta2_2)]

def subproblem3(p, q, k, d):
    
    """
    Solves canonical geometric subproblem 3,solve for theta in
    an elbow joint according to
    
        || q + rot(k, theta)*p || = d
        
    may have 0, 1, or 2 solutions
    
    :type    p: numpy.array
    :param   p: 3 x 1 position vector of point p
    :type    q: numpy.array
    :param   q: 3 x 1 position vector of point q
    :type    k: numpy.array
    :param   k: 3 x 1 rotation axis for point p
    :type    d: number
    :param   d: desired distance between p and q after rotation
    :rtype:  list of numbers
    :return: list of valid theta angles in radians        
    
    """
    
    norm=np.linalg.norm
    
    pp = np.subtract(p,np.dot(np.dot(p, k),k))
    qp = np.subtract(q,np.dot(np.dot(q, k),k))
    dpsq = d**2 - ((np.dot(k, np.add(p,q)))**2)
    
    bb=-(np.dot(pp,pp) + np.dot(qp,qp) - dpsq)/(2*norm(pp)*norm(qp))
    
    if dpsq < 0 or np.abs(bb) > 1:
        warnings.warn("No solution - no rotation can achieve specified distance")
        return []
    
    theta = subproblem1(pp/norm(pp), qp/norm(qp), k)
    
    phi = np.arccos(bb)
    if np.abs(phi) > 0:
        return [theta + phi, theta - phi]
    else:
        return [theta]
    
def subproblem4(p, q, k, d):
    """
    Solves canonical geometric subproblem 4, theta for static
    displacement from rotation axis according to
    
        d = p.T*rot(k, theta)*q
        
    may have 0, 1, or 2 solutions
        
    :type    p: numpy.array
    :param   p: 3 x 1 position vector of point p
    :type    q: numpy.array
    :param   q: 3 x 1 position vector of point q
    :type    k: numpy.array
    :param   k: 3x1 rotation axis for point p
    :type    d: number
    :param   d: desired displacement
    :rtype:  list of numbers
    :return: list of valid theta angles in radians    
    """
        
    a = np.matmul(np.matmul(p,hat(k)),q)
    b = -np.matmul(p, np.matmul(hat(k),hat(k).dot(q)))
    c = np.subtract(d, (np.dot(p,q) -b))
    
    phi = np.arctan2(b, a)
    
    d = c / np.linalg.norm([a,b])
    
    if d > 1:
        return []
    
    psi = np.arcsin(d)
    
    return [-phi+psi, -phi-psi+np.pi]


def apply_robot_aux_transforms(robot, T):
    """
    Apply R_tool,p_tool, T_flange, and T_base to T

    :type  robot: Robot
    :param robot: Robot instance with aux transforms
    :type  T: Transform
    :param T: Transform to apply aux transforms
    :rtype: Transform
    :return: Returns transform with applied aux transforms
    """

    T_ret = T
    if robot.T_flange is not None:
        T_ret = T_ret * robot.T_flange
    if robot.R_tool is not None and robot.p_tool is not None:
        T_ret = T_ret * Transform(robot.R_tool, robot.p_tool)
    if robot.T_base is not None:
        T_ret = robot.T_base * T_ret

    return T_ret

def unapply_robot_aux_transforms(robot, T):
    """
    Unapply R_tool,p_tool, T_flange, and T_base to T

    :type  robot: Robot
    :param robot: Robot instance with aux transforms
    :type  T: Transform
    :param T: Transform to unapply aux transforms
    :rtype: Transform
    :return: Returns transform with unapplied aux transforms
    """

    T_ret = T
    if robot.R_tool is not None and robot.p_tool is not None:
        T_ret = T_ret * Transform(robot.R_tool, robot.p_tool).inv()
    if robot.T_flange is not None:
        T_ret = T_ret * robot.T_flange.inv()
    if robot.T_base is not None:
        T_ret = robot.T_base.inv() * T_ret

    return T_ret

def identity_transform():
    """Returns an identity transform"""
    return Transform(np.eye(3,dtype=np.float64),np.zeros((3,),dtype=np.float64))

def random_R():
    q=np.random.rand(4)
    q=q/np.linalg.norm(q)
    return q2R(q)

def random_p():
    return np.random.rand(3)

def random_transform():
    return Transform(random_R(), random_p())


    