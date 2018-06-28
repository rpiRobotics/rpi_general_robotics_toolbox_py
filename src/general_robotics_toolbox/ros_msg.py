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

from geometry_msgs.msg import Quaternion, Point, Vector3, Pose, Twist, Wrench
from . import general_robotics_toolbox as rox
import numpy as np

"""Provides convenience functions for converting between general_robotics_toolbox
and ROS geometry_msgs types"""

def msg2q(ros_quaternion):
    """
    Converts a geometry_msgs/Quaternion message into a 4x1 quaternion vector
    
    :type ros_quaternion: geometry_msgs.msg.Quaternion
    :param ros_quaternion: ROS Quaternion message
    :rtype: numpy.array
    :return The 4x1 quaternion matrix
    """
    return np.array([ros_quaternion.w, ros_quaternion.x, ros_quaternion.y, ros_quaternion.z])

def q2msg(q):
    """
    Converts a 4x1 quaternion vector into a geometry_msgs/Quaternion message
    
    :type q: numpy.array
    :param q: 4x1 quaternion matrix
    :rtype: geometry_msgs.msg.Quaternion
    :return The ROS Quaternion message
    """
    q2=np.reshape(q, (4,))
    return Quaternion(q2[1], q2[2], q2[3], q2[0])

def msg2R(ros_quaternion):
    """
    Converts a geometry_msgs/Quaternion message into a 3x3 rotation matrix
    
    :type ros_quaternion: geometry_msgs.msg.Quaternion
    :param ros_quaternion: ROS Quaternion message
    :rtype: numpy.array
    :return The 3x3 rotation matrix
    """
    return rox.q2R(msg2q(ros_quaternion))

def R2msg(R):
    """
    Converts a 3x3 rotation matrix into a geometry_msgs/Quaternion message
    
    :type R: numpy.array
    :param R: 3x3 rotation matrix
    :rtype: geometry_msgs.msg.Quaternion
    :return The ROS Quaternion message
    """
    return q2msg(rox.R2q(R))

def msg2p(ros_point):
    """
    Converts a geometry_msgs/Point message into a 3x1 vector
    
    :type ros_point: geometry_msgs.msg.Point
    :param ros_point: ROS Point message
    :rtype: numpy.array
    :return The 3x1 point vector
    """
    return np.array([ros_point.x, ros_point.y, ros_point.z])

def p2msg(p):
    """
    Converts a 3x1 point vector into a geometry_msgs/Point message
    
    :type p: numpy.array
    :param p: 3x1 point matrix
    :rtype: geometry_msgs.msg.Point
    :return The ROS Point message
    """
    p2=np.reshape(p, (3,))
    return Point(p2[0], p2[1], p2[2])

def msg2pose(ros_pose):
    """
    Converts a geometry_msgs/Pose message into a general_robotics_toolbox.Pose
    
    :type ros_pose: geometry_msgs.msg.Pose
    :param ros_pose: ROS Pose message
    :rtype: general_robotics_toolbox.Pose
    :return The Pose class instance
    """
    R=msg2R(ros_pose.orientation)
    p=msg2p(ros_pose.position)
    return rox.Pose(R,p)

def pose2msg(pose):
    """
    Converts a general_robotics_toolbox.Pose into a geometry_msgs/Pose message
    
    :type pose: general_robotics_toolbox.Pose
    :param pose: general_robotics_toolbox.Pose class instance
    :rtype: geometry_msgs.msg.Pose
    :return The ROS Pose message
    """
    return Pose(p2msg(pose.p), R2msg(pose.R))    

def msg2twist(ros_twist):
    """
    Converts a geometry_msgs/Twist message into a 6x1 vector
    
    :type ros_twist: geometry_msgs.msg.Twist
    :param ros_twist: ROS Twist message
    :rtype: numpy.array
    :return The 6x1 twist vector
    """
    return np.array([ros_twist.angular.x, ros_twist.angular.y, ros_twist.angular.z, \
                     ros_twist.linear.x, ros_twist.linear.y, ros_twist.linear.z])

def twist2msg(twist):
    """
    Converts a 6x1 twist vector into a geometry_msgs/Twist message
    
    :type twist: numpy.array
    :param twist: 6x1 twist matrix
    :rtype: geometry_msgs.msg.Twist
    :return The ROS Twist message
    """
    twist2=np.reshape(twist, (6,))
    return Twist(Vector3(twist2[3], twist2[4], twist2[5]),
                 Vector3(twist2[0], twist2[1], twist2[2]))

def msg2wrench(ros_wrench):
    """
    Converts a geometry_msgs/Wrench message into a 6x1 vector
    
    :type ros_wrench: geometry_msgs.msg.Wrench
    :param ros_wrench: ROS Wrench message
    :rtype: numpy.array
    :return The 6x1 wrench vector
    """
    return np.array([ros_wrench.torque.x, ros_wrench.torque.y, ros_wrench.torque.z, \
                     ros_wrench.force.x, ros_wrench.force.y, ros_wrench.force.z])

def wrench2msg(wrench):
    """
    Converts a 6x1 wrench vector into a geometry_msgs/Twist message
    
    :type twist: numpy.array
    :param twist: 6x1 wrench matrix
    :rtype: geometry_msgs.msg.Wrench
    :return The ROS Wrench message
    """
    wrench2=np.reshape(wrench, (6,))
    return Wrench(Vector3(wrench2[3], wrench2[4], wrench2[5]),
                 Vector3(wrench2[0], wrench2[1], wrench2[2]))





