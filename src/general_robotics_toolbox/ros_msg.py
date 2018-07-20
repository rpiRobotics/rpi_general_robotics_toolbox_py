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

from geometry_msgs.msg import Quaternion, Point, Vector3, Pose, Twist, Wrench, \
 Transform, TransformStamped, PoseStamped
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

def msg2p(ros_vector3):
    """
    Converts a geometry_msgs/Vector3 message into a 3x1 vector
    
    :type ros_vector3: geometry_msgs.msg.Vector3
    :param ros_vector3: ROS Vector3 message
    :rtype: numpy.array
    :return The 3x1 point vector
    """
    return np.array([ros_vector3.x, ros_vector3.y, ros_vector3.z])

def p2msg(p):
    """
    Converts a 3x1 point vector into a geometry_msgs/Vector3 message
    
    :type p: numpy.array
    :param p: 3x1 point matrix
    :rtype: geometry_msgs.msg.Point
    :return The ROS Point message
    """
    p2=np.reshape(p, (3,))
    return Vector3(p2[0], p2[1], p2[2])

def point_msg2p(ros_point):
    """
    Converts a geometry_msgs/Point message into a 3x1 vector
    
    :type ros_point: geometry_msgs.msg.Point
    :param ros_point: ROS Point message
    :rtype: numpy.array
    :return The 3x1 point vector
    """
    return np.array([ros_point.x, ros_point.y, ros_point.z])

def p2point_msg(p):
    """
    Converts a 3x1 point vector into a geometry_msgs/Point message
    
    :type p: numpy.array
    :param p: 3x1 point matrix
    :rtype: geometry_msgs.msg.Point
    :return The ROS Point message
    """
    p2=np.reshape(p, (3,))
    return Point(p2[0], p2[1], p2[2])

def msg2transform(ros_transform):
    """
    Converts a geometry_msgs/Pose message into a general_robotics_toolbox.Pose
    
    :type ros_transform: geometry_msgs.msg.Transform, geometry_msgs.msg.Pose, 
          geometry_msgs.msg.TransformStamped, or geometry_msgs.msg.PoseStamped
    :param ros_transform: ROS Transform, Pose, TransformStamped, or PoseStamped message
    :rtype: general_robotics_toolbox.Pose
    :return The Pose class instance
    """
    
    if hasattr(ros_transform, 'translation') and hasattr(ros_transform, 'rotation'):
        #geometry_msgs/Transform
        R=msg2R(ros_transform.rotation)
        p=msg2p(ros_transform.translation)
        return rox.Transform(R,p)
    elif hasattr(ros_transform, 'header') and hasattr(ros_transform, 'transform'):
        #geometry_msgs/TransformStamped
        R=msg2R(ros_transform.transform.rotation)
        p=msg2p(ros_transform.transform.translation)
        parent_frame_id=ros_transform.header.frame_id
        child_frame_id=ros_transform.child_frame_id
        return rox.Transform(R,p,parent_frame_id,child_frame_id)
    if hasattr(ros_transform, 'position') and hasattr(ros_transform, 'orientation'):
        #geometry_msgs/Pose
        R=msg2R(ros_transform.orientation)
        p=msg2p(ros_transform.position)
        return rox.Transform(R,p)
    elif hasattr(ros_transform, 'header') and hasattr(ros_transform, 'pose'):
        #geometry_msgs/PoseStamped
        R=msg2R(ros_transform.pose.orientation)
        p=msg2p(ros_transform.pose.position)
        parent_frame_id=ros_transform.header.frame_id        
        return rox.Transform(R,p,parent_frame_id)
    else:
        assert False, "Invalid data type for ros_transform"

def transform2msg(transform):
    """
    Converts a general_robotics_toolbox.Transform into a geometry_msgs/Transform message
    
    :type pose: general_robotics_toolbox.Transform
    :param pose: general_robotics_toolbox.Transform class instance
    :rtype: geometry_msgs.msg.Transform
    :return The ROS Transform message
    """
    return Transform(p2msg(transform.p), R2msg(transform.R))    

def transform2transform_stamped_msg(transform):
    """
    Converts a general_robotics_toolbox.Transform into a geometry_msgs/TransformStamped message
    
    :type pose: general_robotics_toolbox.Transform
    :param pose: general_robotics_toolbox.Transform class instance
    :rtype: geometry_msgs.msg.TransformStamped
    :return The ROS Transform message
    """
    r=TransformStamped()
    r.transform=Transform(p2msg(transform.p), R2msg(transform.R))
    if transform.child_frame_id is not None:
        r.child_frame_id=transform.child_frame_id
    if transform.parent_frame_id is not None:
        r.header.frame_id=transform.parent_frame_id        
    return r

def transform2pose_msg(transform):
    """
    Converts a general_robotics_toolbox.Transform into a geometry_msgs/Pose message
    
    :type pose: general_robotics_toolbox.Transform
    :param pose: general_robotics_toolbox.Transform class instance
    :rtype: geometry_msgs.msg.Pose
    :return The ROS Pose message
    """
    return Pose(p2point_msg(transform.p), R2msg(transform.R))    

def transform2pose_stamped_msg(transform):
    """
    Converts a general_robotics_toolbox.Transform into a geometry_msgs/PoseStamped message
    
    :type pose: general_robotics_toolbox.Transform
    :param pose: general_robotics_toolbox.Transform class instance
    :rtype: geometry_msgs.msg.PoseStamped
    :return The ROS Pose message
    """
    r=PoseStamped()
    r.pose=Pose(p2point_msg(transform.p), R2msg(transform.R))    
    if transform.parent_frame_id is not None:
        r.header.frame_id=transform.parent_frame_id        
    return r

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





