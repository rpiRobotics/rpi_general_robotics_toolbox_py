ROS 1 Utilities
===============

ROS 1 utility functions for ROS messages and TF2. These modules must be used inside a ROS 1 workspace.

general_robotics_toolbox.ros_msg
--------------------------------

.. module:: general_robotics_toolbox.ros_msg

    Provides convenience functions for converting between general_robotics_toolbox
    and ROS geometry_msgs types

    .. function:: msg2q(ros_quaternion)

        Converts a geometry_msgs/Quaternion message into a 4x1 quaternion vector
    
        :type ros_quaternion: geometry_msgs.msg.Quaternion
        :param ros_quaternion: ROS Quaternion message
        :rtype: numpy.array
        :return: The 4x1 quaternion matrix

    .. function:: q2msg(q)

        Converts a 4x1 quaternion vector into a geometry_msgs/Quaternion message
    
        :type q: numpy.array
        :param q: 4x1 quaternion matrix
        :rtype: geometry_msgs.msg.Quaternion
        :return: The ROS Quaternion message

    .. function:: msg2R(ros_quaternion)

        Converts a geometry_msgs/Quaternion message into a 3x3 rotation matrix
    
        :type ros_quaternion: geometry_msgs.msg.Quaternion
        :param ros_quaternion: ROS Quaternion message
        :rtype: numpy.array
        :return: The 3x3 rotation matrix

    .. function:: R2msg(R)

        Converts a 3x3 rotation matrix into a geometry_msgs/Quaternion message
    
        :type R: numpy.array
        :param R: 3x3 rotation matrix
        :rtype: geometry_msgs.msg.Quaternion
        :return: The ROS Quaternion message

    .. function:: msg2p(ros_vector3)

        Converts a geometry_msgs/Vector3 message into a 3x1 vector
    
        :type ros_vector3: geometry_msgs.msg.Vector3
        :param ros_vector3: ROS Vector3 message
        :rtype: numpy.array
        :return: The 3x1 point vector

    .. function:: p2msg(p)

        Converts a 3x1 point vector into a geometry_msgs/Vector3 message
    
        :type p: numpy.array
        :param p: 3x1 point matrix
        :rtype: geometry_msgs.msg.Point
        :return: The ROS Point message

    .. function:: point_msg2p(ros_point)

        Converts a geometry_msgs/Point message into a 3x1 vector
    
        :type ros_point: geometry_msgs.msg.Point
        :param ros_point: ROS Point message
        :rtype: numpy.array
        :return: The 3x1 point vector

    .. function:: p2point_msg(p)
        
        Converts a 3x1 point vector into a geometry_msgs/Point message
    
        :type p: numpy.array
        :param p: 3x1 point matrix
        :rtype: geometry_msgs.msg.Point
        :return: The ROS Point message

    .. function:: msg2transform(ros_transform)

        Converts a geometry_msgs/Pose message into a general_robotics_toolbox.Pose
    
        :type ros_transform: geometry_msgs.msg.Transform, geometry_msgs.msg.Pose, 
            geometry_msgs.msg.TransformStamped, or geometry_msgs.msg.PoseStamped
        :param ros_transform: ROS Transform, Pose, TransformStamped, or PoseStamped message
        :rtype: general_robotics_toolbox.Pose
        :return: The Pose class instance

    .. function:: transform2msg(transform)

        Converts a general_robotics_toolbox.Transform into a geometry_msgs/Transform message
    
        :type pose: general_robotics_toolbox.Transform
        :param pose: general_robotics_toolbox.Transform class instance
        :rtype: geometry_msgs.msg.Transform
        :return: The ROS Transform message

    .. function:: transform2transform_stamped_msg(transform)

        Converts a general_robotics_toolbox.Transform into a geometry_msgs/TransformStamped message
    
        :type pose: general_robotics_toolbox.Transform
        :param pose: general_robotics_toolbox.Transform class instance
        :rtype: geometry_msgs.msg.TransformStamped
        :return: The ROS Transform message

    .. function:: transform2pose_msg(transform)

        Converts a general_robotics_toolbox.Transform into a geometry_msgs/Pose message
    
        :type pose: general_robotics_toolbox.Transform
        :param pose: general_robotics_toolbox.Transform class instance
        :rtype: geometry_msgs.msg.Pose
        :return: The ROS Pose message

    .. function:: transform2pose_stamped_msg(transform)

        Converts a general_robotics_toolbox.Transform into a geometry_msgs/PoseStamped message
    
        :type pose: general_robotics_toolbox.Transform
        :param pose: general_robotics_toolbox.Transform class instance
        :rtype: geometry_msgs.msg.PoseStamped
        :return The ROS Pose message

    .. function:: msg2twist(ros_twist)

        Converts a geometry_msgs/Twist message into a 6x1 vector
    
        :type ros_twist: geometry_msgs.msg.Twist
        :param ros_twist: ROS Twist message
        :rtype: numpy.array
        :return: The 6x1 twist vector

    .. function:: twist2msg(twist)

        Converts a 6x1 twist vector into a geometry_msgs/Twist message
    
        :type twist: numpy.array
        :param twist: 6x1 twist matrix
        :rtype: geometry_msgs.msg.Twist
        :return: The ROS Twist message

    .. function:: msg2wrench(ros_wrench)

        Converts a geometry_msgs/Wrench message into a 6x1 vector
    
        :type ros_wrench: geometry_msgs.msg.Wrench
        :param ros_wrench: ROS Wrench message
        :rtype: numpy.array
        :return: The 6x1 wrench vector

    .. function:: wrench2msg(wrench)

        Converts a 6x1 wrench vector into a geometry_msgs/Twist message
    
        :type twist: numpy.array
        :param twist: 6x1 wrench matrix
        :rtype: geometry_msgs.msg.Wrench
        :return: The ROS Wrench message

general_robotics_toolbox.ros_tf
-------------------------------

.. module:: general_robotics_toolbox.ros_tf
    
    .. class:: TransformListener

        Class to use a ROS TF2 listener and retrieve transforms

        All arguments to __init__ are passed to tf.TransformListener

        .. method:: canTransform(self, target_frame, source_frame, time = rospy.Time(0))

            Check if transform is available

        .. method:: canTransformFull(self, target_frame, target_time, source_frame, source_time, fixed_frame)

            Extended version of canTransform

        .. method:: waitForTransform(self, target_frame, source_frame, time, timeout, polling_sleep_duration=None)

            Wait for transform to be available

        .. method:: waitForTransformFull(self, target_frame, target_time, source_frame, source_time, fixed_frame, timeout, polling_sleep_duration=None)

            Extended version of waitForTransform

        .. method:: clear(self)

            Clear the listener

        .. method:: lookupTransform(self, target_frame, source_frame, time = rospy.Time(0))

            Lookup a transform. Returns rox.Transform

        .. method:: lookupTransformFull(self, target_frame, target_time, source_frame, source_time, fixed_frame)

            Extended version of lookupTransform