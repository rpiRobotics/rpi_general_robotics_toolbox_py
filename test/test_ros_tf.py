import general_robotics_toolbox as rox
import numpy as np
import pytest

rospy = pytest.importorskip('rospy')

import general_robotics_toolbox.ros_tf as rox_tf
import general_robotics_toolbox.ros_msg as rox_msg

#Numeric precision reduced for literals
eps = 1e-6 #np.finfo(np.float64).eps

@pytest.mark.skip
def test_ros_tf_listener():
    l=rox_tf.TransformListener()
    rox_tf1=rox.random_transform()
    rox_tf1.parent_frame_id='world'
    rox_tf1.child_frame_id='link1'
    rox_tf2=rox.random_transform()
    rox_tf2.parent_frame_id='link1'
    rox_tf2.child_frame_id='link2'
    
    l.ros_listener.setTransform(rox_msg.transform2transform_stamped_msg(rox_tf1))
    l.ros_listener.setTransform(rox_msg.transform2transform_stamped_msg(rox_tf2))
    
    assert l.canTransform('world','link1')
    assert l.canTransformFull('world', rospy.Time(0), 'link1', rospy.Time(0), 'world')
    assert not l.canTransform('world','link3')
    
    l.waitForTransform('world','link1',rospy.Time(0), rospy.Duration(5))
    l.waitForTransformFull('world', rospy.Time(0), 'link1', rospy.Time(0), 'world', rospy.Duration(5))
    
    l_tf1=l.lookupTransform('world','link1')
    assert l_tf1 == rox_tf1
    l_tf1_full=l.lookupTransformFull('world', rospy.Time(0), 'link1', rospy.Time(0), 'world')
    assert l_tf1_full == rox_tf1
    
    l_tf1_2=l.lookupTransform('world','link2')
    assert l_tf1_2 == rox_tf1*rox_tf2
