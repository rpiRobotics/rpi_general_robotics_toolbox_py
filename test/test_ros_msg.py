import general_robotics_toolbox as rox
import numpy as np
import pytest

rospy = pytest.importorskip('rospy')

import general_robotics_toolbox.ros_msg as rox_msg

#Numeric precision reduced for literals
eps = 1e-6 #np.finfo(np.float64).eps


def test_ros_msg():
    
    #msg2q, q2msg
    q=np.random.rand(4)
    q=q/np.linalg.norm(q)
    
    q_msg=rox_msg.q2msg(q)
    q_msg._check_types()
    np.testing.assert_allclose(q, [q_msg.w, q_msg.x, q_msg.y, q_msg.z], atol=1e-4)
    q2=rox_msg.msg2q(q_msg)
    np.testing.assert_allclose(q, q2, atol=1e-4)
    
    #msg2R, R2msg
    R=rox.q2R(q)
    q_msg_R=rox_msg.R2msg(R)
    q_msg_R._check_types()
    np.testing.assert_allclose(q, [q_msg_R.w, q_msg_R.x, q_msg_R.y, q_msg_R.z], atol=1e-4)
    R2=rox_msg.msg2R(q_msg_R)
    np.testing.assert_allclose(R, R2, atol=1e-4)
    
    #msg2p, p2msg
    p=np.random.rand(3)
    p_msg=rox_msg.p2msg(p)
    p_msg._check_types()
    np.testing.assert_allclose(p, [p_msg.x, p_msg.y, p_msg.z], atol=1e-4)
    p2=rox_msg.msg2p(p_msg)
    np.testing.assert_allclose(p, p2, atol=1e-4)
    
    #transform messages of varying types
    tf=rox.Transform(R,p)
    pose_msg=rox_msg.transform2pose_msg(tf)
    pose_msg._check_types()
    np.testing.assert_allclose(R, rox_msg.msg2R(pose_msg.orientation), atol=1e-4)
    np.testing.assert_allclose(p, rox_msg.msg2p(pose_msg.position), atol=1e-4)
    pose2=rox_msg.msg2transform(pose_msg)
    np.testing.assert_allclose(R, pose2.R, atol=1e-4)
    np.testing.assert_allclose(p, pose2.p, atol=1e-4)
    tf_msg=rox_msg.transform2msg(tf)
    tf_msg._check_types()
    np.testing.assert_allclose(R, rox_msg.msg2R(tf_msg.rotation), atol=1e-4)
    np.testing.assert_allclose(p, rox_msg.msg2p(tf_msg.translation), atol=1e-4)
    tf2=rox_msg.msg2transform(tf_msg)
    np.testing.assert_allclose(R, tf2.R, atol=1e-4)
    np.testing.assert_allclose(p, tf2.p, atol=1e-4)
    
    #transform stamped messages of varying types
    tf3=rox.Transform(R,p,'parent_link','child_link')
    print(tf3)
    print(str(tf3))
    pose_stamped_msg=rox_msg.transform2pose_stamped_msg(tf3)
    pose_stamped_msg._check_types()
    np.testing.assert_allclose(R, rox_msg.msg2R(pose_stamped_msg.pose.orientation), atol=1e-4)
    np.testing.assert_allclose(p, rox_msg.msg2p(pose_stamped_msg.pose.position), atol=1e-4)
    assert pose_stamped_msg.header.frame_id=='parent_link'
    pose3=rox_msg.msg2transform(pose_stamped_msg)
    np.testing.assert_allclose(R, pose3.R, atol=1e-4)
    np.testing.assert_allclose(p, pose3.p, atol=1e-4)
    assert pose3.parent_frame_id=='parent_link'
    tf_stamped_msg=rox_msg.transform2transform_stamped_msg(tf3)
    tf_stamped_msg._check_types()
    np.testing.assert_allclose(R, rox_msg.msg2R(tf_stamped_msg.transform.rotation), atol=1e-4)
    np.testing.assert_allclose(p, rox_msg.msg2p(tf_stamped_msg.transform.translation), atol=1e-4)
    assert tf_stamped_msg.header.frame_id=='parent_link'
    assert tf_stamped_msg.child_frame_id=='child_link'
    tf4=rox_msg.msg2transform(tf_stamped_msg)
    np.testing.assert_allclose(R, tf4.R, atol=1e-4)
    np.testing.assert_allclose(p, tf4.p, atol=1e-4)
    assert tf4.parent_frame_id=='parent_link'
    assert tf4.child_frame_id=='child_link'
    
    #msg2twist, twist2msg
    twist=np.random.rand(6)
    twist_msg=rox_msg.twist2msg(twist)
    twist_msg._check_types()
    np.testing.assert_allclose(twist, [twist_msg.angular.x, twist_msg.angular.y, twist_msg.angular.z, \
                                        twist_msg.linear.x, twist_msg.linear.y, twist_msg.linear.z], \
                                        atol=1e-4)
    twist2=rox_msg.msg2twist(twist_msg)
    np.testing.assert_allclose(twist, twist2, atol=1e-4)
    
    #msg2wrench, wrench2msg
    wrench=np.random.rand(6)
    wrench_msg=rox_msg.wrench2msg(wrench)
    wrench_msg._check_types()
    np.testing.assert_allclose(wrench, [wrench_msg.torque.x, wrench_msg.torque.y, wrench_msg.torque.z, \
                                        wrench_msg.force.x, wrench_msg.force.y, wrench_msg.force.z], \
                                        atol=1e-4)
    wrench2=rox_msg.msg2wrench(wrench_msg)
    np.testing.assert_allclose(wrench, wrench2, atol=1e-4)      



