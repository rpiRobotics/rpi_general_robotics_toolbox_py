import general_robotics_toolbox as rox
import numpy as np
import pytest
import os


from general_robotics_toolbox import robotraconteur as rr_rox
import numpy.testing as nptest


def _get_absolute_path(fname):
    dirname = os.path.dirname(os.path.realpath(__file__))
    return dirname + "/" + fname


def test_robotinfo_yaml_loader_abb1200():
    with open(_get_absolute_path("abb_1200_5_90_robot_default_config.yml"), "r") as f:
        robot = rr_rox.load_robot_info_yaml_to_robot(f)

    nptest.assert_allclose(robot.H, np.transpose(
        [[0, 0, 1], [0, 1, 0], [0, 1, 0], [1, 0, 0], [0, 1, 0], [1, 0, 0]]))
    nptest.assert_allclose(robot.P, np.transpose([
        [0.0, 0.0, 0.3991], [0.0, 0.0, 0.0], [0.0, 0.0, 0.448],
        [0.0, 0.0, 0.042], [0.451, 0.0, 0.0], [0.082, 0.0, 0.0], [0.0, 0.0, 0.0]]))

    nptest.assert_allclose(robot.joint_type, [0, 0, 0, 0, 0, 0])
    nptest.assert_allclose(robot.joint_lower_limit,
                           [-2.967, -1.745, -3.491, -4.712, -2.269, -6.283])
    nptest.assert_allclose(robot.joint_upper_limit, [
                           2.967, 2.269, 1.222, 4.712, 2.269, 6.283])
    nptest.assert_allclose(robot.joint_vel_limit, [
                           5.027, 4.189, 5.236, 6.981, 7.069, 10.472])
    nptest.assert_allclose(robot.joint_acc_limit, [10, 15, 15, 20, 20, 20])
    assert robot.joint_names == ['joint_1', 'joint_2',
                                 'joint_3', 'joint_4', 'joint_5', 'joint_6']
    assert robot.root_link_name is None
    assert robot.tip_link_name == "tool0"
    assert robot.R_tool is None and robot.p_tool is None
    assert robot.T_base is None
    nptest.assert_allclose(robot.T_flange.R, rox.rot(
        [0, 1, 0], np.pi/2.), atol=1e-4)
    nptest.assert_allclose(robot.T_flange.p, [0, 0, 0])


def test_robotinfo_yaml_loader_sawyer_with_tool():
    with open(_get_absolute_path("sawyer_robot_with_electric_gripper_config.yml"), "r") as f:
        robot = rr_rox.load_robot_info_yaml_to_robot(f)
    _assert_sawyer_robot(robot, T_base_expected=rox.Transform(rox.rot([0,0,1],-np.pi/2), [0.5, 0.23, 0.8]))


def _assert_sawyer_robot(robot, T_base_expected = None):

    nptest.assert_allclose(robot.H, np.transpose(
        [[0, 0, 1], [0, 1, 0], [1, 0, 0], [0, 1, 0], [1, 0, 0], [0, 1, 0], [1, 0, 0]]), atol=1e-4)
    nptest.assert_allclose(robot.P, np.transpose([
        [0.0, 0.0, 0.08], [0.081, 0.05, 0.237], [
            0.14, 0.1425, 0.0], [0.26, -0.042, 0.0],
        [0.125, -0.1265, 0.0], [0.275, 0.031, 0.0], [0.11, 0.1053, 0.0], [0.0245, 0.0, 0.0]]), atol=1e-4)

    nptest.assert_allclose(robot.joint_type, [0, 0, 0, 0, 0, 0, 0])
    nptest.assert_allclose(robot.joint_lower_limit,
                           [-3.0503, -3.8095, -3.0426, -3.0439, -2.9761, -2.9761, -4.7124])
    nptest.assert_allclose(robot.joint_upper_limit, [
                           3.0503, 2.2736, 3.0426, 3.0439, 2.9761, 2.9761, 4.7124])
    nptest.assert_allclose(robot.joint_vel_limit, [
                           1.74, 1.328, 1.957, 1.957, 3.485, 3.485, 4.545])
    nptest.assert_allclose(robot.joint_acc_limit, [
                           3.5, 2.5, 5.0, 5.0, 5.0, 5.0, 5.0])
    assert robot.joint_names == [
        'right_j0', 'right_j1', 'right_j2', 'right_j3', 'right_j4', 'right_j5', 'right_j6']
    assert robot.root_link_name == "right_arm_base_link"
    assert robot.tip_link_name == "sawyer_electric_gripper"
    # assert robot.R_tool is None and robot.p_tool is None
    nptest.assert_allclose(robot.R_tool, rox.rot(
        [0, 0, 1], np.deg2rad(5)), atol=1e-4)
    nptest.assert_allclose(robot.p_tool, [0, 0, 0.1577], atol=1e-4)
    if T_base_expected is None:
        assert robot.T_base is None
    else:
        assert T_base_expected.isclose(robot.T_base, tol=1e-4)
    nptest.assert_allclose(rox.R2q(
        robot.T_flange.R), [-0.45451851, 0.54167662, -0.45452185, 0.54167264], atol=1e-4)
    nptest.assert_allclose(robot.T_flange.p, [0, 0, 0])


def test_robotinfo_toolinfo_yaml_loader_sawyer():
    with \
            open(_get_absolute_path("sawyer_robot_default_config.yml"), "r") as f_robot, \
            open(_get_absolute_path("sawyer_electric_gripper_default_config.yml"), "r") as f_tool:

        robot = rr_rox.load_robot_and_tool_info_yaml_to_robot(f_robot, f_tool)
        _assert_sawyer_robot(robot)
