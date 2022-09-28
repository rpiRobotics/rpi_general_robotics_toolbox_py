
import pytest
import sys

if sys.version_info[0] < 3:
    pytest.skip("skipping Tesseract tests on Python 2", allow_module_level=True)

import general_robotics_toolbox as rox
import numpy as np
import numpy.testing as nptest
import os

tesseract_common = pytest.importorskip('tesseract_robotics.tesseract_common')

from general_robotics_toolbox import tesseract as rox_tesseract
from general_robotics_toolbox import robotraconteur as rr_rox

from tesseract_robotics.tesseract_scene_graph import \
    JointType_FIXED, JointType_REVOLUTE, JointType_PRISMATIC
from tesseract_robotics.tesseract_environment import Environment

def _assert_rox_eig_transform_close(T, eig_iso):
    H_eig_iso = eig_iso.matrix()
    H_T = np.zeros((4,4),dtype=np.float64)
    H_T[0:3,0:3] = T.R
    H_T[0:3,3] = T.p
    H_T[3,3] = 1.0

    nptest.assert_allclose(H_eig_iso, H_T, atol = 1e-4)

def test_transform_to_isometry3d():
    T = rox.random_transform()
    eig_iso = rox_tesseract._transform_to_isometry3d(T)
    _assert_rox_eig_transform_close(T, eig_iso)

def test_get_fixed_link_command():
    T = rox.random_transform()
    link, joint = rox_tesseract._get_fixed_link_and_joint(T, "my_link", "my_joint", "my_parent_link")
    assert link.getName() == "my_link"
    assert joint.parent_link_name == "my_parent_link"
    assert joint.child_link_name == "my_link"
    assert joint.getName() == "my_joint"
    _assert_rox_eig_transform_close(T, joint.parent_to_joint_origin_transform)

def test_get_link_command():
    p = rox.random_p()
    h = rox.random_p()
    h = h/np.linalg.norm(h)

    link, joint = rox_tesseract._get_link_and_joint(h, p, 0, -np.rad2deg(24), np.rad2deg(34),
        7.92, 8.92, 6.432, "my_link4", "my_joint8", "my_link3")

    assert link.getName() == "my_link4"
    assert joint.parent_link_name == "my_link3"
    assert joint.child_link_name == "my_link4"
    assert joint.getName() == "my_joint8"
    _assert_rox_eig_transform_close(rox.Transform(np.eye(3),p), 
        joint.parent_to_joint_origin_transform)
    nptest.assert_allclose(h, joint.axis.flatten(), atol=1e-6)
    assert joint.type == JointType_REVOLUTE

    limits = joint.limits
    assert limits.lower == -np.rad2deg(24)
    assert limits.upper == np.rad2deg(34)
    assert limits.velocity == 7.92
    assert limits.acceleration == 8.92
    assert limits.effort == 6.432

def _get_absolute_path(fname):
    dirname = os.path.dirname(os.path.realpath(__file__))
    return dirname + "/" + fname

def test_robot_to_tesseract_env_commands():
    with open(_get_absolute_path("sawyer_robot_with_electric_gripper_config.yml"), "r") as f:
        robot = rr_rox.load_robot_info_yaml_to_robot(f)

    tesseract_env_commands = rox_tesseract.robot_to_tesseract_env_commands(robot, "sawyer")
    env = Environment()
    env.init(tesseract_env_commands)

    kin_info = env.getKinematicsInformation()

    print(kin_info)