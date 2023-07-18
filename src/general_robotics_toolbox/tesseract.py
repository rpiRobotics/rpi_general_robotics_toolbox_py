# Copyright (c) 2022, Rensselaer Polytechnic Institute, Wason Technology LLC
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

"""
Tesseract Robotics integration and support functions for the General Robotics Toolbox

Tesseract Robotics is a high performance robot planning framework. This module allows the high performance
kinematics functions to be used in place of the slow Python functions available in the rest of this module.

Most users will only need the TesseractRobotics class, which is initialized using a Robot structure. These
can either be entered in Python, or use the yaml loaders an the general_robotics_toolbox.robotraconteur module. The
TesseractRobotics class has the functions fwdkin(), jacobian(), invkin(), and redundant_solutions() available.

A simple example of using the TesseractRobotics class::

    # Import reqired modules
    import numpy as np
    import general_robotics_toolbox as rox
    from general_robotics_toolbox import tesseract as rox_tesseract
    from general_robotics_toolbox import robotraconteur as rr_rox

    with open("abb_1200_5_90_robot_default_config.yml", "r") as f:
        robot = rr_rox.load_robot_info_yaml_to_robot(f)
    
    tesseract_robot = rox_tesseract.TesseractRobot(robot, "robot", invkin_solver="OPWInvKin")

    # Random joint angles
    q = np.array([0.1, 0.11, 0.12, 0.13, 0.14, 0.15])

    # Compute forward kinematics
    T_tip_link = tesseract_robot.fwdkin(q)

    # Compute robot Jacobian
    J = tesseract_robot.jacobian(q)

    # Solve inverse kinematics for T_des pose
    T_des = rox.Transform(rox.rot([0,1,0], np.deg2rad(80)), np.array([0.5,0.1,0.71]))
    invkin1 = tesseract_robot.invkin(T_des,q*0.7)

    # Find redundant solutions for first candidate joint angle
    invkin1_redun = []
    for invkin1_i in invkin1:
        invkin1_redun.extend(tesseract_robot.redundant_solutions(invkin1_i))
"""


import sys

if sys.version_info < (3,6):
    raise Exception("Python version 3.6 or higher required for Tesseract")


import numpy as np
from . import general_robotics_toolbox as rox

from tesseract_robotics.tesseract_common import Translation3d, AngleAxisd, Isometry3d, VectorPairString
from tesseract_robotics.tesseract_environment import Environment, Commands, \
    AddLinkCommand, AddKinematicsInformationCommand, AddSceneGraphCommand
from tesseract_robotics.tesseract_scene_graph import Link, Joint, JointLimits, \
    JointType_FIXED, JointType_REVOLUTE, JointType_PRISMATIC, SceneGraph
from tesseract_robotics.tesseract_common import FilesystemPath, ManipulatorInfo, KinematicsPluginInfo, \
    PluginInfoContainer
from tesseract_robotics.tesseract_kinematics import KinGroupIKInput, KinGroupIKInputs, getRedundantSolutions
from tesseract_robotics.tesseract_srdf import KinematicsInformation, parseKinematicsPluginConfigString
import yaml
import io
from typing import NamedTuple
 
def transform_to_isometry3d(T):
    """
    Convert a general_robotics_toolbox.Transform to tesseract_common.Isometry3d

    :type  T: general_robotics_toolbox.Transform
    :param T: The transform to convert
    :rtype  : tesseract_common.Isometry3d
    :return : The converted transform
    """
    # TODO: Use better Isometry3d constructor
    return Translation3d(T.p) * AngleAxisd(T.R)

def isometry3d_to_transform(eig_iso):
    """
    Convert a tesseract_common.Isometry3d to general_robotics_toolbox.Transform

    :type  T: tesseract_common.Isometry3d
    :param T: The transform to convert
    :rtype  : general_robotics_toolbox.Transform
    :return : The converted transform
    """
    H = eig_iso.matrix()
    R = H[0:3,0:3]
    p = H[0:3,3].flatten()
    return rox.Transform(R,p)

def get_link_and_joint(h, p, joint_type, joint_lower_limit, joint_upper_limit, joint_vel_limit,
    joint_acc_limit, joint_effort_limit, link_name, joint_name, parent_link_name):
    """
    Create a tesseract link and joint from parameters

    All units are expected to be SI

    :type h: np.array
    :param h: Joint unit vector, corresponds to H[:,i]
    :type p: np.array
    :param p: Joint origin vector, corresponds to P[:,i+1]
    :type joint_type: int
    :param joint_type: Joint type. Supported values 0 for revolute, 1 for prismatic
    :type joint_lower_limit: float
    :param joint_lower_limit: Joint lower position limit
    :type joint_upper_limit: float
    :param joint_upper_limit: Joint upper position limit
    :type joint_vel_limit: float
    :param joint_vel_limit: Joint velocity limit
    :type joint_acc_limit: float
    :param joint_acc_limit: Joint acceleration limit
    :type joint_effort_limit: float
    :param joint_effort_limit: Joint effort limit
    :type link_name: str
    :param link_name: Link name
    :type joint_name: str
    :param joint_name: Joint name
    :type parent_link_name: str
    :param parent_link_name: Parent link name
    :rtype: Tuple[tesseract_scene_graph.Link,tesseract_scene_graph.Joint]
    """
    
    link = Link(link_name)
    joint = Joint(joint_name)

    if joint_type == 0:      
        joint.type = JointType_REVOLUTE
    elif joint_type == 1:
        joint.type = JointType_PRISMATIC
    else:
        assert False, "Unsupported Tesseract joint type: " + str(joint.type)
    joint.parent_to_joint_origin_transform = transform_to_isometry3d(rox.Transform(np.eye(3), p))
    joint.axis = h
    joint.parent_link_name = parent_link_name
    joint.child_link_name = link_name
    joint.limits = JointLimits(joint_lower_limit, joint_upper_limit, joint_effort_limit,
         joint_vel_limit, joint_acc_limit)
    return link,joint

def get_fixed_link_and_joint(T, link_name, joint_name, parent_link_name):
    """
    Create a tesseract link and fixed joint from parameters

    All units are expected to be SI

    :type T: general_robotics_toolbox.Transform
    :param T: Transform for fixed link origin
    :type link_name: str
    :param link_name: Link name
    :type joint_name: str
    :param joint_name: Joint name
    :type parent_link_name: str
    :param parent_link_name: Parent link name
    :rtype: Tuple[tesseract_scene_graph.Link,tesseract_scene_graph.Joint]
    """
    link = Link(link_name)
    joint = Joint(joint_name)

    joint.type = JointType_FIXED
    joint.parent_to_joint_origin_transform = transform_to_isometry3d(T)
    joint.parent_link_name = parent_link_name
    joint.child_link_name = link_name
    return link,joint

def get_robot_world_to_base_joint(robot, robot_name):
    """
    Create fixed joint from world to robot base origin. Uses robot.T_base if specified, otherwise
    return identity transform.

    :type robot: general_robotics_toolbox.Robot
    :param robot: Input Robot structure containing robot parameters
    :type robot_name: str
    :param robot_name: The name of the robot. Must match name used to initialize other parameters
    :rtype: tesseract_scene_graph.Joint
    :return: The fixed joint from world to robot origin
    """
    

    joint = Joint(f"world_to_{robot_name}")

    joint.type = JointType_FIXED
    if robot.T_base:
        joint.parent_to_joint_origin_transform = transform_to_isometry3d(robot.T_base)
    else:
        iso_base = Isometry3d()
        iso_base.setIdentity()
        joint.parent_to_joint_origin_transform = iso_base
    joint.parent_link_name = "world"
    joint.child_link_name = f"{robot_name}_base_link"
    return joint

def _prefix_names(names, robot_name):
    return [f"{robot_name}_{n}" for n in names]

def robot_to_scene_graph(robot, return_names = False):
    """
    Convert a general_robotics_toolbox.Robot structure to tesseract_scene_graph.Graph. Does not create world
    link, or use T_base.

    :type    robot: general_robotics_toolbox.Robot
    :param   robot: Input Robot structure containing robot parameters
    :type    return_names: bool
    :param   return_names: Return the names of created links and joints. Optional, default false
    :rtype:  tesseract_scene_graph.Graph or (tesseract_scene_graph.Graph, List[str], List[str], List[str])
    :return: The scene graph, and optionally the scene graph, link names, joint names, and chain link names if
             return_names is True
    """


    sg = SceneGraph()

    sg_link_names = ["base_link"]

    assert sg.addLink(Link("base_link"))
    n_joints = len(robot.joint_type)

    if robot.joint_names is None:
        joint_names = [f"joint_{i+1}" for i in range(n_joints)]
    else:
        joint_names = robot.joint_names

    # if robot.link_names is None:
    link_names = [f"link_{i+1}" for i in range(n_joints)]
    # else:
        # link_names = [_p(s) for s in robot.link_names]

    assert len(joint_names) == n_joints

    for i in range(n_joints):
        if i == 0:
            parent_link_name = "base_link"
        else:
            parent_link_name = link_names[i-1]
        
        assert sg.addLink(*get_link_and_joint(robot.H[:,i], robot.P[:,i], robot.joint_type[i], 
            robot.joint_lower_limit[i], robot.joint_upper_limit[i], robot.joint_vel_limit[i], 
            robot.joint_acc_limit[i], 1000.0, link_names[i], joint_names[i],
            parent_link_name))
        sg_link_names.append(link_names[i])
    
    assert sg.addLink(*get_fixed_link_and_joint(rox.Transform(np.eye(3), robot.P[:,n_joints]), 
        "chain_tip", link_names[n_joints-1] + "_to_chain_tip", link_names[n_joints-1]))
    sg_link_names.append("chain_tip")

    tip_link = "chain_tip"

    flange_link_name = "flange"

    if robot.T_flange is not None:
        assert sg.addLink(*get_fixed_link_and_joint(robot.T_flange,
            flange_link_name, "tip_to_flange", tip_link))

        tip_link = flange_link_name
        sg_link_names.append(flange_link_name)

    tool_link_name = "tool_tcp"

    if robot.R_tool is not None and robot.p_tool is not None:
        assert sg.addLink(*get_fixed_link_and_joint(rox.Transform(robot.R_tool, robot.p_tool),
            tool_link_name, "tip_to_tool", tip_link))

        tip_link = tool_link_name
        sg_link_names.append(tool_link_name)
    
    if not return_names:
        return sg
    else:
        return sg, sg_link_names, joint_names, ["base_link"] + link_names

def world_scene_graph(world_link_name = "world"):
    """
    Returns a scene graph containing only a world link

    :type world_link_name: str
    :param world_name_name: The name of the world link. Optional, defaults to "world"
    :rtype: tesseract_scene_graph.Graph
    :return: Scene graph with world link
    """
    sg = SceneGraph()
    sg.addLink(Link(world_link_name))
    return sg


def tesseract_kinematics_information(robot, robot_name, link_names, joint_names, chain_link_names, invkin_solver = None, 
    invkin_plugin_info = None, base_link = None, tip_link = None):
    """
    Creates a tesseract_kinematics.KinematicsInformation structure from parameters

    :type robot: general_robotics_toolbox.Robot
    :param robot: Robot structure containing robot parameters
    :type robot_name: str
    :param robot_name: The name of the robot. Must match name used to initialize other parameters
    :type link_names: List[str]
    :param link_names: The names of the links in the current robot
    :type joint_names: List[str]
    :param joint_names: The names of the joints in the current robot
    :type chain_link_names: List[str]
    :param chain_link_names: The names of the chain links of the current robot
    :type invkin_solver: str
    :param invkin_solver: The name of the inverse kinematics solver to use. Defaults to KDLInvKinChainLMA. Supports
                        KDLInvKinChainLMA, KDLInvKinChainNR, OPWInvKin, and URInvKin
    :type invkin_plugin_info: str
    :param invkin_plugin_info: Override plugin_info yaml string. invkin_solver is ignored if used
    :type base_link: str
    :param base_link: The name of the robot base link. Optional, defaults to link_names[0]
    :type tip_link: str
    :param tip_link: The name of the tip link. Optional, default to link_names[-1]
    :rtype: tesseract_kinematics.KinematicsInformation
    :return: The information structure
    """
    
    if base_link is None:
        base_link = link_names[0]
    if tip_link is None:
        tip_link = link_names[-1]

    kinematics_information = KinematicsInformation()
    chain_group = VectorPairString()
    chain_group.append((base_link, tip_link))
    kinematics_information.addChainGroup(robot_name, chain_group)
    
    plugin_info_str = kinematics_plugin_info_string(robot, robot_name, link_names, joint_names, chain_link_names,
        invkin_solver, invkin_plugin_info, base_link, tip_link)

    
    plugin_info = parseKinematicsPluginConfigString(plugin_info_str)

    kinematics_information.kinematics_plugin_info = plugin_info

    return kinematics_information


def robot_to_tesseract_env_commands(robot, robot_name = "robot", include_world = True, return_names = False, 
    invkin_solver = None, invkin_plugin_info = None):
    """
    Creates a set of Tesseract Environment commands to initialize the environment with a specified robot structure.

    :type robot: general_robotics_toolbox.Robot
    :param robot: Input Robot structure containing robot parameters
    :type robot_name: str
    :param robot_name: The name of the robot. Optional, defaults to "robot"
    :type include_world: bool
    :param include_world: Include the world link in commands. Omit if adding another robot to an environment.
                          Optional, defaults to True.
    :type return_names: bool
    :param return_names: Return names of links in joints. Optional, defaults to false
    :type invkin_solver: str
    :param invkin_solver: The name of the inverse kinematics solver to use. Defaults to KDLInvKinChainLMA. Supports
                        KDLInvKinChainLMA, KDLInvKinChainNR, OPWInvKin, and URInvKin
    :type invkin_plugin_info: str
    :param invkin_plugin_info: Override plugin_info yaml string. invkin_solver is ignored if used
    :rtype:  tesseract_environment.Commands or (tesseract_environment.Commands, List[str], List[str], List[str])
    :return: The environment commands, and optionally the commands, link names and joint names if
             return_names is True
    """
        
    commands = Commands()
    
    if include_world:
        world_sg = world_scene_graph()
        commands.append(AddSceneGraphCommand(world_sg))

    robot_sg, sg_link_names, sg_joint_names, sg_chain_link_names = robot_to_scene_graph(robot, return_names = True)
    robot_base_joint = get_robot_world_to_base_joint(robot, robot_name)
    commands.append(AddSceneGraphCommand(robot_sg, robot_base_joint, f"{robot_name}_"))

    link_names = _prefix_names(sg_link_names, robot_name)
    joint_names = _prefix_names(sg_joint_names, robot_name)
    chain_link_names = _prefix_names(sg_chain_link_names, robot_name)
    
    # if robot.joint_names is None:
    #     joint_names = [f"{robot_name}_joint_{i+1}" for i in range(len(robot.joint_type))]
    # else:
    #     joint_names = [f"{robot_name}_{s}" for s in robot.joint_names]
    
    kinematics_information = tesseract_kinematics_information(robot, robot_name, link_names, joint_names, 
        chain_link_names, invkin_solver, invkin_plugin_info, "world", link_names[-1])
    commands.append(AddKinematicsInformationCommand(kinematics_information))

    if not return_names:
        return commands
    else:
        if include_world:
            return commands, ["world"] + link_names, joint_names
        else:
            return commands, link_names, joint_names

def kinematics_plugin_fwdkin_kdl_plugin_info_dict(robot_name, base_link, tip_link):
    """
    Create dictionary of yaml parameters for KDL forward kinematics

    :type robot_name: str
    :param robot_name: The name of the robot
    :type base_link: str
    :param base_link: The name of the robot base link
    :type tip_link: str
    :param tip_link: The name of the robot tip link
    :rtype: dict
    :return: KDL plugin info as dict to convert to yaml
    """
    plugin_info = {        
        "fwd_kin_plugins": {
            robot_name: {
                "default": "KDLFwdKinChain",
                "plugins": {
                    "KDLFwdKinChain": {
                        "class": "KDLFwdKinChainFactory",
                        "config": {
                            "base_link": base_link,
                            "tip_link": tip_link
                        }
                    }
                }
            }
        }
    }
    
    return plugin_info, [] #["tesseract_kinematics_kdl_factories"]

def kinematics_plugin_invkin_kdl_plugin_info_dict(robot_name, base_link, tip_link, default_solver = "KDLInvKinChainLMA"):
    """
    Create dictionary of yaml parameters for KDL inverse kinematics. Supported solvers are
    KDLInvKinChainLMA and KDLInvKinChainNR

    :type robot_name: str
    :param robot_name: The name of the robot
    :type base_link: str
    :param base_link: The name of the robot base link
    :type tip_link: str
    :param tip_link: The name of the robot tip link
    :type default_solver: str
    :param default_solver: The default KDL solver. Optional, defaults to KDLInvKinChainLMA
    :rtype: dict
    :return: KDL plugin info as dict to convert to yaml
    """
    plugin_info = {
        "inv_kin_plugins": {
            robot_name: {
                "default": default_solver,
                "plugins": {
                    "KDLInvKinChainLMA": {
                        "class": "KDLInvKinChainLMAFactory",
                        "config": {
                            "base_link": base_link,
                            "tip_link": tip_link
                        }
                    },
                    "KDLInvKinChainNR": {
                        "class": "KDLInvKinChainNRFactory",
                        "config": {
                            "base_link": base_link,
                            "tip_link": tip_link
                        }
                    }
                }
            }
        }        
    }

    return plugin_info, ["tesseract_kinematics_kdl_factories"]

def kinematics_plugin_info_dict(robot, robot_name, link_names, joint_names, chain_link_names, invkin_solver = None, 
    invkin_plugin_info = None, base_link = None, tip_link = None):
    """
    Creates dictionary of kinematics plugin info from parameters

    :type robot: general_robotics_toolbox.Robot
    :param robot: Robot structure containing robot parameters
    :type robot_name: str
    :param robot_name: The name of the robot. Must match name used to initialize other parameters
    :type link_names: List[str]
    :param link_names: The names of the links in the current robot
    :type joint_names: List[str]
    :param joint_names: The names of the joints in the current robot
    :type chain_link_names: List[str]
    :param chain_link_names: The names of the chain links of the current robot
    :type invkin_solver: str
    :param invkin_solver: The name of the inverse kinematics solver to use. Defaults to KDLInvKinChainLMA. Supports
                        KDLInvKinChainLMA, KDLInvKinChainNR, OPWInvKin, and URInvKin
    :type invkin_plugin_info: str
    :param invkin_plugin_info: Override plugin_info yaml string. invkin_solver is ignored if used
    :type base_link: str
    :param base_link: The name of the robot base link. Optional, defaults to link_names[0]
    :type tip_link: str
    :param tip_link: The name of the tip link. Optional, default to link_names[-1]
    :rtype: dict
    :return: Kinematics plugin info as dict to convert to yaml
    """
    
    if base_link is None:
        base_link = link_names[0]

    if tip_link is None:
        tip_link = link_names[-1]

    # fwdkin_solver = "KDLFwdKinChain"
    fwdkin_plugin_info, fwdkin_libs = kinematics_plugin_fwdkin_kdl_plugin_info_dict(
        robot_name, base_link, tip_link)

    if invkin_solver is None:
        invkin_solver = "KDLInvKinChainLMA"

    if invkin_plugin_info is not None:
        invkin_plugin_info, invkin_libs = invkin_plugin_info
    else:

        if invkin_solver == "KDLInvKinChainLMA" or invkin_solver == "KDLInvKinChainNR":
            invkin_plugin_info, invkin_libs = kinematics_plugin_invkin_kdl_plugin_info_dict(robot_name,
                base_link, tip_link, invkin_solver)
        elif invkin_solver == "OPWInvKin":
            opw_params = robot_to_opw_inv_kin_parameters(robot)
            invkin_plugin_info, invkin_libs = kinematics_plugin_invkin_opw_plugin_info_dict(robot_name, chain_link_names[0],
                robot_name + "_flange", opw_params)
        elif invkin_solver == "URInvKin":
            ur_params = robot_to_ur_inv_kin_parameters(robot)
            invkin_plugin_info, invkin_libs = kinematics_plugin_invkin_ur_plugin_info_dict(robot_name, base_link,
                tip_link, ur_params)
        else:
            assert False, "Unknown inverse kinematics solver"

    libs = set(fwdkin_libs + invkin_libs)

    plugin_info_dict = {
        "kinematic_plugins": {
            "search_libraries": list(libs),        
            "fwd_kin_plugins": fwdkin_plugin_info["fwd_kin_plugins"],
            "inv_kin_plugins": invkin_plugin_info["inv_kin_plugins"],
        }
    }

    return plugin_info_dict
        
def kinematics_plugin_info_string(robot, robot_name, link_names, joint_names, chain_link_names, invkin_solver = None, 
    invkin_plugin_info = None, base_link = None, tip_link = None):
    """
    Creates string of kinematics plugin info from parameters

    :type robot: general_robotics_toolbox.Robot
    :param robot: Robot structure containing robot parameters
    :type robot_name: str
    :param robot_name: The name of the robot. Must match name used to initialize other parameters
    :type link_names: List[str]
    :param link_names: The names of the links in the current robot
    :type joint_names: List[str]
    :param joint_names: The names of the joints in the current robot
    :type chain_link_names: List[str]
    :param chain_link_names: The names of the chain links of the current robot
    :type invkin_solver: str
    :param invkin_solver: The name of the inverse kinematics solver to use. Defaults to KDLInvKinChainLMA. Supports
                        KDLInvKinChainLMA, KDLInvKinChainNR, OPWInvKin, and URInvKin
    :type invkin_plugin_info: str
    :param invkin_plugin_info: Override plugin_info yaml string. invkin_solver is ignored if used
    :type base_link: str
    :param base_link: The name of the robot base link. Optional, defaults to link_names[0]
    :type tip_link: str
    :param tip_link: The name of the tip link. Optional, default to link_names[-1]
    :rtype: str
    :return: Kinematics plugin info as yaml string
    """

    plugin_info_dict = kinematics_plugin_info_dict(robot, robot_name, link_names, joint_names, chain_link_names,
        invkin_solver, invkin_plugin_info, base_link, tip_link)

    f = io.StringIO()
    yaml.safe_dump(plugin_info_dict, f)
    plugin_info_str = f.getvalue()

    return plugin_info_str

class OPWInvKinParameters(NamedTuple):
    """
    OPW inverse kinematics solver parameters. See https://github.com/Jmeyer1292/opw_kinematics and
    robot_to_opw_inv_kin_parameters()
    """
    a1: float
    a2: float
    b: float
    c1: float
    c2: float
    c3: float
    c4: float
    offsets: np.array
    sign_corrections: np.array

def robot_to_opw_inv_kin_parameters(robot):
    """
    Convert "Robot" structure to OPW Kinematics parameters
    a1, a2, b, c1, c2, c3, c4, offsets, and sign_corrections.

    See https://github.com/Jmeyer1292/opw_kinematics for definitions of these parameters.

    This function uses a heuristic method to guess the correct configuration. The following must be true:

    * Robot has six joints, with a vertical first axis, two parallel and orthogonal joints, and a spherical wrist.
    (Orthogonal, parallel, wrist OPW) format.

    The heuristic algorithm requires that h1 = z, h2 = h3 = y, and h5 = y. h4 = h5 may be either along x or z, depending
    if the robot home position is outstretched along X or Z. Most ROS Industrial robots use the X outstretched format,
    with either Joint 2 or Joint 3 bent 90 degrees from the vertical configuration. The heuristic will check to see
    if the bend is in one of these joints, and attempt to align with the expected OPW configuration with the robot
    outstretched vertically.
    """

    ex = np.array([1.,0.,0.])
    ey = np.array([0.,1.,0.])
    ez = np.array([0.,0.,1.])

    assert robot.T_flange
    assert np.allclose(robot.T_flange.R, rox.rot([0,1,0], np.pi/2.0))
    assert np.allclose(robot.T_flange.p.flatten(), np.array([0.,0.,0.]))

    assert len(robot.joint_type) == 6

    def axis_sign(_h, _e):
        if np.allclose(_h, _e):
            return 1
        elif np.allclose(_h, -_e):
            return -1
        else:
            return 0

    assert axis_sign(robot.H[:,0], ez) != 0
    assert axis_sign(robot.H[:,1], ey) != 0
    assert axis_sign(robot.H[:,2], ey) != 0
    assert axis_sign(robot.H[:,4], ey) != 0
    assert axis_sign(robot.H[:,3], robot.H[:,5]) != 0
    assert axis_sign(robot.H[:,5], ez) != 0 or axis_sign(robot.H[:,5], ex) != 0

    H2 = np.copy(robot.H)
    P2 = np.copy(robot.P)

    offset = np.zeros((6,))
    
    if axis_sign(robot.H[:,5], ex) != 0:
        # Check if bend is at joint 2 or 3
        if np.abs(P2[0,2]) > np.abs(P2[2,2]):
            # x is greater than z for link 2, assume bend at joint 2
            R = rox.rot(ey, -np.pi/2.0)
            P2[:,2:] = np.matmul(R, P2[:,2:])
            H2[:,2:] = np.matmul(R, H2[:,2:])
            offset[1] = -np.pi/2.0
        elif np.abs(P2[0,3] + P2[0,4]) > np.abs(P2[2,3] + P2[2,4]):
            # x is greater than z for link 3, assume bend at joint 3
            R = rox.rot(ey, -np.pi/2.0)
            P2[:,3:] = np.matmul(R, P2[:,3:])
            H2[:,3:] = np.matmul(R, H2[:,3:])
            offset[2] = -np.pi/2.0
        else:
            # unclear where bend is, assume Joint 2
            R = rox.rot(ey, -np.pi/2.0)
            P2[:,2:] = np.matmul(R, P2[:,2:])
            H2[:,2:] = np.matmul(R, H2[:,2:])
            offset[1] = -np.pi/2.0

    if not np.isclose(P2[0,2],0.0):
        q2_op = -np.arctan2(P2[0,2], P2[2,2])
        P2[:,2] = np.matmul(rox.rot(ey,q2_op),P2[:,2])
        offset[1] += q2_op
        offset[2] -= q2_op

    assert np.allclose(P2[0:2,0], 0.0)
    assert np.allclose(P2[0:2,4:], 0.0)
    assert np.allclose(P2[0,2], 0.0)

    c1 = np.round(P2[2,0] + P2[2,1],6).item()
    c2 = np.round(P2[2,2],6).item()
    c3 = np.round(P2[2,3] + P2[2,4],6).item()
    c4 = np.round(P2[2,5] + P2[2,6],6).item()

    a1 = np.round(P2[0,1],6).item()
    a2 = np.round(P2[0,3],6).item()
    b = np.round(P2[1,1] + P2[1,2] + P2[1,3],6).item()

    sign_corrections = np.ones((6,))
    expected_axes = [ez,ey,ey,ez,ey,ez]
    for i in range(len(expected_axes)):
        sign_corrections[i] = axis_sign(H2[:,i], expected_axes[i])

    return OPWInvKinParameters(
        a1,
        a2,
        b,
        c1,
        c2,
        c3,
        c4,
        np.round(offset,8),
        np.round(sign_corrections,8)
    )
    
def kinematics_plugin_invkin_opw_plugin_info_dict(robot_name, base_link, tip_link, opw_params):
    """
    Create dictionary of yaml parameters for OPW inverse kinematics.

    :type robot_name: str
    :param robot_name: The name of the robot
    :type base_link: str
    :param base_link: The name of the robot base link
    :type tip_link: str
    :param tip_link: The name of the robot tip link
    :type opw_params: OPWInvKinParameters
    :param default_solver: Structure containing OPW parameters. See robot_to_opw_inv_kin_parameters()
    :rtype: dict
    :return: OPWInvKin plugin info as dict to convert to yaml
    """
    plugin_info = {
        "inv_kin_plugins": {
            robot_name: {
                "default": "OPWInvKin",
                "plugins": {
                    "OPWInvKin": {
                        "class": "OPWInvKinFactory",
                        "config": {
                            "base_link": base_link,
                            "tip_link": tip_link,
                            "params": {
                                "a1": opw_params.a1,
                                "a2": opw_params.a2,
                                "b": opw_params.b,
                                "c1": opw_params.c1,
                                "c2": opw_params.c2,
                                "c3": opw_params.c3,
                                "c4": opw_params.c4,
                                "offsets": opw_params.offsets.tolist(),
                                "sign_corrections": np.asarray(opw_params.sign_corrections,dtype=np.int8).tolist()

                            }
                        }
                    },
                }
            }
        }        
    }

    return plugin_info, ["tesseract_kinematics_opw_factories"]

def robot_to_tesseract_env(robot, robot_name = "robot", include_world = True, return_names = False, 
    invkin_solver = None, invkin_plugin_info = None):
    """
    Creates a TesseractEnvironment initialized with a specified robot structure.

    :type robot: general_robotics_toolbox.Robot
    :param robot: Input Robot structure containing robot parameters
    :type robot_name: str
    :param robot_name: The name of the robot. Optional, defaults to "robot"
    :type include_world: bool
    :param include_world: Include the world link in commands. Omit if adding another robot to an environment.
                          Optional, defaults to True.
    :type return_names: bool
    :param return_names: Return names of links in joints. Optional, defaults to false
    :type invkin_solver: str
    :param invkin_solver: The name of the inverse kinematics solver to use. Defaults to KDLInvKinChainLMA. Supports
                        KDLInvKinChainLMA, KDLInvKinChainNR, OPWInvKin, and URInvKin
    :type invkin_plugin_info: str
    :param invkin_plugin_info: Override plugin_info yaml string. invkin_solver is ignored if used
    :rtype:  tesseract_environment.Commands or (tesseract_environment.Commands, List[str], List[str], List[str])
    :return: The environment commands, and optionally the commands, link names and joint names if
             return_names is True
    """

    tesseract_env_commands, link_names, joint_names = robot_to_tesseract_env_commands(robot, 
        robot_name, include_world, True, invkin_solver, invkin_plugin_info)

    env = Environment()
    assert env.init(tesseract_env_commands)

    if not return_names:
        return env
    else:
        return env, link_names, joint_names


class TesseractRobot:
    """
    Robot class that uses Tesseract for kinematic solvers. A tesseract_environment.Environment class is populated
    using a general_robotics_toolbox.Robot structure using the utility functions in the 
    general_robotics_toolbox.tesseract module. These solvers use high performance solvers, and are significantly
    faster than the Python based solvers in general_robotics_toolbox. The functions of this class should return
    identical results to the Python based solvers.
    """
    def __init__(self, robot = None, robot_name = "robot", invkin_solver = "KDLInvKinChainLMA", tesseract_env = None):
        """
        Construct a TesseractRobot robot class using a general_robotics_toolbox.Robot structure. Specify a solver
        that best matches the robot. The OPWInvKin and URInvKin solvers use closed form solutions, while
        KDLInvKinChainLMA and KDLInvKinChainNR are iterative solvers. OPWInvKin should be used for six-axis industrial 
        robots with spherical wrists, while URInvKin should be used for Universal Robot UR or URe series robots.

        :type robot: general_robotics_toolbox.Robot
        :param robot: Robot structure containing robot parameters
        :type robot_name: str
        :param robot_name: The name of the robot. Optional, defaults to "robot"
        :type invkin_solver: str
        :param invkin_solver: The name of the inverse kinematics solver to use. Optional, defaults to KDLInvKinChainLMA. 
                        Supports KDLInvKinChainLMA, KDLInvKinChainNR, OPWInvKin, and URInvKin
        :type tesseract_env: tesseract_environment.Environment
        :param tesseract_env: A prepared Tesseract Environment. Use if an existing environment is available. Must
                              be None if using the robot parameter. Optional, defaults to None.

        """

        assert robot or tesseract_env
        assert not (robot  and tesseract_env)

        self.robot_name = robot_name

        if robot is not None:
            self.tesseract_env, link_names, joint_names = robot_to_tesseract_env(robot, robot_name, True, True, 
                invkin_solver)
            self.base_link_name = link_names[0]
            self.tip_link_name = link_names[-1]
        else:
            self.tesseract_env = tesseract_env
            kin_grp = self.tesseract_env.getKinematicGroup(robot_name)
            self.base_link_name = kin_grp.getBaseLinkName()
            self.tip_link_name = kin_grp.getAllPossibleTipLinkNames()[0]
    
    def fwdkin(self, theta, base_link_name = None, tip_link_name = None):
        """
        Compute robot forward kinematics at specified joint angles.

        :type theta: np.array
        :param theta: The N vector of joint angles. Length must match number of joints. Expects radians or meters.
        :type base_link_name: str
        :param base_link_name: Base frame to compute kinematics. Optional, defaults to "world"
        :type tip_link_name: str
        :param tip_link_name: The tip link to compute forward kinematics. Optional, defaults to last link.
        :rtype: general_robotics_toolbox.Transform
        :return: The pose of tip link in base frame
        """
        kin_group = self.tesseract_env.getKinematicGroup(self.robot_name)
        frames = kin_group.calcFwdKin(theta)

        if tip_link_name is None:
            tip_link_name = self.tip_link_name

        tip_link = frames[tip_link_name]

        if base_link_name is None:
            return isometry3d_to_transform(tip_link)
        else:
            base_link = frames[base_link_name]

            res = base_link.inverse() * tip_link
            return isometry3d_to_transform(res)

    def jacobian(self, theta, base_link_name = None, link_name = None):
        """
        Compute robot jacobian at specified joint angles.

        :type theta: np.array
        :param theta: The N vector of joint angles. Length must match number of joints. Expects radians or meters.
        :type base_link_name: str
        :param base_link_name: Base frame to compute jacobian. Optional, defaults to "world"
        :type tip_link_name: str
        :param tip_link_name: The tip link to compute jacobian. Optional, defaults to last link.
        :rtype: np.array
        :return: The 6x6 Jacobian array. Note that this array has angular velocity on the first three rows.
        """
        kin_group = self.tesseract_env.getKinematicGroup(self.robot_name)
        if base_link_name is None:
            base_link_name = self.base_link_name
        if link_name is None:
            link_name = self.tip_link_name

        J_kdl =  kin_group.calcJacobian(theta, base_link_name, link_name)

        # KDL has vel on top, SOA has ang on top
        J = np.vstack((J_kdl[3:6,:],J_kdl[0:3,:]))
        return J

    def invkin(self, tip_link_pose, theta_seed, base_link_name = None, tip_link_name = None):
        """
        Solve inverse kinematics of robot at specified tip link pose.

        :type tip_link_pose: general_robotics_toolbox.Transform
        :param tip_link_pose: The desired pose of the robot tip link
        :type theta_seet: np.array
        :param theta_seed: The N vector of seed joint angles. Used as initial position of robot joints for iterative
                           solvers. Length must match number of joints. Expects radians or meters.
        :type base_link_name: str
        :param base_link_name: Base frame to solve kinematics. Optional, defaults to "world"
        :type tip_link_name: str
        :param tip_link_name: The tip link to solve kinematics. Optional, defaults to last link.
        :rtype: List[np.array]
        :return: A list of joint angle candidate solutions, or empty if no solution possible
        """
        kin_group = self.tesseract_env.getKinematicGroup(self.robot_name)
        if base_link_name is None:
            base_link_name = self.base_link_name
        if tip_link_name is None:
            tip_link_name = self.tip_link_name
        ik = KinGroupIKInput()
        ik.pose = transform_to_isometry3d(tip_link_pose)
        ik.tip_link_name = tip_link_name
        ik.working_frame = base_link_name
        iks = KinGroupIKInputs()
        iks.append(ik)

        invkin1= kin_group.calcInvKin(iks,theta_seed)

        ret = []
        for i in range(len(invkin1)):
            ret.append(invkin1[i].flatten())
        return ret

    def redundant_solutions(self, theta):
        """
        Return "redundant" joint angle solutions. Some robot joints can spin more than 360 degrees, resulting in
        multiple redundant solutions with the joints rotated plus or minus 360 degrees. Return these potential
        solutions.

        :type theta: np.array
        :param theta: Robot joint angles
        :rtype: List[np.array]
        :return: List of redundant joint angles
        """
        kin_group = self.tesseract_env.getKinematicGroup(self.robot_name)
        return redundant_solutions(kin_group, theta)

class URInvKinParameters(NamedTuple):
    """
    UR inverse kinematics solver parameters. See robot_to_ur_inv_kin_parameters()
    """
    d1: float
    a2: float
    a3: float
    d4: float
    d5: float
    d6: float

def robot_to_ur_inv_kin_parameters(robot):
    """
    Convert "Robot" structure to Universal Robots DH parameters for inverse kinematics.
    Determines d1, a2, a3, d4, d5, d6

    See https://www.universal-robots.com/articles/ur/application-installation/dh-parameters-for-calculations-of-kinematics-and-dynamics/
    for more details.

    Robot must be aligned with +x or -x. Factory definition is along -x. URDF definition is along +x, rotated 180
    degrees from factory home position.
    """

    ex = np.array([1.,0.,0.])
    ey = np.array([0.,1.,0.])
    ez = np.array([0.,0.,1.])

    assert robot.T_flange
    assert np.allclose(robot.T_flange.R, np.array([ex, ez, -ey]).T, atol=1e-6)
    assert np.allclose(robot.T_flange.p.flatten(), np.array([0.,0.,0.]))

    assert robot.T_base
    assert np.allclose(robot.T_base.R, rox.rot([0,0,1], np.pi), atol=1e-6)
    assert np.allclose(robot.T_base.p.flatten(), np.array([0.,0.,0.]))

    minus_x = np.allclose(robot.H, np.array([ez,-ey,-ey,-ey,-ez,-ey]).T)
    plus_x = np.allclose(robot.H, -np.array([ez,-ey,-ey,-ey,-ez,-ey]).T)

    P2 = robot.P

    assert minus_x or plus_x

    if plus_x:
        R = rox.rot(ez,np.pi)
        P2 = np.matmul(R, P2)

    assert np.allclose(P2[0:2,0], 0.0)
    d1 = P2[2,0].item()

    assert np.allclose(P2[0,1], 0.0)
    assert np.allclose(P2[2,1:4], 0.0)
    a2 = P2[0,2].item()
    a3 = P2[0,3].item()
    assert np.allclose(P2[0,4:], 0.0)
    d4 = -np.sum(P2[1,1:5]).item()
    d5 = -np.sum(P2[2,4:5]).item()
    assert np.allclose(P2[2,5:], 0.0)
    assert np.allclose(P2[0,5:], 0.0)
    d6 = -np.sum(P2[1,5:6]).item()

    ret = URInvKinParameters(d1,a2,a3,d4,d5,d6)

    return ret

def kinematics_plugin_invkin_ur_plugin_info_dict(robot_name, base_link, tip_link, ur_params):
    """
    Create dictionary of yaml parameters for UR inverse kinematics.

    :type robot_name: str
    :param robot_name: The name of the robot
    :type base_link: str
    :param base_link: The name of the robot base link
    :type tip_link: str
    :param tip_link: The name of the robot tip link
    :type ur_params: URInvKinParameters
    :param default_solver: Structure containing UR parameters. See robot_to_ur_inv_kin_parameters()
    :rtype: dict
    :return: URInvKin plugin info as dict to convert to yaml
    """
    plugin_info = {
        "inv_kin_plugins": {
            robot_name: {
                "default": "URInvKin",
                "plugins": {
                    "URInvKin": {
                        "class": "URInvKinFactory",
                        "config": {
                            "base_link": base_link,
                            "tip_link": tip_link,
                            "params": {
                                "d1": ur_params.d1,
                                "a2": ur_params.a2,
                                "a3": ur_params.a3,
                                "d4": ur_params.d4,
                                "d5": ur_params.d5,
                                "d6": ur_params.d6
                            }
                        }
                    },
                }
            }
        }        
    }

    return plugin_info, ["tesseract_kinematics_ur_factories"]

def redundant_solutions(tesseract_kin_group, theta):
    """
    Return "redundant" joint angle solutions. Some robot joints can spin more than 360 degrees, resulting in
    multiple redundant solutions with the joints rotated plus or minus 360 degrees. Return these potential
    solutions.

    :type tesseract_kin_group: tesseract_kinematics.KinematicGroup
    :param tesseract_kin_group: Tesseract KinematicGroup instance
    :type theta: np.array
    :param theta: Robot joint angles
    :rtype: List[np.array]
    :return: List of redundant joint angles
    """
    limits = tesseract_kin_group.getLimits()
    redundancy_indices = list(tesseract_kin_group.getRedundancyCapableJointIndices())

    redun_sol = getRedundantSolutions(theta, limits.joint_limits, redundancy_indices)

    return [x.flatten() for x in redun_sol]
