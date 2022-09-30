import sys

if sys.version_info < (3,6):
    raise Exception("Python version 3.6 or higher required for Tesseract")


import numpy as np
from . import general_robotics_toolbox as rox

from tesseract_robotics.tesseract_common import Translation3d, AngleAxisd, Isometry3d, vector_pair_string
from tesseract_robotics.tesseract_environment import Environment, Commands, \
    AddLinkCommand, AddKinematicsInformationCommand, AddSceneGraphCommand
from tesseract_robotics.tesseract_scene_graph import Link, Joint, JointLimits, \
    JointType_FIXED, JointType_REVOLUTE, JointType_PRISMATIC, SceneGraph
from tesseract_robotics.tesseract_common import FilesystemPath, ManipulatorInfo, KinematicsPluginInfo, \
    PluginInfoContainer
from tesseract_robotics.tesseract_kinematics import KinGroupIKInput, KinGroupIKInputs
from tesseract_robotics.tesseract_srdf import KinematicsInformation, parseKinematicsPluginConfigString
import yaml
import io
from typing import NamedTuple

def transform_to_isometry3d(T):
    # TODO: Use better Isometry3d constructor
    return Translation3d(T.p) * AngleAxisd(T.R)

def isometry3d_to_transform(eig_iso):
    H = eig_iso.matrix()
    R = H[0:3,0:3]
    p = H[0:3,3].flatten()
    return rox.Transform(R,p)

def get_link_and_joint(h, p, joint_type, joint_lower_limit, joint_upper_limit, joint_vel_limit,
    joint_acc_limit, joint_effort_limit, link_name, joint_name, parent_link_name):
    
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
    link = Link(link_name)
    joint = Joint(joint_name)

    joint.type = JointType_FIXED
    joint.parent_to_joint_origin_transform = transform_to_isometry3d(T)
    joint.parent_link_name = parent_link_name
    joint.child_link_name = link_name
    return link,joint

def get_robot_world_to_base_joint(robot, robot_name):
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
    sg = SceneGraph()
    sg.addLink(Link(world_link_name))
    return sg


def tesseract_kinematics_information(robot, robot_name, link_names, joint_names, chain_link_names, invkin_solver = None, 
    invkin_plugin_info = None, base_link = None, tip_link = None):
    
    if base_link is None:
        base_link = link_names[0]
    if tip_link is None:
        tip_link = link_names[-1]

    kinematics_information = KinematicsInformation()
    chain_group = vector_pair_string()
    chain_group.append((base_link, tip_link))
    kinematics_information.addChainGroup(robot_name, chain_group)
    
    plugin_info_str = kinematics_plugin_info_string(robot, robot_name, link_names, joint_names, chain_link_names,
        invkin_solver, invkin_plugin_info, base_link, tip_link)

    
    plugin_info = parseKinematicsPluginConfigString(plugin_info_str)

    kinematics_information.kinematics_plugin_info = plugin_info

    return kinematics_information


def robot_to_tesseract_env_commands(robot, robot_name = "robot", include_world = True, return_names = False, 
    invkin_solver = None, invkin_plugin_info = None):
        
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
        return commands, link_names, joint_names

def kinematics_plugin_fwdkin_kdl_plugin_info_dict(robot_name, base_link, tip_link):
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
    
    return plugin_info, ["tesseract_kinematics_kdl_factories"]

def kinematics_plugin_invkin_kdl_plugin_info_dict(robot_name, base_link, tip_link, default_solver = "KDLInvKinChainLMA"):
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
            invkin_plugin_info, invkin_libs = kinematics_plugin_invkin_opw_plugin_info_dict(robot_name, base_link,
                tip_link, opw_params)
        elif invkin_solver == "URInvKin":
            assert False, "URInvKin not implemented"
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

    plugin_info_dict = kinematics_plugin_info_dict(robot, robot_name, link_names, joint_names, chain_link_names,
        invkin_solver, invkin_plugin_info, base_link, tip_link)

    f = io.StringIO()
    yaml.safe_dump(plugin_info_dict, f)
    plugin_info_str = f.getvalue()

    return plugin_info_str

class OPWInvKinParameters(NamedTuple):
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

    assert len(robot.joint_type) == 6

    assert np.allclose(robot.H[:,0], ez) 
    assert np.allclose(robot.H[:,1], ey)
    assert np.allclose(robot.H[:,2], ey)
    assert np.allclose(robot.H[:,4], ey)
    assert np.allclose(robot.H[:,3], robot.H[:,5])
    assert np.allclose(robot.H[:,5], ez) or np.allclose(robot.H[:,5], ex)

    P2 = np.copy(robot.P)

    offset = np.zeros((6,))
    sign_corrections = np.ones((6,))

    if np.allclose(robot.H[:,5], ex):
        # Check if bend is at joint 2 or 3
        if np.abs(P2[0,2]) > np.abs(P2[2,2]):
            # x is greater than z for link 2, assume bend at joint 2
            R = rox.rot(ey, -np.pi/2.0)
            P2[:,2:] = np.matmul(R, P2[:,2:])
            offset[1] = -np.pi/2.0
        elif np.abs(P2[0,3] + P2[0,4]) > np.abs(P2[2,3] + P2[2,4]):
            # x is greater than z for link 3, assume bend at joint 3
            R = rox.rot(ey, -np.pi/2.0)
            P2[:,3:] = np.matmul(R, P2[:,3:])
            offset[2] = -np.pi/2.0
        else:
            # unclear where bend is, assume Joint 2
            R = rox.rot(ey, -np.pi/2.0)
            P2[:,2:] = np.matmul(R, P2[:,2:])
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
                                "sign_corrections": np.asarray(opw_params.sign_corrections,dtype=np.uint8).tolist()

                            }
                        }
                    },
                }
            }
        }        
    }

    return plugin_info, ["tesseract_kinematics_opw_factories"]

def robot_to_tesseract_env(robot, robot_name = "robot", include_world = True, 
    invkin_solver = None, invkin_plugin_info = None):

    tesseract_env_commands = robot_to_tesseract_env_commands(robot, robot_name, include_world, False, invkin_solver, 
        invkin_plugin_info)

    env = Environment()
    assert env.init(tesseract_env_commands)

    return env