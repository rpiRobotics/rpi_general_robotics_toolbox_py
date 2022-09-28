import numpy as np
from . import general_robotics_toolbox as rox

from tesseract_robotics.tesseract_common import Translation3d, AngleAxisd, Isometry3d
from tesseract_robotics.tesseract_environment import Environment, Commands, \
    AddLinkCommand, AddKinematicsInformationCommand, AddSceneGraphCommand
from tesseract_robotics.tesseract_scene_graph import Link, Joint, JointLimits, \
    JointType_FIXED, JointType_REVOLUTE, JointType_PRISMATIC, SceneGraph
from tesseract_robotics.tesseract_common import FilesystemPath, ManipulatorInfo, KinematicsPluginInfo, \
    PluginInfoContainer
from tesseract_robotics.tesseract_kinematics import KinGroupIKInput, KinGroupIKInputs
from tesseract_robotics.tesseract_srdf import KinematicsInformation

def _transform_to_isometry3d(T):
    # TODO: Use better Isometry3d constructor
    return Translation3d(T.p) * AngleAxisd(T.R)

def _get_link_and_joint(h, p, joint_type, joint_lower_limit, joint_upper_limit, joint_vel_limit,
    joint_acc_limit, joint_effort_limit, link_name, joint_name, parent_link_name):
    
    link = Link(link_name)
    joint = Joint(joint_name)

    if joint_type == 0:      
        joint.type = JointType_REVOLUTE
    elif joint_type == 1:
        joint.type = JointType_PRISMATIC
    else:
        assert False, "Unsupported Tesseract joint type: " + str(joint.type)
    joint.parent_to_joint_origin_transform = _transform_to_isometry3d(rox.Transform(np.eye(3), p))
    joint.axis = h
    joint.parent_link_name = parent_link_name
    joint.child_link_name = link_name
    joint.limits = JointLimits(joint_lower_limit, joint_upper_limit, joint_effort_limit,
         joint_vel_limit, joint_acc_limit)
    return link,joint

def _get_fixed_link_and_joint(T, link_name, joint_name, parent_link_name):
    link = Link(link_name)
    joint = Joint(joint_name)

    joint.type = JointType_FIXED
    joint.parent_to_joint_origin_transform = _transform_to_isometry3d(T)
    joint.parent_link_name = parent_link_name
    joint.child_link_name = link_name
    return link,joint

def _get_robot_world_to_base_joint(robot, robot_name):
    joint = Joint(f"world_to_{robot_name}")

    joint.type = JointType_FIXED
    if robot.T_base:
        joint.parent_to_joint_origin_transform = _transform_to_isometry3d(robot.T_base)
    else:
        joint.parent_to_joint_origin_transform = Isometry3d()
    joint.parent_link_name = "world"
    joint.child_link_name = f"{robot_name}_base_link"
    return joint

def robot_to_scene_graph(robot, return_names = False, invkin_solver = ""):
    sg = SceneGraph()

    sg_link_names = ["base_link"]

    assert sg.addLink(Link("base_link"))
    n_joints = len(robot.joint_type)

    if robot.joint_names is None:
        joint_names = [f"joint_{i+1}" for i in range(n_joints)]
    else:
        joint_names = robot.joint_names

    # if robot.link_names is None:
    link_names = [f"link_{i}" for i in range(n_joints+1)]
    # else:
        # link_names = [_p(s) for s in robot.link_names]

    assert len(joint_names) == n_joints

    for i in range(n_joints):
        if i == 0:
            parent_link_name = "base_link"
        else:
            parent_link_name = link_names[i-1]
        
        assert sg.addLink(*_get_link_and_joint(robot.H[:,i], robot.P[:,i], robot.joint_type[i], 
            robot.joint_lower_limit[i], robot.joint_upper_limit[i], robot.joint_vel_limit[i], 
            robot.joint_acc_limit[i], 1000.0, link_names[i], joint_names[i],
            parent_link_name))
        sg_link_names.append(link_names[i])
    
    assert sg.addLink(*_get_fixed_link_and_joint(rox.Transform(np.eye(3), robot.P[:,n_joints-1]), 
        link_names[n_joints], link_names[n_joints] + "_chain_tip", link_names[n_joints-1]))
    sg_link_names.append(link_names[n_joints])

    tip_link = link_names[n_joints]

    flange_link_name = "flange"

    if robot.T_flange is not None:
        assert sg.addLink(*_get_fixed_link_and_joint(robot.T_flange,
            flange_link_name, "tip_to_flange", tip_link))

        tip_link = flange_link_name
        sg_link_names.append(flange_link_name)

    tool_link_name = "tool_tcp"

    if robot.R_tool is not None and robot.p_tool is not None:
        assert sg.addLink(*_get_fixed_link_and_joint(rox.Transform(robot.R_tool, robot.p_tool),
            tool_link_name, "tip_to_tool", tip_link))

        tip_link = tool_link_name
        sg_link_names.append(tool_link_name)
    
    if not return_names:
        return sg
    else:
        return sg, sg_link_names, joint_names

def world_scene_graph(world_link_name = "world"):
    sg = SceneGraph()
    sg.addLink(Link(world_link_name))
    return sg


def robot_to_tesseract_env_commands(robot, robot_name = "robot", include_world = True, return_names = False):
        
    commands = Commands()
    
    if include_world:
        world_sg = world_scene_graph()
        commands.append(AddSceneGraphCommand(world_sg))

    robot_sg, sg_link_names, sg_joint_names = robot_to_scene_graph(robot, return_names = True)
    robot_base_joint = _get_robot_world_to_base_joint(robot, robot_name)
    commands.append(AddSceneGraphCommand(robot_sg, robot_base_joint, f"{robot_name}_"))

    link_names = [f"{robot_name}_{s}" for s in sg_link_names]
    joint_names = [f"{robot_name}_{s}" for s in sg_joint_names]
    
    # if robot.joint_names is None:
    #     joint_names = [f"{robot_name}_joint_{i+1}" for i in range(len(robot.joint_type))]
    # else:
    #     joint_names = [f"{robot_name}_{s}" for s in robot.joint_names]
    
    kinematics_information = KinematicsInformation()
    kinematics_information.addJointGroup(robot_name, joint_names)
    commands.append(AddKinematicsInformationCommand(kinematics_information))

    if not return_names:
        return commands
    else:
        return commands, link_names, joint_names

def kinematics_plugin_fwdkin_kdl_command(plugin_factory, robot_name, base_link, tip_link):
    plugin_config = {
        "kinematic_plugins": {
            "search_libraries": [
                "tesseract_kinematics_kdl_factories"
            ],
            "fwk_kin_plugins": {
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
    }

    PluginInfoContainer()

    plugin_info = KinematicsPluginInfo()

 
