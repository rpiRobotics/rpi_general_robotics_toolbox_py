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

from __future__ import absolute_import

import yaml
from . import general_robotics_toolbox as rox
import numpy as np

"""Provides utilities to read Robot Raconteur Yaml files containing robot and tool definition.
   See RobotRaconteurCompanion.Util classes for conversions between Robot Raconteur and robotics toolbox types."""


def _check_list(l, error_msg, expected_count=-1):
    if l is None:
        raise ValueError(error_msg)

    if expected_count >= 0:
        if len(l) != expected_count:
            raise ValueError(error_msg)


def _identifier_name(identifier):
    if isinstance(identifier, str):
        return identifier
    else:
        return identifier["name"]


def _to_vec3(d):
    return np.array([d["x"], d["y"], d["z"]], dtype=np.float64)


def _to_qvec(d):
    return np.array([d["w"], d["x"], d["y"], d["z"]], dtype=np.float64)


def _to_transform(d):
    if "orientation" in d:
        q = _to_qvec(d["orientation"])
        p = _to_vec3(d["position"])
    else:
        q = _to_qvec(d["rotation"])
        p = _to_vec3(d["translation"])

    return rox.Transform(rox.q2R(q), p)


def load_robot_info_yaml_to_robot(robot_info_file, chain_number=0):
    """
    Parse a YAML robot info file and return a populated Robot structure. This function will also include the
    ``current_tool`` entry if present.

    :param robot_info_file: The robot info file to parse
    :type robot_info_file: TextIO | dict
    :param chain_number: The index of the chain to parse. For single robots, use default 0
    :type chain_number: int    
    """
    if isinstance(robot_info_file, dict):
        robot_yml = robot_info_file
    else:
        robot_yml = yaml.safe_load(robot_info_file)
    _check_list(
        robot_yml["chains"], "could not find kinematic chain number " + str(chain_number))
    if chain_number >= len(robot_yml["chains"]):
        raise ValueError("invalid kinematic chain number " + str(chain_number))

    chain = robot_yml["chains"][chain_number]
    joint_count = len(chain["joint_numbers"])
    for i in range(1, joint_count):
        if chain["joint_numbers"][i-1] >= chain["joint_numbers"][i]:
            raise ValueError(
                "joint numbers must be increasing in chain number " + str(chain_number))

        if chain["joint_numbers"][i] >= len(robot_yml["joint_info"]):
            raise ValueError(
                "joint number out of bounds in chain number " + str(chain_number))

    _check_list(chain["H"], "invalid shape for H in chain number " +
                str(chain_number), joint_count)
    _check_list(chain["P"], "invalid shape for P in chain number " +
                str(chain_number), joint_count + 1)

    H = np.zeros((3, joint_count), dtype=np.float64)
    for i in range(joint_count):
        H[:, i] = _to_vec3(chain["H"][i])

    P = np.zeros((3, joint_count + 1), dtype=np.float64)
    for i in range(joint_count+1):
        P[:, i] = _to_vec3(chain["P"][i])

    joint_type = [0]*joint_count
    joint_lower_limit = np.zeros((joint_count,), dtype=np.float64)
    joint_upper_limit = np.zeros((joint_count,), dtype=np.float64)
    joint_vel_limit = np.zeros((joint_count,), dtype=np.float64)
    joint_acc_limit = np.zeros((joint_count,), dtype=np.float64)
    joint_names = [None]*joint_count

    for i in range(joint_count):
        j = robot_yml["joint_info"][i]
        if j["joint_type"] == "revolute":
            # Revolute joint
            joint_type[i] = 0
        elif j["joint_type"] == "prismatic":
            # Prismatic joint
            joint_type[i] = 1
        else:
            raise ValueError("invalid joint type: " + str(j['joint_type']))

        if j["joint_limits"] is None:
            raise ValueError("joint_limits must not be null")
        joint_lower_limit[i] = j["joint_limits"]["lower"]
        joint_upper_limit[i] = j["joint_limits"]["upper"]
        joint_vel_limit[i] = j["joint_limits"]["velocity"]
        joint_acc_limit[i] = j["joint_limits"]["acceleration"]
        if j["joint_identifier"] is not None:
            joint_names[i] = _identifier_name(j["joint_identifier"])
        else:
            joint_names[i] = ""

    root_link_name = None
    if "link_identifiers" in chain and len(chain["link_identifiers"]) > 0 and chain["link_identifiers"][0] is not None:
        root_link_name = _identifier_name(chain["link_identifiers"][0])

    tip_link_name = None
    if chain["flange_identifier"] is not None:
        tip_link_name = _identifier_name(chain["flange_identifier"])

    T_flange = _to_transform(chain["flange_pose"])

    r_tool = None
    p_tool = None

    current_tool = chain.get("current_tool", None)
    if current_tool is not None:
        tcp = current_tool.get("tcp")
        if tcp is not None:
            T_tool = _to_transform(tcp)
            r_tool = T_tool.R
            p_tool = T_tool.p
            tool_device_info = current_tool.get("device_info")
            if tool_device_info is not None:
                tip_link_name = _identifier_name(tool_device_info["device"])

    T_base = None
    robot_device_info = robot_yml.get("device_info", None)
    if robot_device_info is not None:
        robot_origin_pose = robot_device_info.get("device_origin_pose", None)
        if robot_origin_pose is not None:
            T_base = _to_transform(robot_origin_pose["pose"])

    rox_robot = rox.Robot(H, P, joint_type, joint_lower_limit, joint_upper_limit, joint_vel_limit,
                          joint_acc_limit, None, r_tool, p_tool, joint_names, root_link_name, tip_link_name,  
                          T_flange=T_flange, T_base = T_base)

    return rox_robot


def load_robot_and_tool_info_yaml_to_robot(robot_info_file, tool_info_file, robot_chain_number=0):
    """
    Parse a YAML robot info file and YAML tool info file, and return a populated Robot structure with the
    tool attached.

    :param robot_info_file: The robot info file to parse
    :type robot_info_file: TextIO | dict
    :param tool_info_file: The tool info file to parse
    :type tool_info_file: TextIO | dict
    :param chain_number: The index of the chain to parse. For single robots, use default 0
    :type chain_number: int    
    """
    robot = load_robot_info_yaml_to_robot(robot_info_file, robot_chain_number)
    if isinstance(tool_info_file, dict):
        tool_yml = tool_info_file
    else:
        tool_yml = yaml.safe_load(tool_info_file)

    tcp = tool_yml["tcp"]

    T_tool = _to_transform(tcp)
    r_tool = T_tool.R
    p_tool = T_tool.p
    tool_device_info = tool_yml["device_info"]
    tip_link_name = _identifier_name(tool_device_info["device"])

    robot.R_tool = r_tool
    robot.p_tool = p_tool
    robot.tip_link_name = tip_link_name

    return robot
