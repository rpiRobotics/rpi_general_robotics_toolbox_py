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

import os
import numpy as np
from urdf_parser_py.urdf import URDF
import rospkg
import xacro
import general_robotics_toolbox as rox

def _resolve_file(fname, package = None):
    if package is not None:
        rospack = rospkg.RosPack()
        package_path=rospack.get_path(package)
        return os.path.join(package_path, fname)
    else:
        return fname

def _load_xacro_to_string(fname, package = None):
    
    fname2 = _resolve_file(fname,package)
    return xacro.process_file(fname2).toprettyxml(indent='  ')

def _load_xml_to_string(fname, package = None):
    fname2 = _resolve_file(fname,package)
    with open(fname2, 'r') as f:
        return f.read()

def _rpy_to_rot(rpy):
    return rox.rot([0,0,1],rpy[2]).dot(rox.rot([0,1,0],rpy[1]))\
        .dot(rox.rot([1,0,0],rpy[0]))
        
        
def robot_from_xml_string(xml_string, root_link = None, tip_link = None):
    """
    Loads a Robot class from a string containing a robot described in URDF XML format.
    The joints, dimensions, and various limits will be extracted from the URDF data. Inertial
    properties and other URDF concepts are not currently supported. Use the root_link and
    tip_link to specify the root and tip of a robot if more than one robot is in the file.
    
    :type  xml_string: str
    :param xml_string: The URDF XML data in a string
    :type  root_link: str
    :param root_link: The name of the root link (base link) of the robot. If None,
                      this will be determined automatically. Optional
    :type  tip_link: The name of the tip link (tool link) of the robot. If None,
                     this will be determined automatically. Must be specified
                     if more than one robot is specified in the URDF. Optional
    :rtype:  general_robotics_toolbox.Robot
    :return: The populated Robot class    
    """    
    
    urdf_robot = URDF.from_xml_string(xml_string)
    return _robot_from_urdf_robot(urdf_robot, root_link, tip_link)
    
def _robot_from_urdf_robot(urdf_robot, root_link = None, tip_link = None):
    
    if root_link is not None:
        assert root_link in urdf_robot.link_map, "Invalid root_link specified"
    else: 
        root_link = urdf_robot.get_root()
    tip_links = []
    
    valid_chains=[]
    
    for l1 in filter(lambda l: l not in urdf_robot.child_map, urdf_robot.link_map):
        try:
            chain = urdf_robot.get_chain(root_link, l1, True, False, True)
            if any(map(lambda c: urdf_robot.joint_map[c].joint_type == 'floating', chain)):
                continue
            if all(map(lambda c: urdf_robot.joint_map[c].joint_type == 'fixed', chain)):
                continue
            tip_links.append(l1)
            valid_chains.append(chain)
        except KeyError:
            pass
    
    if tip_link is None:    
        assert len(tip_links) == 1, "Multiple robots detected, specify tip link of desired robot"
        tip_link = tip_links[0]
                
    chain = urdf_robot.get_chain(root_link, tip_link, True, False, True)
    assert len(chain) > 0, "Invalid robot chain found"        
        
    n = len(filter(lambda c: urdf_robot.joint_map[c].joint_type != 'fixed', chain))
    P = np.zeros((3, n+1))
    H = np.zeros((3, n))
    joint_type = np.zeros(n)
    joint_lower_limit = [None]*n
    joint_upper_limit = [None]*n
    joint_vel_limit = [None]*n
    joint_effort_limit = [None]*n
    joint_names = [None]*n
    M = [None]*(n+1)
    
    i = 0
    
    R=np.identity(3)
    
    for c in chain:
        j = urdf_robot.joint_map[c]
        if j.origin is not None:
            if j.origin.xyz is not None:
                P[:,i] += R.dot(j.origin.xyz)
            if j.origin.rpy is not None:
                R = R.dot(_rpy_to_rot(j.origin.rpy))
        if urdf_robot.link_map[j.child].inertial is not None:
            M[i] = _append_inertia(M[i], _convert_inertial(urdf_robot.link_map[j.parent].inertial,R))
        if j.joint_type == 'fixed':
            pass            
        elif j.joint_type == 'revolute' or j.joint_type == 'prismatic':            
            H[:,i] = R.dot(j.axis)
            joint_type[i] = 0 if j.joint_type == 'revolute' else 1
            if j.limit is not None:
                joint_lower_limit[i] = j.limit.lower
                joint_upper_limit[i] = j.limit.upper
                joint_vel_limit[i] = j.limit.velocity
                joint_effort_limit[i] = j.limit.effort
            joint_names[i] = j.name
            i += 1            
        else:        
            assert False, "Only revolute, prismatic, fixed, and floating joints supported"
    
    M[-1] = _append_inertia(M[-1],_convert_inertial(urdf_robot.link_map[j.child].inertial,R))

    if None in joint_lower_limit or None in joint_upper_limit:
        joint_lower_limit = None
        joint_upper_limit = None
    else:
        joint_lower_limit = np.array(joint_lower_limit)
        joint_upper_limit = np.array(joint_upper_limit)
    
    if None in joint_vel_limit:
        joint_vel_limit = None
    else:
        joint_vel_limit = np.array(joint_vel_limit)
        
    if np.allclose(np.zeros((3,3)), R, atol=1e-5):
        R_tool = None
        p_tool = None
    else:
        R_tool = R
        p_tool = np.zeros((3,))             
    
    if None in M:
        M = None

    robot = rox.Robot(H, P, joint_type, joint_lower_limit, joint_upper_limit, \
                        joint_vel_limit, R_tool=R_tool, p_tool=p_tool, \
                        joint_names = joint_names, root_link_name = root_link, \
                        tip_link_name = tip_link, M = M )

    # Tack on joint_effort_limit if available
    if not None in joint_effort_limit:
        robot.joint_effort_limit = joint_effort_limit
    
    return robot

def _convert_inertial(urdf_inertial, R):
    if urdf_inertial is None:
        return None

    if (urdf_inertial.mass is None or urdf_inertial.mass <=1e-6):
        return None

    m = urdf_inertial.mass

    p_com = np.zeros((3,))
    R_com = np.eye(3)

    if (urdf_inertial.origin is not None):
        if (urdf_inertial.origin.xyz is not None):
            p_com[:] = urdf_inertial.origin.xyz
        if (urdf_inertial.origin.rpy is not None):
            R_com = _rpy_to_rot(urdf_inertial.origin.rpy)

    I = np.zeros((3,3))
    if (urdf_inertial.inertia is not None):
        inr = urdf_inertial.inertia
        I = np.matrix([[inr.ixx, inr.ixy, inr.ixz],
                      [inr.ixy, inr.iyy, inr.iyz],
                      [inr.ixz, inr.iyz, inr.izz]])

    # Rotate inertia matrix
    I = R_com.dot(I).dot(R_com.transpose())
    # Translate inertia matrix (parallel axis theorem)
    I += m * np.inner(p_com,p_com) * np.eye(3) - m * np.outer(p_com,p_com)

    M = np.block([[I, m*rox.hat(p_com)],
                  [-m*rox.hat(p_com), m*np.eye(3)]])

    return M



def _append_inertia(old_inertia,new_inertia):
    if (new_inertia is None):
        return old_inertia
    if (old_inertia is None):
        return new_inertia
    return np.sum(old_inertia, new_inertia)



def robot_from_xml_file(fname, package = None, root_link = None, tip_link = None):
    """
    Loads a Robot class from a file containing a robot described in URDF XML format.
    The joints, dimensions, and various limits will be extracted from the URDF data. Inertial
    properties and other URDF concepts are not currently supported. Use the root_link and
    tip_link to specify the root and tip of a robot if more than one robot is in the file.
    
    :type  fname: str
    :param fname: The filename to load. If using the package parameter, this should
                  be relative to the root of the package.
    :type  package: str
    :param package: The name of a ROS package containing the file. If specified, fname
                    will be relative to this package. Optional
    :type  root_link: str
    :param root_link: The name of the root link (base link) of the robot. If None,
                      this will be determined automatically. Optional
    :type  tip_link: The name of the tip link (tool link) of the robot. If None,
                     this will be determined automatically. Must be specified
                     if more than one robot is specified in the URDF. Optional
    :rtype:  general_robotics_toolbox.Robot
    :return: The populated Robot class    
    """
    
    xml_string = _load_xml_to_string(fname, package)
    return robot_from_xml_string(xml_string, root_link, tip_link)

def robot_from_xacro_file(fname, package = None, root_link = None, tip_link = None):
    """
    Loads a Robot class from a xacro file containing a robot described in URDF XML format.
    The xacro file will be executed into URDF XML format automatically.
    The joints, dimensions, and various limits will be extracted from the URDF data. Inertial
    properties and other URDF concepts are not currently supported. Use the root_link and
    tip_link to specify the root and tip of a robot if more than one robot is in the file.
    
    :type  fname: str
    :param fname: The filename to load. If using the package parameter, this should
                  be relative to the root of the package.
    :type  package: str
    :param package: The name of a ROS package containing the file. If specified, fname
                    will be relative to this package. Optional
    :type  root_link: str
    :param root_link: The name of the root link (base link) of the robot. If None,
                      this will be determined automatically. Optional
    :type  tip_link: The name of the tip link (tool link) of the robot. If None,
                     this will be determined automatically. Must be specified
                     if more than one robot is specified in the URDF. Optional
    :rtype:  general_robotics_toolbox.Robot
    :return: The populated Robot class    
    """
    xml_string = _load_xacro_to_string(fname, package)
    return robot_from_xml_string(xml_string, root_link, tip_link)

def robot_from_parameter_server(key='robot_description', root_link = None, tip_link = None):    
    """
    Loads a Robot class from the ROS parameter server. The joints, dimensions, and various 
    limits will be extracted from the URDF data. Inertial properties and other URDF concepts 
    are not currently supported. Use the root_link and tip_link to specify the root and tip 
    of a robot if more than one robot is in the file.
    
    :type  key: str
    :param key: The ROS parameter server key. Defaults to 'robot_description'. Optional
    :type  root_link: str
    :param root_link: The name of the root link (base link) of the robot. If None,
                      this will be determined automatically. Optional
    :type  tip_link: The name of the tip link (tool link) of the robot. If None,
                     this will be determined automatically. Must be specified
                     if more than one robot is specified in the URDF. Optional
    :rtype:  general_robotics_toolbox.Robot
    :return: The populated Robot class    
    """
       
    urdf_robot = URDF.from_parameter_server(key)
    return _robot_from_urdf_robot(urdf_robot, root_link, tip_link)   
    
    
