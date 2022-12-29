# general_robotics_toolbox

[![](https://img.shields.io/badge/python-2.7|3.5+-blue.svg)](https://github.com/rpiRobotics/rpi_general_robotics_toolbox_py)
[![](https://img.shields.io/pypi/v/general-robotics-toolbox)](https://pypi.org/project/general-robotics-toolbox/)

The `general_robotics_toolbox` package provides a toolbox of Python functions for geometry, forward kinematics, inverse kinematics, and dynamics of robots. The functions are based on "A Mathematical Introduction to Robotic Manipulation" by Richard Murray, Zexiang Li, and S. Shankar Sastry (1994), "A spatial operator algebra for manipulator modeling and control" by G. Rodriguez, A. Jain, and K. Kreutz-Delgad, and lecture notes by Dr. John Wen, Rensselaer Polytechnic Institute.

Documentation can be found at: https://general-robotics-toolbox.readthedocs.io/

License: BSD

## Installation

`general-robotics-toolbox` is avaliable on PyPi. To install with all features available, use:

```
pip install general-robotics-toolbox[urdf,tesseract]
```

For minimal installation with just the base Python kinematic functions, use:

```
pip install general-robotics-toolbox
```

## Features

The base `general_robotics_toolbox` module provides the following:

* Rotation: `hat`, `invhat`, `rot`, `R2rot`, `screw_matrix`, `q2r`, `R2q`, `q2rot`, `rot2q`, `quatcomplement`,  
  `quatproduct`, `quatjacobian`, `rpy2R`, `R2rpy`
* Transform: `Transform` class
* Robot Parameters: `Robot` class
* Forward Kinematics: `fwdkin`, `robotjacobian`
* Canonical Geometric Subproblems: `subproblems 0 - 3`
* Inverse Kinematics: `robot6_sphericalwrist_invkin` (OPW), `ur_invkin`, `iterative_invkin`, `equivalent_configurations`

### URDF Parser

The `general_robotics_toolbox.urdf` module provides functions to parse URDF files into `Robot` structures.

### Tesseract Robotics Acceleration

The `general_robotics_toolbox.tesseract` module uses the Tesseract Robot Planning Framework to accelerate
kinematics functions.

### Robot Raconteur Info Parsers

The `general_robotics_toolbox.robotraconteur` module provides functions to parse Robot Raconteur device info YAML
files into `Robot` structures.

### ROS 1 Utilities

The `general_robotics_toolbox.ros_msg` and `general_robotics_toolbox.ros_tf` modules provide utility functions to 
convert ROS 1 messages and listen for TF2 messages.

## Acknowledgment

This work was supported in part by Subaward No. ARM-17-QS-F-01, ARM-TEC-18-01-F-19, ARM-TEC-19-01-F-24, and ARM-TEC-21-02-F19 from the Advanced Robotics for Manufacturing ("ARM") Institute under Agreement Number W911NF-17-3-0004 sponsored by the Office of the Secretary of Defense. ARM Project Management was provided by Christopher Adams. The views and conclusions contained in this document are those of the authors and should not be interpreted as representing the official policies, either expressed or implied, of either ARM or the Office of the Secretary of Defense of the U.S. Government. The U.S. Government is authorized to reproduce and distribute reprints for Government purposes, notwithstanding any copyright notation herein.

This work was supported in part by the New York State Empire State Development Division of Science, Technology and Innovation (NYSTAR) under contract C160142. 

![](docs/figures/arm_logo.jpg) ![](docs/figures/nys_logo.jpg)