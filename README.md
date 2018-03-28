# rpi_general_robotics_toolbox_py
This package provides a toolbox of Python functions for geometry, forward kinematics, inverse kinematics, and dynamics of robots. The functions are based on "A Mathematical Introduction to Robotic Manipulation" by Richard Murray, Zexiang Li, and S. Shankar Sastry (1994), "A spatial operator algebra for manipulator modeling and control" by G. Rodriguez, A. Jain, and K. Kreutz-Delgad, and lecture notes by Dr. John Wen, Rensselaer Polytechnic Institute.

The following operations are currently implemented:

* Rotation and Pose: hat, rot, q2r, R2q, quatcomplement, quatproduct, quatjacobian, Pose class, screw_matrix
* Robot Parameters: Robot class
* Forward Kinematics: fwdkin, robotjacobian
* Paden-Kahan geometry subproblems: subproblems 0 - 3

License: BSD
