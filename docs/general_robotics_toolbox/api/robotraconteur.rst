Robot Raconteur Info Parsers
=========================================

Robot Raconteur drivers use YAML info files to provide metadata to clients. These YAML files also typically
include physical parameters about the devices. For robots and tools, these files contain kinematic and dynamic
parameters. The utility functions in this module parse the YAML files and return :class:`general_robotics_toolbox.Robot`
structures populated with the parameters read from the YAML files.

general_robotics_toolbox.robotraconteur
---------------------------------------

.. automodule:: general_robotics_toolbox.robotraconteur
    :members:
