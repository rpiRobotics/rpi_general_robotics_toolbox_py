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
# POSSIBILITY OF SUCH DAMAGE.

from __future__ import absolute_import

from . import general_robotics_toolbox as rox
import tf
import rospy

#Import exceptions used by ROS tf module
from tf2_ros import TransformException as Exception, ConnectivityException, LookupException, ExtrapolationException

class TransformListener(object):
    def __init__(self, *args, **kwargs):
        self.ros_listener=tf.TransformListener(*args, **kwargs)
        
    def canTransform(self, target_frame, source_frame, time = rospy.Time(0)):
        return self.ros_listener.canTransform(target_frame,source_frame, time)

    def canTransformFull(self, target_frame, target_time, source_frame, source_time, fixed_frame):
        return self.ros_listener.canTransformFull(target_frame, target_time, source_frame, source_time, fixed_frame)
    
    def waitForTransform(self, target_frame, source_frame, time, timeout, polling_sleep_duration=None):
        return self.ros_listener.waitForTransform(target_frame, source_frame, time, timeout, polling_sleep_duration)
        
    def waitForTransformFull(self, target_frame, target_time, source_frame, source_time, fixed_frame, timeout, polling_sleep_duration=None):
        return self.ros_listener.waitForTransformFull(target_frame, target_time, source_frame, source_time, fixed_frame, timeout, polling_sleep_duration)
        
    def clear(self):
        self.ros_listener.clear()
        
    def lookupTransform(self, target_frame, source_frame, time = rospy.Time(0)):
        t,r = self.ros_listener.lookupTransform(target_frame, source_frame, time)
        q=[r[3], r[0], r[1], r[2]]
        return rox.Transform(rox.q2R(q), t, target_frame, source_frame)
     
    def lookupTransformFull(self, target_frame, target_time, source_frame, source_time, fixed_frame):
        t,r = self.ros_listener.lookupTransformFull(target_frame, target_time, source_frame, source_time, fixed_frame)
        q=[r[3], r[0], r[1], r[2]]
        return rox.Transform(rox.q2R(q), t, target_frame, source_frame)
        
    