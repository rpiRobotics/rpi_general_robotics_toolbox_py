#!/usr/bin/env python

import unittest
import general_robotics_toolbox as rox
import numpy as np

#Numeric precision reduced for literals
eps = 1e-6 #np.finfo(np.float64).eps

#inches to meters conversion factor
#(use Pint package for any real programs)
in_2_m = 0.0254

class Test_hat(unittest.TestCase):
   
    def runTest(self):        
        k=[1,2,3]
        k_hat=rox.hat(k)
        k_hat_t=np.array([[0, -3, 2], [3, 0, -1], [-2, 1,0]])
        np.testing.assert_allclose(k_hat, k_hat_t)

class Test_rot(unittest.TestCase):
    
    def _rot_test(self, k, theta, rot_t):
        rot=rox.rot(k, theta)
        np.testing.assert_allclose(rot, rot_t, atol=1e-5)
    
    def runTest(self):
        rot_1_t=np.array([[1,0,0], [0,0,1], [0,-1,0]]).T
        self._rot_test(np.array([1,0,0]), np.pi/2.0, rot_1_t)
        
        rot_2_t=np.array([[0,0,-1], [0,1,0], [1,0,0]]).T
        self._rot_test(np.array([0,1,0]), np.pi/2.0, rot_2_t)
        
        rot_3_t=np.array([[0,1,0], [-1,0,0], [0,0,1]]).T
        self._rot_test(np.array([0,0,1]), np.pi/2.0, rot_3_t)
        
        #Random rotation
        rot_4=np.array([[-0.5057639, -0.1340537,0.8521928], \
                        [0.6456962,-0.7139224,0.2709081], \
                        [0.5720833,0.6872731,0.4476342]])        
        self._rot_test(np.array([0.4490221,0.30207945,0.84090853]), 2.65949884, rot_4)
        
        
class Test_screwmatrix(unittest.TestCase):
    
    def runTest(self):
        k=[1, 2, 3]
        G=rox.screw_matrix(k)
        G_t=np.array([[ 1,  0,  0,  0, -3,  2], \
             [ 0,  1,  0,  3,  0, -1], \
             [ 0,  0,  1, -2,  1,  0], \
             [ 0,  0,  0,  1,  0,  0], \
             [ 0,  0,  0,  0,  1,  0], \
             [ 0,  0,  0,  0,  0,  1,]])
        
        np.testing.assert_allclose(G, G_t)
        
class Test_R2q(unittest.TestCase):
    
    def runTest(self):
        rot=np.array([[-0.5057639,-0.1340537,0.8521928], \
                      [0.6456962,-0.7139224,0.2709081], \
                      [0.5720833,0.6872731,0.4476342]])
        
        q_t=np.array([0.2387194, 0.4360402, 0.2933459, 0.8165967])
        q=rox.R2q(rot)
        np.testing.assert_allclose(q_t, q, atol=1e-6)
     
class Test_q2R(unittest.TestCase):
    
    def runTest(self):
        rot_t=np.array([[-0.5057639,-0.1340537,0.8521928], \
                      [0.6456962,-0.7139224,0.2709081], \
                      [0.5720833,0.6872731,0.4476342]])
        
        q=np.array([0.2387194, 0.4360402, 0.2933459, 0.8165967])
        rot=rox.q2R(q)
        np.testing.assert_allclose(rot, rot_t, atol=1e-6)

class Test_quatcomplement(unittest.TestCase):
    
    def runTest(self):
        
        q=np.array([[ 0.2387194, 0.4360402, 0.2933459, 0.8165967]]).T
        q_c=rox.quatcomplement(q)
        np.testing.assert_allclose(q[0], q_c[0])
        np.testing.assert_allclose(q[1:3], -q_c[1:3])

class Test_quatproduct(unittest.TestCase):
    
    def runTest(self):
        
        q_1=np.array([0.63867877, 0.52251797, 0.56156573, 0.06089615])
        q_2=np.array([0.35764716, 0.61051424, 0.11540801, 0.69716703])
        R_t=rox.q2R(q_1).dot(rox.q2R(q_2))
        q_t=rox.R2q(R_t)
        
        q = rox.quatproduct(q_1).dot(q_2).reshape(4,1)
                        
        np.testing.assert_allclose(q, q_t, atol=1e-6)
        
class Test_quatjacobian(unittest.TestCase):
    
    def runTest(self):
        #TODO: test against better control case
        q=np.array([0.63867877, 0.52251797, 0.56156573, 0.06089615])
        J=rox.quatjacobian(q)
        J_t=np.array([[-0.26125898, -0.28078286, -0.03044808], \
             [ 0.31933938,  0.03044808, -0.28078286], \
             [-0.03044808,  0.31933938,  0.26125898], \
             [ 0.28078286, -0.26125898,  0.31933938]])
        
        np.testing.assert_allclose(J, J_t, atol=1e-6)
        
        
class Test_fwdkin(unittest.TestCase):
    
    def runTest(self):
        
        #TODO: other joint types
        
        #Home configuration (See Page 2-2 of Puma 260 manual)
        puma=puma260b_robot()
        pose=rox.fwdkin(puma, np.zeros(6))
        np.testing.assert_allclose(pose.R, np.identity(3))
        np.testing.assert_allclose(pose.p, np.array([10,-4.9,4.25])*in_2_m, atol=1e-6)
        
        #Another right-angle configuration
        joints2=np.array([180,-90,-90, 90, 90, 90])*np.pi/180.0
        pose2=rox.fwdkin(puma, joints2)
        np.testing.assert_allclose(pose2.R, rox.rot([0,0,1],np.pi).dot(rox.rot([0,1,0], -np.pi/2)), atol=1e-6)
        np.testing.assert_allclose(pose2.p, np.array([-0.75, 4.9, 31])*in_2_m)
        
        #Random configuration
        joints3=np.array([50, -105, 31, 4, 126, -184])*np.pi/180
        pose3=rox.fwdkin(puma,joints3)
        
        pose3_R_t=np.array([[0.4274, 0.8069, -0.4076],\
                            [0.4455, -0.5804,-0.6817], \
                            [-0.7866, 0.1097, -0.6076]])
        pose3_p_t=[0.2236, 0.0693, 0.4265]
        
        np.testing.assert_allclose(pose3.R, pose3_R_t, atol=1e-4)
        np.testing.assert_allclose(pose3.p, pose3_p_t, atol=1e-4)
        
        puma_tool=puma260b_robot_tool()
        
        pose4=rox.fwdkin(puma_tool, joints3)
        pose4_R_t=np.array([[0.4076, 0.8069, 0.4274],\
                            [0.6817, -0.5804,0.4455], \
                            [0.6076, 0.1097, -0.7866]])
        
        pose4_p_t=[0.2450, 0.0916, 0.3872]
        
        np.testing.assert_allclose(pose4.R, pose4_R_t, atol=1e-4)
        np.testing.assert_allclose(pose4.p, pose4_p_t, atol=1e-4)
        
class Test_robotjacobian(unittest.TestCase):
    
    def runTest(self):
        #Home configuration (See Page 2-2 of Puma 260 manual)
        puma=puma260b_robot()        
        J=rox.robotjacobian(puma, np.zeros(6))
        np.testing.assert_allclose(J[0:3,:], puma.H, atol=1e-4)
        J_t_v=np.array([[4.9,10,0],[-8.75,0,-10],[-8,0,-2.2], \
                        [0,2.2,0],[0,0,-2.2],[0,0,0]]).T*in_2_m
        np.testing.assert_allclose(J[3:6,:], J_t_v, atol=1e-4)


        #Another right-angle configuration
        joints2=np.array([180,-90,-90, 90, 90, 90])*np.pi/180.0
        J2=rox.robotjacobian(puma, joints2)
        J2_t=np.array([[0,0,0,0,-1,0],                  \
                       [0,-1,-1,0,0,0],                 \
                       [1,0,0,-1,-0,1],                 \
                       [-0.1245,-0.4572,-0.2591,0,0,0], \
                       [-0.0191,0,0,0,0.0559,0],        \
                       [0,-0.0191,0,0,0,0,]])       
        np.testing.assert_allclose(J2, J2_t, atol=1e-4)
        
        #Random configuration
        joints3=np.array([50, -105, 31, 4, 126, -184])*np.pi/180
        J3=rox.robotjacobian(puma, joints3)
        J3_t=np.array([[0, -0.766, -0.766,-0.6179, -0.7765, 0.4274], \
                       [0, 0.6428, 0.6428, -0.7364, 0.6265, 0.4456], \
                       [1, 0, 0, 0.2756, -0.0671, -0.7866], \
                       [-0.0693, 0.0619, -0.0643, 0.0255, -0.0259, 0], \
                       [0.2236, 0.0738, -0.0766, -0.0206, -0.0357,  0], \
                       [0, -0.1969, -0.2298, 0.0022, -0.0343,  0]])
        
        np.testing.assert_allclose(J3, J3_t, atol=1e-4)

class Test_subproblems(unittest.TestCase):
    
    def runTest(self):
        x=[1,0,0]
        y=[0,1,0]
        z=[0,0,1]
        
        #subproblem0
        assert(rox.subproblem0(x,y,z) == np.pi/2)


        #subproblem1
        k1=(np.add(x,z))/np.linalg.norm(np.add(x,z))
        k2=(np.add(y,z))/np.linalg.norm(np.add(y,z))
        
        assert(rox.subproblem1(k1, k2, z) == np.pi/2)
        
        #subproblem2
        p2=x
        q2=np.add(x, np.add(y,z))
        q2=q2/np.linalg.norm(q2)
        
        a2 = rox.subproblem2(p2, q2, z, y)
        assert len(a2) == 2
        
        
        r1=rox.rot(z,a2[0][0]).dot(rox.rot(y,a2[0][1]))[:,0]
        r2=rox.rot(z,a2[1][0]).dot(rox.rot(y,a2[1][1]))[:,0]
        
        np.testing.assert_allclose(r1, q2, atol=1e-4)
        np.testing.assert_allclose(r2, q2, atol=1e-4)
        
        a3 = rox.subproblem2(x, z, z, y)
        assert len(a3) == 1
        
        r3=rox.rot(z,a3[0][0]).dot(rox.rot(y,a3[0][1]))[:,0]        
        np.testing.assert_allclose(r3, z, atol=1e-4)
        
        #subproblem3
        p4=[.5, 0, 0]
        q4=[0, .75, 0]
        
        a4=rox.subproblem3(p4, q4, z, .5) 
        a5=rox.subproblem3(p4, q4, z, 1.25)
        
        assert len(a4) == 2
        np.testing.assert_allclose(np.linalg.norm(np.add(q4, rox.rot(z, a4[0]).dot(p4))),0.5)
        np.testing.assert_allclose(np.linalg.norm(np.add(q4, rox.rot(z, a4[1]).dot(p4))),0.5)
        
        assert len(a5) == 1
        np.testing.assert_allclose(np.linalg.norm(np.add(q4, rox.rot(z, a5[0]).dot(p4))),1.25)
        
        #subproblem4
        
        p6=y
        q6=[.8, .2, .5]
        d6=.3
        
        a6=rox.subproblem4(p6, q6, z, d6)
                
        np.testing.assert_allclose(np.dot(p6, rox.rot(z,a6[0]).dot(q6)), d6, atol=1e-4)
        np.testing.assert_allclose(np.dot(p6, rox.rot(z,a6[1]).dot(q6)), d6, atol=1e-4)
                
class Test_robot6_sphericalwrist_invkin(unittest.TestCase):
    
    def runTest(self):
        
        robot1=puma260b_robot()
        robot2=abb_irb6640_180_255_robot()
        robot3=puma260b_robot_tool()
        
        def _test_configuration(robot, theta):
            
            pose_1 = rox.fwdkin(robot, theta)
            theta2 = rox.robot6_sphericalwrist_invkin(robot, pose_1)
            
            if not len(theta2) > 0:
                return False
            for theta2_i in theta2:
                pose_2 = rox.fwdkin(robot, theta2_i)
                if not pose_1 == pose_2:
                    return False
            return True
        
        def _test_last_configuration(robot, theta, last_theta):
            
            pose_1 = rox.fwdkin(robot, theta)
            theta2 = rox.robot6_sphericalwrist_invkin(robot, pose_1, last_theta)
            pose_2 = rox.fwdkin(robot, theta2[0])
            if not pose_1 == pose_2:
                return False
            if not np.allclose(theta2[0], last_theta, atol=np.deg2rad(10)):
                return False
            return True
            
        
        assert _test_configuration(robot1, np.zeros(6))
        #Previous failed value, add to unit test
        assert _test_configuration(robot2, [-0.09550528, -0.43532822, -2.35369965, -2.42324955, -1.83659391, -4.00786639])    
        
        for robot in (robot1,robot2,robot3):
            for _ in xrange(100):
                theta = np.random.rand(6)*(robot.joint_upper_limit - robot.joint_lower_limit) \
                   + robot.joint_lower_limit
                                                    
                assert _test_configuration(robot, theta)
          
        theta_test1 = np.zeros(6)
        assert _test_last_configuration(robot1, theta_test1, theta_test1 + np.deg2rad(4))
        assert _test_last_configuration(robot2, theta_test1 - np.deg2rad(4), np.array([0,0,0,0,0,np.pi*2]))
                        
        for robot in (robot1,robot2,robot3):
            for _ in xrange(100):
                theta = np.random.rand(6)*(robot.joint_upper_limit - robot.joint_lower_limit - np.deg2rad(30)) \
                   + robot.joint_lower_limit + np.deg2rad(15)
                                                    
                last_theta = theta + (np.random.rand(6)-0.5)*2*np.deg2rad(4)
                assert _test_last_configuration(robot, theta, last_theta)
    
                
def puma260b_robot():
    """Returns an approximate Robot instance for a Puma 260B robot"""
    
    x=np.array([1,0,0])
    y=np.array([0,1,0])
    z=np.array([0,0,1])
    a=np.array([0,0,0])
    
    H = np.array([z,y,y,z,y,x]).T    
    P = np.array([13*z, a, (-4.9*y + 7.8*x -0.75*z), -8.0*z, a, a, 2.2*x]).T*in_2_m
    joint_type=[0,0,0,0,0,0]
    joint_min=np.deg2rad(np.array([-5, -256, -214, -384, -32, -267]))
    joint_max=np.deg2rad(np.array([313, 76, 34, 194, 212, 267]))
    
    return rox.Robot(H, P, joint_type, joint_min, joint_max)    
    
def puma260b_robot_tool():
    
    robot=puma260b_robot()
    robot.R_tool=rox.rot([0,1,0], np.pi/2.0)
    robot.p_tool=[0.05, 0, 0]
    return robot
    
def abb_irb6640_180_255_robot():
    """Return a Robot instance for the ABB IRB6640 180-255 robot"""
    
    x=np.array([1,0,0])
    y=np.array([0,1,0])
    z=np.array([0,0,1])
    a=np.array([0,0,0])
    
    H = np.array([z,y,y,x,y,x]).T
    P = np.array([0.78*z, 0.32*x, 1.075*z, 0.2*z, 1.142*x, 0.2*x, a]).T
    joint_type=[0,0,0,0,0,0]
    joint_min=np.deg2rad(np.array([-170, -65, -180, -300, -120, -360]))
    joint_max=np.deg2rad(np.array([170,  85, 70,  300,  120,  360]))
    
    p_tool=np.array([0,0,0])
    R_tool=rox.rot([0,1,0], np.pi/2.0)
    
    return rox.Robot(H, P, joint_type, joint_min, joint_max, R_tool=R_tool, p_tool=p_tool) 
    
    
class GeneralRoboticsToolboxTestSuite(unittest.TestSuite):
    def __init__(self):
        super(GeneralRoboticsToolboxTestSuite, self).__init__()
        self.addTest(Test_hat())
        self.addTest(Test_rot())
        self.addTest(Test_screwmatrix())
        self.addTest(Test_R2q())
        self.addTest(Test_q2R())
        self.addTest(Test_quatcomplement())
        self.addTest(Test_quatjacobian())
        self.addTest(Test_fwdkin())
        self.addTest(Test_robotjacobian())
        self.addTest(Test_subproblems())
        self.addTest(Test_robot6_sphericalwrist_invkin())

     
if __name__ == '__main__':
    import rosunit
    rosunit.unitrun('rpi_general_robotics_toolbox_py', \
                 'test_general_robotics_toolbox', \
                 'test_general_robotics_toolbox.GeneralRoboticsToolboxTestSuite')
