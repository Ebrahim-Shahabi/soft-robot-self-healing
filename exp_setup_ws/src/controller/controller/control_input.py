import rclpy
from rclpy.node import Node

import cv2
import numpy as np
import math
import jax
import jax.numpy as jnp
import haiku as hk
from functools import partial
from controller import DeLaN_model_v4 as delan
import pickle
import time

# from apriltag_msgs.msg import AprilTagDetectionArray
from pneumatic_actuation_msgs.msg import FluidPressures
from std_msgs.msg import Float32
from sensor_msgs.msg import FluidPressure

# Load model
with open(f"./src/controller/controller/models/one_segment_spatial_soft_robot_delan_experiment.jax", 'rb') as f:
    data = pickle.load(f)


hyper = data["hyper"]
params = data["params"]

activations = {
    'tanh': jnp.tanh,
    'softplus': jax.nn.softplus,
    'sigmoid': jax.nn.sigmoid,
}

lagrangian_fn = hk.transform(partial(
    delan.structured_lagrangian_fn,
    n_dof=hyper['n_dof'],
    shape=(hyper['n_width'],) * hyper['n_depth'],
    activation=activations[hyper['activation1']],
    epsilon=hyper['diagonal_epsilon'],
    shift=hyper['diagonal_shift'],
))

#dissipative_matrix(qd, n_dof, shape, activation)
dissipative_fn = hk.transform(partial(
    delan.dissipative_matrix,
    n_dof=hyper['n_dof'],
    shape=(5,) * 3,
    activation=activations[hyper['activation2']]
))

# potential matrix
potential_fn = hk.transform(partial(
    delan.potential_energy_fn,
    shape=(hyper['n_width'],) * hyper['n_depth'],
    activation=activations[hyper['activation1']],
))

V = lambda q: potential_fn.apply(params["lagrangian"], None, q).squeeze()
# dV_dq = jax.grad(V)(q)                 # shape (n_dof,)


#input_transform_matrix(q, n_dof, actuator_dof, shape, activation)
input_mat_fn = hk.transform(partial(
    delan.input_transform_matrix,
    n_dof=hyper['n_dof'],
    actuator_dof=hyper['actuator_dof'],
    shape=(hyper['n_width']//2,) * (hyper['n_depth']-1),
    activation=activations[hyper['activation1']]
))

lagrangian = lagrangian_fn.apply
# dissipative_mat = dissipative_fn.apply
input_mat = input_mat_fn.apply


# --- JAX helpers (outside Controller, so they're compiled once) ---
def AL(q):
    # A(q) : (n_dof, actuator_dof)
    return input_mat(params['input_transform'], None, q)

def GL(q):
    # G(q) = dV/dq : (n_dof,)
    return jax.grad(V)(q)

# jit for speed
AL_jit = jax.jit(AL)
GL_jit = jax.jit(GL)

def feedforward_control(qd):
    Al = AL_jit(qd)
    Gl = GL_jit(qd)
    u_ff = np.linalg.pinv(np.array(Al)) @ np.array(Gl)
    return float(np.array(u_ff).squeeze())

def calculate_angle(origin, tip, angle0=0.0, dt=0.01):
    dx = tip[0] - origin[0]
    dz = tip[2] - origin[2]
    angle = -math.atan2(dx, dz)          # radians
    dangle = (angle - angle0) / dt if dt > 1e-9 else 0.0
    return angle, dangle



class controllerNode(Node):

    def __init__(self):
        super().__init__('controllerNode')


        self.create_subscription(
            Float32,
            '/ref',
            self.reference_callback,
            10
        )
        self.control_input = 0

        # could be that i have to change the topic to the vtem one.
        self.publisher_ = self.create_publisher(FluidPressures, '/vtem_control/input_pressures', 10)
        timer_period = 0.01  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)

    
    # control here
    def reference_callback(self,msg):
        self.get_logger().info(f'[INFO] Received reference input {msg.data}.\n')
        q_ref = msg.data
        self.control_input = feedforward_control(np.array([q_ref], dtype=float))
        # truncate input
        if self.control_input > 300:
            self.get_logger().info(f'[ERROR] Control Input exceeds upper bound: {self.control_input}.\n')
            self.control_input = 0
        elif self.control_input <= 0:
            self.get_logger().info(f'[ERROR] Control Input exceeds lower bound: {self.control_input}.\n')
            self.control_input = 0


    # send control input
    def timer_callback(self):
        self.get_logger().info(f'[INFO] Timer callback. Sending pressure {self.control_input}.\n')
        msg = FluidPressures()
        fp = FluidPressure()
        fp.header.stamp = self.get_clock().now().to_msg()
        fp.fluid_pressure = float(self.control_input*100)
        fp.variance = 0.0
        msg.data = [fp]
        self.publisher_.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = controllerNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()