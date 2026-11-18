import rclpy
from rclpy.node import Node

import cv2
import numpy as np
import math
import jax
import jax.numpy as jnp
import haiku as hk
from functools import partial
from controller import DeLaN_model_static as delan
import pickle
import time

# from apriltag_msgs.msg import AprilTagDetectionArray
from pneumatic_actuation_msgs.msg import FluidPressures
from std_msgs.msg import Float32
from sensor_msgs.msg import FluidPressure
from std_msgs.msg import Float32MultiArray

# Load model
with open(f"./src/controller/controller/models/test.jax", 'rb') as f:
    data = pickle.load(f)


hyper = data["hyper"]
params = data["params"]

activations = {
    'tanh': jnp.tanh,
    'softplus': jax.nn.softplus,
    'sigmoid': jax.nn.sigmoid,
}


# potential matrix
potential_fn = hk.transform(partial(
    delan.potential_energy,
    shape=(hyper['n_width'],) * hyper['n_depth'],
    activation=activations[hyper['activation1']],
))

V = lambda q: potential_fn.apply(params["potential"], None, q).squeeze()
# dV_dq = jax.grad(V)(q)                 # shape (n_dof,)


#input_transform_matrix(q, n_dof, actuator_dof, shape, activation)
input_mat_fn = hk.transform(partial(
    delan.input_transform_matrix,
    n_dof=hyper['n_dof'],
    actuator_dof=hyper['actuator_dof'],
    shape=(hyper['n_width']//2,) * (hyper['n_depth']-1),
    activation=activations[hyper['activation1']]
))


input_mat = input_mat_fn.apply

model = delan.forward_model(
    params=params,
    key=None,
    pot_mat=potential_fn.apply,
    input_mat=input_mat_fn.apply
)

def feedforward_control(qd):
    qd = jnp.array(qd)

    A, tau_hat = model(qd)

    tau_hat = np.array(tau_hat).squeeze()

    return float(tau_hat)

def calculate_angle(origin, tip, angle0=0.0, dt=0.01):
    dx = tip[0] - origin[0]
    dz = tip[2] - origin[2]
    angle = -math.atan2(dx, dz)          # radians
    dangle = (angle - angle0) / dt if dt > 1e-9 else 0.0
    return angle, dangle



class StaticControllerNode(Node):

    def __init__(self):
        super().__init__('StaticControllerNode')


        self.create_subscription(
            Float32MultiArray,
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
    def reference_callback(self, msg):
        self.get_logger().info(f'[INFO] Received reference input {msg.data}.\n')

        angle_ref, dx_ref, dy_ref = msg.data

        # Network expects: [dx, dy, angle]
        qd = np.array([dx_ref/100, dy_ref/100, angle_ref], dtype=float)

        tau_scaled = feedforward_control(qd)

        # Training target was pressure[mBar] / 100
        self.control_input = tau_scaled * 100.0

        self.get_logger().info(
            f'[INFO] qd sent to network: {qd}, raw tau: {tau_scaled}, pressure: {self.control_input} mBar'
        )

        if self.control_input > 300:
            self.get_logger().info(f'[ERROR] Control Input exceeds upper bound: {self.control_input}.\n')
            self.control_input = 0
        elif self.control_input <= 0:
            self.get_logger().info(f'[ERROR] Control Input exceeds lower bound: {self.control_input}.\n')
            self.control_input = 0


    # send control input
    def timer_callback(self):
        msg = FluidPressures()
        now = self.get_clock().now().to_msg()
        msg.header.stamp = now
        msg.header.frame_id = ''

        pressures = []

        # convert mBar → Pa
        pressure_pa = float(self.control_input)*100

        for i in range(3):
            fp = FluidPressure()
            fp.header.stamp = now
            fp.header.frame_id = f'valve_{i}'
            fp.variance = 0.0

            if i == 2: # change this depending on which valvule you want to have output
                fp.fluid_pressure = pressure_pa  # 200 mBar → 20000 Pa
            else:
                fp.fluid_pressure = 0.0

            fp.variance = 0.0
            pressures.append(fp)

        msg.data = pressures
        self.publisher_.publish(msg)

        self.get_logger().info(
            f"[INFO] Sending pressure: {self.control_input} mBar ({pressure_pa} Pa)"
        )

def main(args=None):
    rclpy.init(args=args)
    node = StaticControllerNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()