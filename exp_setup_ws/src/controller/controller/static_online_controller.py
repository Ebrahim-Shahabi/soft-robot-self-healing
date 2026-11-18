import threading
import rclpy
from rclpy.node import Node

import numpy as np
import jax
import jax.numpy as jnp
import haiku as hk
import optax
import pickle
from pathlib import Path
import time
from controller import DeLaN_model_static as DeLan
from controller.library import subspace as sb

from pneumatic_actuation_msgs.msg import FluidPressures
from sensor_msgs.msg import FluidPressure
from std_msgs.msg import Bool, Float32MultiArray
from my_msgs.msg import RecordingMsg


SCRIPT_DIR = Path(__file__).resolve().parent
MODEL_DIR   = "./src/controller/controller/models/online_controller/only_angle/"

TASK_MODEL_FILES = {
    "r1d1_45deg" : MODEL_DIR + "robot1_damage1_healed_45deg.jax",
    "r1d1_135deg": MODEL_DIR + "robot1_damage1_healed_135deg.jax",
    "r1d2_0deg"  : MODEL_DIR + "robot1_damage2_healed_0deg.jax",
    "r1d3_0deg"  : MODEL_DIR + "robot1_damage3_healed_0deg.jax",
    "r1d3_135deg": MODEL_DIR + "robot1_damage3_healed_135deg.jax",
    "r2d0_0deg"  : MODEL_DIR + "robot2_damage0_0deg.jax",
    "r2d0_135deg": MODEL_DIR + "robot2_damage0_135deg.jax",
}


BATCH_SIZE        = 64
EPOCHS_PER_ROUND  = 100
LR                = 1e-2
PRESSURE_MAX_MBAR = 300.0


def _load_base_model(path):
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data["hyper"], data["params"]


def _build_apply_fns(hyper: dict):
    activations = {
        "tanh":     jnp.tanh,
        "softplus": jax.nn.softplus,
        "sigmoid":  jax.nn.sigmoid,
    }
    activation   = activations[hyper["activation1"]]
    hidden_pot   = (hyper["n_width"],) * hyper["n_depth"]
    hidden_inp   = (hyper["n_width"] // 2,) * (hyper["n_depth"] - 1)
    n_dof        = hyper["n_dof"]
    actuator_dof = hyper["actuator_dof"]

    pot_t = hk.without_apply_rng(hk.transform(
        lambda q: DeLan.potential_energy(q, hidden_pot, activation)
    ))
    inp_t = hk.without_apply_rng(hk.transform(
        lambda q: DeLan.input_transform_matrix(q, n_dof, actuator_dof, hidden_inp, activation)
    ))

    def _potential(params, key, q):
        return pot_t.apply(params, q)

    def _input_mat(params, key, q):
        return inp_t.apply(params, q)

    return _potential, _input_mat


def _build_subspace(W0: dict, task_model_files: dict):
    task_weights = {}
    for name, fpath in task_model_files.items():
        with open(fpath, "rb") as f:
            task_weights[name] = pickle.load(f)["params"]

    subspace = sb.compute_task_subspace(
        w0_star=W0,
        task_weights=task_weights,
        variance_threshold=0.90,
        max_rank=None,
        free_layer_names=None,
    )
    return sb.SubspaceParams(subspace=subspace, w0_star=W0, free_layer_names=None)



class OnlineControllerNode(Node):

    def __init__(self):
        super().__init__("OnlineControllerNode")
        self.t_system = time.time()

        base_path = MODEL_DIR + "robot1_damage1_healed_0deg.jax"
        self.hyper, self.W0 = _load_base_model(base_path)
        self._potential, self._input_mat = _build_apply_fns(self.hyper)
        self.sp = _build_subspace(self.W0, TASK_MODEL_FILES)


        self._lock             = threading.Lock()
        self._params           = self.W0
        self._trainable        = self.sp.init_trainable()
        self._opt              = optax.adam(LR)
        self._opt_state        = self._opt.init(self._trainable)
        self._data_buffer      = []          # [[dx, dy, angle, tau], ...]
        self._control_active   = True
        self._model_trained    = True
        self._total_epochs     = 0
        self._current_tau_mbar = 0.0
        self.K_GAINS = np.array([0.0, 0.0, 4.0], dtype=np.float32)
        self.D_GAINS = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        self.dstate = np.zeros(3, dtype=np.float32)
        # q_current: updated every RecordingMsg; used for feedback term
        # q_ref:     updated every /ref message
        self._q_current = np.zeros(3, dtype=np.float32)   # [dx, dy, angle]
        self._q_ref     = np.zeros(3, dtype=np.float32)   # [dx, dy, angle]
        

        self._jit_step = self._make_jit_step()


        self._new_data_event = threading.Event()
        self._shutdown_event = threading.Event()
        self._train_thread   = threading.Thread(
            target=self._training_loop, daemon=True
        )
        self._train_thread.start()


        self.create_subscription(
            RecordingMsg,
            "/data_recorder_online/info",
            self._state_callback,       # updates q_current AND buffer
            10,
        )

        self.create_subscription(
            Float32MultiArray,
            "/ref",
            self._reference_callback,   # updates q_ref
            10,
        )
        self.create_subscription(
            Bool,
            "/dummy_listener",
            self.initiate_callback,   # updates q_ref
            10,
        )

        self.init = False
        self._pub   = self.create_publisher(FluidPressures, "/vtem_control/input_pressures", 10)
        self._timer = self.create_timer(0.01, self._timer_callback)
        self.train_pub = self.create_publisher(Float32MultiArray, "/training_info", 10)
        self.controller_pub = self.create_publisher(Float32MultiArray, '/controller_output', 10)
        self.tau_ff = 0.0
        self.tau_fb = 0.0
        self.previous_control = 0
        self.get_logger().info("[OnlineController] Node ready.")

    def _make_jit_step(self):
        sp         = self.sp
        _potential = self._potential
        _input_mat = self._input_mat
        opt        = optax.adam(LR)

        @jax.jit
        def _step(trainable, opt_state, q_b, tau_b):
            def _loss(tr):
                full_p = sp.reconstruct(tr)
                loss, _ = DeLan.loss_fn(full_p, q_b, tau_b, _potential, _input_mat)
                return loss + 1e-3 * jnp.sum(tr["alpha"] ** 2)

            l, g               = jax.value_and_grad(_loss)(trainable)
            upd, new_opt_state = opt.update(g, opt_state)
            new_tr             = optax.apply_updates(trainable, upd)
            return new_tr, new_opt_state, l

        return _step


    def _training_loop(self):
        while not self._shutdown_event.is_set():
            self._new_data_event.wait(timeout=1.0)
            self._new_data_event.clear()

            with self._lock:
                n = len(self._data_buffer)


            if n < BATCH_SIZE:
                continue

            with self._lock:
                data      = np.array(self._data_buffer, dtype=np.float32)
                self._data_buffer.clear()
                trainable = self._trainable
                opt_state = self._opt_state

            # q_data   = jnp.array(data[:, :3])
            # tau_data = jnp.array(data[:, 3:4])
            q_data   = jnp.array(data[:,0:1])
            tau_data = jnp.array(data[:,1:2])
            N        = q_data.shape[0]

            last_loss = float("inf")
            for _ in range(EPOCHS_PER_ROUND):
                bi = np.random.permutation(N)[:BATCH_SIZE]
                trainable, opt_state, last_loss = self._jit_step(
                    trainable, opt_state, q_data[bi], tau_data[bi]
                )

            with self._lock:
                self._trainable     = trainable
                self._opt_state     = opt_state
                self._params        = self.sp.reconstruct(trainable)
                self._total_epochs += EPOCHS_PER_ROUND

                if self._total_epochs > 100 and last_loss < 0.2:   # arbitrary threshold for "good enough"
                    self._model_trained = True

            self.get_logger().info(
                f"[Training] total_epochs={self._total_epochs}  "
                f"loss={float(last_loss):.6f}  buffer={N}"
            )
            self.train_pub.publish(Float32MultiArray(data=[float(last_loss), float(self._total_epochs)]))

 
    def initiate_callback(self, msg: Bool):
        self.init = msg.data
        if self.init:
            self.get_logger().info(f"[INFO] Received initiation signal: {self.init}")

    def _state_callback(self, msg: RecordingMsg):
        if self.init:
            """
            Two jobs:
            1. Update q_current for the feedback term  (happens every message)
            2. Append to training buffer               (happens every message)
            """
            dx    = float(msg.xe - msg.x0)
            dy    = float(msg.ye - msg.y0)
            angle = float(msg.angle)
            tau   = float(msg.avg_pressure)
            
            with self._lock:
                t_system = time.time() - self.t_system
                self.t_system = time.time()
                old_q = self._q_current.copy()
                self._q_current = np.array([dx, dy, angle], dtype=np.float32)
                self.dstate = (self._q_current - old_q) / t_system

                # self._data_buffer.append([
                #         dx, dy, angle,   # q_current (trustworthy because error is small)
                #         tau/100          # total pressure (trustworthy because fb ≈ 0 when error ≈ 0)
                #     ])
                # self._data_buffer.append([angle, tau/100])  # angle-only data to encourage learning angle control
                self._data_buffer.append([angle, self._current_tau_mbar / 100.0])

                n = len(self._data_buffer)

            self._new_data_event.set()
            self.get_logger().debug(f"[Data] buffer={n}  angle={angle:.4f}  tau={tau:.2f}")

            # Recompute control output immediately with the fresh q_current
            # so the timer always has an up-to-date command ready.
            self._update_control_output()

    def _reference_callback(self, msg: Float32MultiArray):
        if self.init:
            """
            Expects [angle_ref, dx_ref, dy_ref]  (same as static controller).
            Stores q_ref and recomputes control output.
            """
            angle_ref, dx_ref, dy_ref = msg.data
            with self._lock:
                self._q_ref = np.array(
                    [dx_ref, dy_ref, angle_ref], dtype=np.float32
                )

            self._update_control_output()



    def _update_control_output(self):
        if self.init:
            with self._lock:
                active    = self._control_active
                trained   = self._model_trained
                params    = self._params
                q_current = self._q_current.copy()
                q_ref     = self._q_ref.copy()

            if not active:
                self.get_logger().info("Controller inactive. Output set to 0.")
                return

            # --- feedforward term ---
            # tau_ff_raw = self._feedforward(params, q_ref[2])  # always compute for logging
            tau_ff_raw = self._feedforward(params, q_ref[2:3])  # shape (1,) not ()

            # --- feedback term ---
            error  = q_ref - q_current
            tau_fb = float(self.K_GAINS @ error) - float(self.D_GAINS @ self.dstate)

            if not trained:
                tau_total = tau_fb + self.previous_control
                tau_total = float(np.clip(tau_total * 100.0, 0.0, PRESSURE_MAX_MBAR))
                self.previous_control = tau_total / 100.0

                # update shared state BEFORE return
                self.tau_fb   = tau_fb
                self.tau_ff  = 0.0

                with self._lock:
                    self._current_tau_mbar = tau_total

                self.get_logger().info(
                    f"[FB only] fb={tau_fb:.4f}  "
                    f"total={tau_total:.2f} mBar  "
                    f"error={np.round(error, 4)}"
                )
                return

            # --- trained path ---
            if self._model_trained:
                self.K_GAINS = np.array([0.0, 0.0, 0.4], dtype=np.float32)  # reduce fb
                self.D_GAINS = np.array([0.0, 0.0, 0.0], dtype=np.float32)

            tau_total = (tau_ff_raw + tau_fb) * 100.0
            tau_total = float(np.clip(tau_total, 0.0, PRESSURE_MAX_MBAR))

            self.tau_fb  = tau_fb
            self.tau_ff = tau_ff_raw

            with self._lock:
                self._current_tau_mbar = tau_total

            self.get_logger().info(
                f"ff={tau_ff_raw:.4f}  fb={tau_fb:.4f}  "
                f"total={tau_total:.2f} mBar  "
                f"error={np.round(error, 4)}"
            )
    def _feedforward(self, params, q_ref: np.ndarray) -> float:
        """Run the adapted DeLaN model and return pressure in mBar."""
        q_jnp   = jnp.array(q_ref, dtype=jnp.float32)
        forward = DeLan.forward_model(params, None, self._potential, self._input_mat)
        _, tau_hat = forward(q_jnp)
        return float(np.array(tau_hat).squeeze()) # network output is /100 of mBar, this is applied later


    def _timer_callback(self):
        if self.init:
            with self._lock:
                active   = self._control_active
                trained  = self._model_trained
                tau_mbar = self._current_tau_mbar

            tau_pa = float(tau_mbar) * 100.0 if (active) else 0.0

            now = self.get_clock().now().to_msg()
            msg = FluidPressures()
            msg.header.stamp    = now
            msg.header.frame_id = ""

            pressures = []
            for i in range(3):
                fp = FluidPressure()
                fp.header.stamp    = now
                fp.header.frame_id = f"valve_{i}"
                fp.variance        = 0.0
                fp.fluid_pressure  = tau_pa if i == 2 else 0.0
                pressures.append(fp)

            msg.data = pressures
            self._pub.publish(msg)
            # Also publish the raw control output for logging/analysis
            control_msg = Float32MultiArray()
            control_msg.data = [
                float(tau_mbar),          # u      — already in mBar
                self.tau_ff * 100.0,     # u_ff   — normalize → mBar  
                self.tau_fb  * 100.0,     # u_fb   — normalize → mBar
            ]
            self.controller_pub.publish(control_msg)

    def destroy_node(self):
        self._shutdown_event.set()
        self._new_data_event.set()
        self._train_thread.join(timeout=2.0)
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = OnlineControllerNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()