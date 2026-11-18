import threading
import rclpy
from rclpy.node import Node

import numpy as np
import jax
import jax.numpy as jnp
import haiku as hk
import optax
import pickle
import time
from functools import partial
from controller import DeLaN_model_static as DeLan
from controller import DeLaN_model_static_svd as DeLan_svd
from controller import DeLaN_model_static_svd as DeLan_svd

from pneumatic_actuation_msgs.msg import FluidPressures
from sensor_msgs.msg import FluidPressure
from std_msgs.msg import Bool, Float32MultiArray
from my_msgs.msg import RecordingMsg


MODEL_DIR         = "./src/controller/controller/models/online_controller/only_angle/"
BATCH_SIZE        = 64
EPOCHS_PER_ROUND  = 100
LR                = 1e-3
RANK              = None   # None = full rank
TRAIN_BIAS        = True
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

    # SVD versions for training
    pot_svd_t = hk.without_apply_rng(hk.transform(
        partial(DeLan_svd.potential_energy,
                shape=hidden_pot, activation=activation, rank=RANK)
    ))
    inp_svd_t = hk.without_apply_rng(hk.transform(
        partial(DeLan_svd.input_transform_matrix,
                n_dof=n_dof, actuator_dof=actuator_dof,
                shape=hidden_inp, activation=activation, rank=RANK)
    ))

    # Dense versions for inference (uses standard hk.MLP)
    pot_t = hk.without_apply_rng(hk.transform(
        lambda q: DeLan.potential_energy(q, hidden_pot, activation)
    ))
    inp_t = hk.without_apply_rng(hk.transform(
        lambda q: DeLan.input_transform_matrix(q, n_dof, actuator_dof, hidden_inp, activation)
    ))

    def _potential_svd(params, key, q):
        return pot_svd_t.apply(params, q)

    def _input_mat_svd(params, key, q):
        return inp_svd_t.apply(params, q)

    def _potential(params, key, q):
        return pot_t.apply(params, q)

    def _input_mat(params, key, q):
        return inp_t.apply(params, q)

    return _potential_svd, _input_mat_svd, _potential, _input_mat


def _init_svd_params(hyper, W0, rank):
    """Convert pretrained dense params into SVD form."""
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

    rng = jax.random.PRNGKey(42)
    key1, key2 = jax.random.split(rng)
    sample_q = jnp.zeros((n_dof,), dtype=jnp.float32)

    pot_svd_t = hk.without_apply_rng(hk.transform(
        partial(DeLan_svd.potential_energy,
                shape=hidden_pot, activation=activation, rank=rank)
    ))
    inp_svd_t = hk.without_apply_rng(hk.transform(
        partial(DeLan_svd.input_transform_matrix,
                n_dof=n_dof, actuator_dof=actuator_dof,
                shape=hidden_inp, activation=activation, rank=rank)
    ))

    init_g = pot_svd_t.init(key1, sample_q)
    init_i = inp_svd_t.init(key2, sample_q)

    g_params = DeLan_svd.load_svd_from_pretrained(init_g, W0["potential"],       rank=rank)
    i_params = DeLan_svd.load_svd_from_pretrained(init_i, W0["input_transform"], rank=rank)

    return {"potential": g_params, "input_transform": i_params}


def _build_svd_masks(svd_params, hyper, train_bias):
    n_layers = hyper["n_depth"] + 1
    pot_dict  = {f"potential_energy/linear_{i}":       True for i in range(n_layers)}
    inp_dict  = {f"input_transform_matrix/linear_{i}": True for i in range(n_layers - 1)}

    mask_g = DeLan_svd.make_svd_mask(svd_params["potential"],       dix=pot_dict,  train_bias=train_bias)
    mask_i = DeLan_svd.make_svd_mask(svd_params["input_transform"], dix=inp_dict,  train_bias=train_bias)
    return mask_g, mask_i


def svd_params_to_dense(svd_params):
    """Reconstruct dense (in, out) weights from SVD params for inference."""
    dense = {}
    for mod_name, leaf in svd_params.items():
        new_leaf = {}
        if "U" in leaf and "log_s" in leaf and "Vt" in leaf:
            W = (leaf["U"] * jnp.exp(leaf["log_s"])[None, :]) @ leaf["Vt"]  # (out, in)
            new_leaf["w"] = W.T                                               # (in, out)
        elif "w" in leaf:
            w = leaf["w"]
            if w.ndim == 1:
                if mod_name.endswith("linear_0"):
                    new_leaf["w"] = w.reshape(1, -1)
                else:
                    new_leaf["w"] = w.reshape(-1, 1)
            else:
                new_leaf["w"] = w
        if "b" in leaf:
            new_leaf["b"] = leaf["b"]
        dense[mod_name.replace("/linear_", "/~/linear_")] = new_leaf
    return dense


class OnlineControllerSVDNode(Node):

    def __init__(self):
        super().__init__("OnlineControllerSVDNode")
        self.t_system = time.time()

        base_path = MODEL_DIR + "robot2_damage0_0deg.jax"
        self.hyper, self.W0 = _load_base_model(base_path)

        (self._potential_svd, self._input_mat_svd,
         self._potential,     self._input_mat) = _build_apply_fns(self.hyper)

        # Initialize SVD params from pretrained W0
        self._svd_params = _init_svd_params(self.hyper, self.W0, RANK)

        # Build masked optimizers (only log_s and bias train)
        mask_g, mask_i = _build_svd_masks(self._svd_params, self.hyper, TRAIN_BIAS)
        self._optimizer1 = optax.masked(optax.adam(LR), mask_g)
        self._optimizer2 = optax.masked(optax.adam(LR), mask_i)
        self._opt_state1 = self._optimizer1.init(self._svd_params["potential"])
        self._opt_state2 = self._optimizer2.init(self._svd_params["input_transform"])

        # Dense params for inference, kept in sync after each training round
        self._dense_params = {
            "potential":       svd_params_to_dense(self._svd_params["potential"]),
            "input_transform": svd_params_to_dense(self._svd_params["input_transform"]),
        }

        self._lock             = threading.Lock()
        self._data_buffer      = []
        self._control_active   = True
        self._model_trained    = False
        self._total_epochs     = 0
        self._current_tau_mbar = 0.0
        self.previous_control  = 0.0
        self.K_GAINS  = np.array([0.0, 0.0, 6.0], dtype=np.float32)
        self.D_GAINS  = np.array([0.0, 0.0, 0.1], dtype=np.float32)
        self.dstate   = np.zeros(3, dtype=np.float32)
        self._q_current = np.zeros(3, dtype=np.float32)
        self._q_ref     = np.zeros(3, dtype=np.float32)
        self.tau_ff = 0.0
        self.tau_fb = 0.0

        self._jit_step = self._make_jit_step()

        self._new_data_event = threading.Event()
        self._shutdown_event = threading.Event()
        self._train_thread   = threading.Thread(target=self._training_loop, daemon=True)
        self._train_thread.start()

        self.create_subscription(RecordingMsg, "/data_recorder_online/info", self._state_callback, 10)
        self.create_subscription(Float32MultiArray, "/ref", self._reference_callback, 10)
        self.create_subscription(Bool, "/dummy_listener", self.initiate_callback, 10)

        self.init = False
        self._pub           = self.create_publisher(FluidPressures, "/vtem_control/input_pressures", 10)
        self._timer         = self.create_timer(0.01, self._timer_callback)
        self.train_pub      = self.create_publisher(Float32MultiArray, "/training_info", 10)
        self.controller_pub = self.create_publisher(Float32MultiArray, "/controller_output", 10)
        self.get_logger().info("[OnlineControllerSVD] Node ready.")

    def _make_jit_step(self):
        optimizer1    = self._optimizer1
        optimizer2    = self._optimizer2
        _potential_svd = self._potential_svd
        _input_mat_svd = self._input_mat_svd

        loss_fn = partial(
            DeLan_svd.loss_fn,
            pot_mat=_potential_svd,
            input_mat=_input_mat_svd,
            lambda_data=1.0,
        )

        @jax.jit
        def _step(svd_params, opt_state1, opt_state2, q_b, tau_b):
            (l, _), grads = jax.value_and_grad(loss_fn, 0, has_aux=True)(
                svd_params, q_b, tau_b
            )
            upd1, opt_state1 = optimizer1.update(grads["potential"],       opt_state1, svd_params["potential"])
            upd2, opt_state2 = optimizer2.update(grads["input_transform"], opt_state2, svd_params["input_transform"])
            new_params = {
                "potential":       optax.apply_updates(svd_params["potential"],       upd1),
                "input_transform": optax.apply_updates(svd_params["input_transform"], upd2),
            }
            return new_params, opt_state1, opt_state2, l

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
                data       = np.array(self._data_buffer, dtype=np.float32)
                self._data_buffer.clear()
                svd_params = self._svd_params
                opt_state1 = self._opt_state1
                opt_state2 = self._opt_state2

            q_data   = jnp.array(data[:, 0:1])
            tau_data = jnp.array(data[:, 1:2])
            N        = q_data.shape[0]

            last_loss = float("inf")
            for _ in range(EPOCHS_PER_ROUND):
                bi = np.random.permutation(N)[:BATCH_SIZE]
                svd_params, opt_state1, opt_state2, last_loss = self._jit_step(
                    svd_params, opt_state1, opt_state2,
                    q_data[bi], tau_data[bi]
                )

            # Reconstruct dense params for inference
            dense_params = {
                "potential":       svd_params_to_dense(svd_params["potential"]),
                "input_transform": svd_params_to_dense(svd_params["input_transform"]),
            }

            with self._lock:
                self._svd_params    = svd_params
                self._opt_state1    = opt_state1
                self._opt_state2    = opt_state2
                self._dense_params  = dense_params
                self._total_epochs += EPOCHS_PER_ROUND
                if self._total_epochs > 100 and last_loss < 0.2:
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
        if not self.init:
            return
        dx    = float(msg.xe - msg.x0)
        dy    = float(msg.ye - msg.y0)
        angle = float(msg.angle)

        with self._lock:
            dt            = time.time() - self.t_system
            self.t_system = time.time()
            old_q         = self._q_current.copy()
            self._q_current = np.array([dx, dy, angle], dtype=np.float32)
            self.dstate   = (self._q_current - old_q) / max(dt, 1e-6)
            self._data_buffer.append([angle, self._current_tau_mbar / 100.0])
            n = len(self._data_buffer)

        self._new_data_event.set()
        self.get_logger().debug(f"[Data] buffer={n}  angle={angle:.4f}")
        self._update_control_output()

    def _reference_callback(self, msg: Float32MultiArray):
        if not self.init:
            return
        angle_ref, dx_ref, dy_ref = msg.data
        with self._lock:
            self._q_ref = np.array([dx_ref, dy_ref, angle_ref], dtype=np.float32)
        self._update_control_output()

    def _update_control_output(self):
        if not self.init:
            return

        with self._lock:
            active       = self._control_active
            trained      = self._model_trained
            dense_params = self._dense_params
            q_current    = self._q_current.copy()
            q_ref        = self._q_ref.copy()

        if not active:
            return

        tau_ff_raw = self._feedforward(dense_params, q_ref[2:3])
        error      = q_ref - q_current
        tau_fb     = float(self.K_GAINS @ error) - float(self.D_GAINS @ self.dstate)

        if not trained:
            tau_total = float(np.clip((tau_fb + self.previous_control) * 100.0, 0.0, PRESSURE_MAX_MBAR))
            self.previous_control = tau_total / 100.0
            self.tau_ff, self.tau_fb = 0.0, tau_fb
            with self._lock:
                self._current_tau_mbar = tau_total
            self.get_logger().info(f"[FB only] fb={tau_fb:.4f}  total={tau_total:.2f} mBar")
            return

        self.K_GAINS = np.array([0.0, 0.0, 6], dtype=np.float32)
        self.D_GAINS = np.array([0.0, 0.0, 0.0], dtype=np.float32)

        tau_total = float(np.clip((tau_ff_raw + tau_fb) * 100.0, 0.0, PRESSURE_MAX_MBAR))
        self.tau_ff, self.tau_fb = tau_ff_raw, tau_fb
        with self._lock:
            self._current_tau_mbar = tau_total
        self.get_logger().info(f"ff={tau_ff_raw:.4f}  fb={tau_fb:.4f}  total={tau_total:.2f} mBar")

    def _feedforward(self, dense_params, q_ref: np.ndarray) -> float:
        q_jnp   = jnp.array(q_ref, dtype=jnp.float32)
        forward = DeLan.forward_model(dense_params, None, self._potential, self._input_mat)
        _, tau_hat = forward(q_jnp)
        return float(np.array(tau_hat).squeeze())

    def _timer_callback(self):
        if not self.init:
            return

        with self._lock:
            active   = self._control_active
            tau_mbar = self._current_tau_mbar

        tau_pa = float(tau_mbar) * 100.0 if active else 0.0
        now    = self.get_clock().now().to_msg()
        msg    = FluidPressures()
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

        control_msg = Float32MultiArray()
        control_msg.data = [float(tau_mbar), self.tau_ff * 100.0, self.tau_fb * 100.0]
        self.controller_pub.publish(control_msg)

    def destroy_node(self):
        self._shutdown_event.set()
        self._new_data_event.set()
        self._train_thread.join(timeout=2.0)
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = OnlineControllerSVDNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()