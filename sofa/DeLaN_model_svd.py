import jax
import jax.numpy as jnp
import haiku as hk
import numpy as np
from functools import partial


class SVDLinear(hk.Module):
    def __init__(self, output_size, rank=None, with_bias=True, name=None):
        super().__init__(name=name)
        self.output_size = output_size
        self.rank = rank
        self.with_bias = with_bias

    def __call__(self, x):
        in_size = x.shape[-1]
        r = self.rank if self.rank is not None else min(self.output_size, in_size)

        if in_size == 1:
            W = hk.get_parameter("w", (self.output_size,), init=hk.initializers.RandomNormal(1e-2))
            if self.with_bias:
                b = hk.get_parameter("b", (self.output_size,), init=jnp.zeros)
                y = x * W + b
        elif self.output_size == 1:
            W = hk.get_parameter("w", (in_size,), init=hk.initializers.RandomNormal(1e-2))
            if self.with_bias:
                b = hk.get_parameter("b", (1,), init=jnp.zeros)
                y = jnp.dot(x, W) + b
        else:
            U = hk.get_parameter("U", (self.output_size, r),
                                init=hk.initializers.RandomNormal(1e-2))
            Vt = hk.get_parameter("Vt", (r, in_size),
                                  init=hk.initializers.RandomNormal(1e-2))
            log_s = hk.get_parameter("log_s", (r,), init=jnp.zeros)
            # freeze bases
            U = jax.lax.stop_gradient(U)
            Vt = jax.lax.stop_gradient(Vt)
            s = jnp.exp(log_s)
            W = (U * s[None, :]) @ Vt
            y = x @ W.T
            if self.with_bias:
                b = hk.get_parameter("b", (self.output_size,), init=jnp.zeros)
                y = y + b

        return y

class SVDMLP(hk.Module):
    def __init__(self, output_sizes, activation, rank=None, name=None):
        super().__init__(name=name)
        self.output_sizes = output_sizes
        self.activation = activation
        self.rank = rank

    def __call__(self, x):
        for i, out_size in enumerate(self.output_sizes[:-1]):
            x = SVDLinear(out_size, rank=self.rank, with_bias=True, name=f"linear_{i}")(x)
            x = self.activation(x)
        x = SVDLinear(self.output_sizes[-1], rank=self.rank, with_bias=True,
                      name=f"linear_{len(self.output_sizes)-1}")(x)
        return x

def dissipative_matrix_svd(q, n_dof, shape, activation, rank=None):
    n_output = int((n_dof ** 2 + n_dof) / 2)

    net = SVDMLP(output_sizes=shape + (n_output,),
                 activation=activation,
                 rank=rank,
                 name="dissipative_matrix")

    scaler = 0.4
    l_diagonal, l_off = jnp.split(net(q), [n_dof], axis=-1)
    l_diagonal = jax.nn.sigmoid(l_diagonal)

    triangular = jnp.zeros((n_dof, n_dof))
    triangular = triangular.at[np.diag_indices(n_dof)].set(l_diagonal)
    triangular = triangular.at[np.tril_indices(n_dof, -1)].set(l_off)

    D = triangular @ triangular.T
    return D * scaler

def mass_matrix_fn_svd(q, n_dof, shape, activation, epsilon, shift, rank = None):
    n_output = int((n_dof ** 2 + n_dof) / 2)  # the number of values of the lower triangle matrix

    # Compute Matrix Indices
    net = SVDMLP(output_sizes=shape + (n_output, ),  # shape defined the layers and their neural numbers
                      activation=activation,
                      rank=rank,
                      name="mass_matrix")

    l_diagonal, l_off_diagonal = jnp.split(net(q), [n_dof, ], axis=-1)

    # Ensure positive diagonal:
    l_diagonal = jax.nn.softplus(l_diagonal + shift) + epsilon

    triangular_mat = jnp.zeros((n_dof, n_dof))
    diagonal_index = np.diag_indices(n_dof)
    tril_index = np.tril_indices(n_dof, -1)
    triangular_mat = triangular_mat.at[diagonal_index].set(l_diagonal[:])
    triangular_mat = triangular_mat.at[tril_index].set(l_off_diagonal[:])

    mass_mat = jnp.matmul(triangular_mat, triangular_mat.transpose())
    return mass_mat


def input_transform_matrix_svd(q, n_dof, actuator_dof, shape, activation, rank=None):
    n_output = n_dof * actuator_dof
    net = SVDMLP(output_sizes=shape + (n_output, ),  # shape defined the layers and their neural numbers
                      activation=activation,
                      rank=rank,
                      name="input_transform_matrix")
    input_mat = net(q).reshape(n_dof, actuator_dof)
    '''
    To easily get the Kp and Kd for the controller of the soft robot, do the following adjustment.
    '''
    # n_output = int(n_dof * actuator_dof / 2)
    # net = hk.nets.MLP(output_sizes=shape + (n_output, ),  # shape defined the layers and their neural numbers
    #                   activation=activation,
    #                   name="input_transform_matrix")
    # elems = jax.nn.tanh(net(q))
    # seg1, seg2 = jnp.split(elems, [int(n_output/2), ], axis=-1)
    # seg1_mat = seg1.reshape(int(n_dof/2), int(n_dof/2))
    # seg2_mat = seg2.reshape(int(n_dof/2), int(n_dof/2))
    # left = jnp.row_stack((seg1_mat, jnp.zeros((int(n_dof/2), int(n_dof/2)))))
    # right = jnp.row_stack((jnp.zeros((int(n_dof / 2), int(n_dof / 2))), seg2_mat))
    # input_mat = jnp.hstack((left, right))
    return input_mat


def potential_energy_fn_svd(q, shape, activation, rank=None):
    net = SVDMLP(output_sizes=shape +(1, ),
                      activation=activation,
                      rank=rank,
                      name="potential_energy")

    # Apply feature transform
    return net(q)

def kinetic_energy_fn_svd(q, qd, n_dof, shape, activation, epsilon, shift, rank=None):
    mass_mat = mass_matrix_fn_svd(q, n_dof, shape, activation, epsilon, shift, rank=rank)
    return 1./2. * jnp.dot(qd, jnp.dot(mass_mat, qd))

def structured_lagrangian_fn_svd(q, qd, n_dof, shape, activation, epsilon, shift, rank=None):
    e_kin = kinetic_energy_fn_svd(q, qd, n_dof, shape, activation, epsilon, shift, rank=rank)
    e_pot = potential_energy_fn_svd(q, shape, activation, rank=rank).squeeze()
    return e_kin - e_pot


def rk4_step(f, x, y, t, h):
    # Runge-kutta integration
    k1 = h * f(x, y, t)
    k2 = h * f(x + k1/2, y, t + h/2)
    k3 = h * f(x + k2/2, y, t + h/2)
    k4 = h * f(x + k3, y, t + h)
    return x + 1/6 * (k1 + 2 * k2 + 2 * k3 + k4)


def mass_matrix_fn(q, n_dof, shape, activation, epsilon, shift):
    n_output = int((n_dof ** 2 + n_dof) / 2)  # the number of values of the lower triangle matrix

    # Compute Matrix Indices
    net = hk.nets.MLP(output_sizes=shape + (n_output, ),  # shape defined the layers and their neural numbers
                      activation=activation,
                      name="mass_matrix")

    l_diagonal, l_off_diagonal = jnp.split(net(q), [n_dof, ], axis=-1)

    # Ensure positive diagonal:
    l_diagonal = jax.nn.softplus(l_diagonal + shift) + epsilon

    triangular_mat = jnp.zeros((n_dof, n_dof))
    diagonal_index = np.diag_indices(n_dof)
    tril_index = np.tril_indices(n_dof, -1)
    triangular_mat = triangular_mat.at[diagonal_index].set(l_diagonal[:])
    triangular_mat = triangular_mat.at[tril_index].set(l_off_diagonal[:])

    mass_mat = jnp.matmul(triangular_mat, triangular_mat.transpose())
    return mass_mat


def dissipative_matrix(q, n_dof, shape, activation):
    assert n_dof > 0
    n_output = int((n_dof ** 2 + n_dof) / 2)  # the number of values of the lower triangle matrix

    # Compute Matrix Indices
    net = hk.nets.MLP(output_sizes=shape + (n_output, ),  # shape defined the layers and their neural numbers
                      activation=activation,
                      name="dissipative_matrix")

    # scaler to constraint the matrix
    scaler = 0.4
    l_diagonal, l_off_diagonal = jnp.split(net(q), [n_dof, ], axis=-1)

    l_diagonal = jax.nn.sigmoid(l_diagonal)

    triangular_mat = jnp.zeros((n_dof, n_dof))
    diagonal_index = np.diag_indices(n_dof)
    tril_index = np.tril_indices(n_dof, -1)
    triangular_mat = triangular_mat.at[diagonal_index].set(l_diagonal[:])
    triangular_mat = triangular_mat.at[tril_index].set(l_off_diagonal[:])

    dissipative_mat = jnp.matmul(triangular_mat, triangular_mat.transpose())
    dissipative_mat *= scaler
    return dissipative_mat


def input_transform_matrix(q, n_dof, actuator_dof, shape, activation):
    n_output = n_dof * actuator_dof
    net = hk.nets.MLP(output_sizes=shape + (n_output, ),  # shape defined the layers and their neural numbers
                      activation=activation,
                      name="input_transform_matrix")
    input_mat = net(q).reshape(n_dof, actuator_dof)
    '''
    To easily get the Kp and Kd for the controller of the soft robot, do the following adjustment.
    '''
    # n_output = int(n_dof * actuator_dof / 2)
    # net = hk.nets.MLP(output_sizes=shape + (n_output, ),  # shape defined the layers and their neural numbers
    #                   activation=activation,
    #                   name="input_transform_matrix")
    # elems = jax.nn.tanh(net(q))
    # seg1, seg2 = jnp.split(elems, [int(n_output/2), ], axis=-1)
    # seg1_mat = seg1.reshape(int(n_dof/2), int(n_dof/2))
    # seg2_mat = seg2.reshape(int(n_dof/2), int(n_dof/2))
    # left = jnp.row_stack((seg1_mat, jnp.zeros((int(n_dof/2), int(n_dof/2)))))
    # right = jnp.row_stack((jnp.zeros((int(n_dof / 2), int(n_dof / 2))), seg2_mat))
    # input_mat = jnp.hstack((left, right))
    return input_mat


def kinetic_energy_fn(q, qd, n_dof, shape, activation, epsilon, shift):
    mass_mat = mass_matrix_fn(q, n_dof, shape, activation, epsilon, shift)
    return 1./2. * jnp.dot(qd, jnp.dot(mass_mat, qd))


def potential_energy_fn(q, shape, activation):
    net = hk.nets.MLP(output_sizes=shape +(1, ),
                      activation=activation,
                      name="potential_energy")

    # Apply feature transform
    return net(q)


def structured_lagrangian_fn(q, qd, n_dof, shape, activation, epsilon, shift):
    e_kin = kinetic_energy_fn(q, qd, n_dof, shape, activation, epsilon, shift)
    e_pot = potential_energy_fn(q, shape, activation).squeeze()
    return e_kin - e_pot


def forward_model(params, key, lagrangian, dissipative_mat, input_mat, n_dof):
    def equation_of_motion(state, tau, t=None):
        # state should be a (n_dof * 3) np.array
        q, qd = jnp.split(state, 2)
        argnums = [2, 3]

        l_params = params["lagrangian"]

        # Compute Lagrangian and Jacobians:
        lagrangian_value_and_grad = jax.value_and_grad(lagrangian, argnums=argnums) # lagrangian_fn.apply(params, rng, q, qd)  ==> positional indices:   0      1   2  3
        L, (dLdq, dLdqd) = lagrangian_value_and_grad(l_params, key, q, qd)

        # Compute Hessian:
        lagrangian_hessian = jax.hessian(lagrangian, argnums=argnums)
        (_, (d2L_dqddq, d2Ld2qd)) = lagrangian_hessian(l_params, key, q, qd)

        # Compute Dissipative term
        d_params = params["dissipative"]
        dissipative = dissipative_mat(d_params, key, q)

        # for A(q) as a net
        i_params = params["input_transform"]
        input_transform = input_mat(i_params, key, q)

        qdd_pred = jnp.linalg.pinv(d2Ld2qd) @ \
                   (input_transform @ tau - d2L_dqddq @ qd + dLdq - dissipative @ qd)

        return jnp.concatenate([qd, qdd_pred])
    return equation_of_motion


def inverse_model(params, key, lagrangian, dissipative_mat, input_mat, n_dof):
    def equation_of_motion(state, qdd=None,  t=None):
        # state should be a (n_dof * 3) np.array
        q, qd = jnp.split(state, 2)
        argnums = [2, 3]

        l_params = params["lagrangian"]

        # Compute Lagrangian and Jacobians:
        # def structured_lagrangian_fn(q, qd, n_dof, shape, activation, epsilon, shift):
        lagrangian_value_and_grad = jax.value_and_grad(lagrangian, argnums=argnums)
        L, (dLdq, dLdqd) = lagrangian_value_and_grad(l_params, key, q, qd)

        # Compute Hessian:
        lagrangian_hessian = jax.hessian(lagrangian, argnums=argnums)
        (_, (d2L_dqddq, d2Ld2qd)) = lagrangian_hessian(l_params, key, q, qd)

        # Compute Dissipative term
        d_params = params["dissipative"]
        # def dissipative_matrix(qd, n_dof, shape, activation):
        dissipative = dissipative_mat(d_params, key, q)

        i_params = params["input_transform"]
        input_transform = input_mat(i_params, key, q)

        # Compute the inverse model
        tau = jnp.linalg.inv(input_transform) @ (d2Ld2qd @ qdd + d2L_dqddq @ qd - dLdq + dissipative @ qd)
        return tau
    return equation_of_motion

def loss_fn(params, q, qd, tau, q_next, qd_next, lagrangian, dissipative_mat, input_mat, n_dof, time_step=None):
    states = jnp.concatenate([q, qd], axis=1)
    targets = jnp.concatenate([q_next, qd_next], axis=1)

    # Forward error:
    f = jax.jit(forward_model(params=params, key=None, lagrangian=lagrangian, dissipative_mat=dissipative_mat, input_mat=input_mat, n_dof=n_dof))
    if time_step is not None:
        preds = jax.vmap(partial(rk4_step, f, t=0.0, h=time_step), (0, 0))(states, tau)
    else:
        preds = jax.vmap(f, (0, 0))(states, tau)

    forward_error = jnp.sum((targets - preds)**2, axis=-1)
    mean_forward_error = jnp.mean(forward_error)
    var_forward_error = jnp.var(forward_error)

    # Compute Loss
    loss =  mean_forward_error

    logs = {
        'loss': loss,
        'forward_mean': mean_forward_error,
        'forward_var': var_forward_error,
    }
    return loss, logs

