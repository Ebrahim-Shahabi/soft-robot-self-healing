import jax
import jax.numpy as jnp
import haiku as hk


def input_transform_matrix(q, n_dof, actuator_dof, shape, activation):
    net = hk.nets.MLP(
        output_sizes=shape + (n_dof * actuator_dof,),
        activation=activation,
        name="input_transform_matrix"
    )

    raw_A = net(q)
    A = raw_A.reshape(n_dof, actuator_dof)

    A = A + 1e-4
    return A

def potential_energy(q, shape, activation):
    net = hk.nets.MLP(
        output_sizes=shape + (1,),
        activation=activation,
        name="potential_energy"
    )
    return net(q).squeeze(-1)  

 
def forward_model(params, key, pot_mat, input_mat):
    def model(q):
        def V_fn(q_):
            return pot_mat(params["potential"], key, q_)

        G = jax.grad(V_fn)(q)                           
        A = input_mat(params["input_transform"], key, q)

        tau_hat = jnp.linalg.pinv(A) @ G

        return A, tau_hat

    return model


def loss_fn(
    params,
    q,
    tau,
    pot_mat,
    input_mat,
    lambda_data=1.0,
):
    f = forward_model(
        params=params,
        key=None,
        pot_mat=pot_mat,
        input_mat=input_mat,
    )

    A_pred, tau_pred = jax.vmap(f, in_axes=0)(q)
    if tau_pred.ndim == 1:
        tau_pred = tau_pred[:, None]


    data_res = tau_pred - tau

    data_loss_per_sample = jnp.sum(data_res**2, axis=-1)

    data_loss = jnp.mean(data_loss_per_sample)

    loss = lambda_data * data_loss 
    mean_forward_error = data_loss
    var_forward_error = jnp.var(data_loss_per_sample)
    logs = {
    'loss': loss,
    'forward_mean': mean_forward_error,
    'forward_var': var_forward_error,
    'A_pred': jnp.mean(A_pred)
    }

    return loss, logs

