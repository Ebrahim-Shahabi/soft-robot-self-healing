import jax
import jax.numpy as jnp
import haiku as hk

# network only takes pressure as input and outputs state.
def network(tau, state_dof, shape, activation):
    net = hk.nets.MLP(output_sizes=shape + (state_dof, ),  
                      activation=activation,
                      name="network")
    return net(tau)

def forward_model(params, key, net):
    def model(tau):
        return net(params['network'], key, tau)
    return model


def loss_fn(
    params,
    q,
    tau,
    net,
    lambda_data=1.0,
):
    f = forward_model(
        params=params,
        key=None,
        net=net,
    )

    q_hat = jax.vmap(f, in_axes=0)(tau)

    data_res = q_hat - q
    data_loss_per_sample = jnp.sum(data_res**2, axis=-1)
    loss = lambda_data*jnp.mean(data_loss_per_sample)
    mean_forward_error = loss
    var_forward_error = jnp.var(data_loss_per_sample)
    logs = {
    'loss': loss,
    'forward_mean': mean_forward_error,
    'forward_var': var_forward_error,
    }

    return loss, logs