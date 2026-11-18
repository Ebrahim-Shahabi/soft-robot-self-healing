import jax
import jax.numpy as jnp
import haiku as hk
import numpy as np
from functools import partial


'''
Creates a mask for selective training of parameters in an SVD-based model.

This function generates a boolean mask with the same structure as the parameter
tree, indicating which parameters should be updated during optimization. The mask
is constructed based on a user-provided dictionary specifying which modules to train,
and selectively enables updates for specific parameter types (e.g. weights, singular
values, and optionally biases).

@param[in] tree : haiku parameter tree (module_name -> parameter_name -> array).
@param[in] dix : dictionary mapping module names to booleans, indicating whether
                 each module should be trained.
@param[in] train_bias : whether to include bias parameters ("b" or "bias") in training.

@return mask : dictionary with the same structure as tree, containing boolean values
               that specify which parameters are trainable.
'''
def make_svd_mask(tree, dix,  train_bias=True):
    # tree is the haiku param tree (module_name -> param_name -> array)
    def leaf_mask(mod, leaf, dix):
        out = {}
        flag = dix[mod]
        # print(mod, flag)
        for k in leaf.keys():
            if k == 'log_s' or (train_bias and k in ("b", "bias")) or k in ("w"):
                out[k] = True and flag
            else:
                out[k] = False
        return out
    
    return {mod: leaf_mask(mod, leaf, dix) for mod, leaf in tree.items()}

'''
SVD factorization function. Takes a weight matrix W and returns its SVD components U, log_s, Vt.
Logarithm of singular values is used for better numerical stability during training. 
The rank parameter allows for truncation of the SVD to a specified number of components.
'''
def svd_factorize(W, rank=None, eps=1e-8):
    U, S, Vt = jnp.linalg.svd(W, full_matrices=False)
    if rank is not None:
        U, S, Vt = U[:, :rank], S[:rank], Vt[:rank, :]
    log_s = jnp.log(S + eps)
    return U, log_s, Vt

# for naming issues.
def _strip_tilde_scope(path: str) -> str:
    return path.replace("/~/", "/")

'''
Loads parameters from a pretrained model into an initialized parameter tree,
handling both dense and SVD-based weight representations.

This function matches modules between the initialized model and the pretrained
model (accounting for naming differences), copies compatible parameters, and
converts dense weights into SVD form when required.

@param[in] init_tree : dictionary containing the initialized model parameters.
@param[in] pretrained_tree : dictionary containing the pretrained model parameters.
@param[in] rank : target rank used when factorizing dense weights into SVD form.

@return out : dictionary with the same structure as init_tree, populated with
              pretrained parameters where possible.
'''
def load_svd_from_pretrained(init_tree, pretrained_tree, rank):
    pretrained_by_norm = {_strip_tilde_scope(k): k for k in pretrained_tree.keys()}

    out = {}
    for mod_name, init_leaf in init_tree.items():
        leaf = dict(init_leaf)  

        pretrained_mod = pretrained_by_norm.get(mod_name)
        if pretrained_mod is None:
            out[mod_name] = leaf
            continue

        pre_leaf = pretrained_tree[pretrained_mod]

        
        if "b" in leaf and "b" in pre_leaf:
            leaf["b"] = pre_leaf["b"]

        if "w" not in pre_leaf:
            out[mod_name] = leaf
            continue

        Wpre = pre_leaf["w"]

        
        init_is_dense = "w" in init_leaf and ("U" not in init_leaf)
        init_is_svd   = "U" in init_leaf and "Vt" in init_leaf and "log_s" in init_leaf

        if init_is_dense:
            # Ensure no SVD keys remain
            leaf.pop("U", None)
            leaf.pop("Vt", None)
            leaf.pop("log_s", None)

            # Load W into the expected shape
            Wexp = leaf["w"]
            
            if getattr(Wpre, "ndim", None) == getattr(Wexp, "ndim", None) and Wpre.shape == Wexp.shape:
                leaf["w"] = Wpre
            elif getattr(Wpre, "ndim", None) == 2 and getattr(Wexp, "ndim", None) == 2 and Wpre.T.shape == Wexp.shape:
                leaf["w"] = Wpre.T
            elif getattr(Wpre, "ndim", None) == 2 and getattr(Wexp, "ndim", None) == 1:
                # handle scalar<->vector special cases if your old model stored 2D
                if Wpre.shape[0] == 1 and Wpre.shape[1] == Wexp.shape[0]:
                    leaf["w"] = Wpre.reshape(-1)          # (1, in) -> (in,)
                elif Wpre.shape[1] == 1 and Wpre.shape[0] == Wexp.shape[0]:
                    leaf["w"] = Wpre.reshape(-1)          # (out, 1) -> (out,)
                else:
                    raise ValueError(f"{mod_name}: cannot map pretrained w{Wpre.shape} to expected w{Wexp.shape}")
            else:
                raise ValueError(f"{mod_name}: cannot map pretrained w{getattr(Wpre,'shape',None)} to expected w{getattr(Wexp,'shape',None)}")

        elif init_is_svd:
            # Ensure no dense 'w' remains
            leaf.pop("w", None)

            if getattr(Wpre, "ndim", None) != 2:
                raise ValueError(f"{mod_name}: init expects SVD (U,Vt,log_s) but pretrained w is not 2D: {getattr(Wpre,'shape',None)}")

            # inside load_svd_from_pretrained, in the init_is_svd branch:
            Wpre_for_svd = Wpre if Wpre.shape[0] == leaf["U"].shape[0] else Wpre.T

            U, log_s, Vt = svd_factorize(Wpre_for_svd, rank=rank)
            leaf["U"] = U
            leaf["log_s"] = log_s
            leaf["Vt"] = Vt

        # else: init leaf is something unexpected; leave as-is

        out[mod_name] = leaf

    return out


'''
Definition of the SVD Linear module. During initialization the module 
will define the weights for each layer in the appropriate form.

@param[in] output_size : the output dimension of the linear layer.
@param[in] rank : the rank for the SVD decomposition. 
@param[in] with_bias : whether to include a bias term in the linear transformation.
@param[in] name : the name of the module.
''' 
class SVDLinear(hk.Module):
    def __init__(self, output_size, rank=None, with_bias=True, name=None):
        super().__init__(name=name)
        self.output_size = output_size
        self.rank = rank
        self.with_bias = with_bias

    def __call__(self, x):
        in_size = x.shape[-1]
        r = self.rank if self.rank is not None else min(self.output_size, in_size)

        if in_size == 1: # definition of first layer.
            W = hk.get_parameter("w", (self.output_size,), init=hk.initializers.RandomNormal(1e-2))
            if self.with_bias:
                b = hk.get_parameter("b", (self.output_size,), init=jnp.zeros)
                y = x * W + b
        elif self.output_size == 1: # definition of last layer.
            W = hk.get_parameter("w", (in_size,), init=hk.initializers.RandomNormal(1e-2))
            if self.with_bias:
                b = hk.get_parameter("b", (1,), init=jnp.zeros)
                y = jnp.dot(x, W) + b
        else: # definition of hidden layers.
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

'''
Definition of the network architecture that takes advantage of the previous SVD module.

@param[in] output_sizes : the output dimension of the layers.
@param[in] activation : the activation function.
@param[in] rank : the rank for the SVD decomposition.
@param[in] name : the name of the network. Could be e.g. potential energy.
''' 
class SVDMLP(hk.Module):
    def __init__(self, output_sizes, activation, rank=None, name=None):
        super().__init__(name=name)
        self.output_sizes = output_sizes
        self.activation = activation
        self.rank = rank

    def __call__(self, x):
        for i, out_size in enumerate(self.output_sizes[:-1]):
            # apply linear and activation for all layers except the last one.
            x = SVDLinear(out_size, rank=self.rank, with_bias=True, name=f"linear_{i}")(x)
            x = self.activation(x) 
        x = SVDLinear(self.output_sizes[-1], rank=self.rank, with_bias=True,
                      name=f"linear_{len(self.output_sizes)-1}")(x)
        return x
    


def input_transform_matrix(q, n_dof, actuator_dof, shape, activation, rank = None):
    n_output = n_dof * actuator_dof
    net = SVDMLP(output_sizes=shape + (n_output, ),  # shape defined the layers and their neural numbers
                      activation=activation,
                      rank=rank,
                      name="input_transform_matrix")

    raw_A = net(q)
    A = raw_A.reshape(n_dof, actuator_dof)

    A = A + 1e-4
    return A

def potential_energy(q, shape, activation, rank = None):
    net = SVDMLP(output_sizes=shape +(1, ),
                      activation=activation,
                      rank=rank,
                      name="potential_energy")
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


