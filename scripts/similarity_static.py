import numpy as np
import pickle
import os
import jax.numpy as jnp
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import time
import sys

# function to make dictionaries in same format for every network.
def make_dict(n):
    source = dict.fromkeys(["potential", "input"])
    source["potential"] = dict.fromkeys(range(0,n[0]))
    source["input"] = dict.fromkeys(range(0,n[1])) # input matrix has one less layer.
    for key_i in source.keys():
        if key_i in "potential":
            source[key_i] = dict.fromkeys(range(0,n[0]))
            for key_j in source[key_i].keys():
                source[key_i][key_j] = dict.fromkeys(["w", "b"])
        elif key_i in "input":
            source[key_i] = dict.fromkeys(range(0,n[1]))
            for key_j in source[key_i].keys():
                source[key_i][key_j] = dict.fromkeys(["w", "b"])
    return source

def copy_dict(n, model):
    temp = make_dict(n)
    # copy the values of the source network
    for k in model.keys():
        if "potential" in k:
            for i, j in enumerate(model["potential"].keys()):
                temp["potential"][i] = model["potential"][j]
        elif "input_transform" in k:
            for i, j in enumerate(model["input_transform"].keys()):
                temp["input"][i] = model["input_transform"][j]
        else:
            print("[ERROR] There are no keys named like that in the source network.")

    return temp


def similarity(dynamic_matrix):
    T = jnp.array(dynamic_matrix).T  # shape: (n_params, n_tasks)
    U_full, S_full, Vt_full = jnp.linalg.svd(T, full_matrices=False)

    results = []

    for j in range(T.shape[1]):
        T_minus = jnp.concatenate([T[:, :j], T[:, j+1:]], axis=1)

        U_minus, S_minus, Vt_minus = jnp.linalg.svd(T_minus, full_matrices=False)
        # top-1 cosine similarity
        cos1 = jnp.abs(jnp.dot(U_full[:, 0], U_minus[:, 0]) /
                    (jnp.linalg.norm(U_full[:, 0]) * jnp.linalg.norm(U_minus[:, 0])))

        # top-2 subspace similarity
        k = 2
        overlap = jnp.linalg.svd(U_full[:, :k].T @ U_minus[:, :k], compute_uv=False)

        results.append({
            "removed_task": j,
            "top1_cosine": cos1,
            "subspace_overlap": overlap
        })
    return results

def plot_similarity(results, title):
    # Extract values
    tasks = [r['removed_task'] for r in results]
    top1 = [float(r['top1_cosine']) for r in results]
    top2 = [float(r['subspace_overlap'][1]) for r in results]

    x = np.arange(len(tasks))   # positions
    width = 0.35                # bar width

    plt.figure()
    plt.bar(x - width/2, top1, width, label="Top-1 direction")
    plt.bar(x + width/2, top2, width, label="2nd direction (k=2)")
    plt.xticks(x, [f"Task {t}" for t in tasks])
    plt.xlabel("Removed Task")
    plt.ylabel("Similarity")
    plt.title("Singular Direction Stability")
    plt.ylim(0, 1)
    plt.title(title)
    # Optional reference line
    plt.axhline(0.9, linestyle='--')
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.5)

def plot_function(dtheta, temp):
    # source parameters (for title)
    E_source = 0.6988  # MPa
    nu_source = 0.44

    # other parameter sets
    E1 = 0.988  # MPa
    nu1 = 0.5

    plt.figure()
    plt.plot(np.abs(dtheta[0]), color="blue", label=f"E={E1} MPa, nu={nu1}")
    plt.axvline(npot_, ls="--", c="red")
    plt.xlabel("Parameter index")
    plt.ylabel("|value|")
    plt.title(f"Source: E={E_source} MPa, nu={nu_source}")
    plt.legend()

    param_labels = [
        f"E={E1} MPa, nu={nu1}",
    ]
    for key_i in temp[0]:

        key_js = list(temp[0][key_i].keys())
        fig, axes = plt.subplots(len(key_js), 1)
        fig.suptitle(key_i)
        if len(key_js) == 1:
            axes = [axes]
        for ax, key_j in zip(axes, key_js):
            for i, dw in enumerate(temp):
                w = np.abs(dw[key_i][key_j]["w"].flatten())
                b = np.abs(dw[key_i][key_j]["b"].flatten())
                vals = np.concatenate([w, b])
                ax.plot(vals, label=f"{key_j} ({param_labels[i]})")
            ax.legend()

    plt.show()


t0 = time.perf_counter()
# Load the weights of all the models.
with open(sys.path[0] + f"/../models/static_models/20260311-125118.jax", 'rb') as f:
    data = pickle.load(f)
    print("[INFO] Source network loaded.")
params_ = {}
params_.update({"S": data["params"]})
print(params_['S'].keys())

# Load models.
with open(sys.path[0] + f"/../models/static_models/20260311-125118.jax", 'rb') as f:
    data = pickle.load(f)
    print("[INFO] Target network 1 loaded.")
params_.update({"T1": data["params"]}) 

with open(sys.path[0] + f"/../models/static_models/20260311-125118.jax", 'rb') as f:
    data = pickle.load(f)
    print("[INFO] Target network 2 loaded.")
params_.update({"T2": data["params"]}) 


# extract number of parameters for each network.
params = data["params"] # get the weights.
npot_, ninp_ = 0, 0
for key in params["potential"].keys():
    npot_+= len(params["potential"][key]["b"])
    npot_+= np.array(params["potential"][key]["w"]).size
for key in params["input_transform"].keys():
    ninp_ += len(params["input_transform"][key]["b"])
    ninp_ += np.array(params["input_transform"][key]["w"]).size

print(f"Number of parameters in potential network is : {npot_}")
print(f"Number of parameters in input network is : {ninp_}")

np_ = len(params_["S"]["potential"].keys())
ni_ = len(params_["S"]["input_transform"].keys())
len_net = [np_, ni_]

dtheta, temp = [], [] # temp is for the plots.
dtheta_pot, dtheta_input = [], []
source = copy_dict(len_net, params_["S"]) 
for i in range(1, len(params_)):
    target = copy_dict(len_net, params_["T"+str(i)])
    print("[INFO] Subtracting the networks.")
    dw = make_dict(len_net)
    for key_i in dw.keys():
        for key_j in dw[key_i].keys():
            dw[key_i][key_j]["w"] = target[key_i][key_j]["w"] - source[key_i][key_j]["w"]
            dw[key_i][key_j]["b"] = target[key_i][key_j]["b"] - source[key_i][key_j]["b"]

    dtheta_task = []
    dp, di = [], []
    for key_i in dw:
        for key_j in dw[key_i]:
            w = dw[key_i][key_j]["w"].flatten()
            b = dw[key_i][key_j]["b"].flatten()
            dtheta_task.append(dw[key_i][key_j]["w"].flatten())
            dtheta_task.append(dw[key_i][key_j]["b"].flatten())
            if "potential" in key_i:
                dp.append(w)
                dp.append(b)
            elif "input" in key_i:
                di.append(w)
                di.append(b)

    dtheta_pot.append(jnp.concatenate(dp))
    dtheta_input.append(jnp.concatenate(di))

    print(f"[INFO] Done subtracting source from Target network {i}.")
    temp.append(dw)
    dtheta.append(jnp.concatenate(dtheta_task))

# plot_function(dtheta, temp)

results_pot = similarity(dynamic_matrix=dtheta_pot)
results_inp = similarity(dynamic_matrix=dtheta_input)
plot_similarity(results_pot, title="potential")
plot_similarity(results_inp, title="input")
plt.show()