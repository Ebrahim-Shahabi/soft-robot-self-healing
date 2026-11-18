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
    source = dict.fromkeys(["mass", "potential", "dissipative", "input"])
    source["mass"] = dict.fromkeys(range(0,n[0])) # plus the output
    source["potential"] = dict.fromkeys(range(0,n[1]))
    source["dissipative"] = dict.fromkeys(range(0,n[2])) # 3 is also the depth of the dissipative network.
    source["input"] = dict.fromkeys(range(0,n[3])) # input matrix has one less layer.
    for key_i in source.keys():
        if key_i in ["mass", "potential", "dissipative"]:
            source[key_i] = dict.fromkeys(range(0,n[0]))
            for key_j in source[key_i].keys():
                source[key_i][key_j] = dict.fromkeys(["w", "b"])
        elif key_i in "input":
            source[key_i] = dict.fromkeys(range(0,n[3]))
            for key_j in source[key_i].keys():
                source[key_i][key_j] = dict.fromkeys(["w", "b"])
    return source

def copy_dict(n, model):
    temp = make_dict(n)
    # copy the values of the source network
    for k in model.keys():
        if "lagrangian" in k:
            for i, j in enumerate(model["lagrangian"].keys()):
                if "mass" in j:
                    temp["mass"][i]["w"] = model["lagrangian"][j]["w"]
                    temp["mass"][i]["b"] = model["lagrangian"][j]["b"]
                elif "potential" in j:
                    temp["potential"][i - n[0]]["w"] = model["lagrangian"][j]["w"]
                    temp["potential"][i - n[0]]["b"] = model["lagrangian"][j]["b"]
        elif "dissipative" in k:
            for i, j in enumerate(model["dissipative"].keys()):
                temp["dissipative"][i] = model["dissipative"][j]
        elif "input_transform" in k:
            for i, j in enumerate(model["input_transform"].keys()):
                temp["input"][i] = model["input_transform"][j]
        else:
            print("[ERROR] There are no keys named like that in the source network.")

    return temp

def plot_function(dtheta, temp):
    # source parameters (for title)
    E_source = 0.6988  # MPa
    nu_source = 0.44

    # other parameter sets
    E1 = 0.988  # MPa
    nu1 = 0.5

    E2 = 0.758  # MPa
    nu2 = 0.3

    E3 = 12
    nu3 = 12

    E4 = 15
    nu4 = 15

    plt.figure()
    plt.plot(np.abs(dtheta[0]), color="blue", label=f"E={E1} MPa, nu={nu1}")
    plt.plot(np.abs(dtheta[1]), color="red", label=f"E={E2} MPa, nu={nu2}")
    plt.plot(np.abs(dtheta[2]), color="green", label=f"E={E3} MPa, nu={nu3}")
    plt.plot(np.abs(dtheta[3]), color="yellow", label=f"E={E4} MPa, nu={nu4}")
    plt.axvline(nmass_, ls="--", c="red")
    plt.axvline(nmass_ + npot_, ls="--", c="blue")
    plt.axvline(nmass_ + npot_ + ndis_, ls="--", c="green")
    plt.xlabel("Parameter index")
    plt.ylabel("|value|")
    plt.title(f"Source: E={E_source} MPa, nu={nu_source}")
    plt.legend()

    param_labels = [
        f"E={E1} MPa, nu={nu1}",
        f"E={E2} MPa, nu={nu2}",
        f"E={E3} MPa, nu={nu3}",
        f"E={E4} MPa, nu={nu4}"
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

t0 = time.perf_counter()
# Load the weights of all the models.
with open(sys.path[0] + f"/../models/one_segment_spatial_soft_robot_delan.jax", 'rb') as f:
    data = pickle.load(f)
    print("[INFO] Source network loaded.")
params_ = {}
params_.update({"S": data["params"]})

# with open(sys.path[0] + f"/../models/one_segment_spatial_soft_robot_delan_no_svd.jax", 'rb') as f:
#     data = pickle.load(f)
#     print("[INFO] Target network 1 loaded.")
# params_.update({"T1": data["params"]}) 

with open(sys.path[0] + f"/../models/one_segment_spatial_soft_robot_delan_no_svd_1.jax", 'rb') as f:
    data = pickle.load(f)
    print("[INFO] Target network 2 loaded.")
params_.update({"T1": data["params"]}) 

with open(sys.path[0] + f"/../models/one_segment_spatial_soft_robot_delan_no_svd_2.jax", 'rb') as f:
    data = pickle.load(f)
    print("[INFO] Target network 3 loaded.")
params_.update({"T2": data["params"]}) 

with open(sys.path[0] + f"/../models/one_segment_spatial_soft_robot_delan_no_svd_3.jax", 'rb') as f:
    data = pickle.load(f)
    print("[INFO] Target network 4 loaded.")
params_.update({"T3": data["params"]}) 

params = data["params"] # get the weights.
print(f"[INFO] Loading all the models took : {time.perf_counter() - t0:.2f}s")
# extract number of parameters for each network.
nmass_, npot_, ndis_, ninp_ = 0, 0, 0, 0
for key in params["lagrangian"].keys():
    if "mass" in key:
        nmass_ += len(params["lagrangian"][key]["b"])
        nmass_ += np.array(params["lagrangian"][key]["w"]).size
    if "potential" in key:
        npot_ += len(params["lagrangian"][key]["b"])
        npot_ += np.array(params["lagrangian"][key]["w"]).size
for key in params["dissipative"].keys():
    ndis_ += len(params["dissipative"][key]["b"])
    ndis_ += np.array(params["dissipative"][key]["w"]).size
for key in params["input_transform"].keys():
    ninp_ += len(params["input_transform"][key]["b"])
    ninp_ += np.array(params["input_transform"][key]["w"]).size

# print(f"Number of parameters in mass network is : {nmass_}")
# print(f"Number of parameters in potential network is : {npot_}")
# print(f"Number of parameters in dissipative network is : {ndis_}")
# print(f"Number of parameters in input network is : {ninp_}")


print("[INFO] Transforming networks to better format.")
# print(params_["S"]['lagrangian'].keys())
nm_ = len([k for k in params_["S"]["lagrangian"].keys() if "mass" in k])
np_ = len([k for k in params_["S"]["lagrangian"].keys() if "potential" in k])
nd_ = len(params_["S"]["dissipative"].keys())
ni_ = len(params_["S"]["input_transform"].keys())
len_net = [nm_, np_, nd_, ni_]

dtheta, temp = [], [] # temp is for the plots.
dtheta_mass, dtheta_pot, dtheta_diss, dtheta_input = [], [], [], []
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
    dm, dp, dd, di = [], [], [], []
    for key_i in dw:
        for key_j in dw[key_i]:
            w = dw[key_i][key_j]["w"].flatten()
            b = dw[key_i][key_j]["b"].flatten()
            dtheta_task.append(dw[key_i][key_j]["w"].flatten())
            dtheta_task.append(dw[key_i][key_j]["b"].flatten())
            if "mass" in key_i:
                dm.append(w)
                dm.append(b)
            elif "potential" in key_i:
                dp.append(w)
                dp.append(b)
            elif "dissipative" in key_i:
                dd.append(w)
                dd.append(b)
            elif "input" in key_i:
                di.append(w)
                di.append(b)

    dtheta_mass.append(jnp.concatenate(dm))
    dtheta_pot.append(jnp.concatenate(dp))
    dtheta_diss.append(jnp.concatenate(dd))
    dtheta_input.append(jnp.concatenate(di))

    print(f"[INFO] Done subtracting source from Target network {i}.")
    temp.append(dw)
    dtheta.append(jnp.concatenate(dtheta_task))
    
# plot_function(dtheta, temp)

# get singular value decomposition of the matrix.
Um, sigma_m, Vtm = jnp.linalg.svd(jnp.array(dtheta_mass).T, full_matrices=False)

# similarity between tasks.
results_mass = similarity(dynamic_matrix=dtheta_mass)
results_pot = similarity(dynamic_matrix=dtheta_pot)
results_diss = similarity(dynamic_matrix=dtheta_diss)
results_inp = similarity(dynamic_matrix=dtheta_input)
plot_similarity(results_mass, title="mass")
plot_similarity(results_pot, title="potential")
plot_similarity(results_diss, title="dissipative")
plot_similarity(results_inp, title="input")
plt.show()


# choose ranks
k_mass = 2
k_pot  = 2
k_diss = 2
k_inp  = 2

# compute SVDs
Um, sigma_m, Vtm = jnp.linalg.svd(jnp.array(dtheta_mass).T, full_matrices=False)
Up, sigma_p, Vtp = jnp.linalg.svd(jnp.array(dtheta_pot).T, full_matrices=False)
Ud, sigma_d, Vtd = jnp.linalg.svd(jnp.array(dtheta_diss).T, full_matrices=False)
Ui, sigma_i, Vti = jnp.linalg.svd(jnp.array(dtheta_input).T, full_matrices=False)

# take top-k
U_mass = np.array(Um[:, :k_mass])
U_pot  = np.array(Up[:, :k_pot])
U_diss = np.array(Ud[:, :k_diss])
U_inp  = np.array(Ui[:, :k_inp])

# save directory
save_dir = os.path.join(sys.path[0], "../models")
os.makedirs(save_dir, exist_ok=True)

# save each separately
np.save(os.path.join(save_dir, "U_mass.npy"), U_mass)
np.save(os.path.join(save_dir, "U_potential.npy"), U_pot)
np.save(os.path.join(save_dir, "U_dissipative.npy"), U_diss)
np.save(os.path.join(save_dir, "U_input.npy"), U_inp)

print("[INFO] Saved U matrices:")
print(" - U_mass.npy")
print(" - U_potential.npy")
print(" - U_dissipative.npy")
print(" - U_input.npy")


