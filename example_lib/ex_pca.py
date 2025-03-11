import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from example_lib.ex_load import pkl_load, pkl_save


def get_pca(rawdata, n_comp, trans=0, save_opt=0, load_opt=0, path=None, name=None):
    pre_model = []
    model = []

    if load_opt:
        try:
            comp, comp_weight, model = pkl_load(path, name)
        except Exception as e:
            raise Exception(f"<get_pca> Something is wrong while making a PCA model. {e}")
    else:
        if trans == 1:
            try:
                pre_model.append(('Scaling', StandardScaler()))
                pre_model = Pipeline(pre_model)
                dataset = pre_model.fit_transform(rawdata)
            except Exception as e:
                raise Exception(f"<get_pca> Something is wrong while preprocessing a dataset. {e}")
        else:
            dataset = rawdata

        try:
            model.append(('PCA', PCA(n_components=n_comp, random_state=42)))
            
            print(f"<get_pca> PCA model: {model}")

            model = Pipeline(model)
            comp = model.fit_transform(dataset)
            comp_weight = model["PCA"].components_

            if save_opt:
                pkl_save(path, name, [comp, comp_weight, model])
        except Exception as e:
            raise Exception(f"<get_pca> Something is wrong while making a PCA model. {e}")

    print(f"<get_pca> W: {comp.shape}, H: {comp_weight.shape}")
    #print(f"<get_pca> variation:", model["PCA"].explained_variance_ratio_)

    return comp, comp_weight, model


def comp_shape(comp, comp_weight, model, n_comp, exp_info, interp_period=10000, sample_size=3):
    fig, axes = plt.subplots(n_comp, sample_size, figsize=(15, 10), squeeze=False)

    for k in range(n_comp):
        tmp = comp[:, k].reshape(-1, 1)
        tmp2 = comp_weight[k, :].reshape(1, -1)

        now_comp = np.dot(tmp, tmp2) # component without mean
        
        for t in range(sample_size):
            axes[k][t].plot(np.arange(interp_period), now_comp[t, :], label=str(k + 1))

            if k == 0:
                axes[k][t].set_title(exp_info[t], fontsize=6)
    
    fig.suptitle(f'Component', fontsize=19)
    plt.spring()

    return comp, comp_weight, model


def plot_comp_vs_comp(data, exp_info, cy_info, capacity, fir, sec, cycle=0, origin=0):
    min_cap, max_cap = min(capacity), max(capacity)
    mn_dict = {}

    for i, (exp, cy) in enumerate(zip(exp_info, cy_info)):
        if exp not in mn_dict:
            mn_dict[exp] = [data[i][fir], data[i][sec], cy]
        else:
            if cycle and mn_dict[exp][2] > cy:
                mn_dict[exp] = [data[i][fir], data[i][sec], cy]
            elif cycle == 0:
                mn_dict[exp] = [min(mn_dict[exp][0], data[i][fir]), min(mn_dict[exp][1], data[i][sec]), cy]

    x_mn, x_mx, y_mn, y_mx = float('inf'), -float('inf'), float('inf'), -float('inf')

    cmap = plt.get_cmap('coolwarm')
    norm = plt.Normalize(vmin=min_cap, vmax=max_cap)

    fig, ax = plt.subplots(1, 1, figsize=(12, 6), constrained_layout=True, squeeze=False)

    for i in range(data.shape[0]):
        firval, secval = data[i][fir], data[i][sec]

        if origin:
            firval -= mn_dict[exp_info[i]][0]
            secval -= mn_dict[exp_info[i]][1]

        x_mn = min(x_mn, firval)
        x_mx = max(x_mx, firval)
        y_mn = min(y_mn, secval)
        y_mx = max(y_mx, secval)

        ax[0][0].scatter(firval, secval, c=capacity[i], cmap=cmap, norm=norm)

    ax[0][0].set_xlim(x_mn - 10, x_mx + 10)
    ax[0][0].set_ylim(y_mn - 10, y_mx + 10)
    ax[0][0].set_xlabel(f'Component {fir + 1}')
    ax[0][0].set_ylabel(f'Component {sec + 1}')

    fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=plt.gca())

    return


def recon_cumul_comp(dataset, comp, comp_weight, model, n_comp, exp_info, interp_period=10000, sample_size=3):
    n_comps = range(1, n_comp + 1)
    fig, axes = plt.subplots(1, sample_size, figsize=(15, 8), squeeze=False)

    for t in range(sample_size):
        axes[0][t].plot(dataset[t, :], np.arange(interp_period), label=str(0))
        axes[0][t].set_title(exp_info[t], fontsize=6)

    for t in range(sample_size):
        for n_comp in n_comps:
            recon_q = np.dot(comp[:, :n_comp], comp_weight[:n_comp, :]) + model["PCA"].mean_
            axes[0][t].plot(recon_q[t, :], np.arange(interp_period), label=str(n_comp))

    fig.suptitle(f'Reconstruction with PCA', fontsize=19)
    plt.spring()
    plt.legend()
    
    return comp, comp_weight, model


def recon_each_comp(dataset, comp, comp_weight, model, n_comp, exp_info, interp_period=10000, sample_size=1):
    n_comps = range(1, n_comp + 1)
    fig, axes = plt.subplots(sample_size, n_comp + 1, figsize=(12, 6), squeeze=False)

    for t in range(sample_size):
        for p in range(0, n_comp + 1):
            axes[t][p].plot(dataset[t, :], np.arange(interp_period), label=str(0))
            axes[t][p].set_title(exp_info[t], fontsize=6)

    for n_comp in n_comps:
        recon_q = np.dot(comp[:, n_comp - 1].reshape(-1, 1), comp_weight[n_comp - 1, :].reshape(1, -1)) + model["PCA"].mean_

        for t in range(sample_size):
            axes[t][n_comp].plot(recon_q[t, :], np.arange(interp_period), label=str(n_comp))

    fig.suptitle(f'Reconstruction with PCA', fontsize=19)
    plt.spring()
    plt.legend()
    
    return comp, comp_weight, model


def correlation(data, capacity, avv_info):
    label = ["Capacity", "Average Voltage", "PC1", "PC2", "PC3", "PC4", "PC5", "PC6", "PC7"]
    corr_feat = np.transpose(np.concatenate([capacity.reshape(-1, 1), avv_info.reshape(-1, 1), data], axis=1))
    correlation_matrix = np.abs(np.corrcoef(corr_feat))

    plt.figure(figsize=(10, 8))
    plt.title('Correlation Matrix Visualization(Abs)')
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5, xticklabels=label, yticklabels=label)

    return


def comp_vs_cap(comp, cy_info, capacity, target_cy):
    fig, ax = plt.subplots(1, 1, figsize=(12, 6), constrained_layout=True, squeeze=False)
    fig.suptitle(f'PC1 Vs. Capacity, {target_cy}th Cycle')

    for i in range(comp.shape[0]):
        if cy_info[i] == target_cy:
            ax[0][0].scatter(comp[i, 0], capacity[i], c=capacity[i])

    return


def do_pca(dataset, comp, comp_weight, model, n_comp, mode, exp_info, interp_period):
    if len(dataset) == 0:
        print(f"<pca_fig> No data is available!")
        
        return

    if mode == "comp_shape":
        comp_shape(comp, comp_weight, model, n_comp, exp_info, interp_period)
    elif mode == "recon_cumul_comp":
        recon_cumul_comp(dataset, comp, comp_weight, model, n_comp, exp_info, interp_period)
    elif mode == "recon_each_comp":
        recon_each_comp(dataset, comp, comp_weight, model, n_comp, exp_info, interp_period)

    return
