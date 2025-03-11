import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.interpolate import griddata
from example_lib.ex_utility import get_feature, capcurve_filter


def dict_condition(bat_dict, channel_name, cycle_norm):
    now_dict = bat_dict[channel_name]
    norm_dict = get_feature(now_dict, cycle_norm)

    if norm_dict == {}:
        return (None for _ in range(6))

    charge_capacity = norm_dict['charge_capacity']
    charge_v = norm_dict['charge_voltage']
    discharge_capacity = norm_dict['discharge_capacity']
    discharge_v = norm_dict['discharge_voltage']

    if type(charge_capacity) == int or type(discharge_capacity) == int:
        return (None for _ in range(6))

    return now_dict, norm_dict, charge_capacity, charge_v, discharge_capacity, discharge_v


def plot_capacity_curve(batch_map, exp_norm, ch_norm, cy_norm):   
    fig, axes = plt.subplots(1, 2, figsize=(12, 6), constrained_layout=True, squeeze=False)

    if exp_norm not in batch_map.keys():
        print(f"No {exp_norm} Experiment exists!")

        return
    
    bat_dict = batch_map[exp_norm]

    if ch_norm not in bat_dict.keys():
        print(f"No {ch_norm} Channel exists!")

        return

    now_dict, _, charge_capacity, charge_v, discharge_capacity, discharge_v = dict_condition(bat_dict, ch_norm, cy_norm)

    if now_dict is None:
        print(f"Something went wrong!")

        return
    
    axes[0][0].plot(charge_capacity, charge_v)
    axes[0][0].set_title(exp_norm, fontsize=6)
    axes[0][0].grid(alpha=0.4)
    axes[0][1].plot(discharge_capacity, discharge_v)
    axes[0][1].set_title(exp_norm, fontsize=6)
    axes[0][1].grid(alpha=0.4)

    fig.suptitle(f'{cy_norm}th cycle charge/discharge capacity vs. voltage', fontsize=19)
    fig.supxlabel('Charge/Discharge capacity', fontsize=12)
    fig.supylabel('Voltage[V]', fontsize=12)

    return


def all_charge_curve_fig(batch_map):
    fig, axes = plt.subplots(1, 1, figsize=(12, 6), constrained_layout=True, squeeze=False)

    for batch_name, bat_dict in batch_map.items():
        print(f"<all_charge_curve_fig> {batch_name}, {len(bat_dict)}")

        channel_names = bat_dict.keys()

        for channel_name in channel_names:
            now_dict = bat_dict[channel_name]
            _, _, validate_cycle = capcurve_filter(now_dict)

            for cy in range(1, 150):
                if cy not in validate_cycle:
                    continue

                _, norm_dict, charge_capacity, charge_v, _, _ = dict_condition(bat_dict, channel_name, cy)

                if norm_dict is None:
                    continue
                
                axes[0][0].plot(charge_capacity, charge_v)

    axes[0][0].grid(alpha=0.4)
    axes[0][0].set_title("charge curve", fontsize=6)  

    axes[0][0].set_xlim(-10, 220)
    axes[0][0].set_ylim(2.75, 4.75)

    fig.suptitle(f'Charge capacity vs. voltage', fontsize=19)
    fig.supxlabel('Charge capacity', fontsize=12)
    fig.supylabel('Voltage[V]', fontsize=12)

    return


def cycle_characteristic(batch_map, cumul=0):
    fig, axes = plt.subplots(1, 1, figsize=(12, 6), constrained_layout=True, squeeze=False)
    cycle_cnt, cycle_mx = [0 for _ in range(1000)], 0

    for bat_dict in batch_map.values():
        channel_names = bat_dict.keys()

        for channel_name in channel_names:
            cycles = np.array(bat_dict[channel_name]["TotlCycle"], dtype=np.int32).reshape(-1)

            if cycles.shape[0] == 0:
                continue

            now_mx = np.max(cycles)
            cycle_mx = max(cycle_mx, now_mx)

            tp = (now_mx - 1) // 20

            if cumul == 0:
                cycle_cnt[tp] += 1
            else:
                for i in range(0, tp + 1):
                    cycle_cnt[i] += 1

    if cycle_mx == 0:
        raise Exception("Wrong Data!")

    tp = (cycle_mx - 1) // 20
    cycle_cnt = cycle_cnt[0:tp + 1]
    
    axes[0][0].set_xticks(range(0, tp + 1), [f"{1 + 20 * i} - {19 + 20 * i}th cycles" for i in range(tp + 1)])
    axes[0][0].bar(range(0, tp + 1), cycle_cnt)

    fig.suptitle('A number of cycles in experiments' + ((", cumulative") if cumul else ""), fontsize=19)
    fig.supxlabel('Cycle', fontsize=12)
    fig.supylabel('# of exps', fontsize=12)

    return


def triangle_figure(df, interp_period=10000, method="cubic"):
    mean_d = df.groupby(["st_pos", "inp_len"])["rmse"].mean().reset_index()

    a_values = np.linspace(mean_d['st_pos'].min(), mean_d['st_pos'].max(), 1000)
    b_values = np.linspace(mean_d['inp_len'].min(), mean_d['inp_len'].max(), 1000)
    A, B = np.meshgrid(a_values, b_values)

    C = griddata((mean_d['st_pos'], mean_d['inp_len']), mean_d['rmse'], (A, B), method=method)

    norm = mcolors.Normalize(vmin=0, vmax=4) #rmse bar range
    plt.figure(figsize=(10, 8)) #figure size

    print(f"<triangle_figure> Now generating figure...")

    sc = plt.imshow(C, extent=(mean_d['st_pos'].min(), mean_d['st_pos'].max(), mean_d['inp_len'].min(), mean_d['inp_len'].max()), origin='lower', aspect='auto', alpha=0.7, norm=norm)

    ticks_location = np.linspace(0, interp_period, 5) #interpolated capacity range
    ticks_labels = np.round(np.linspace(3.0, 4.5, 5), 3) #voltage range

    plt.xticks(ticks_location, ticks_labels)

    plt.jet()
    plt.xlabel('Starting Voltage(V)')
    plt.ylabel('Input Length(%)')

    cbar = plt.colorbar(sc)
    cbar.set_label("mean of rmse")

    start_pos_percent = [0, 100]
    input_len_percent = [0, 100] #100 - start_pos[1] makes isosceles triangle
    
    plt.xlim(interp_period * start_pos_percent[0] * 0.01, interp_period * start_pos_percent[1] * 0.01) #x-axis visual cutting range(ex, 0, 3850)
    plt.ylim(input_len_percent[0], input_len_percent[1]) #y-axis visual cutting range(ex, 44, 92)

    print("<triangle_figure> figure generated")

    return


def plot_comp_with_target_cycle(data, data_cy_info, data2, data2_cy_info, fir, sec, target_cycle=[1]):
    fig, axes = plt.subplots(1, 1, figsize=(12, 6), constrained_layout=True, squeeze=False)

    for i in range(len(data_cy_info)):
        if data_cy_info[i] in target_cycle:
            axes[0][0].scatter(data[i][fir], data[i][sec], c='black')
    
    for i in range(len(data2_cy_info)):
        if data2_cy_info[i] in target_cycle:
            axes[0][0].scatter(data2[i][fir], data2[i][sec], c='orange')

    fig.suptitle('Component Prediction Result', fontsize=19)
    fig.supxlabel(f'Component {fir + 1}', fontsize=12)
    fig.supylabel(f'Component {sec + 1}', fontsize=12)

    return
