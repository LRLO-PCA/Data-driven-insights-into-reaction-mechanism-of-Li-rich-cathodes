import numpy as np
from tqdm import tqdm


def get_feature(raw_dict, cycle_norm=1):
    if len(raw_dict['Current[mA]']) < cycle_norm:
        return {}

    now_dict = raw_dict
    zero_out = np.where(now_dict['Voltage[V]'][cycle_norm - 1] == 0)[0].copy()

    if len(zero_out):
        for col in now_dict.columns:
            if col not in ["TotlCycle", "constant[g]"]: # hard-coded feature selection
                now_dict[col][cycle_norm - 1] = np.delete(now_dict[col][cycle_norm - 1], zero_out)

    battery_constant = now_dict['constant[g]'][0]
    current_g = now_dict['Current[mA]'][cycle_norm - 1] / battery_constant

    passtime_diff = np.hstack(([0], np.diff(now_dict['PassTime[Sec]'][cycle_norm - 1])))
    delta_capacity = passtime_diff * current_g / 3600
    delta_power = now_dict['Voltage[V]'][cycle_norm - 1] * delta_capacity

    condition = now_dict['Condition'][cycle_norm - 1]
    delta_sign_capacity = np.copy(delta_capacity)
    delta_sign_capacity[condition == 2] = -delta_capacity[condition == 2]

    delta_sign_power = delta_power
    delta_sign_power[condition == 2] = -delta_power[condition == 2]

    accumulated_power = np.cumsum(delta_sign_power)

    try:
        acc_capacity_stidx = np.where(condition == 1)[0][0]
        acc_capacity_edidx = np.where(condition == 1)[0][-1]

        charge_range = range(acc_capacity_stidx, acc_capacity_edidx + 1)
        charge_passtime = now_dict['PassTime[Sec]'][cycle_norm - 1][charge_range]
        charge_voltage = now_dict['Voltage[V]'][cycle_norm - 1][charge_range]
        charge_capacity = np.cumsum(delta_capacity[charge_range])
    except:
        charge_range = range(-1, 0)
        charge_passtime, charge_voltage, charge_capacity = 0, 0, 0

    try:
        acc_capacity_upidx = np.where(condition == 2)[0][0]
        acc_capacity_downidx = np.where(condition == 2)[0][-1]

        discharge_range = range(acc_capacity_upidx, acc_capacity_downidx + 1)
        discharge_passtime = now_dict['PassTime[Sec]'][cycle_norm - 1][discharge_range]
        discharge_voltage = now_dict['Voltage[V]'][cycle_norm - 1][discharge_range]
        discharge_capacity = np.cumsum(delta_capacity[discharge_range])
    except:
        discharge_range = range(-1, 0)
        discharge_passtime, discharge_voltage, discharge_capacity = 0, 0, 0

    return {'battery_constant': battery_constant, 'current_g':current_g, 'passtime_diff':passtime_diff, 'delta_capacity':delta_capacity, 'delta_power':delta_power, 'condition':condition, 
            'delta_sign_capacity':delta_sign_capacity, 'charge_capacity':charge_capacity, 'discharge_capacity':discharge_capacity,
            'delta_sign_power':delta_sign_power, 'accumulated_power':accumulated_power, 'charge_range':charge_range, 'discharge_range':discharge_range,
            'charge_voltage': charge_voltage, 'discharge_voltage': discharge_voltage, 'charge_passtime': charge_passtime, 'discharge_passtime': discharge_passtime}


def dqdv_uniform_and_interpolation(norm_dict, period=50, opt=1, ch_cut=[3.0, 4.5], dch_cut=[3.0, 4.5]):
    dty = "float64"

    charge_passtime = norm_dict['charge_passtime']
    charge_v = norm_dict['charge_voltage']
    charge_capacity = norm_dict['charge_capacity']

    discharge_passtime = norm_dict['discharge_passtime']
    discharge_v = norm_dict['discharge_voltage']
    discharge_capacity = norm_dict['discharge_capacity']

    if period <= 0:
        return (charge_v, charge_capacity, discharge_v, discharge_capacity)
    
    if opt == 1:
        charge_norm, discharge_norm = charge_passtime, discharge_passtime
    elif opt == 2:
        charge_norm, discharge_norm = charge_v, discharge_v

    if type(charge_norm) == int or type(discharge_norm) == int:
        return ([] for _ in range(4))

    if opt == 1:
        charge_uniform_interval = np.linspace(max(np.min(charge_norm), ch_cut[0]), min(np.max(charge_norm), ch_cut[1]), num=period)
        discharge_uniform_interval = np.linspace(max(np.min(discharge_norm), dch_cut[0]), min(np.max(discharge_norm), dch_cut[1]), num=period)
    elif opt == 2:
        charge_uniform_interval = np.linspace(ch_cut[0], ch_cut[1], num=period)
        discharge_uniform_interval = np.linspace(dch_cut[0], dch_cut[1], num=period)

    charge_voltage_interpolate = np.interp(charge_uniform_interval, np.array(charge_norm, dtype=dty), np.array(charge_v, dtype=dty))
    charge_capacity_interpolate = np.interp(charge_uniform_interval, np.array(charge_norm, dtype=dty), np.array(charge_capacity, dtype=dty))

    discharge_voltage_interpolate = np.interp(discharge_uniform_interval, np.array(discharge_norm, dtype=dty), np.array(discharge_v, dtype=dty))
    discharge_capacity_interpolate = np.interp(discharge_uniform_interval, np.array(discharge_norm, dtype=dty), np.array(discharge_capacity, dtype=dty))

    if opt == 1:
        return (charge_voltage_interpolate, charge_capacity_interpolate, discharge_voltage_interpolate, discharge_capacity_interpolate)
    else:
        return (charge_uniform_interval, charge_capacity_interpolate, discharge_uniform_interval, discharge_capacity_interpolate)


def capcurve_filter(now_dict):
    x_list, disc_list = [], []
    first_cycle = 10
    pre_cap = -1
    validate_cycle = []

    for i in range(len(now_dict)):
        if i + 1 < first_cycle:
            continue

        norm_dict = get_feature(now_dict, i + 1)

        if norm_dict == {}:
            break

        ch_cap = np.max(norm_dict['charge_capacity'])
        
        if ((i + 1 == first_cycle and ch_cap <= 160) or (ch_cap >= 250 or ch_cap <= 150)):
            break

        if ch_cap >= 0:
            if pre_cap != -1 and pre_cap < ch_cap and ch_cap >= pre_cap * 1.025:
                continue
            elif pre_cap != -1 and pre_cap * 1.25 < ch_cap:
                break
            elif pre_cap * 0.8 > ch_cap:
                break
                       
            x_list.append(i + 1)
            disc_list.append(ch_cap)
            validate_cycle.append(i + 1)

            pre_cap = ch_cap

    if len(x_list):
        x_list.pop()
        disc_list.pop()
        validate_cycle.pop()

    return x_list, disc_list, validate_cycle


def avg_capacity(cap, vol):
    sm = np.sum(np.diff(cap) * vol[:-1])
    sm /= cap[-1]

    return sm


def get_curvedata(batch_map, channel_wise=0, channel_mode=1, interp_period=10000):
    dataset = []
    exp_info, ch_info, cy_info, avv_info, bat_info = [], [], [], [], []

    for batch_name, bat_dict in tqdm(batch_map.items()):
        channel_names = bat_dict.keys()

        for channel_name in channel_names:
            ch_sample = []

            if channel_name not in bat_dict.keys():
                continue

            now_dict = bat_dict[channel_name]

            _, _, validate_cycle = capcurve_filter(now_dict)

            for cycle_norm in range(1, len(now_dict) + 1):
                if cycle_norm not in validate_cycle:
                    continue

                norm_dict = get_feature(now_dict, cycle_norm)

                if norm_dict == {}:
                    break

                charge_voltage, charge_capacity, _, _ = dqdv_uniform_and_interpolation(norm_dict, interp_period, 2)

                if len(charge_voltage) >= 2:
                    if channel_wise == 0:
                        dataset.append(charge_capacity)
                        exp_info.append(batch_name)
                        ch_info.append(channel_name)
                        cy_info.append(cycle_norm)
                        avv_info.append(avg_capacity(charge_capacity, charge_voltage))
                        bat_info.append(norm_dict["battery_constant"])
                    elif channel_wise > 0:
                        if channel_mode == 0:
                            now_dat = [np.max(charge_capacity)]
                        elif channel_mode == 1:
                            now_dat = charge_capacity

                        ch_sample.extend(now_dat)

            if channel_wise > 0:
                if channel_mode == 0 and len(ch_sample) >= channel_wise:
                    ch_sample = ch_sample[:channel_wise]
                elif channel_mode == 1 and (len(ch_sample) // interp_period) >= channel_wise:
                    ch_sample = ch_sample[:channel_wise * interp_period]

                dataset.append(ch_sample)
                exp_info.append(batch_name)
                ch_info.append(channel_name)

    dataset = np.array(dataset)
    cy_info = np.array(cy_info)
    ch_info = np.array(ch_info)
    exp_info = np.array(exp_info)
    avv_info = np.array(avv_info)
    bat_info = np.array(bat_info)

    return dataset, exp_info, ch_info, cy_info, avv_info, bat_info


def valid_indexing(dl, mx_sample=5):
    cnt = np.array([True for _ in range(dl.x_val.shape[0])])
    cycle_num = np.where(cnt == True)[0]

    if len(cycle_num) > mx_sample:
        cycle_num = cycle_num[:mx_sample]

    return cycle_num


def get_sort_value(data, norm="mean"):
    if norm == "mean":
        print(f"<get_sort_value> mean rmse: {np.mean(data)}")
    elif norm == "median":
        print(f"<get_sort_value> median rmse: {np.median(data)}")
    elif norm == "all":
        print(f"<get_sort_value> mean rmse: {np.mean(data)}")
        print(f"<get_sort_value> median rmse: {np.median(data)}")

        lst = [25, 75, 90, 95]

        for norm in lst:
            print(f"<get_sort_value> Above About {norm}% of the rmse: {np.percentile(data, norm, method='nearest')}")
    else:
        try:
            percent_value = int(norm)
        except:
            raise Exception(f"<get_sort_value> No measurity is matched. Check the measurity option.")
        
        target_rmse = np.percentile(data, percent_value, method="nearest")

        print(f"<get_sort_value> Above About {norm}% of the rmse: {target_rmse}")

    return
