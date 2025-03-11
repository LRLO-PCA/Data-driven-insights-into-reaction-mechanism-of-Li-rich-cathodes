import pandas as pd
import numpy as np
from example_lib.ex_utility import valid_indexing, get_sort_value
from example_lib.ex_figure import triangle_figure
from example_lib.ex_model import Model


def predict(pred_model, train_H, pca_model, q0, q_rate, mx_sample=5, verbose=0):
    is_valid, train_idx = pred_model.get_train_idx(q0, q_rate)

    if is_valid == 0:
        if verbose:
            print(f"<operate> {q0, q_rate} pair is not a valid constraint.")

        return np.nan, 0, None

    print(f"<operate> {q0}, {q_rate}; model train/valid begins...")

    mean_curve = pca_model["PCA"].mean_.reshape(1, -1)
    cnt = valid_indexing(pred_model.dl, mx_sample)

    valid_scores, now_curves, recon_curves, weights = [[] for _ in range(4)]

    for i, curve_idx in enumerate(cnt):
        now_curve = pred_model.dl.x_val[curve_idx, :] - mean_curve

        pred_model.model = pred_model.train(now_curve, train_H, train_idx)
        now_valid_score, recon_curve = pred_model.valid(now_curve, train_H, pred_model.model, mean_curve)

        valid_scores.append(now_valid_score)
        now_curves.append(now_curve)
        recon_curves.append(recon_curve)
        weights.append(pred_model.get_mlr_weight()[1])

        if verbose and (len(cnt) < 100 or i % 20 == 0):
            print(f"<operate> {100 * i / len(cnt)} / 100% Done...")

    if len(valid_scores) == 0:
        raise Exception(f"<operate> No result is available. Check the filtering option.")
    
    if verbose:
        get_sort_value(valid_scores)

    return np.mean(valid_scores), 1, np.array(weights).squeeze(1)


def get_triangle(dataloader, train_H, pca_model, xmax, xinv, ymax, yinv, mx_sample=5, method="cubic", interp_period=10000, fail_thres=100):
    column = ["st_pos", "inp_len", "rmse"]
    res_df = pd.DataFrame([], columns=column)

    for q0 in range(1, xmax + 1, xinv):
        for q_rate in range(1, ymax + 1, yinv):
            pred_model = Model(dataloader)

            res, suc, _ = predict(pred_model, train_H, pca_model, q0, q_rate, mx_sample)

            if suc == 0:
                break
            
            res = fail_thres if res > fail_thres else res

            now_row = [q0, q_rate, res]
            res_df = pd.concat([res_df, pd.DataFrame([now_row], columns=column)], axis=0)

    triangle_figure(res_df, interp_period, method)

    return res_df
