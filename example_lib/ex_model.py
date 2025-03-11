import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import root_mean_squared_error
from sklearn import linear_model


class Model:
    def __init__(self, dataloader):
        self.dl = dataloader
        self.model = None

        return


    def get_mlr_weight(self):
        return (self.model.intercept_, self.model.coef_) 


    def get_linear(self, n_jobs=-1, fit_intercept=True):
        return linear_model.LinearRegression(fit_intercept=fit_intercept, n_jobs=n_jobs)
    

    def get_linear_param(self):
        return {}, {}


    def get_param(self, model_type):
        param = None

        if model_type == "linear":
            param, cv_param = self.get_linear_param()
        else:
            raise Exception(f"No model named {model_type} available!")

        return param, cv_param


    def get_model(self, model_type, param={}):
        model = None

        if model_type == "linear":
            model = self.get_linear(**param)
        else:
            raise Exception(f"No model named {model_type} available!")

        return model

    
    def get_train_idx(self, train_q0, train_rate):
        sample_len = self.dl.x_val.shape[1]

        if train_q0 < 0 or train_q0 >= sample_len:
            raise Exception(f"<get_train_idx> Not a valid q0 point! should be {0} <= {train_q0} <= {self.dl.x_val.shape}")
        
        now_percentile = ((train_q0 + 1) / sample_len) * 100

        if train_rate < 0:
            print(f"<get_train_idx> Not a valid range! should be {0} <= {train_rate}")
        elif train_rate >= 100 - now_percentile:
            print(f"<get_train_idx> Beyond the valid length; Skip this point.")

            return 0, None

        train_len = min(sample_len - (train_q0 + 1), round((sample_len / 100) * train_rate))
        train_idx = np.arange(train_q0, min(sample_len, train_q0 + train_len))

        return 1, train_idx


    def train(self, now_curve, train_H, train_idx, model_type="linear", verbose=0):
        model_x_train = train_H[:, train_idx].T
        model_y_train = now_curve[:, train_idx].T

        param, cv_param = self.get_param(model_type)
        self.model = self.get_model(model_type, param)

        clf = GridSearchCV(self.model, param_grid=cv_param, cv=min(len(model_x_train), 5), scoring="neg_root_mean_squared_error", n_jobs=-1, refit=True, verbose=0)
        clf.fit(model_x_train, model_y_train)

        if verbose:
            print(f"<train> score: {-clf.cv_results_['mean_test_score']}, best_model: {clf.best_estimator_.coef_}")

        return clf.best_estimator_

    
    def valid(self, now_curve, train_H, best_model, mean_curve, mx_cap=0, interp_period=10000):
        train_H = train_H.T
        recon_curve = best_model.predict(train_H).reshape(1, -1)

        recon_curve += mean_curve
        now_curve += mean_curve

        if mx_cap == 1:
            valid_score = root_mean_squared_error(now_curve[-1], recon_curve[-1])
        else:
            valid_score = root_mean_squared_error(now_curve[:, -interp_period:].reshape(-1), recon_curve[:, -interp_period:].reshape(-1))

        return valid_score, recon_curve
