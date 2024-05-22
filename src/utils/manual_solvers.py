import numpy as np
from scipy import stats
from numpy.linalg import inv
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error


class bayes_calculator:

    def __init__(self, 
                 pool_ws, # poolsize * X_dim
                 pool_sigs, # poolsize
                 bat_size,
                 max_context_len
                 ):
        self.pool_ws = pool_ws
        self.pool_size = len(pool_ws)
        self.x_dim = pool_ws.shape[1]
        self.pool_sigs = pool_sigs
        self.bat_size = bat_size
        self.max_context_len = max_context_len

        self.posterior_records = np.zeros((bat_size, max_context_len, self.pool_size))

        self.log_like_records = np.zeros((bat_size, max_context_len, self.pool_size))

        self.seen_context_sample = 0

    def posterior_calculator(self, num_samp):
        assert num_samp <= self.seen_context_sample

        # loglike_list = self.log_like_records[:, :num_samp]

        # log_aggre = np.exp(loglike_list.sum(axis=1))
        # log_norm = log_aggre.sum(axis=1)

        # prob = log_aggre / np.tile(log_norm.reshape(-1,1), (1, log_aggre.shape[1]))

        final_loglike = self.log_like_records[:, num_samp-1, :]

        if num_samp == 1:

            past_posterior = np.ones((self.bat_size, self.pool_size)) / self.pool_size

        else:

            past_posterior = self.posterior_records[:, num_samp-2, :]

        raw_prob = np.clip(past_posterior * np.exp(final_loglike), a_min = 1e-8, a_max = 1e6)

        raw_norm = raw_prob.sum(axis=1)

        # print(min(raw_norm))

        prob = raw_prob / np.tile(raw_norm.reshape(-1,1), (1, raw_prob.shape[1]))

        return prob
    
    
    def update(self, xs, ys):

        assert xs.shape == (self.bat_size, self.x_dim)
        assert len(ys) == self.bat_size

        # update self.log_like_records[:, self.seen_context_sample, :]
        preds = xs @ self.pool_ws.T  # bat_size * pool_size
        ys = np.tile(ys.reshape(-1,1), (1, self.pool_size))
        full_sigs = np.tile(self.pool_sigs.reshape(1, -1), (self.bat_size, 1))

        self.log_like_records[:, self.seen_context_sample, :] = stats.norm.logpdf(preds - ys, scale=full_sigs)

        # update self.posterior_records[:, self.seen_context_sample, :]

        self.seen_context_sample += 1


        self.posterior_records[:, self.seen_context_sample-1, :] = self.posterior_calculator(self.seen_context_sample)

        

    def pred_mu_sig(self, xs):

        assert xs.shape == (self.bat_size, self.x_dim)

        if self.seen_context_sample >= 1:

            posteriors = self.posterior_records[:, self.seen_context_sample-1, :]
        
        else:

            posteriors = np.ones((self.bat_size, self.pool_size)) / self.pool_size

        # pred_w, bat_size * x_dim
        pred_w = posteriors @ self.pool_ws

        pred_ys = np.diag(pred_w @ xs.T)

        # calculate sigs
        respective_preds = xs @ self.pool_ws.T
        pred_ys_expand = np.tile(pred_ys.reshape(-1, 1), (1, self.pool_size))

        bias_terms = (respective_preds - pred_ys_expand)**2  # bat_size * pool_size
        var_terms = np.tile((self.pool_sigs**2).reshape(1, -1), (self.bat_size, 1))

        estimate_sigs = np.sqrt(np.diag((bias_terms + var_terms) @ posteriors.T))

        return pred_ys, estimate_sigs
    


def solve_linear_regression(common_X, diff_y):

    samp_num, dim = common_X.shape

    samp_num_2, batch_num = diff_y.shape

    assert samp_num == samp_num_2
    
    # Estimate parameters using closed-form solution
    W = np.linalg.inv(common_X.T @ common_X) @ common_X.T @ diff_y  # shape dim * batch_num


    # Calculate residuals
    residuals = diff_y - common_X @ W
    
    # Estimate noise variance (sigma)
    sigma_squared = np.sum(residuals**2, axis = 0) / (common_X.shape[0] - common_X.shape[1])
    
    return W, np.sqrt(sigma_squared)




def w_sig_NGgenerator(a0, b0, mu0, x_dim, N_samples):

    k = a0
    theta = 1/b0


    taus = np.random.gamma(shape = k, scale =  theta, size = N_samples)

    sigs = np.sqrt(1/taus)

    ws_raw = np.random.normal(size=(N_samples, x_dim))

    ws = ((ws_raw.T) @ np.diag(sigs) + np.tile(mu0, (N_samples, 1)).T).T

    # ws = np.expand_dims(ws, axis = -1)

    return ws, sigs


def w_PriorGenerator(mu0, x_dim, N_samples, Sigma0 = None):

    if Sigma0 is None:
        Sigma0 = np.eye(x_dim)

    ws_raw = np.random.multivariate_normal(mu0, Sigma0, N_samples)

    return ws_raw


def w_Posterior(Xs, Ys, mu0, Sigma_noise, Sigma0=None):

    prompt_len, x_dim = Xs.shape
    if Sigma0 is None:
        Sigma0 = np.eye(x_dim)

    estimated_ws = np.zeros((prompt_len, x_dim))
    
    current_sigma = Sigma0
    current_sumXY = np.linalg.inv(Sigma0) @ mu0
    
    for i in range(prompt_len):

        update_sigma = np.linalg.inv(
            (1/Sigma_noise[0,0]) * np.array([Xs[i]]).T @ np.array([Xs[i]]) + np.linalg.inv(current_sigma)
        )

        update_sumXY = current_sumXY + Ys[i] * np.linalg.inv(Sigma_noise) @ Xs[i]

        estimated_ws[i] = update_sigma @ update_sumXY

        current_sigma = update_sigma
        current_sumXY = update_sumXY

    return estimated_ws







def w_sig_NGposterior(Xs, Ys, a0, b0, mu0, MAP_sig=False, bias_adjust=False, pred_Ys=None):

    if bias_adjust:
        assert len(Ys) == len(pred_Ys)

    current_a = a0
    current_b = b0
    current_mu = mu0

    prompt_len, x_dim = Xs.shape
    current_Lambda = np.diag(np.ones(x_dim))

    # records
    estimated_as = np.zeros(prompt_len)
    estimated_bs = np.zeros(prompt_len)
    estimated_mus = np.zeros((prompt_len, x_dim))
    estimated_Lambdas = np.zeros((prompt_len, x_dim, x_dim))


    estimated_ws = np.zeros((prompt_len, x_dim))
    estimated_sigs = np.zeros(prompt_len)
    estimated_sigs_foreseeNext = np.zeros(prompt_len)
    estimated_sigs_foreseeNext_sigAdjust = np.zeros(prompt_len)

    for i in range(prompt_len):

        # update with new sample
        update_Lambda = np.array([Xs[i]]).T @ np.array([Xs[i]]) + current_Lambda
        update_a = current_a + 1/2
        update_mu = inv(update_Lambda) @ (current_Lambda @ current_mu + Ys[i] * Xs[i])
        update_b = current_b + 1/2 * (Ys[i]**2 + np.array([current_mu]) @ current_Lambda @ current_mu - np.array([update_mu]) @ update_Lambda @ update_mu)

        MAP_w = update_mu
        MAP_sigma = np.sqrt(update_b/(update_a-1))


        estimated_as[i] = update_a
        estimated_bs[i] = update_b
        estimated_mus[i] = update_mu
        estimated_Lambdas[i] = update_Lambda
        estimated_ws[i] = MAP_w
        estimated_sigs[i] = MAP_sigma

        current_a = update_a
        current_b = update_b
        current_mu = update_mu
        current_Lambda = update_Lambda

        if i <= prompt_len-2:
            next_x = Xs[i+1]
            
            estimated_sigs_foreseeNext[i] = np.sqrt(current_b/(current_a-1) * (1+ np.trace(np.reshape(next_x, (-1, 1))  @ np.reshape(next_x, (1, -1)) @ np.linalg.inv(current_Lambda))))

            if bias_adjust:
                next_pred_y = pred_Ys[i+1]
                estimated_sigs_foreseeNext_sigAdjust[i] = np.sqrt( \
                    current_b/(current_a-1) * (1+ np.trace(np.reshape(next_x, (-1, 1))  @ np.reshape(next_x, (1, -1)) @ np.linalg.inv(current_Lambda))) \
                        + (np.dot(next_x, update_mu) -next_pred_y)**2)


    if MAP_sig:
        return estimated_ws, estimated_sigs
    elif bias_adjust:
        return estimated_ws, estimated_sigs_foreseeNext_sigAdjust
    else:
        return estimated_ws, estimated_sigs_foreseeNext


def w_sig_ridge(Xs, Ys, alpha = 0.1):

    prompt_len, x_dim = Xs.shape

    estimated_ws = np.zeros((prompt_len, x_dim))
    estimated_sigs = np.zeros(prompt_len)

    for i in range(prompt_len):

        ridge = Ridge(alpha=alpha, fit_intercept=False)
        X_train = Xs[:i+1]
        y_train = Ys[:i+1]
        ridge.fit(X_train, y_train)
        beta = ridge.coef_

        estimated_ws[i] = beta

        # Get predicted values
        y_pred = ridge.predict(X_train)

        # Compute residuals
        residuals = y_train - y_pred

        # Compute residual sum of squares
        rss = np.sum(residuals ** 2)

        # Compute degrees of freedom
        df = max(1, len(y_train) - X_train.shape[1] - 1)  # Number of observations - Number of parameters (including intercept) - 1

        # Estimate variance of noise term
        variance_noise = rss / df
        estimated_sigs[i] = np.sqrt(variance_noise)

    

    
    return estimated_ws, estimated_sigs