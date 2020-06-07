import matplotlib.pyplot as plt
import numpy.matlib as matlib
from scipy.stats import multivariate_normal
import numpy as np
import support_code


def likelihood_func(w, X, y_train, likelihood_var):
    '''
    Implement likelihood_func. This function returns the data likelihood
    given f(y_train | X; w) ~ Normal(Xw, likelihood_var).

    Args:
        w: Weights
        X: Training design matrix with first col all ones (np.matrix)
        y_train: Training response vector (np.matrix)
        likelihood_var: likelihood variance

    Returns:
        likelihood: Data likelihood (float)
    '''
    
    sigma = np.eye(X.shape[0])*likelihood_var

    likelihood = np.prod(multivariate_normal.pdf(
        y_train.reshape(-1),
        mean = np.asarray(np.dot(X,w)).reshape(-1),
        cov=sigma)
    )


    return likelihood


def get_posterior_params(X, y_train, prior, likelihood_var = 0.2**2):
    '''
    Implement get_posterior_params. This function returns the posterior
    mean vector \mu_p and posterior covariance matrix \Sigma_p for
    Bayesian regression (normal likelihood and prior).

    Note support_code.make_plots takes this completed function as an argument.

    Args:
        X: Training design matrix with first call all ones (np.matrix)
        y_train: Training response vector (np.matrix)
        prior: Prior parameters; dict with 'mean' (prior mean np.matrix)
               and 'var' (prior covariance np.matrix)
        likelihood_var: likelihood variance- default (0.2**2) per the lecture slides

    Returns:
        post_mean: Posterior mean (np.matrix)
        post_var: Posterior mean (np.matrix)
    '''

    post_var = (1.0/likelihood_var * X.T*X + prior['var'].getI()).getI()

    post_mean = (X.T * X + likelihood_var * prior['var'].getI()).getI() * X.T * y_train

    return post_mean, post_var

def get_predictive_params(X_new, post_mean, post_var, likelihood_var = 0.2**2):
    '''
    Implement get_predictive_params. This function returns the predictive
    distribution parameters (mean and variance)

    Args:
        X_new: New observation (np.matrix)
        post_mean, post_var: Returned from get_posterior_params
        likelihood_var: likelihood variance (0.2**2) per the lecture slides

    Returns:
        - pred_mean: Mean of predictive distribution
        - pred_var: Variance of predictive distribution
    '''

    pred_var = X_new.T * post_var * X_new + likelihood_var

    pred_mean = post_mean.T * X_new

    return pred_mean, pred_var

def print_predictive_interval(pred_mean, pred_var):
    '''
    Implement print_predictive_interval. This function simply prints a 95%
    prediction interval given the pred_mean and pred_var returned by get_predictive_params

    Returns:
        - None: just prints interval to stdout.
    '''
    lower = (pred_mean - 1.96*pred_var)[0,0]
    upper = (pred_mean + 1.96*pred_var)[0,0]
    print('95% prediction interval: [{0},{1}]'.format(lower,upper))


if __name__ == '__main__':

    '''
    If your implementations are correct, running
        python problem.py
    inside the Bayesian Regression directory will, for each sigma in sigams_to-test:
        - Generate and show plots
        - Generate 95% predictive interval for the new x_new = 0.1, and
        print to stdout.
    '''

    np.random.seed(46134)
    actual_weights = np.matrix([[0.3], [0.5]])
    data_size = 40
    noise = {"mean":0, "var":0.2 ** 2}
    likelihood_var = noise["var"]
    xtrain, ytrain = support_code_py3.generate_data(data_size,
                                                    noise,
                                                    actual_weights)

    #Question (b)
    sigmas_to_test = [1/2, 1/(2**5), 1/(2**10)]
    for sigma_squared in sigmas_to_test:
        prior = {"mean":np.matrix([[0], [0]]),
                 "var":matlib.eye(2) * sigma_squared}

        post_mean, post_var = get_posterior_params(xtrain, ytrain, prior)

        x_new = actual_weights = np.matrix([[1], [0.1]])
        pred_mean, pred_var = get_predictive_params(x_new, post_mean, post_var)

        print_predictive_interval(pred_mean, pred_var)

        support_code_py3.make_plots(actual_weights,
                                xtrain,
                                ytrain,
                                likelihood_var,
                                prior,
                                likelihood_func,
                                get_posterior_params)
