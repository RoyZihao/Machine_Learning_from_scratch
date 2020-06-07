import numpy as np
from scipy.stats import beta,t,norm

#define a random funciton
def y(x):
    ret = .15*np.sin(40*x)
    ret = ret + .25*np.sin(10*x)
    step_fn1 = np.zeros(len(x))
    step_fn1[x >= .25] = 1
    step_fn2 = np.zeros(len(x))
    step_fn2[x >= .75] = 1
    ret = ret - 0.3*step_fn1 + 0.8 *step_fn2
    return ret

x = np.arange(0.0, 1.0, 0.001)
plt.plot(x, y(x))
plt.show()


#generate x from beta prior
def data_sampler( a = 1,b = 1):
    while True:
        yield beta.rvs(a, b)

#Add gaussian noise
def noise_normal(mean = 0, var = 0.1):
    while True:
        yield norm.rvs(mean, var)


n_data = 150
beta_a,beta_b = 1,1
noise_mean , noise_var=0, 0.1

#generate training and testing data
data_x = np.array([data_sampler(beta_a,beta_b).next() for i in range(n_data)])
data_y = y(data_x) + np.array([noise_normal(noise_mean,noise_var).next() for i in range(n_data)])

x_train,x_test = data_x[:100],data_x[100:]
y_train,y_test = data_y[:100],data_y[100:]

np.savetxt("train.txt",np.vstack([x_train,y_train]).transpose())
np.savetxt("test.txt",np.vstack([x_test,y_test]).transpose())