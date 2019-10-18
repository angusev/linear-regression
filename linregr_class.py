import numpy as np
import scipy
from sklearn.base import BaseEstimator
from tqdm import tqdm_notebook
from sklearn.metrics import mean_squared_error


class LinearReg(BaseEstimator):
    def __init__(self, gd_type='stochastic', loss_func='mse', opt=None, tolerance=1e-4, max_iter=1000, 
                 w0=None, alpha=1e-3, eta=1e-3, batch_size=1, eta_degree=None, history=False):
        """
        gd_type: 'full' or 'stochastic'
        opt: None, 'momentum' or 'adam'
        loss_func: 'mse' or 'log_cosh'
        tolerance: for stopping gradient descent
        max_iter: maximum number of steps in gradient descent
        w0: np.array of shape (d) - init weights
        eta: if learning rate is set constantly
        eta_degree: if learning rate changes dynamically: 1 / self.iter ** eta_degree
        alpha: momentum coefficient
        history: save losses at each iteration or not
        """
        self.gd_type = gd_type
        self.opt = opt
        self.loss_func = loss_func
        self.w0 = w0
        
        self.eta = eta
        self.eta_degree = eta_degree
        self.tolerance = tolerance
        self.max_iter = max_iter
        self.w = None
        self.loss_history = None      # list of loss function values at each training iteration
        self.batch_size = batch_size
        self.history = history
        self.iter = 1
        
        #parametrs for Momentum
        self.alpha = alpha
        self.moment = None
        
        # parametrs for Adam
        self.m = None
        self.v = None
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.eps = 1e-8
    
    def fit(self, X_without_bias, y):
        """
        X: np.array of shape (ell, d)
        y: np.array of shape (ell)
        ---
        output: self
        """
        X = np.hstack((X_without_bias, np.ones((X_without_bias.shape[0], 1))))
        if self.w0 is None:
            self.w0 = np.zeros(X.shape[1])
        else:
            self.w0 = np.append(self.w0, [[0]])
            self.w0 = np.array(self.w0)
        self.w0 = np.array([self.w0])
        if self.history:
            self.loss_history = []
        if not(self.eta_degree is None):
            self.eta = 1
        self.w = self.w0
        prev_w = self.w0
        if self.history:
            self.loss_history.append(self.calc_loss(X, y, True))
        self.w = self.step(X, y)
        
        print("{} GD with {} loss founction learning start:".format(self.gd_type, self.loss_func))
        
        for iter_num in tqdm_notebook(range(1, self.max_iter + 1)):
            self.iter = iter_num
            if np.linalg.norm(self.w - prev_w) < self.tolerance:
                print("Distance between w_{} and w_{} is smaller than tolerance.".format(iter_num - 1, iter_num))
                return self
            if not(self.eta_degree is None):
                self.eta = 1 / iter_num ** self.eta_degree
            new_w = self.step(X, y)
            prev_w = self.w
            self.w = new_w
            if self.history:
                self.loss_history.append(self.calc_loss(X, y, True))
        return self
    
    def step(self, X, y):
        gradient = self.calc_gradient(X, y)
        if self.opt == 'momentum':        
            columns = X.shape[1]
            if self.moment is None:
                self.moment = np.zeros((1, columns))
            new_moment = self.alpha * self.moment + self.eta * gradient
            self.moment = new_moment
            new_w = self.w - self.moment
        elif self.opt == 'adam':
            if self.m is None:
                self.m = np.zeros((1, X.shape[1]))
            if self.v is None:
                self.v = np.zeros((1, X.shape[1]))
            self.m = self.beta1 * self.m + (1 - self.beta1) * gradient
            self.v = self.beta2 * self.v + (1 - self.beta2) * gradient ** 2
            m_hat = self.m / (1 - self.beta1 ** self.iter)
            v_hat = self.v / (1 - self.beta2 ** self.iter)
            new_w = self.w - self.eta * m_hat / (v_hat ** 0.5 + self.eps)
        elif self.opt is None:
            new_w = self.w - self.eta * gradient
        else:
            raise Exception('Unknown optimizer')
        return new_w
    
    def predict(self, X, with_bias=False):
        if self.w is None:
            raise Exception('Not trained yet')
        if not(with_bias):
            X_with_bias = np.hstack((X, np.ones((X.shape[0], 1))))
        else:
            X_with_bias = X
        y_pred = X_with_bias.dot(self.w.T).T
        y_pred = np.ravel(y_pred)
        return y_pred
    
    def calc_gradient(self, X_orig, y_orig):
        """
        X: np.array of shape (ell, d) (ell can be equal to 1 if stochastic)
        y: np.array of shape (ell)
        ---
        output: np.array of shape (d)
        """
        if self.gd_type == 'full':
            y = np.array([y_orig]).T
            X = X_orig
        elif self.gd_type == 'stochastic':
            rows = X_orig.shape[0]
            random_rows = np.random.randint(rows, size=self.batch_size)
            X = X_orig[random_rows]
            y = np.array([y_orig]).T[random_rows]
        else:
            raise Exception('Unknown GD type')
        if self.loss_func == 'mse':
            gradient = 2 * X.T.dot(X.dot(self.w.T) - y).T / X.shape[0]
        elif self.loss_func == 'log_cosh':
            gradient = X.T.dot(np.tanh(X.dot(self.w.T) - y)).T / X.shape[0]
        return gradient

    def calc_loss(self, X, y, with_bias=False):
        """
        X: np.array of shape (ell, d)
        y: np.array of shape (ell)
        ---
        output: float 
        """
        y_pred = self.predict(X, with_bias)
        if self.loss_func == 'mse':
            loss = mean_squared_error(y_pred.ravel(), y)
        elif self.loss_func == 'log_cosh':
            loss = np.sum(np.log(np.cosh(y_pred - y))) / len(y)
        else:
            raise Exception('Unknown loss function')
        return loss