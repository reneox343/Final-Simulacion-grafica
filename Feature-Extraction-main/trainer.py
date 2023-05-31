import copy, math
import numpy as np
import pandas as pd
import csv
import ast

import matplotlib.pyplot as plt
from ipywidgets import Output
from plt_overfit import overfit_example, output
from lab_utils_logistic import sigmoid
from features import *
import cv2
def compute_cost_logistic_reg(X, y, w, b, lambda_ = 1):
    """
    Computes the cost over all examples
    Args:
    Args:
      X (ndarray (m,n): Data, m examples with n features
      y (ndarray (m,)): target values
      w (ndarray (n,)): model parameters  
      b (scalar)      : model parameter
      lambda_ (scalar): Controls amount of regularization
    Returns:
      total_cost (scalar):  cost 
    """

    m,n  = X.shape
    cost = 0.
    for i in range(m):
        z_i = np.dot(X[i], w) + b                                      #(n,)(n,)=scalar, see np.dot
        f_wb_i = sigmoid(z_i)                                          #scalar
        cost +=  -y[i]*np.log(f_wb_i) - (1-y[i])*np.log(1-f_wb_i)      #scalar
             
    cost = cost/m                                                      #scalar

    reg_cost = 0
    for j in range(n):
        reg_cost += (w[j]**2)                                          #scalar
    reg_cost = (lambda_/(2*m)) * reg_cost                              #scalar
    
    total_cost = cost + reg_cost                                       #scalar
    return total_cost    

def compute_gradient_logistic_reg(X, y, w, b, lambda_): 
    """
    Computes the gradient for linear regression 
 
    Args:
      X (ndarray (m,n): Data, m examples with n features
      y (ndarray (m,)): target values
      w (ndarray (n,)): model parameters  
      b (scalar)      : model parameter
      lambda_ (scalar): Controls amount of regularization
    Returns
      dj_dw (ndarray Shape (n,)): The gradient of the cost w.r.t. the parameters w. 
      dj_db (scalar)            : The gradient of the cost w.r.t. the parameter b. 
    """
    m,n = X.shape
    dj_dw = np.zeros((n,))                            #(n,)
    dj_db = 0.0                                       #scalar

    for i in range(m):
        f_wb_i = sigmoid(np.dot(X[i],w) + b)          #(n,)(n,)=scalar
        err_i  = f_wb_i  - y[i]                       #scalar
        for j in range(n):
            dj_dw[j] = dj_dw[j] + err_i * X[i,j]      #scalar
        dj_db = dj_db + err_i
    dj_dw = dj_dw/m                                   #(n,)
    dj_db = dj_db/m                                   #scalar

    for j in range(n):
        dj_dw[j] = dj_dw[j] + (lambda_/m) * w[j]

    return dj_db, dj_dw  

def gradient_descent(X, y, w_in, b_in, alpha, r_lambda, num_iters): 
    """
    Performs batch gradient descent
    
    Args:
      X (ndarray (m,n)   : Data, m examples with n features
      y (ndarray (m,))   : target values
      w_in (ndarray (n,)): Initial values of model parameters  
      b_in (scalar)      : Initial values of model parameter
      alpha (float)      : Learning rate
      r_lambda (float)     : Regularization rate
      num_iters (scalar) : number of iterations to run gradient descent
      
    Returns:
      w (ndarray (n,))   : Updated values of parameters
      b (scalar)         : Updated value of parameter 
    """
    # An array to store cost J and w's at each iteration primarily for graphing later
    J_history = []
    w = copy.deepcopy(w_in)  #avoid modifying global w within function
    b = b_in
    
    for i in range(num_iters):
        # Calculate the gradient and update the parameters
        dj_db, dj_dw = compute_gradient_logistic_reg(X, y, w, b, r_lambda)   

        # Update Parameters using w, b, alpha and gradient
        w = w - alpha * dj_dw               
        b = b - alpha * dj_db               
      
        # Save cost J at each iteration
        if i<100000:      # prevent resource exhaustion 
            J_history.append( compute_cost_logistic_reg(X, y, w, b, r_lambda) )

        # Print cost every at intervals 10 times or as many iterations if < 10
        if i% math.ceil(num_iters / 10) == 0:
            print(f"Iteration {i:4d}: Cost {J_history[-1]}   ")
        
    return w, b, J_history

def y_change(y, cl):
    """
    Creates an independent y vector that only holds 1's for
    the selected class and zero for the rest
    
    Args:
      y (ndarray (m,)) : target values
      cl (scalar)      : The class we are studying.
      
    Returns:
      y_pr (ndarray (n,))   : Array holding only 1's for the 
                              analyzed class.
    """
    y_pr=[]
    for i in range(0, len(y)):
        if y[i] == cl:
            y_pr.append(1)
        else:
            y_pr.append(0)
    return y_pr


def find_param(X, y):
    """
    Creates the w_i vector for the given class.
    
    Args:
      X (ndarray (m,n)    : Data, m examples with n features
      y (ndarray (m,))    : Target values
      
    Returns:
      theta_list (ndarray (n,)) : This is a matrix that will hold a row for the w values
                                  for every i class. 
    """
    alph = 0.1
    r_lambda = 0.7
    iters = 1000

    y_uniq = list(set(y.flatten()))
    theta_list = []
    for i in y_uniq:
        w_in = np.random.rand(X.shape[1])
        b_in = 0.5

        y_tr = pd.Series(y_change(y, i))
        # y_tr = y_tr[:, np.newaxis]
        np.array(y_tr)[:, np.newaxis]
        print(f"\n\nWe will find the weights for class: {i}")
        theta1, _ , _ = gradient_descent(X, y_tr, w_in, b_in, alph, r_lambda, iters) 
        theta_list.append(theta1)
    return theta_list


def trainModel():
    #load images
    labels = ['Cubism', 'Graffiti', 'PopArt']
    #features
    x_train = []
    #labels
    y_train = []
    norm = 0
    with open("ImagesData.csv", 'r') as file:
        csv_reader = csv.reader(file)
        flag = True
        flagNorm = True
        
        for row in csv_reader:
            if(flag):
                flag = False
                continue
            if(row == []):
                continue
            x = np.array(ast.literal_eval(row[2]))
            # x+=1
            # x = np.log10(x)+1
            #norm = np.linalg.norm(x)
            if(flagNorm):
                flagNorm = False
                norm = np.linalg.norm(x)
            
            x_train.append(x/norm)
            if(row[1] == 'Cubism'):
                y_train.append(0)
            if(row[1] == 'Graffiti'):
                y_train.append(1)
            if(row[1] == 'PopArt'):
                y_train.append(2)
        y_train = np.array(y_train)
        x_train = np.array(x_train)
        theta_list = find_param(x_train, y_train)
        f = open("data.py", "w")
        f.write("import numpy as np\n")
        f.write("norm = "+str(norm)+"\n")
        f.write("theta_list = "+str(theta_list).replace("array","np.array"))
        f.close()

trainModel()