#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 11 01:33:22 2024

@author: thomastaylor

Version 0.0.3 of Xtrapolate functions for prompting, graphing, and scoring.

"""
#function imports
from matplotlib import pyplot as plt

import pandas as pd

import numpy as np

from sklearn.linear_model import LinearRegression

import random


#dummy data 

X_train = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

X_test = [10, 11, 12, 13, 14]

Y_train = [2, 5, 6, 9, 10, 12, 14, 15, 18, 20]
Y_test = [22, 24, 26, 28, 30]

dummy_dict = {}


dummy_dict['xlabs'] = ['Dummy X']

dummy_dict['ylabs'] = ['Dummy Y']

dummy_dict['plot_title'] = 'Graph of Dummy Data'

dummy_dict['DS_Name'] = 'dummy_dict'

dummy_dict['X_data'] = range(0, 15)

dummy_dict['Y_data'] = [2, 5, 6, 9, 10, 14, 14, 15, 18, 20, 22, 24, 26, 28, 30]



db_dict = {}

db_dict['dummy_dict'] = dummy_dict

dummy_dict2 = dummy_dict.copy()


dummy_dict2['plot_title'] = 'Graph of Dummy Data 2'

dummy_dict2['Y_data'] = [21, 49, 62, 93, 104, 141, 149, 156, 180, 200, 220, 240, 260, 280, 300]

db_dict['dummy_dict2'] = dummy_dict2


##########################
'''
Section 2: Back end function
'''

def g_prompt(X_test):
    guess_list = []
    for i in X_test:
        while True:
            try: 
                guess = float(input(f"What do you predict Y's value will be at {i}?: "))
                print("\n")
            except ValueError:
                print('Input must be a number.\n')
                continue
            else:
                break
            
        guess_list.append(guess)
            
        
    return guess_list



def scoring(pred, actual, weights, ybar):
    #This is the equivalent of the R2 score just out of 100
    
    score = 0
    #avoid divide by 0
    mean_var = 0.0000000000001
    for i in range(0, len(pred)):
        
        score += ((pred[i]-actual[i])**2)*weights[i]
        mean_var += ((actual[i]-ybar)**2)*weights[i]
        
    norm_score = score/mean_var
    
    grade = (1-norm_score)*100
    
    grade = round(grade, 3)
    
    return grade
   




def first_plot_scatter(X_train, Y_train, xlab, ylab, plot_title):
    
    plt.scatter(X_train, Y_train, c='blue')
    
    plt.xlabel(xlab)
    
    plt.ylabel(ylab)
    
    plt.title(plot_title)
    
    plt.show()
    
    
    return


def auto_reg_lin(x, y, X_test):
    
    x = np.array(x)
    y = np.array(y)
    
    X_test = np.array(X_test)
    
    while True:
        try: 
            model = LinearRegression().fit(x, y)
            
            
            intercept, coefficients = model.intercept_, model.coef_
            
            pred = model.predict(X_test) 
        except ValueError:
            #if x is one dimensional
            x = x.reshape(-1, 1)
            
            X_test = X_test.reshape(-1, 1)
            continue
        else:
            break

    
    return intercept, coefficients, pred


def game_v_1(X_train, X_test, Y_train, Y_test, xlab = 'X', ylab = 'Y', plot_title = None ):
    
    #inital plot to show user
    
    plt.Figure()
    first_plot_scatter(X_train, Y_train, xlab, ylab, plot_title)
    
    guess_list = g_prompt(X_test)
    
    plt.cla()
    
    
    ybar = (sum(Y_train)+sum(Y_test))/(len(Y_train)+len(Y_test))
    
    #weights will eventually be modifable based on if score bonus should be given to accuracy further out from source data.
    weights = np.ones(len(X_test))

        
    
    user_score = scoring(guess_list, Y_test, weights, ybar)
    
    
    intercept, coefficients, ML_pred = auto_reg_lin(X_train, Y_train, X_test)
    
    
    ML_Score = scoring(ML_pred, Y_test, weights, ybar)
    
    #plot after user guesses
    
    plt.scatter(X_train, Y_train, c='blue')
    
    plt.scatter(X_test, Y_test, c='blue', label = 'Actual')
    
    plt.scatter(X_test, guess_list, c='green', label = 'User Pred')
    
    plt.scatter(X_test, ML_pred, c='red', label = 'ML Pred')
    
    plt.xlabel(xlab)
    
    plt.ylabel(ylab)
    
    plt.title(plot_title)
    
    plt.legend()
    
    #plt.legend([Y_test, guess_list, ML_pred], ['Actual', 'Player guess', 'ML Guess'])
    
    plt.show()
    
    print(f"User score was {user_score}. ML score was {ML_Score}.\n")
    
    
    
    if user_score >= ML_Score: 
        print('User wins.')
    else:
        print('Regression wins.')
    
    
    return
    
    
####################################
'''
Section 3: Initalizing a round
'''

def game_init(chosen_db = None):
    
    if chosen_db == None:
        chosen_db = random.choice(list(db_dict.values()))
    
    cdb = chosen_db.copy()
    
    xlab = cdb['xlabs'][0]
    
    ylab = cdb['ylabs'][0]
    
    plot_title = cdb['plot_title']
    
    
    
    X_train = cdb['X_data'][0:-5]
    X_test = cdb['X_data'][-5:]
    
    Y_train = cdb['Y_data'][0:-5]
    Y_test = cdb['Y_data'][-5:]
    
    
    
    
    

    game_v_1(X_train, X_test, Y_train, Y_test, xlab, ylab, plot_title)
    
    
    
    return
    
    
        
game_init()   
    
