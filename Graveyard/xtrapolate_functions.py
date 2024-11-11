#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 13 15:58:35 2024

@author: thomastaylor
"""

from tkinter import *
from tkinter.simpledialog import askfloat


import numpy as np

from sklearn.linear_model import LinearRegression

import random

#dummy data 





#functions: to do move over to oop structure

    
    
def game_start():
    
    
    chosen_db = random.choice(list(db_dict.values()))
    
    
    cdb = chosen_db.copy()
    
    xlab = cdb['xlabs'][0]
    
    ylab = cdb['ylabs'][0]
    
    plot_title = cdb['plot_title']
    
    
    
    X_train = cdb['X_data'][0:-5]
    X_test = cdb['X_data'][-5:]
    
    Y_train = cdb['Y_data'][0:-5]
    Y_test = cdb['Y_data'][-5:]
    
    
    
    
    

    game_v_3(X_train, X_test, Y_train, Y_test, xlab, ylab, plot_title)
    
    
    return


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


def m_scoring(pred, actual, weights):
    #mapes scoring, not yet implemented
    
    score = 0
    #avoid divide by 0
   
    for i in range(0, len(pred)):
        if actual[i] == 0:
            actual[i] = 0.0000000000001
            
        score +=abs((actual[i] - pred[i])/actual[i])*weights[i]

    
    
    grade = (score)*100/len(pred)
    
    grade = round(grade, 3)
    
    return grade
   

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


def game_v_3(X_train, X_test, Y_train, Y_test, xlab = 'X', ylab = 'Y', plot_title = None ):
    
    #inital plot to show user
    
    fig = plot_init()
    plot(fig, X_train, Y_train, xlab, ylab, plot_title, 'blue')
    
    canvas, toolbar = canvas_init(fig)
    guess_list = get_guess(ylab, X_test)
    
    
    
    
    
    
    ybar = (sum(Y_train)+sum(Y_test))/(len(Y_train)+len(Y_test))
    
    #weights will eventually be modifable based on if score bonus should be given to accuracy further out from source data.
    weights = np.ones(len(X_test))

    
    
    user_score = scoring(guess_list, Y_test, weights, ybar)
    
    
    intercept, coefficients, ML_pred = auto_reg_lin(X_train, Y_train, X_test)
    
    

    
    ML_Score = scoring(ML_pred, Y_test, weights, ybar)
    
    #remove previous plot
    
    close_canvas(canvas, fig, toolbar)
    
    #plot new values
    
    after_plot(fig, X_train, Y_train, X_test, Y_test, guess_list, ML_pred, xlab, ylab, plot_title)
    
    canvas, toolbar = canvas_init(fig)
    #plt.legend([Y_test, guess_list, ML_pred], ['Actual', 'Player guess', 'ML Guess'])
    

    
    score = Label(window, text=f"User score was {user_score}. ML score was {ML_Score}.")
    
    score.pack()
    
    
    if user_score >= ML_Score: 
        winner = Label(window, text='User wins.')
    else:
        winner = Label(window, text='Regresion wins.')
    
    
    winner.pack()
    
    return
    
