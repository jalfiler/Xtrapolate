#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 13 15:58:35 2024

@author: thomastaylor
"""

from tkinter import *
from tkinter.simpledialog import askfloat
from matplotlib.figure import Figure 
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg, 
NavigationToolbar2Tk) 

from matplotlib import pyplot as plt

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




#functions: to do move over to oop structure

def close_canvas(canvas, fig, toolbar):
    def _clear():
                canvas.get_tk_widget().pack_forget()
                toolbar.pack_forget()
    _clear()
    
    plt.close(fig)
    return 


def redraw(canvas):
    canvas.draw()
    
    return

def canvas_init(fig):

	#creating the Tkinter canvas 
	# containing the Matplotlib figure 
    canvas = FigureCanvasTkAgg(fig, 
							master = window) 
    canvas.draw() 

	# placing the canvas on the Tkinter window 
    canvas.get_tk_widget().pack() 

	# creating the Matplotlib toolbar 
    toolbar = NavigationToolbar2Tk(canvas, 
								window) 
    toolbar.update() 

	# placing the toolbar on the Tkinter window 
    canvas.get_tk_widget().pack() 
    
    return canvas, toolbar




def plot_init():
    fig = Figure(figsize = (5, 5), dpi = 100)
    
    return fig



def plot(fig, X, Y, xlab, ylab, plot_title, color): 


	# adding the subplot 
    plot1 = fig.add_subplot(111) 

	# plotting the graph 
    plot1.scatter(X, Y, c=color)
    
    '''
    
    fig.add_xlabel(xlab)
    
    fig.ylabel(ylab)
    
    fig.title(plot_title)
    '''
    
def after_plot(fig, X_train, Y_train, X_test, Y_test, guess_list, ML_pred, xlab, ylab, plot_title):
    
    plot1 = fig.add_subplot(111) 
    
    
    plot1.scatter(X_train, Y_train, c='blue')
    plot1.scatter(X_test, Y_test, c='blue')
    
    plot1.scatter(X_test, guess_list, c='green')
    
    plot1.scatter(X_test, ML_pred, c='red')
    
    return
    




def get_guess(YLab, X_test):
    


    guess_list = []
    for i in X_test:
        g = askfloat("Input", f"Enter your guess for {YLab} {i}")
        
        while g == None:
            g = askfloat("Input", f"Enter your guess for {YLab} {i}")
            
    
        guess_list.append(g)
    
    
    return guess_list

    
    
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
    


window = Tk() 

# setting the title 
window.title('Xtrapolate') 

# dimensions of the main window 
window.geometry("1000x800") 




#guess = Entry(window)

# button that displays the plot 
'''plot_button = Button(master = window, 
					command = plot, 
					height = 2, 
					width = 10, 
					text = "Plot") '''


g_button = Button(master = window, 
					command = game_start, 
					height = 2, 
					width = 10, 
					text = "Start guessing")


# place the button 
# in main window 
#plot_button.pack() 

#guess.pack()


g_button.pack()

# run the gui 
window.mainloop() 
