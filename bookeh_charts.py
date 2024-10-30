#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 25 15:22:27 2024

@author: thomastaylor
"""

# Importing required functions
import random

from flask import Flask
from flask import Flask,render_template,request
from bokeh.embed import components
from bokeh.plotting import figure

import numpy as np


from sklearn.linear_model import LinearRegression

import random

from xtrapolate_functions import scoring, auto_reg_lin
# Flask constructor
app = Flask(__name__)



#dummy data 


X_train = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

X_test = [10, 11, 12, 13, 14]

Y_train = [2, 5, 6, 9, 10, 12, 14, 15, 18, 20]
Y_test = [22, 24, 26, 28, 30]


Plot_Title = 'Dummy Data'
# Root endpoint
@app.route('/')
def homepage():

	# Creating Plot Figure
	p = figure(height=350, sizing_mode="stretch_width")

	# Defining Plot to be a Scatter Plot
	p.circle(
		[i for i in X_train],
		[j for j in Y_train],
		size=20,
		color="navy",
		alpha=0.5
	)

	# Get Chart Components
	script, div = components(p)

	# Return the components to the HTML template
	return f'''
	<html lang="en">
		<head>
			<script src="https://cdn.bokeh.org/bokeh/release/bokeh-3.4.3.min.js"></script>
			<title>Bokeh Charts</title>
		</head>
		<body>
			<h1> Graph of {Plot_Title} </h1>
			{ div }
			{ script }
            
            
            <h1> Submit a prediction for Y at the following X values </h1>
            
            <form action="/display" method = "POST">
    <p> {X_test[0]} <input type = "text" name = "g1" /></p>
    <p> {X_test[1]} <input type = "text" name = "g2" /></p>
    <p> {X_test[2]} <input type = "text" name = "g3" /></p>
    <p> {X_test[3]} <input type = "text" name = "g4" /></p>
    <p> {X_test[4]} <input type = "text" name = "g5" /></p>
    <p><input type = "submit" value = "Submit" /></p>
    </form>
            
		</body>
	</html>
	'''


@app.route('/display/', methods = ['POST', 'GET'])
def display():
    if request.method == 'GET':
        return f"The URL /data is accessed directly. Try going to '/form' to submit form"
    if request.method == 'POST':
        user_guesses = []
        
        user_guesses.append(float(request.form.get("g1")))
        
        user_guesses.append(float(request.form.get("g2")))
        
        user_guesses.append(float(request.form.get("g3")))
        
        user_guesses.append(float(request.form.get("g4")))
        
        user_guesses.append(float(request.form.get("g5")))
        
        
        weights = np.ones(len(X_test))
        
        ybar = (sum(Y_train)+sum(Y_test))/(len(Y_train)+len(Y_test))
        
        user_score = scoring(user_guesses, Y_test, weights, ybar)
        
        
        intercept, coefficients, ML_pred = auto_reg_lin(X_train, Y_train, X_test)
        
        

        
        ML_Score = scoring(ML_pred, Y_test, weights, ybar)
        
        
        p = figure(height=350, sizing_mode="stretch_width")

    	# Defining Plot to be a Scatter Plot
        p.circle(
    		[i for i in X_train],
    		[j for j in Y_train],
    		size=20,
    		color="navy",
    		alpha=0.5
    	)
        
        p.circle(
    		[i for i in X_test],
    		[j for j in Y_test],
    		size=20,
    		color="navy",
    		alpha=0.5
    	)
        
        p.circle(
    		[i for i in X_test],
    		[j for j in user_guesses],
    		size=20,
    		color="green",
    		alpha=0.5
    	)
        
        p.circle(
    		[i for i in X_test],
    		[j for j in ML_pred],
    		size=5,
    		color="red",
    		alpha=0.8
    	)
        
        

    	# Get Chart Components
        script, div = components(p)
        
        
    
        if user_score >= ML_Score: 
            
            return  f'''
    	<html lang="en">
    		<head>
    			<script src="https://cdn.bokeh.org/bokeh/release/bokeh-3.4.3.min.js"></script>
    			<title>Bokeh Charts 2</title>
    		</head>
    		<body>
    			<h1> After </h1>
    			{ div }
    			{ script }
                <h2> User Score was: {user_score}, ML Score Was {ML_Score } <h2>
                
                <h2> User Wins </h2>
                
    		</body>
    	</html>
    	'''
           
        else:
        
        
            return  f'''
    	<html lang="en">
    		<head>
    			<script src="https://cdn.bokeh.org/bokeh/release/bokeh-3.4.3.min.js"></script>
    			<title>Bokeh Charts 2</title>
    		</head>
    		<body>
    			<h1> After </h1>
    			{ div }
    			{ script }
               <h2> User Score was: {user_score}, ML Score Was {ML_Score } <h2>
                
                <h2> ML Wins </h2>
                
    		</body>
    	</html>
    	'''





# Main Driver Function
if __name__ == '__main__':
	# Run the application on the local development server
	app.run(debug=True)
