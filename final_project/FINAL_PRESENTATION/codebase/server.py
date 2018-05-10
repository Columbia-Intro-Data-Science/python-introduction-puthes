#!/usr/bin/env python2.7

"""

To run locally:

    python server.py

Go to http://localhost:8111 in your browser.

A debugger such as "pdb" may be helpful for debugging.
Read about it online.
"""

import os
import os.path
import sqlite3
import string
from sqlalchemy import *
from sqlalchemy.pool import NullPool
from flask import Flask, request, render_template, g, redirect, Response
from werkzeug.utils import secure_filename
from flask import send_from_directory, url_for
import sys
import numpy 
import scipy
import math
import random
from sklearn.decomposition import PCA
from sklearn.cross_validation import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn import svm
from sklearn.svm import LinearSVC
from sklearn.svm import SVR
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV

from scipy.misc import imread
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from pylab import *
#from skimage import img_as_float
from PIL import Image
import seaborn as sns
from sklearn.utils import shuffle
import random
import json


DATABASEURI = "postgresql://puthes:data85@localhost/postgres"
#DATABASEURI = "sqlite:///test.db"
engine = create_engine(DATABASEURI)



tmpl_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'templates')
app = Flask(__name__, template_folder=tmpl_dir)

UPLOAD_FOLDER1 = os.path.join(os.environ['HOME'], 'upload/')
print (UPLOAD_FOLDER1)
ALLOWED_EXTENSIONS = set(['jpg','jpeg'])

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER1 

@app.before_request
def before_request():
    """
    This function is run at the beginning of every web request 
    (every time you enter an address in the web browser).
    We use it to setup a database connection that can be used throughout the request.

    The variable g is globally accessible.
    """
    try:
    	g.conn = engine.connect()
    except:
	print "uh oh, problem connecting to database"
	import traceback; traceback.print_exc()
	g.conn = None

@app.teardown_request
def teardown_request(exception):
    """
    At the end of the web request, this makes sure to close the database connection.
    If you don't, the database could run out of memory!
    """
    try:
	g.conn.close()
    except Exception as e:
	pass




#
@app.route('/')
def index():
    """
    request is a special object that Flask provides to access web request information:

    request.method:   "GET" or "POST"
    request.form:     if the browser submitted a form, this contains the data in the form
    request.args:     dictionary of URL arguments, e.g., {a:1, b:2} for http://localhost?a=1&b=2
    See its API: http://flask.pocoo.org/docs/0.10/api/#incoming-request-data
    """  
   
    print (request.args)
    return render_template('index.html')
  
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def processing():
	n=46
	z=100
	z1=200
	prod=z*z1
	data = np.zeros((z1,z))
	for s in range(0,n+1):   #last sample plus 1
	    im =Image.open("/home/sp/data/data/%d.png"%s)
	    im = im.resize((z, z1), Image.ANTIALIAS)
	    im = im.convert('RGB')
    
	    im=numpy.array(im)
    
	    grayscale = numpy.zeros((z,z1))
  
	    grayscale = np.dot(im, [ 0.2989,  0.5870, 0.1140])   
	    grayscale = (grayscale - np.mean(grayscale))/np.std(grayscale)   #standardiz
	    data = np.dstack((data,grayscale))
    

	y = [0,1,1,1,1,2,2,2,2,1,2,2,2,3,3,3,4,4,4,4,3,3,3,1,1,1,1,3,2,3,2,3,2,4,4,1,1,2,2,4,4,2,2,3,4,3,2]

	# 0 is empty

	# 1 is circle

	#2 is triangle

	#3 heart

	#4 is star


	y=np.array((y))
	data=data[:,:,1:n+2]  #last sample plus 2
	data2=reshape(data[:,:,0], prod, 1) #

	for i in range(1,n+1):  #last sample plus 1
    		data2=np.vstack((data2,reshape(data[:,:,i], prod, 1)))

	data3=data2[0:n+2,:]  #last sample plus 2

	data3 = (data3 - np.mean(data3))/np.std(data3)   #standardize
	print (numpy.mean(data3))
	print (numpy.std(data3))

	print ("end")
	################
	#########################################################################
	m=4
	testdata = np.zeros((z1,z))
	#for s in range(1,m+1):   #last sample plus 1
	im =Image.open("/home/sp/upload/circle.jpg")
	im = im.resize((z, z1), Image.ANTIALIAS)
	im = im.convert('RGB')
    	cc = np.array(im)
	grayscale = np.dot(cc, [0.2126, 0.7152, 0.0722])  
  	#print (grayscale.shape)
   	grayscale = (grayscale - np.mean(grayscale))/np.std(grayscale)   #standardize
   	testdata = np.dstack((testdata,grayscale))
     
        im =Image.open("/home/sp/upload/triangle.jpg")
	im = im.resize((z, z1), Image.ANTIALIAS)
	im = im.convert('RGB')
    	cc = np.array(im)
	grayscale = np.dot(cc, [0.2126, 0.7152, 0.0722])  
  	#print (grayscale.shape)
   	grayscale = (grayscale - np.mean(grayscale))/np.std(grayscale)   #standardize
   	testdata = np.dstack((testdata,grayscale))

        im =Image.open("/home/sp/upload/heart.jpg")
	im = im.resize((z, z1), Image.ANTIALIAS)
	im = im.convert('RGB')
    	cc = np.array(im)
	grayscale = np.dot(cc, [0.2126, 0.7152, 0.0722])  
  	#print (grayscale.shape)
   	grayscale = (grayscale - np.mean(grayscale))/np.std(grayscale)   #standardize
   	testdata = np.dstack((testdata,grayscale))

        im =Image.open("/home/sp/upload/star.jpg")
	im = im.resize((z, z1), Image.ANTIALIAS)
	im = im.convert('RGB')
    	cc = np.array(im)
	grayscale = np.dot(cc, [0.2126, 0.7152, 0.0722])  
  	#print (grayscale.shape)
   	grayscale = (grayscale - np.mean(grayscale))/np.std(grayscale)   #standardize
   	testdata = np.dstack((testdata,grayscale))



    

	testdata=testdata[:,:,1:m+2]  #last sample plus 2 
	testdata2=reshape(testdata[:,:,0], prod, 1)

	for i in range(0,m): 
	    #print (i)
	    testdata2=np.vstack((testdata2,reshape(testdata[:,:,i], prod, 1)))
	
	testdata3=testdata2[1:m+1,:]  #last sample plus 1

	testset=testdata3
	testset = (testset - np.mean(testset))/np.std(testset)   #standardize
	

	return data3, y, testset


def svm(data3,y,testset):   #linear support vector machine
	X=data3
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

	from sklearn import svm
	from sklearn.svm import LinearSVC
	from sklearn.svm import SVR

	k_range = list(numpy.logspace(-3, 0, 30))
	param_grid = dict(C=k_range)
                
	clf = LinearSVC(penalty='l2')
	grid = GridSearchCV(clf, param_grid, cv=5, scoring='accuracy')    # find the best k using grid search with cv = 5
	grid.fit(X_train,y_train)
	
	print(grid.best_params_)



	clf = LinearSVC(C=grid.best_params_['C'],random_state=0,penalty='l2')

	#print (clf)
	clf.fit(X_train,y_train)
	svm1 = clf.predict(testset)

	y_pred = clf.predict(X_test)
	cm = confusion_matrix(y_test, y_pred)
	fig = plt.figure()
	plt.matshow(cm)
	plt.title('Confusion Matrix Linear SVM')
	plt.colorbar()
	plt.ylabel('True Label')
	plt.xlabel('Predicated Label')
	plt.savefig('/home/sp/static/svm.png')
        plt.show()
	return svm1

def pca_svm(data3,y,testset):
	X = data3
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

	pca = PCA(n_components=15)
	print (X.shape)
	new=pca.fit(X)
	X_pca = pca.transform(X)
	testset_pca = pca.transform(testset)
	X=X_pca
	from sklearn import svm
	from sklearn.svm import LinearSVC
	from sklearn.svm import SVR
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

	k_range = list(numpy.logspace(-3, 0, 30))
	param_grid = dict(C=k_range)
                
	clf = LinearSVC(penalty='l2')
	grid = GridSearchCV(clf, param_grid, cv=5, scoring='accuracy')    # find the best k using grid search with cv = 5
	grid.fit(X_train,y_train)
	print(grid.best_params_)
	clf = LinearSVC(C=grid.best_params_['C'],random_state=0,penalty='l2')
	clf.fit(X_train,y_train)
	pca_svm1= clf.predict(testset_pca)
	y_pred = clf.predict(X_test)
	cm = confusion_matrix(y_test, y_pred)
	fig = plt.figure()
	plt.matshow(cm)
	plt.title('Confusion Matrix SVM PCA')
	plt.colorbar()
	plt.ylabel('True Label')
	plt.xlabel('Predicated Label')
	plt.savefig('/home/sp/static/pca_svm.png')
        plt.show()
	return pca_svm1

def logistic(data3,y,testset):   #logreg
	X = data3
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

	k_range = list(numpy.logspace(-5, 1, 30))
	param_grid = dict(C=k_range)
                
	clf = LogisticRegression(penalty='l2')
	grid = GridSearchCV(clf, param_grid, cv=5, scoring='accuracy')    # find the best k using grid search with cv = 5
	grid.fit(X_train,y_train)
	
	print(grid.best_params_)

	alpha_best = grid.best_params_['C']
	clf = LogisticRegression(penalty='l2', C=alpha_best)  # c = 1.0 default
	clf.fit(X_train,y_train)
	coef=clf.coef_
	logisticregression_pred = clf.predict(testset)
	print (logisticregression_pred)

	y_pred = clf.predict(X_test)
	cm = confusion_matrix(y_test, y_pred)

	
	fig = plt.figure()
	plt.matshow(cm)
	plt.title('Confusion Matrix Logistic regression')
	plt.colorbar()
	plt.ylabel('True Label')
	plt.xlabel('Predicated Label')
	plt.savefig('/home/sp/static/logreg.png')
	plt.show()	

	return logisticregression_pred



def knn(data3,y,testset):
	X = data3
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

 
	k_range = list(range(1, 7))
	param_grid = dict(n_neighbors=k_range)
                
	knn = KNeighborsClassifier()
	grid = GridSearchCV(knn, param_grid, cv=5, scoring='accuracy')    # find the best k using grid search with cv = 5
	grid.fit(X_train,y_train)
	
	print(grid.best_params_)
	k = (grid.best_params_['n_neighbors'])
       
	model = KNeighborsClassifier(n_neighbors=k)
	model.fit(X_train,y_train)
	prediction1 = model.predict(X_test)
	prediction = model.predict(testset)
	cm = confusion_matrix(y_test, prediction1)

	fig = plt.figure()
	plt.matshow(cm)
	plt.title('Confusion Matrix KNN ')
	plt.colorbar()
	plt.ylabel('True Label')
	plt.xlabel('Predicated Label')
	plt.savefig('/home/sp/static/knn.png')
        plt.show()
	return prediction

@app.route('/', methods=['GET','POST'])
def upload_file():
    name_a = request.form['name_a1']   # name
    name_b = request.form['name_b1']  # location
    name_c = request.form['name_c1']   # neurological difficult
    name_d = request.form['name_d1']   # special details
    g.conn.execute("INSERT INTO submission VALUES(%s, %s, %s, %s)",(name_a, name_b, name_c, name_d))
                                     
    temp = 0
    name=0
    name1=0
    name2=0
    name3=0

    file1 = '/home/sp/static/knn.png'
    file2 = '/home/sp/static/logreg.png'
    file3 = '/home/sp/static/static.png'
    file4 = '/home/sp/static/svm.png'
    try:
    	os.remove(file1)
    except OSError:
    	pass
    try:
    	os.remove(file2)
    except OSError:
    	pass
    try:
    	os.remove(file3)
    except OSError:
    	pass
    try:
    	os.remove(file4)
    except OSError:
    	pass
    context = dict(data=str(name),data1=str(name1),data2=str(name2),data3=str(name3))
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            #return redirect(request.url)
            return render_template("index.html", **context)
        file = request.files['file']
        # if user does not select file, browser also
        # submit a empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            temp = temp+1
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            x = str(filename)
            y = str(temp)
            name = y + ', ' + x 

	if 'file1' not in request.files:
            flash('No file part')
            #return redirect(request.url)
            return render_template("index.html", **context)
        file1 = request.files['file1']
        # if user does not select file, browser also
        # submit a empty part without filename
        if file1.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file1 and allowed_file(file1.filename):
            temp = temp+1
            filename = secure_filename(file1.filename)
            file1.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            x1 = str(filename)
            y1 = str(temp)
            name1 = y1 + ', ' + x1 

	if 'file2' not in request.files:
            flash('No file part')
            #return redirect(request.url)
            return render_template("index.html", **context)
        file2 = request.files['file2']
        # if user does not select file, browser also
        # submit a empty part without filename
        if file2.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file2 and allowed_file(file2.filename):
            temp = temp+1
            filename = secure_filename(file2.filename)
            file2.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            x1 = str(filename)
            y1 = str(temp)
            name2 = y1 + ', ' + x1 

	if 'file3' not in request.files:
            flash('No file part')
            #return redirect(request.url)
            return render_template("index.html", **context)
        file3 = request.files['file3']
        # if user does not select file, browser also
        # submit a empty part without filename
        if file3.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file3 and allowed_file(file3.filename):
            temp = temp+1
            filename = secure_filename(file3.filename)
            file3.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            x1 = str(filename)
            y1 = str(temp)
            name3 = y1 + ', ' + x1 
            print (file3)
            #return redirect('/')
        
        context = results(name_a)
          
	#context = dict(data=str(name),data1=str(name1),data2=str(name2),data3=str(name3))
	#return name, render_template("index.html", **context)
        return render_template("results.html", **context)

@app.route('/examples')
def examples():            
   	return render_template("examples.html")

@app.route('/pca')
def pca():            
   	return render_template("pca.html")

@app.route('/background')
def background():            
   	return render_template("background.html")

@app.route('/All_submissions')
def All_submissions():        
	names = []
        cursor = g.conn.execute("SELECT * FROM submission")
        for result in cursor:
   	 	dict1 = dict(result.items());
    		z = str(dict1['name'])
    		x = str(dict1['location'])
  	  	w = str(dict1['movement_disorder'])
       		a = str(dict1['special_details'])
        	b = str(dict1['knn_score'])
		c = str(dict1['logistic_regression_score'])
		d = str(dict1['svm_score'])
		e = str(dict1['pca_svm_score'])
        	f = str(dict1['id'])
    		name = z + ' ,  ' + x + '  , ' + w + '  , ' + a + '  , ' + b + '  , ' + c + '  , ' + d + '  , ' + e + '  , ' + f
    		names.append(name)
  	cursor.close()
        context=dict(data=names)
        print (request.args)
    
   	return render_template("All_submissions.html",**context)

import glob
@app.route('/results')
def results(name_a): 
      
        data3,y,testset = processing()
        prediction = knn(data3, y, testset)
	svm1 = svm(data3, y, testset)
        logisticregression_pred = logistic(data3, y, testset)
        pca_svm1 = pca_svm(data3,y,testset)
        
        values = ("circle","triangle","heart","star","nothing")
    	ss=prediction[0]
    	if ss == 1:
        	data = values[0]
    	if ss == 2:
        	data = values[1]
    	if ss == 3:
        	data = values[2]
    	if ss == 4:
        	data = values[3]
	if ss == 0:
        	data = values[4]
        ss=prediction[1]
    	if ss == 1:
        	data1 = values[0]
    	if ss == 2:
        	data1 = values[1]
    	if ss == 3:
        	data1 = values[2]
    	if ss == 4:
        	data1 = values[3]
	if ss == 0:
        	data1 = values[4]
        ss=prediction[2]
    	if ss == 1:
        	data2 = values[0]
    	if ss == 2:
        	data2 = values[1]
    	if ss == 3:
        	data2 = values[2]
    	if ss == 4:
        	data2 = values[3]
	if ss == 0:
        	data2 = values[4]
        ss=prediction[3]
    	if ss == 1:
        	data3 = values[0]
    	if ss == 2:
        	data3 = values[1]
    	if ss == 3:
        	data3 = values[2]
    	if ss == 4:
        	data3 = values[3]
	if ss == 0:
        	data3 = values[4]
        
        score = 0
        if data == values[0]:
        	score = score + 1
	if data1 == values[1]:
        	score = score + 1
        if data2 == values[2]:
        	score = score + 1
        if data3 == values[3]:
        	score = score + 1
        dexterity = (float(score)/4.0 * 100)
############################################################
	ss=svm1[0]
    	if ss == 1:
        	data4 = values[0]
    	if ss == 2:
        	data4 = values[1]
    	if ss == 3:
        	data4 = values[2]
    	if ss == 4:
        	data4 = values[3]
	if ss == 0:
        	data4 = values[4]


	ss=svm1[1]
    	if ss == 1:
        	data5 = values[0]
    	if ss == 2:
        	data5 = values[1]
    	if ss == 3:
        	data5 = values[2]
    	if ss == 4:
        	data5 = values[3]
	if ss == 0:
        	data5 = values[4]


	ss=svm1[2]
    	if ss == 1:
        	data6 = values[0]
    	if ss == 2:
        	data6 = values[1]
    	if ss == 3:
        	data6 = values[2]
    	if ss == 4:
        	data6 = values[3]
	if ss == 0:
        	data6 = values[4]

	ss=svm1[3]
    	if ss == 1:
        	data7 = values[0]
    	if ss == 2:
        	data7 = values[1]
    	if ss == 3:
        	data7 = values[2]
    	if ss == 4:
        	data7 = values[3]
	if ss == 0:
        	data7 = values[4]

	score = 0
        if data4 == values[0]:
        	score = score + 1
	if data5 == values[1]:
        	score = score + 1
        if data6 == values[2]:
        	score = score + 1
        if data7 == values[3]:
        	score = score + 1
        dexterity1 = (float(score)/4.0 * 100)
#############################################################################################################

	ss=logisticregression_pred[0]
    	if ss == 1:
        	data8 = values[0]
    	if ss == 2:
        	data8 = values[1]
    	if ss == 3:
        	data8 = values[2]
    	if ss == 4:
        	data8 = values[3]
	if ss == 0:
        	data8 = values[4]

	ss=logisticregression_pred[1]
    	if ss == 1:
        	data9 = values[0]
    	if ss == 2:
        	data9 = values[1]
    	if ss == 3:
        	data9 = values[2]
    	if ss == 4:
        	data9 = values[3]
	if ss == 0:
        	data9 = values[4]


	ss=logisticregression_pred[2]
    	if ss == 1:
        	data10 = values[0]
    	if ss == 2:
        	data10 = values[1]
    	if ss == 3:
        	data10 = values[2]
    	if ss == 4:
        	data10 = values[3]
	if ss == 0:
        	data10 = values[4]

	ss=logisticregression_pred[3]
    	if ss == 1:
        	data11 = values[0]
    	if ss == 2:
        	data11 = values[1]
    	if ss == 3:
        	data11 = values[2]
    	if ss == 4:
        	data11 = values[3]
	if ss == 0:
        	data11 = values[4]


	score = 0
        if data8 == values[0]:
        	score = score + 1
	if data9 == values[1]:
        	score = score + 1
        if data10 == values[2]:
        	score = score + 1
        if data11 == values[3]:
        	score = score + 1
        dexterity2 = (float(score)/4.0 * 100)

        ########################################################
	ss=pca_svm1[0]
    	if ss == 1:
        	data12 = values[0]
    	if ss == 2:
        	data12 = values[1]
    	if ss == 3:
        	data12 = values[2]
    	if ss == 4:
        	data12 = values[3]
	if ss == 0:
        	data12 = values[4]


	ss=pca_svm1[1]
    	if ss == 1:
        	data13 = values[0]
    	if ss == 2:
        	data13 = values[1]
    	if ss == 3:
        	data13 = values[2]
    	if ss == 4:
        	data13 = values[3]
	if ss == 0:
        	data13 = values[4]


	ss=pca_svm1[2]
    	if ss == 1:
        	data14 = values[0]
    	if ss == 2:
        	data14 = values[1]
    	if ss == 3:
        	data14 = values[2]
    	if ss == 4:
        	data14 = values[3]
	if ss == 0:
        	data14 = values[4]

	ss=pca_svm1[3]
    	if ss == 1:
        	data15 = values[0]
    	if ss == 2:
        	data15 = values[1]
    	if ss == 3:
        	data15 = values[2]
    	if ss == 4:
        	data15 = values[3]
	if ss == 0:
        	data15 = values[4]


	score = 0
        if data12 == values[0]:
        	score = score + 1
	if data13 == values[1]:
        	score = score + 1
        if data14 == values[2]:
        	score = score + 1
        if data15 == values[3]:
        	score = score + 1
        dexterity3 = (float(score)/4.0 * 100)


        #################################################################################

        files = glob.glob('/home/sp/upload/*')
	for f in files:
   		os.remove(f)

        print (name_a)
        print (name_a)
        g.conn.execute("UPDATE submission SET knn_score = %d WHERE name = '%s'" % (dexterity, name_a))
        g.conn.execute("UPDATE submission SET svm_score = %d WHERE name = '%s'" % (dexterity1, name_a))
        g.conn.execute("UPDATE submission SET logistic_regression_score = %d WHERE name = '%s'" % (dexterity2, name_a)) 
        g.conn.execute("UPDATE submission SET pca_svm_score = %d WHERE name = '%s'" % (dexterity3, name_a))

	context = dict(data=data,data1=data1,data2=data2,data3=data3,dexterity=dexterity,data4=data4,data5=data5,data6=data6,data7=data7,data8=data8,data9=data9,data10=data10,data11=data11,dexterity1=dexterity1,dexterity2=dexterity2,dexterity3=dexterity3,data12=data12,data13=data13,data14=data14,data15=data15,name=name_a)
	return context
            


	
# Example of adding new data to the database
@app.route('/add', methods=['POST'])
def add():
  name = request.form['name']
  g.conn.execute('INSERT INTO test VALUES (NULL, ?)', name)
  return redirect('/')



@app.route('/login')
def login():
    abort(401)
    this_is_never_executed()


if __name__ == "__main__":
  import click

  @click.command()
  @click.option('--debug', is_flag=True)
  @click.option('--threaded', is_flag=True)
  @click.argument('HOST', default='0.0.0.0')
  @click.argument('PORT', default=8111, type=int)
  def run(debug, threaded, host, port):
    """
    This function handles command line parameters.
    Run the server using:

        python server.py

    Show the help text using:

        python server.py --help

    """

    HOST, PORT = host, port
    print "running on %s:%d" % (HOST, PORT)
    app.run(host=HOST, port=PORT, debug=debug, threaded=threaded)


  run()
