import argparse
import numpy as np
import tensorflow.compat.v1 as tf
import matplotlib.pyplot as plt
import warnings
import csv
import sys
from get_data import vehicles_data

warnings.filterwarnings('ignore')
plt.rcParams['figure.figsize'] = (20.0, 10.0)
plt.style.use('ggplot')
tf.disable_v2_behavior()

def predict(X_t, y_t, x_t, k_t):
    neg_one = tf.constant(-1.0, dtype=tf.float64)    
    distances =  tf.reduce_sum(tf.abs(tf.subtract(X_t, x_t)), 1)    
    neg_distances = tf.multiply(distances, neg_one)    
    vals, indx = tf.nn.top_k(neg_distances, k_t)    
    y_s = tf.gather(y_t, indx)
    return y_s
    
    
def get_label(preds):
    counts = np.bincount(preds.astype('int64'))
    return np.argmax(counts)

def cycle_data(data_time,delta,j):     
        
    while  data_time[j+1].hour-data_time[j].hour<=1 \
        and data_time[j+1].day==data_time[j].day:
            delta+=data_time[j+1].second-data_time[j].second
            delta+=(data_time[j+1].minute-data_time[j].minute)*60
            delta+=(data_time[j+1].hour-data_time[j].hour)*3600
            j+=1
    return delta,j
    
def parse_data(vehicle,delta_fuel,delta_time,time_unit,new_delta):

    if time_unit=="m":
        delta_time=delta_time*60
    elif time_unit=="h":
        delta_time=delta_time*3600
    elif time_unit!="s":
        print("Wrong time_unit argument")
        print("Should be <s> or <m> or <h>")
        print("Restart program with correct time_unit argument")
        sys.exit("Error")
        
    f1=open(vehicle.upper()+"_data_delta_time.txt","w")
    f2=open(vehicle.upper()+"_data_delta_fuel_+.txt","w")
    f3=open(vehicle.upper()+"data_delta_fuel_-.txt","w")
    
    data_time=[]
    data_fuel_level=[]
    data_delta_time=[]
    data_delta_fuel_level=[]
    data_delta_fuel_level_neg=[]
    delta=0
    i,j,k=0,0,0
    permission=False
    
    data_time=vehicles_data[vehicle].fuelLevel.DTIME
    data_fuel_level=list(vehicles_data[vehicle].fuelLevel.BEVALUE)
    
    if new_delta:
        print("It will take some time")
        print("Please wait")
        while j!=10:
            k=j
            delta,j=cycle_data(data_time,0,j)      
            data_delta_time.append(delta)                        
            data_delta_fuel_level.append(data_fuel_level[j+1]-data_fuel_level[k])
            delta=0
            f1.write(str(data_delta_time[i])+"\n")
            if data_delta_fuel_level[i]>=0:
                f2.write(str(data_delta_fuel_level[i])+"\n")
            else:
                f3.write(str(data_delta_fuel_level[i])+"\n")
            i+=1
            j+=1
            print(j)
        f1.close()
        f2.close()
        f3.close()
    
    else:
        data_delta_time=f1.readlines()
        data_delta_time=[x.strip() for x in data_delta_time]
        data_delta_fuel_level=f2.readlines()
        data_delta_fuel_level=[x.strip() for x in data_delta_fuel_level]
        f1.close()
        f2.close()
        
        
                
            
    
#def show_data(X1,X2,Y1,Y2,testX,testY):
    
#generated data
#num_points_each_cluster = 100
#mu1 = [-0.4, 3]
#covar1 = [[1.3,0],[0,1]]
#mu2 = [0.5, 0.75]
#covar2 = [[2.2,1.2],[1.8,2.1]]
#X1 = np.random.multivariate_normal(mu1, covar1, num_points_each_cluster)
#X2 = np.random.multivariate_normal(mu2, covar2, num_points_each_cluster)
#y1 = np.ones(num_points_each_cluster)
#y2 = np.zeros(num_points_each_cluster)


#plt.plot( X1[:, 0], X1[:,1], 'ro', label='class 1')
#plt.plot(X2[:, 0], X2[:,1], 'bo', label='class 0')
#plt.legend(loc='best')
#plt.show()

#X = np.vstack((X1, X2))
#y = np.hstack((y1, y2))
#print(X.shape, y.shape)


#X_tf = tf.constant(X)
#y_tf = tf.constant(y)


    
#example = np.array([0, 0])
#example_tf = tf.constant(example,dtype=tf.float64)

#plt.plot( X1[:, 0], X1[:,1], 'ro', label='class 1')
#plt.plot(X2[:, 0], X2[:,1], 'bo', label='class 0')
#plt.plot(example[0], example[1], 'g', marker='D', markersize=10, label='test point')
#plt.legend(loc='best')
#plt.show()   

#k_tf = tf.constant(3)
#pr = predict(X_tf, y_tf, example_tf, k_tf)
#sess = tf.Session()
#y_index = sess.run(pr)
#print (get_label(y_index))

#example_2 = np.array([0.1, 2.5])
#example_2_tf = tf.constant(example_2)
#plt.plot( X1[:, 0], X1[:,1], 'ro', label='class 1')
#plt.plot(X2[:, 0], X2[:,1], 'bo', label='class 0')
#plt.plot(example_2[0], example_2[1], 'g', marker='D', markersize=10, label='test point')
#plt.legend(loc='best')
#plt.show() 

#pr = predict(X_tf, y_tf, example_2_tf, k_tf)
#y_index = sess.run(pr)
#print (get_label(y_index))

def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--parse", action="store_true")   
    parser.add_argument("--name", help="Name")
    parser.add_argument("--df",help="delta fuel")
    parser.add_argument("--dt",help="delta time")
    parser.add_argument("--tu",help="time units")
    parser.add_argument("--nd",help="new values delta")
    

    return parser

def main():
    parser = create_parser()
    args = parser.parse_args()
    
    if args.parse:
        parse_data(args.name,args.df,args.dt,args.tu,args.nd)
       



if __name__=="__main__":
    main()