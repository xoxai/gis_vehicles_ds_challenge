#Program solution of first and second tasks
#k-NN classifier with 3 classes:
#0 - refueling, 1 - fuel consumption, 2 - fuel dump
#this code must be put in directory with given .csv files

#Import necessary modules
import argparse
import numpy as np
import tensorflow.compat.v1 as tf
import matplotlib.pyplot as plt
import warnings
import csv
import sys
from get_data import vehicles_data

#Module option setting
warnings.filterwarnings('ignore')
plt.rcParams['figure.figsize'] = (20.0, 10.0)
plt.style.use('ggplot')
tf.disable_v2_behavior()

#The main classification function,
#to the input of which 
#the sampling data tensors - X_t,
# class label tensors - y_t,
# argument data - x_t, 
#number of considered neighbors are fed - k_t
def predict(X_t, y_t, x_t, k_t):

    neg_one = tf.constant(-1.0, dtype=tf.float64) 
    #using L-1 distance
    distances =  tf.reduce_sum(tf.abs(tf.subtract(X_t, x_t)), 1) 
    #searching for the farthest points based on negative distances    
    neg_distances = tf.multiply(distances, neg_one)
    #index production
    vals, indx = tf.nn.top_k(neg_distances, k_t) 
    #slicing labels of points
    y_s = tf.gather(y_t, indx)
    return y_s
    
#Auxiliary function to obtain a class label,
#to the input of which
#array of predicted labels - preds     
def get_label(preds):

    counts = np.bincount(preds.astype('int64'))
    return np.argmax(counts)
    
#Function of passing through fuel level data
#at every moment of time
def cycle_data(data_time,delta,j):
     
    try:
        while  data_time[j+1].hour-data_time[j].hour<=1 \#delta selection condition
            and data_time[j+1].day==data_time[j].day:
                delta+=data_time[j+1].second-data_time[j].second
                delta+=(data_time[j+1].minute-data_time[j].minute)*60
                delta+=(data_time[j+1].hour-data_time[j].hour)*3600       
                j+=1           
        return delta,j
    except Exception:
        return 0,-1
#Main function of preparing data for further classification,
#to the input of which   
#type of vehicle  - vehicle
#argument data of delta fuel - delta_fuel
#argument data of delta time - delta_time
#time unit of delta time - time unit
#flag of parsing new data - new_delta
#value of scatter between fuelconsumtion and dump - scatter 
def parse_data(vehicle,delta_fuel,delta_time,time_unit,new_delta,scatter):

    if time_unit=="m":
        delta_time=delta_time*60
    elif time_unit=="h":
        delta_time=delta_time*3600
    elif time_unit!="s":
        print("Wrong time_unit argument")
        print("Should be <s> or <m> or <h>")
        print("Restart program with correct time_unit argument")
        sys.exit("Error") 
        
    #initialization    
    data_time=[]
    data_fuel_level=[]
    data_delta_time=[]
    data_delta_fuel_level=[]
    data_delta_fuel_level_neg=[]
    data_delta_time_neg=[]
    fuel_dumps=[]
    time_dumps=[]
    
    delta=0
    i,j,k=0,0,0  
    average_delta_time_neg=0
    average_delta_fuel_level_neg=0
    
    try:
        data_time=vehicles_data[vehicle].fuelLevel.DTIME
        data_fuel_level=list(vehicles_data[vehicle].fuelLevel.BEVALUE)
    except Exception:
        print("Wrong name of data argument")       
        print("Restart program with correct name data argument")
        sys.exit("Error")
        
    #parsing new data
    if new_delta=="Yes":
        f1=open(vehicle.upper()+"_data_delta_time_+.txt","w")#file with value of delta time with positive values of delta
        f2=open(vehicle.upper()+"_data_delta_fuel_+.txt","w")#file with positive values of delta
        f3=open(vehicle.upper()+"_data_delta_fuel_-.txt","w")#file with negative values of delta
        f4=open(vehicle.upper()+"_data_delta_time_-.txt","w")#file with value of delta time with negative values of delta
        
        print("It will take some time")
        print("Please wait")
        
        while j!=len(data_time)-1:
            k=j
            delta,j=cycle_data(data_time,0,j)
            if j!=-1:            
                data_delta_time.append(delta)                        
                data_delta_fuel_level.append(data_fuel_level[j+1]-data_fuel_level[k])
                delta=0                
                if data_delta_fuel_level[i]>=0:#adding value according to its sign
                    f1.write(str(data_delta_time[i])+"\n")
                    f2.write(str(data_delta_fuel_level[i])+"\n")
                else:
                    f4.write(str(data_delta_time[i])+"\n")
                    f3.write(str(data_delta_fuel_level[i])+"\n")
                i+=1
                j+=1                
            else:
                break
                
        f1.close()
        f2.close()
        f3.close()
        f4.close()
    #
    elif new_delta=="No":
        f1=open(vehicle.upper()+"_data_delta_time_+.txt","r")
        f2=open(vehicle.upper()+"_data_delta_fuel_+.txt","r")
        f3=open(vehicle.upper()+"_data_delta_fuel_-.txt","r")
        f4=open(vehicle.upper()+"_data_delta_time_-.txt","r")
        
        data_delta_time=f1.readlines()
        data_delta_time=[x.strip() for x in data_delta_time]
        data_delta_fuel_level=f2.readlines()
        data_delta_fuel_level=[x.strip() for x in data_delta_fuel_level]
        data_delta_fuel_level_neg=f3.readlines()
        data_delta_fuel_level_neg=[x.strip() for x in data_delta_fuel_level_neg]
        data_delta_time_neg=f4.readlines()
        data_delta_time_neg=[x.strip() for x in data_delta_time_neg]
        #formation of the conditions for classification of the received data 
        #based on the difference between the received fuel delta values and
        #the average consumption for a specific value of the time delta, 
        #taking into account the scatter from the average value
        for i in range(0,len(data_delta_time),1):
            data_delta_time[i]=int(data_delta_time[i])
            data_delta_fuel_level[i]=float(data_delta_fuel_level[i])          
        for i in range(0,len(data_delta_fuel_level_neg),1):
            data_delta_fuel_level_neg[i]=float(data_delta_fuel_level_neg[i])
            data_delta_time_neg[i]=int(data_delta_time_neg[i])
            average_delta_fuel_level_neg+=data_delta_fuel_level_neg[i]
            average_delta_time_neg+=data_delta_time_neg[i]
        
        average_delta_fuel_level_neg=abs(average_delta_fuel_level_neg/len(data_delta_fuel_level_neg))
        average_delta_time_neg=average_delta_time_neg/len(data_delta_time_neg)
        
        try:
            scatter=float(scatter)
        except Exception:
            print("Wrong scatter argument")       
            print("Restart program with correct scatter argument")
            sys.exit("Error")
        
        if scatter>100 or scatter<0:
            print("Wrong scatter argument")       
            print("Restart program with correct scatter argument")
            sys.exit("Error")  
            
        elif scatter==0:#no data will be added, just show recommended values of scatter
            print("Founded scatter values:")
            while scatter!=100:
                for i in range(0,len(data_delta_fuel_level_neg),1):
                    if abs(data_delta_fuel_level_neg[i])- average_delta_fuel_level_neg>scatter*average_delta_fuel_level_neg \
                    and data_delta_time_neg[i]-average_delta_time_neg<scatter*average_delta_time_neg:                        
                        if scatter!=0:
                            print(scatter)
                scatter+=1
        else:        
            for i in range(0,len(data_delta_fuel_level_neg),1):
                if abs(data_delta_fuel_level_neg[i])- average_delta_fuel_level_neg>scatter*average_delta_fuel_level_neg \
                and data_delta_time_neg[i]-average_delta_time_neg<scatter*average_delta_time_neg:
                    fuel_dumps.append(data_delta_fuel_level_neg[i])
                    time_dumps.append(data_delta_time_neg[i]) 
                
        
              
                      
        f1.close()
        f2.close()
        f3.close()
        f4.close()
        
    else:
        print("Wrong new values of delta argument")       
        print("Restart program with correct new valuse of delta argument")
        sys.exit("Error")
    
    
    show_data(\
    data_delta_time,\
    data_delta_time_neg,\
    time_dumps,\
    data_delta_fuel_level,\
    data_delta_fuel_level_neg,\
    fuel_dumps,\
    delta_time,\
    delta_fuel)            
            
#The function of displaying the obtained data,
#as well as forming data for correct prediction
#taking into account the rules of TensorFlow,
#to the input of which 
#X1,X2,X3 - values of time delta for three classes
#Y1,Y2,Y3 - values of fuel delta for three classes
#testX and testY are value of argument data   
def show_data(X1,X2,X3,Y1,Y2,Y3,testX,testY):

    #test constants
    mu1 = [-0.4, 3]
    covar1 = [[1.3,0],[0,1]]
    result=0
    #chart preparation
    plt.plot(X1,Y1,'go',label='refueling')
    plt.plot(X2,Y2,'bo',label='fuel consumption')
    plt.plot(X3,Y3,'ro',label='fuel dump')
    plt.plot(float(testX), float(testY), 'm', marker='D', markersize=10, label='argument')
    plt.legend(loc='best')
    plt.xlabel("Values of time delta")
    plt.ylabel("Values of fuel delta")
    
    #data preparation
    N=max(len(X1), len(X2), len(X3))  
    X1.extend([0]*(N-len(X1)))    
    X2.extend([0]*(N-len(X2)))         
    X3.extend([0]*(N-len(X3)))      
   
    Y1.extend([0]*(N-len(Y1)))    
    Y2.extend([0]*(N-len(Y2)))         
    Y3.extend([0]*(N-len(Y3))) 
    
    X1=np.array(X1)
    X2=np.array(X2)
    X3=np.array(X3)
    Y1=np.array(Y1)
    Y2=np.array(Y2)
    Y3=np.array(Y3)
    
    C1=np.zeros(N)
    C2=np.ones(N)
    C3=np.full(N,2)   
    
    T1=np.random.multivariate_normal(mu1,covar1,N)       
    T1[:, 0],T1[:,1]=X1,Y1
    
    T2=np.random.multivariate_normal(mu1,covar1,N)
    T2[:, 0],T2[:,1]=X2,Y2
    
    T3=np.random.multivariate_normal(mu1,covar1,N)
    T3[:, 0],T3[:,1]=X3,Y3   
 
    X=np.vstack((T1,T2,T3))
    Y=np.hstack((C1,C2,C3))    

    X_tf=tf.constant(X)
    Y_tf=tf.constant(Y)
    
    argument=tf.constant(np.array([testX,testY]),dtype=tf.float64)
    
    K_neighbourhood=tf.constant(4)#number of neighbours
    
    prediction=predict(X_tf,Y_tf,argument,K_neighbourhood)
    session = tf.Session()
    current_index = session.run(prediction)
    result=get_label(current_index)
    if result==0:
        print("Argument is predicted as refueling")
    elif result==1:
        print("Argument is predicted as fuel consumption")
    else:
        print("Argument is predicted as fuel dump")
    
    plt.show()  
    

#Command line argument parsing function
def create_parser():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("-p", "--parse", action="store_true")   
    parser.add_argument("--name", help="Name")
    parser.add_argument("--df",help="delta fuel")
    parser.add_argument("--dt",help="delta time")
    parser.add_argument("--tu",help="time units")
    parser.add_argument("--nvd",help="new values of delta") 
    parser.add_argument("--sc",help="scatter") 

    return parser
#Example of usage:python knn.py -p --name vehicle1 --df 40 --dt 50 --tu s --nvd No --sc 1
#-p - use parse_data function
#--name - enter vehicle type
#--df -value of delta fuel argument
#--dt -value of delta time argument
#--tu - 's' for seconds, 'm' for minutes and 'h' for hours
#--nvd - parse new data or not ('Yes' or 'No')
#--sc - value of scatter to create classified data

def main():
    parser = create_parser()
    args = parser.parse_args()
    
    if args.parse:
        parse_data(args.name,args.df,args.dt,args.tu,args.nvd,args.sc)    

if __name__=="__main__":
    main()