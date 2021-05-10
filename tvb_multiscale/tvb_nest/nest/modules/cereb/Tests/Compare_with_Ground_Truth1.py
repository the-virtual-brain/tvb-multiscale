import sys
import numpy as np
import os
import glob

# Building the Weight Matrix Ground Truth
LTP1 = 1.0e-3
LTD1 = -1.0e-2
Init_PFPC = 0.07

IO_Spikes = [10.0, 100.0, 300.0, 310.0, 320.0, 500.0, 600.0, 700.0, 800.0, 999.0]

Weight_Matrix = []
GR_Activity=np.loadtxt("GR_65600.dat")
for t in range(65600):
    t_array=GR_Activity[t]
    count=0
    LTD_amount = []
    LTD_time = []
    CurrentWeight = []
    LTD_index = 0
    for GRspike in t_array:
        if count == 0:
            CurrentWeight.append(Init_PFPC+LTP1)
            spike_diff = IO_Spikes-(GRspike+2)
            spike_diff = spike_diff[np.where((spike_diff>0) & (spike_diff <= 200))]
            for sds in spike_diff:
                LTD_amount.append(LTD1*np.exp(-(-sds-150.0)/1000.0)*pow((np.sin(2*3.1415*(-sds-150)/1000.0)),20)/1.2848)
                LTD_time.append(sds+GRspike+1)
            count+=1
        else:
            CurrentWeight.append(CurrentWeight[count-1]+LTP1)
            spike_diff = IO_Spikes-(GRspike+2)
            spike_diff = spike_diff[np.where((spike_diff>0) & (spike_diff <= 200))]
            for sds in spike_diff:
                LTD_amount.append(LTD1*np.exp(-(-sds-150.0)/1000.0)*pow((np.sin(2*3.1415*(-sds-150)/1000.0)),20)/1.2848)
                LTD_time.append(sds+GRspike+1)
            count+=1   
    CurrentWeight=np.array(CurrentWeight)      
    LTD_amount=np.array([x for _,x in sorted(zip(LTD_time,LTD_amount))])
    LTD_time = sorted(LTD_time)
    LTD_time_u = np.unique(LTD_time)
    amou = []
    for LTD_times_u in LTD_time_u:
        ind = np.where(LTD_time == LTD_times_u)[0]
        amou.append(np.sum(LTD_amount[ind]))
    count = 0
    for n,LTD_times_u in enumerate(LTD_time_u):
        ind =  np.where(t_array>LTD_times_u)[0]
        for index in ind:
            CurrentWeight[index]=CurrentWeight[index]+amou[n]
    for n,w in enumerate(CurrentWeight): 
        Weight_Matrix.append([t+1, 65601, t_array[n]+1, round(w,8)])

IO_Spikes = [20.0, 200.0, 260.0, 360.0, 370.0, 460.0, 560.0, 660.0, 760.0, 959.0]
        
for t in range(65600):
    t_array=GR_Activity[t]
    count=0
    LTD_amount = []
    LTD_time = []
    CurrentWeight = []
    LTD_index = 0
    for GRspike in t_array:
        if count == 0:
            CurrentWeight.append(Init_PFPC+LTP1)
            spike_diff = IO_Spikes-(GRspike+2)
            spike_diff = spike_diff[np.where((spike_diff>0) & (spike_diff <= 200))]
            for sds in spike_diff:
                LTD_amount.append(LTD1*np.exp(-(-sds-150.0)/1000.0)*pow((np.sin(2*3.1415*(-sds-150)/1000.0)),20)/1.2848)
                LTD_time.append(sds+GRspike+1)
            count+=1
        else:
            CurrentWeight.append(CurrentWeight[count-1]+LTP1)
            spike_diff = IO_Spikes-(GRspike+2)
            spike_diff = spike_diff[np.where((spike_diff>0) & (spike_diff <= 200))]
            for sds in spike_diff:
                LTD_amount.append(LTD1*np.exp(-(-sds-150.0)/1000.0)*pow((np.sin(2*3.1415*(-sds-150)/1000.0)),20)/1.2848)
                LTD_time.append(sds+GRspike+1)
            count+=1   
    CurrentWeight=np.array(CurrentWeight)      
    LTD_amount=np.array([x for _,x in sorted(zip(LTD_time,LTD_amount))])
    LTD_time = sorted(LTD_time)
    LTD_time_u = np.unique(LTD_time)
    amou = []
    for LTD_times_u in LTD_time_u:
        ind = np.where(LTD_time == LTD_times_u)[0]
        amou.append(np.sum(LTD_amount[ind]))
    count = 0
    for n,LTD_times_u in enumerate(LTD_time_u):
        ind =  np.where(t_array>LTD_times_u)[0]
        for index in ind:
            CurrentWeight[index]=CurrentWeight[index]+amou[n]
    for n,w in enumerate(CurrentWeight): 
        Weight_Matrix.append([t+1, 65602, t_array[n]+1, round(w,8)])

Weight_Matrix=np.array(Weight_Matrix)

SimResults1 = np.loadtxt("PFPC1.csv")
SimResults4 = np.loadtxt("PFPC4.csv")

print( ["Length Ground Truth: " +  str(len(Weight_Matrix)) + " Length Simulation1 Results: " + str(len(SimResults1)) + " Length Simulation4 Results: " + str(len(SimResults4)) ])

SimResults1=np.matrix(SimResults1)
SimResults4=np.matrix(SimResults4)
Weight_Matrix=np.matrix(Weight_Matrix)

Weight_Matrix.sort(axis=0)
SimResults1.sort(axis=0)
SimResults4.sort(axis=0)

Difference1 = SimResults1-Weight_Matrix
# Get rid of approximation errors
Error1 = np.sum(Difference1)

Difference4 = SimResults4-Weight_Matrix
# Get rid of approximation errors
Error4 = np.sum(Difference4)

Error = Error1 + Error4

if Error < 3.0:
    sys.exit(0)
else:
   sys.exit(-1)
