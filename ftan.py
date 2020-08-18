#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 28 09:23:42 2018

@author: Josiah
"""
from obspy import read, read_inventory
from obspy.geodetics.base import gps2dist_azimuth
import matplotlib.pyplot as plt
import csv
from numpy import fft
import numpy as np

def fold_trace(tr):
    """fold_trace() takes a trace and computes the symmetric component by 
    summing the time-reversed acausal component with the causal coponent"""
    # Pick some constants we'll need from the trace:
    npts2 = tr.stats.npts
    npts = int((npts2-1)/2)

    # Fold the CCFs:
    tr_folded= tr.copy()
    causal =  tr.data[npts:-1]
    acausal = tr.data[npts:0:-1]
    tr_folded.data =  (causal+acausal)/2
    
    return tr_folded

def interstation_distance(filename):
    """compute the interstation distance from file name.
    filenames should as stationA.stationB.components.SAC
    e.g. ABAZ.KBAZ.ZT.SAC"""

    
    #read inventory to get station coordinates and calculate interstation distance
    inv = read_inventory("inv.XML")
    
    x=filename.split("_")
    stationA=inv.select(station=x[0])
    stationB=inv.select(station=x[1])
    distance, az, baz = gps2dist_azimuth(stationA[0][0].latitude,stationA[0][0].longitude,stationB[0][0].latitude,stationB[0][0].longitude)
    
    return distance

def make_gaussian_filters(array,samp_rate, dist, freqmin=0.01,freqmax=10,filt_no=50,alpha=10):
    """Make a set of Gaussian filters.
    :array : Real component of FFT of seismic signal. Filter lengths need to be the same length as this trace
    :freqmin : lowest center frequency
    :freqmax : maximum center frequency
    :filt_no : number of filters
    :alpha : adjustable parameter that sets the resolution in the frequency and time domains. Default = 1
    """
    frequencies = np.arange(len(array))*samp_rate/len(array)
    filters=[]
    central_frequencies = np.logspace(np.log10(freqmin),np.log10(freqmax),filt_no)
    #omega_0 is the center frequency of each Gaussian filter
    for omega_0 in central_frequencies:
        gaussian=[]
        if 10 < dist < 100:
            alpha=10
        elif 100 < dist < 250:
            alpha= 30
        elif dist > 250:
            alpha = 100
        elif dist < 10:
           alpha = 0.2
        for omega in frequencies:
                gaussian.append(np.e**(-alpha*((omega-omega_0)/omega_0)**2))
        filters.append([[omega_0],gaussian])
    
    return filters, frequencies
    
def ftan(filename,freqmin=0.1,freqmax=10,vmin=1.0,vmax=5.0,fold=False,alpha=10):
    """returns the filename, and computes and saves FTAN amplitudes and group velocities.
    :filename : name of sac file e.g. ABAZ_ETAZ_ZZ.SAC
    :freqmin : minimum frequency for filter
    :freqmax : maximum frequency for filter
    :alpha : adjustable parameter that sets the resolution in the frequency and time domains. Default = 10
    """
    st=read(filename)
    tr=st[0]
    if fold==True:
        tr = fold_trace(tr)
    samp_rate = tr.stats.sampling_rate #sampling rate
    samp_int = tr.stats.delta #sampling interval
    t = np.arange(0,len(tr.data)*samp_int,samp_int) # time vector
    dist=tr.stats.sac["dist"]/1000
    
    if freqmax > samp_rate/2:
        print("Maximum frequency exceeded the Nyquist frequency")
        print("Freqmax = {}, Nyquist = {}".format(str(freqmax),str(samp_rate/2)))
        print("Maximum frequency reset to {}".format(str(samp_rate/2)))
        freqmax = samp_rate/2
        
    tr_ft = fft.fft(tr.data) # Fourier transform tr into the frequency domain
    
    #take the analytic component by...
    tr_af = tr_ft.copy()
    tr_af[:len(tr_af)//2]*=2.0 #multiplying all the positive frequencies by two
    tr_af[len(tr_af)//2:]*=0.0 #multiplying all the negative frequencies by zero

    gaussian_filters, frequencies = make_gaussian_filters(tr_ft.real,freqmin=freqmin,
                                    freqmax=freqmax,samp_rate=samp_rate,dist=dist, alpha=alpha)
    
    filename_amplitudes = ".".join([filename,"amplitudes.csv"])
    headers=["Speed (km/s)","Centre Period (s)","Instaneous Period (s)",
        "Amplitude"]

    with open(filename_amplitudes,"w",newline='') as csvfile:
        writer=csv.writer(csvfile)
        writer.writerow(headers)
        group_speeds=[]
        gs_c_period=[]
        gs_inst_period=[]
        envelope_max=[]
        phase_at_group_time = []
        neg_errors = []
        pos_errors = []
        
        for g_filter in gaussian_filters:
            tr_aff = tr_af*np.array(g_filter[1]) #applying the filters to the analytic signal in frequency domain
            tr_at = fft.ifft(tr_aff) # inverse Fourier transform of the filtered analytic signal
            envelope=np.log(np.absolute(tr_at)**2)# taking the logarithm of the square of the filtered analytic signal in the time domain
#            phase_function = np.unwrap(np.angle(tr_at)) # compute the unwrapped phase function
            phase_function = np.angle(tr_at) # compute the phase function
            phase_at_group_time.append(phase_function[np.argmax(envelope)]) # save phase at group time
            omega_inst = np.diff(phase_function)/(samp_int*2*np.pi) # compute instaneous frequencies for phase function
            omega_inst = omega_inst[np.argmax(envelope)-1]

#            omega_inst = omega_inst[np.argmin(np.abs(omega_inst-g_filter[0]))] # compute instaneous frequencies
            center_period=np.around(float(1/g_filter[0][0]),decimals=2)
            instantaneous_period = np.around(float(1/omega_inst),decimals=2)
#            phi_s = 0 #source phase is zero for ambient noise cross-correlations

        
            for i, amplitude in enumerate(envelope.real):
                if t[i] != 0 and vmin < dist/t[i] < vmax:
                    speed=float(dist/t[i])
                    writer.writerow([speed,center_period,instantaneous_period,amplitude])
                else:
                    pass
                
            #compute group speeds and related data
            if t[np.argmax(envelope)] != 0 and vmin < dist/t[np.argmax(envelope)] < vmax:
                envelope_max.append(max(envelope))
                group_speeds.append(dist/t[np.argmax(envelope)])
                gs_c_period.append(center_period)
                gs_inst_period.append(instantaneous_period)
                
#                Error Analysis
                #upper error
                e_up=envelope[np.argmax(envelope):]
                t_up=t[np.argmax(envelope):]
                amp_up=[]
                for i, amp in enumerate(e_up):
                    if amp >= max(envelope)-0.5:
                        amp_up.append([t_up[i],amp])
                    else:
                        break
                if len(amp_up)>1:
                    pos_error=dist/amp_up[-1][0]-dist/amp_up[0][0]
                else:
                    pos_error=10
                if pos_error==float("Inf"):
                    neg_error=10
                pos_errors.append(abs(pos_error))
                
                #lower error
                e_dwn=envelope[:np.argmax(envelope)+1]
                e_dwn=e_dwn[::-1]
                t_dwn=t[:np.argmax(envelope)+1]
                t_dwn=t_dwn[::-1]
                amp_dwn=[]
                for i, amp in enumerate(e_dwn):
                    if amp >= max(envelope)-0.5:
                        amp_dwn.append([t_dwn[i],amp])
                    else:
                        break

                if amp_dwn[-1][0] !=0 and amp_dwn[0][0] !=0:
                    neg_error = dist/amp_dwn[-1][0]-dist/amp_dwn[0][0]
                else:
                    neg_error = 10
#                print(center_period)
                neg_errors.append(abs(neg_error))               

    filename_group_speeds = ".".join([filename,"group_speeds","csv"])
    headers=["Group Speed (km/s)","Centre Period (s)","Instaneous Period (s)","Negative Error","Postive Error","time (s)","distance (km)"]
    # headers=["Group Speed (km/s)","Centre Period (s)"]
    group_speeds,gs_c_period,gs_inst_period,phase_at_group_time = trim(group_speeds,gs_c_period,gs_inst_period,envelope_max,phase_at_group_time)
    with open(filename_group_speeds,"w",newline='') as csvfile:
        writer=csv.writer(csvfile)
        writer.writerow(headers)
        for i, group_speed in enumerate(group_speeds):
                    writer.writerow([group_speed,gs_c_period[i],gs_inst_period[i],neg_errors[i],pos_errors[i], dist/group_speed, dist])  
#                    writer.writerow([group_speed,gs_c_period[i]])
    
def trim(group_speeds,gs_c_period,gs_inst_period,envelope_max,phase_at_group_time,jthresh=0.2,ampthresh=-15):
    """Returns group speed as a function of period trimmed to avoid large jumps caused by noise and low amplitude envelopes.
    :group_speeds :
    :gs_c_period :
    :gs_inst_period : 
    :envelope_max : List of envelope maxima (track of the amplitude ridge in FTAN plot)
    :jthresh : Jump threshold. Discards groups speeds to avoid jumps larger than jthresh. Default = 0.2  
    :ampthresh : Amplitude threshold. Discards groups speeds if the envelope maximum is below he threshold  . Default = -15              
    """
    del_array=[]
    for i in range(len(envelope_max)):
        if envelope_max[i] < ampthresh:
            del_array.append(i)
    for i in del_array[::-1]:
        del group_speeds[i]
        del gs_c_period[i]
        del gs_inst_period[i]
        del envelope_max[i]
        del phase_at_group_time[i]

    if len(np.where(abs(np.diff(group_speeds))>jthresh)[0]) > 0:
        gs1=group_speeds[envelope_max.index(max(envelope_max)):]
        cp1=gs_c_period[envelope_max.index(max(envelope_max)):]
        ip1=gs_inst_period[envelope_max.index(max(envelope_max)):]
        ph1=phase_at_group_time[envelope_max.index(max(envelope_max)):]
        
        gs2=group_speeds[:envelope_max.index(max(envelope_max))]
        cp2=gs_c_period[:envelope_max.index(max(envelope_max))]
        ip2=gs_inst_period[:envelope_max.index(max(envelope_max))]
        ph2=phase_at_group_time[:envelope_max.index(max(envelope_max))]
        gs1_diff=abs(np.diff(gs1))
        gs2_diff=abs(np.diff(gs2))
    
        if len(np.where(gs1_diff>jthresh)[0]) == 0:
            step_int1 = len(gs1_diff)
        elif len(np.where(gs1_diff>jthresh)[0]) > 0:
            step_int1 = min(np.where(gs1_diff>jthresh)[0])+1
            
        if len(np.where(gs2_diff>jthresh)[0]) == 0:
            step_int2 = 0
        elif len(np.where(gs2_diff>jthresh)[0]) > 0:
            step_int2=max(np.where(gs2_diff>jthresh)[0])+1
            
        group_speeds =np.append(gs1[:step_int1],gs2[step_int2:])
        gs_c_period =np.append(cp1[:step_int1],cp2[step_int2:])
        gs_inst_period =np.append(ip1[:step_int1],ip2[step_int2:])
        phase_at_group_time=np.append(ph1[:step_int1],ph2[step_int2:])
        gs_inst_period,gs_c_period, group_speeds,  phase_at_group_time= zip(*sorted(zip(gs_inst_period, gs_c_period, group_speeds, phase_at_group_time)))
        
    return group_speeds,gs_c_period,gs_inst_period,phase_at_group_time
    
def plot(filename,group_speeds=False,phase_speeds=False,fplot=False,three_d=False):
    import numpy as np
    import matplotlib.pyplot as plt
    array1=np.loadtxt(filename+".amplitudes.csv",delimiter=',',skiprows=1)
    speeds=array1[:,0]
#    if inst_period==True:
#        periods=array1[:,2]
#    else:
#        periods=array1[:,1]   
    centre_periods=array1[:,1]
    instantaneous_periods=array1[:,2]
    if fplot==True:
#        periods=np.reciprocal(periods)
        instantaneous_periods = np.reciprocal(instantaneous_periods)
        centre_periods= np.reciprocal(centre_periods)
    amplitudes=array1[:,3]
    if fplot==False:
        x = np.unique(centre_periods)[::-1]
    else:
        x = np.unique(centre_periods)
    
    y = np.unique(speeds)[::-1]
    X, Y = np.meshgrid(x, y)
    Z = amplitudes.reshape(len(x),len(y)).T
    
    plt.figure(figsize=(5,5))
    plt.title(filename)
    plt.contourf(X,Y,Z,50,cmap="jet")
    
    if group_speeds is True:
        try:
            array2=np.loadtxt(filename+".group_speeds.csv",delimiter=',',skiprows=1)
            gspeeds=array2[:,0]
            centre_periods=array2[:,1]
            instantaneous_periods=array2[:,2]

            if fplot==False:
                plt.scatter(centre_periods,gspeeds,label="group speed",color="w",marker="+")
                plt.scatter(instantaneous_periods, gspeeds, label="instantaneous group speed", color="k", marker="x")
            else:
                plt.scatter(np.reciprocal(centre_periods),gspeeds,label="group speed",color="w",marker="+")
            plt.legend(loc=1)
        except:
            print("Input file may be empty")
            
    if phase_speeds is True:
        try:
            array2=np.loadtxt(filename+".phase_speeds.csv",delimiter=',',skiprows=1)
            
            centre_periods=array2[:,0]
            instantaneous_periods=array2[:,1]
    #        plt.scatter(instantaneous_periods,gspeeds,label="group speed",color="k",marker="x")
            for N_column in range(2,10):
                if fplot==False:
                    plt.scatter(centre_periods,array2[:,N_column],label="phase speed",color="k",marker=".")
                else:
                    plt.scatter(np.reciprocal(centre_periods),array2[:,N_column],label="phase speed",color="k",marker=".")
                plt.legend(loc=1)
        except:
            print("Input file may be empty")
            
    plt.ylim(1.0,5.0)
    plt.xlim(1,10)
    plt.xlabel("Period (s)")
    if fplot == True:
        plt.xlabel("Frequency (Hz)")
        plt.xlim(0.1,1)
    plt.ylabel("Speed (kms$^{-1}$)")
    plt.grid() 
    plt.savefig(filename+"_ftanplot.pdf")
    plt.show()  

    if phase_speeds is True:
        plt.figure(figsize=(5,5))
        plt.title(filename)
        try:
            array2=np.loadtxt(filename+".phase_speeds.csv",delimiter=',',skiprows=1)
            
            centre_periods=array2[:,0]
            instantaneous_periods=array2[:,1]
    #        plt.scatter(instantaneous_periods,gspeeds,label="group speed",color="k",marker="x")
            for N_column in range(2,10):
                if fplot==False:
                    plt.scatter(centre_periods,array2[:,N_column],color="k",marker=".")
                else:
                    plt.scatter(np.reciprocal(centre_periods),array2[:,N_column],color="k",marker=".") 
        except:
            print("Input file may be empty") 
        plt.ylim(1.0,5.0)
        plt.xlim(1,10)
#        plt.xscale('log')
        plt.xlabel("Period (s)")
        if fplot == True:
            plt.xlabel("Frequency (Hz)")
#            plt.xlim(0.1,1)
            print('done')
        plt.ylabel("Speed (kms$^{-1}$)")
        plt.grid()
        plt.savefig(filename+"_phase_speed.pdf")
        plt.show()

    if three_d==True:
        import matplotlib as mpl
        from mpl_toolkits.mplot3d import Axes3D
        import numpy as np
        import matplotlib.pyplot as plt

        mpl.rcParams['legend.fontsize'] = 10

        fig = plt.figure()
        ax = fig.gca(projection='3d')
        ax.set_xlabel("Period (s)")
        ax.set_ylabel("Speed (kms$^{-1}$)")
        ax.set_zlabel("Amplitude")
        X = centre_periods.reshape(len(x), len(y)).T
        Y = speeds.reshape(len(x), len(y)).T
        Z = amplitudes.reshape(len(x), len(y)).T
        # for column in range(0,Z.shape[1]):
        #     ax.plot(X[:,column],Y[:,column], Z[:,column])
        # ax.plot(centre_periods,speeds, amplitudes, label='parametric curve')
        ax.plot_wireframe(X, Y, Z, rstride=30, cstride=3)
        # rotate the axes and update
        # ax.view_init(30, 0)
        # for angle in range(0, 360):
        #     ax.view_init(30, angle)
        #     plt.draw()
        #     plt.pause(.001)

        # if group_speeds is True:
        #     try:
        #         array2 = np.loadtxt(filename + ".group_speeds.csv", delimiter=',', skiprows=1)
        #         gspeeds = array2[:, 0]
        #         centre_periods = array2[:, 1]
        #         instantaneous_periods = array2[:, 2]
        #         amplitudes=array2[:, 3]
        #         #            plt.scatter(instantaneous_periods,gspeeds,label="instantaneous group speed",color="k",marker="x")
        #         ax.scatter(centre_periods, gspeeds, amplitudes, color="r")
        #     except:
        #         print("Input file may be empty")
        plt.show()