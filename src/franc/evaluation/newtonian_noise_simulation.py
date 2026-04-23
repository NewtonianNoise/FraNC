# -*- coding: utf-8 -*-
"""
Created on Tue Jan 27 12:45:15 2026

@author: schillings
"""

import numpy as np
import math
import torch
import random as rd
import scipy.constants as co
import time as systime
import os
import ast
from sys import argv
import matplotlib.pyplot as plt
import matplotlib.animation as ani
import scipy.special as sp
from scipy.optimize import differential_evolution
from scipy import linalg

#plotting defaults
plt.rc('legend',fontsize=15)
plt.rc('axes',labelsize=16,titlesize=16)
plt.rc("xtick",labelsize=13)
plt.rc("ytick",labelsize=13)
plt.rc('figure',figsize=(10,9))

total_start_time=systime.time()


pi=torch.pi
G=co.gravitational_constant


#########################################
#~~~~~~~~~~~~~~File Reader~~~~~~~~~~~~~~#
#########################################

class ReadData:
    tag=""
    folder=""
    fileType=""
    
    dictionary={}
    
    def __init__(self, tag, folder,fileType="settingFile"):
        self.tag=tag
        self.folder=folder
        self.fileType=fileType
        
        dataFile=np.loadtxt(folder+"/"+fileType+tag+".txt",dtype=str,delimiter="ö", comments="//")
        
        for line in dataFile:
            key=line.split(" = ")[0]
            value=line.split(" = ")[1]
            if value=="True" or value=="False":
                value=value=="True"
            elif "np.array" in value or "torch.tensor" in value:
                value=np.array(ast.literal_eval(value.split("(")[1].split(")")[0]))
            elif "min" in value:
                key=key[1:]
                value=float(value.split(" min")[0])
            else:
                try:
                    value=float(value)
                except:
                    pass
            self.dictionary.update({key: value})
            
        

#############################################
#~~~~~~~~~~~~~~Event Generator~~~~~~~~~~~~~~#
#############################################

class NewtonianNoiseDataGenerator:
    
    
    ###########################################################
    #~~~~~~~~~~~~~~Predefined Parameter Settings~~~~~~~~~~~~~~#
    ###########################################################

    single_mirror_position = [[0, 0, 0]]
    single_mirror_direction = [[1, 0, 0]]
    
    four_mirror_corner_position = [[64.12, 0, 0],
                                   [536.35, 0, 0],
                                   [64.12*0.5, 64.12*math.sqrt(3)/2, 0],
                                   [536.35*0.5, 536.35*math.sqrt(3)/2, 0]]      
    four_mirror_corner_direction = [[1, 0, 0],
                                    [1, 0, 0],
                                    [0.5, math.sqrt(3)/2, 0],
                                    [0.5, math.sqrt(3)/2, 0]]     


    ##############################################
    #~~~~~~~~~~~~~~Setting defaults~~~~~~~~~~~~~~#
    ##############################################
    
    def __init__(self,
        #~~~~~~~~~~~~~~Save management~~~~~~~~~~~~~~#
        
        ID = 5, #int(argv[1])
        tag = "X",                         #Dataset identifier
        folder = "testset/dataset",        #Dataset path
        saveas = "testset/resultFile",     #Identifier for all savefiles produced

        #~~~~~~~~~~~~~~General~~~~~~~~~~~~~~#
        
        useGPU = False,                    #Set True if you have and want to use GPU-resources
        randomSeed = 1,                    #If None, use no seed
        isMonochromatic = False,           #Toggles between monochromatic plane waves and Gaussian wave packets
        NoR = 10,                          #Number of runs/realizations/wave events
        
        #~~~~~~~~~~~~~~Constants~~~~~~~~~~~~~~#
        
        M = 211,                           #Mirror mass of LF-ET in kg
        rho = 3000,                        #Density of rock in kg/m³
        c_p = 6000,                        #Sound velocity of rock #6000 m/s
        
        #~~~~~~~~~~~~~~Domain~~~~~~~~~~~~~~#
        
        L = 6000,                          #Length of simulation box in m
        Nx = 200,                          #Number of spatial steps for force calculation, choose even number
        
        xmax = 6000,                       #Distance of wave starting point from 0
        
        tmax = None,                       #Time of simulation in s
        Nt = 200,                          #Number of time steps
        
        #~~~~~~~~~~~~~~Experiment config~~~~~~~~~~~~~~#
        
        depth = 6000,                      #Upper domain cutoff (=L for full space)
        cavity_r = 5,                      #Radius of spherical cavern in m
        
        mirror_positions = [[0, 0, 0]],    #Array of mirror position [x,y,z] in m
        mirror_directions = [[1, 0, 0]],   #Array of mirror free moving axis unit vectors
        
        point_source_positions = [[-1000, -1000, 0]], #Array of point source position [x,y,z] in m
        
        #~~~~~~~~~~~~~~Event parameters~~~~~~~~~~~~~~#
        
        Awavemin = 1e-11,             
        Awavemax = 1e-11,                 #Basically delta_rho/rho
        
        fmin = 1,
        fmax = 10,                        #frequency of seismic wave in Hz
        fmono = 1,                        #The frequency of all monochromatic plane waves in Hz
        
        sigmafmin = 0.5,
        sigmafmax = 1,                    #width of frequency of Gaussian wave-packet in Hz
        
        anisotropy = "none",              #"none" for isotropy, 
                                          #"quad" for more waves from above, 
                                          #"left" for only waves from -x, 
                                          #"p000"-"p100" for 0%-100% point source contribution
        
        #~~~~~~~~~~~~~~Dataset Parameters~~~~~~~~~~~~~~#
                                       
        state = [[400, 350, 0], 
                 [-250, 250, 0], 
                 [200, -250, 0], 
                 [-100, -100, 0]],        #Seismometer positions
                                        
        NoS = 4,                          #Number of Seismometers
        freq = 10,                        #Frequency of the Wiener filter in Hz
        SNR = 1e10,                       #SNR as defined in earlier optimization attempts
        p = 1,                            #Ratio of P- and S-waves
        c_ratio = 2/3,                    #Ratio c_s/c_p
    
        #~~~~~~~~~~~~~~Window Parameters~~~~~~~~~~~~~~#
    
        NoW = 10,                         #Number of total time windows
        NoT = 1,                          #Number of runs without update of WF (test)
        NoE = 1,                          #Number of wave events per time window
    
        time_window_multiplier = 1,       #Time length of a time window to be evaluated in units of t_max
        twindow = None,                   #If not none, used instead of time_window_multiplier to determine window length in s
    
        randomlyPlaced = False,           #Determines if events are shifted around inside the window or locked in place
        
        ):
            #~~~~~~~~~~~~~~Save management~~~~~~~~~~~~~~#
            
            self.default_ID = ID
            self.default_tag = tag
            self.default_folder = folder
            self.default_saveas = saveas
            
            #~~~~~~~~~~~~~~General~~~~~~~~~~~~~~#
            
            self.default_useGPU = useGPU
            self.default_randomSeed = randomSeed
            self.default_isMonochromatic = isMonochromatic
            self.default_NoR = NoR
                
            #~~~~~~~~~~~~~~Constants~~~~~~~~~~~~~~#
                
            self.default_M = M
            self.default_rho = rho
            self.default_c_p = c_p
                
            #~~~~~~~~~~~~~~Domain~~~~~~~~~~~~~~#
                
            self.default_L = L
            self.default_Nx = Nx
            self.default_dx = 2*L/Nx                  #Spacial stepwidth in m, should be <c_P/10/max(f_0)
                
            self.default_xmax = xmax
                
            self.default_tmax = tmax or 2*xmax/c_p
            self.default_Nt = Nt
            self.default_dt = self.default_tmax/Nt    #Temporal stepwidth in s
                
                
            #~~~~~~~~~~~~~~Experiment config~~~~~~~~~~~~~~#
                
            self.default_depth = depth
            self.default_cavity_r = cavity_r
                
            self.default_mirror_positions = mirror_positions
            self.default_mirror_directions = mirror_directions
            self.default_mirror_count = len(mirror_positions)
                
            self.default_point_source_positions = point_source_positions
            self.default_source_count = len(point_source_positions)+1
                
            #~~~~~~~~~~~~~~Event parameters~~~~~~~~~~~~~~#
              
            self.default_Awavemin = Awavemin             
            self.default_Awavemax = Awavemax
                
            self.default_fmin = fmin
            self.default_fmax = fmax
            self.default_fmono=fmono
                
            self.default_sigmafmin = sigmafmin
            self.default_sigmafmax = sigmafmax
                
            self.default_anisotropy = anisotropy  
            
            #~~~~~~~~~~~~~~Dataset Parameters~~~~~~~~~~~~~~#
                                           
            self.default_state = state
                                            
            self.default_NoS = NoS or len(state)
            self.default_freq = freq
            self.default_SNR = SNR
            self.default_p = p
            self.default_c_ratio = c_ratio
        
            #~~~~~~~~~~~~~~Window Parameters~~~~~~~~~~~~~~#
        
            self.default_NoW = NoW
            self.default_NoT = NoT
            self.default_NoE = NoE
        
            self.default_time_window_multiplier = time_window_multiplier
            self.default_twindow = twindow
        
            self.default_randomlyPlaced = randomlyPlaced
            
            
        
    ##############################################
    #~~~~~~~~~~~~~~Event Generation~~~~~~~~~~~~~~#
    ##############################################
    
    def generateEventSet(self,
        tag,
        NoR = None,
        
        #~~~~~~~~~~~~~~Save management~~~~~~~~~~~~~~#
        
        ID = None,
        folder = None,
        
        #~~~~~~~~~~~~~~General~~~~~~~~~~~~~~#
        
        useGPU = None,
        randomSeed = None,
        isMonochromatic = None,
        
        #~~~~~~~~~~~~~~Constants~~~~~~~~~~~~~~#
        
        M = None,
        rho = None,
        c_p = None,
        
        #~~~~~~~~~~~~~~Domain~~~~~~~~~~~~~~#
        
        L = None,
        Nx = None,
        
        xmax = None,
        
        tmax = None,
        Nt = None,
        
        #~~~~~~~~~~~~~~Experiment config~~~~~~~~~~~~~~#
        
        depth = None,
        cavity_r = None,
        
        mirror_positions = None,
        mirror_directions = None,
        point_source_positions = None,
        
        #~~~~~~~~~~~~~~Event parameters~~~~~~~~~~~~~~#
        
        Awavemin = None,             
        Awavemax = None, 
        
        fmin = None,
        fmax = None,
        fmono = None,
        
        sigmafmin = None,
        sigmafmax = None,
        
        anisotropy = None
        ):
            NoR = NoR if not NoR==None else self.default_NoR
            
            #~~~~~~~~~~~~~~Save management~~~~~~~~~~~~~~#
            
            ID = ID if not ID==None else self.default_ID
            folder = folder if not folder==None else self.default_folder
            
            #~~~~~~~~~~~~~~General~~~~~~~~~~~~~~#
            
            useGPU = useGPU if not useGPU==None else self.default_useGPU
            randomSeed = randomSeed if not randomSeed==None else self.default_randomSeed
            isMonochromatic = isMonochromatic if not isMonochromatic==None else self.default_isMonochromatic
            
            #~~~~~~~~~~~~~~Constants~~~~~~~~~~~~~~#
            
            M = M if not M==None else self.default_M
            rho = rho if not rho==None else self.default_rho
            c_p = c_p if not c_p==None else self.default_c_p
            
            #~~~~~~~~~~~~~~Domain~~~~~~~~~~~~~~#
            
            L = L if not L==None else self.default_L
            Nx = Nx if not Nx==None else self.default_Nx
            dx=2*L/Nx 
             
            xmax = xmax if not xmax==None else self.default_xmax
            
            tmax = tmax if not tmax==None else 2*xmax/c_p
            Nt = Nt if not Nt==None else self.default_Nt
            dt=tmax/Nt
            
            #~~~~~~~~~~~~~~Experiment config~~~~~~~~~~~~~~#
            
            depth = depth if not depth==None else self.default_depth
            cavity_r = cavity_r if not cavity_r==None else self.default_cavity_r
            
            mirror_positions = mirror_positions if not mirror_positions==None else self.default_mirror_positions
            mirror_directions = mirror_directions if not mirror_directions==None else self.default_mirror_directions
            mirror_count=len(mirror_positions)
            
            point_source_positions = point_source_positions if not point_source_positions==None else self.default_point_source_positions
            source_count=len(point_source_positions)+1
            
            #~~~~~~~~~~~~~~Event parameters~~~~~~~~~~~~~~#
            
            Awavemin = Awavemin if not Awavemin==None else self.default_Awavemin
            Awavemax = Awavemax if not Awavemax==None else self.default_Awavemax
            
            fmin = fmin if not fmin==None else self.default_fmin
            fmax = fmax if not fmax==None else self.default_fmax
            fmono = fmono if not fmono==None else self.default_fmono
            
            sigmafmin = sigmafmin if not sigmafmin==None else self.default_sigmafmin
            sigmafmax = sigmafmax if not sigmafmax==None else self.default_sigmafmax
            
            anisotropy = anisotropy if not anisotropy==None else self.default_anisotropy
        
            #~~~~~~~~~~~~~~Some Setting Checks~~~~~~~~~~~~~~#
            
            if not os.path.exists(folder):  
                os.makedirs(folder)
            if os.path.exists(folder+"/settingFile"+tag+".txt"):
                raise NameError("better not overwrite your data!")
            
            if randomSeed!=None and randomSeed!="none":
                rd.seed(randomSeed)
                np.random.seed(randomSeed)
                
            if useGPU:
                device=torch.device("cuda")
            else:
                device=torch.device("cpu")
                
            #~~~~~~~~~~~~~~Wave event parameter generation~~~~~~~~~~~~~~#
            
            #wave direction
            polar_angles=torch.tensor(np.random.random(NoR) * 2*pi, device=device)
            azimuthal_angles=torch.tensor(np.arccos(2*np.random.random(NoR)-1), device=device)
            sources=torch.zeros(NoR,device=device,dtype=torch.int32)
            
            if anisotropy=="quad":
                azimuthal_angles=torch.tensor(np.arccos(2*np.random.random(NoR)**2-1), device=device)
            elif anisotropy=="left":
                polar_angles=torch.tensor(np.random.random(NoR) * pi - pi/2, device=device)
            elif "p" in anisotropy:
                point_probability=float(anisotropy.split("p")[1])/100
                sources=torch.tensor(np.random.choice(np.arange(0,source_count),NoR,p=[1-point_probability]+[point_probability/(source_count-1)]*(source_count-1)),dtype=torch.int32)
            
            #packet properties
            As=torch.tensor(np.random.random(NoR) * (Awavemax-Awavemin)+Awavemin, device=device)
            phases=torch.zeros(NoR, device=device)
            x0s=torch.ones(NoR, device=device) * (-xmax)
            t0s=torch.zeros(NoR, device=device)
            
            if isMonochromatic:
                fs=torch.ones(NoR, device=device) * fmono
                sigmafs=torch.zeros(NoR, device=device)
            else:
                fs=torch.tensor(np.random.random(NoR) * (fmax-fmin)+fmin, device=device)
                sigmafs=torch.tensor(np.random.random(NoR) * (sigmafmax-sigmafmin)+sigmafmin, device=device)
            
            #S-wave only
            s_polarisations=np.random.random(NoR) * 2*pi
            
            #precalculation
            exp_const=-2*pi**2 * sigmafs**2
            sin_const=2*pi * fs
            
            force_const=rho * G * M * dx**3
            
            sin_polar=torch.sin(polar_angles)
            cos_polar=torch.cos(polar_angles)
            sin_azi=torch.sin(azimuthal_angles)
            cos_azi=torch.cos(azimuthal_angles)
            
            
            #~~~~~~~~~~~~~~Domain preparation~~~~~~~~~~~~~~#
            
            #time and space
            time=torch.linspace(0, tmax, Nt+1, device=device)[:-1]
            x=torch.linspace(-L+dx/2, L-dx/2, Nx, device=device)
            y=torch.linspace(-L+dx/2, L-dx/2, Nx, device=device)
            z=torch.linspace(-L+dx/2, L-dx/2, Nx, device=device)
            xyz=torch.meshgrid(x, y, z, indexing="ij")
            x3d=xyz[1]
            y3d=xyz[0]
            z3d=xyz[2]
                
            #integration constants from mirror geometry
            r3d=torch.sqrt(x3d**2+y3d**2+z3d**2) + 1e-20
            cavity_kernel=r3d<L
            cavity_kernel*=z3d<depth
            
            r3ds=[]
            geo_facts=torch.zeros((mirror_count,Nx,Nx,Nx),device=device)
            for mirror in range(mirror_count):
                pos=mirror_positions[mirror]
                di=mirror_directions[mirror]
                r3ds.append(torch.sqrt((x3d-pos[0])**2+(y3d-pos[1])**2+(z3d-pos[2])**2)+1e-20)
                cavity_kernel*=r3ds[mirror]>cavity_r
                geo_facts[mirror]=((x3d-pos[0])*di[0]+(y3d-pos[1])*di[1]+(z3d-pos[2])*di[2])/r3ds[mirror]**3
            for mirror in range(mirror_count):
                if useGPU:
                    geo_facts[mirror].to(device=device)
                geo_facts[mirror]*=cavity_kernel
        
        
            #~~~~~~~~~~~~~~Function definitions~~~~~~~~~~~~~~#
        
            def gaussian_wave_packet(x, t, x0, t0, A, exp_const, sin_const, phase=0):
                
                diff = (x - x0) / c_p - (t - t0)
                exp_term = torch.exp(exp_const * diff**2)
                sin_term = torch.sin(sin_const * diff + phase)
                
                wave = A * exp_term * sin_term
                return wave
            
            def ricker_wavelet(x, t, x0, t0, A, sigma):
                
                diff = (x -x0) / c_p - (t - t0)
                
                wave = A * (1-(diff/sigma)**2) * np.exp(-diff**2/2/sigma**2)
                return wave
            
            def calc_force(drho, mirror):
                F = force_const * torch.sum(geo_facts[mirror] * drho)
                return F
            
        
            #~~~~~~~~~~~~~~Calculation of Newtonian noise~~~~~~~~~~~~~~#
        
            
            forces=torch.zeros((NoR,mirror_count,Nt), device=device)
            
            for R in range(NoR):
            
                #preparation    
                if sources[R]==0:
                    kx3D=cos_polar[R]*sin_azi[R]*x3d+sin_polar[R]*sin_azi[R]*y3d+cos_azi[R]*z3d
                else:
                    kx3D=torch.sqrt((x3d-point_source_positions[sources[R]-1][0])**2+(y3d-point_source_positions[sources[R]-1][1])**2+(z3d-point_source_positions[sources[R]-1][2])**2)
                
                if useGPU:
                    kx3D.to(device=device)
                
                #force calculation
                for i,t in enumerate(time):
                    density_fluctuations=gaussian_wave_packet(kx3D, t, x0s[R], t0s[R], As[R], exp_const[R], sin_const[R], phases[R])
                    if useGPU:
                        density_fluctuations.to(device=device)
                    for mirror in range(mirror_count):
                        forces[R][mirror][i]=calc_force(density_fluctuations,mirror)
                
        
            #~~~~~~~~~~~~~~Write settings file~~~~~~~~~~~~~~#
        
            if not os.path.exists(folder+"/settingFile"+tag+".txt"):
                f = open(folder+"/settingFile"+tag+".txt",'a+') 
                f.write("NoR = "+str(NoR)+"\n")
                f.write("isMonochromatic = "+str(isMonochromatic)+"\n")
                f.write("randomSeed = "+str(randomSeed)+"\n")
                f.write("M = "+str(M)+"\n")
                f.write("rho = "+str(rho)+"\n")
                f.write("c_p = "+str(c_p)+"\n")
                f.write("L = "+str(L)+"\n")
                f.write("x_max = "+str(xmax)+"\n")
                f.write("t_max = "+str(tmax)+"\n")
                f.write("Nx = "+str(Nx)+"\n")
                f.write("Nt = "+str(Nt)+"\n")
                f.write("depth = "+str(depth)+"\n")
                f.write("cavity_r = "+str(cavity_r)+"\n")
                f.write("mirror_positions = np.array("+str(list(mirror_positions))+")\n")
                f.write("mirror_directions = np.array("+str(list(mirror_directions))+")\n")
                f.write("point_source_positions = np.array("+str(list(point_source_positions))+")\n")
                f.write("Awave_max = "+str(Awavemax)+"\n")
                f.write("Awave_min = "+str(Awavemin)+"\n")
                f.write("f_min = "+str(fmin)+"\n")
                f.write("f_max = "+str(fmax)+"\n")
                f.write("f_mono = "+str(fmono)+"\n")
                f.write("sigma_f_min = "+str(sigmafmin)+"\n")
                f.write("sigma_f_max = "+str(sigmafmax)+"\n")
                f.write("anisotropy = "+str(anisotropy)+"\n")
                f.write("useGPU = "+str(useGPU)+"\n")
                
                f.write("#runtime = "+str(np.round((systime.time()-total_start_time)/60,2))+" min\n")
                f.close()        
                    
                
            #~~~~~~~~~~~~~~Write dataset~~~~~~~~~~~~~~#
            
                CPUdevice=torch.device("cpu")
                
                #mirror forces
                forces=np.array(forces.to(device=CPUdevice))
                for mirror in range(mirror_count):
                    np.save(folder+"/wave_event_result_force_"+str(mirror)+"_"+tag+".npy", forces[:,mirror])
                   
                #wave events
                np.save(folder+"/wave_event_data_polar_angle_"+tag+".npy", polar_angles.to(device=CPUdevice))
                np.save(folder+"/wave_event_data_azimuthal_angle_"+tag+".npy", azimuthal_angles.to(device=CPUdevice))
                np.save(folder+"/wave_event_data_A_"+tag+".npy", As.to(device=CPUdevice))
                np.save(folder+"/wave_event_data_phase_"+tag+".npy", phases.to(device=CPUdevice))
                np.save(folder+"/wave_event_data_x0_"+tag+".npy", x0s.to(device=CPUdevice))
                np.save(folder+"/wave_event_data_t0_"+tag+".npy", t0s.to(device=CPUdevice))
                np.save(folder+"/wave_event_data_f0_"+tag+".npy", fs.to(device=CPUdevice))
                np.save(folder+"/wave_event_data_sigmaf_"+tag+".npy", sigmafs.to(device=CPUdevice))
                np.save(folder+"/wave_event_data_s_polarization_"+tag+".npy", s_polarisations)
                np.save(folder+"/wave_event_data_source_"+tag+".npy", sources)
                
                total_time=(systime.time()-total_start_time)/60
                print("#total time: "+str(total_time)+" min")
    
    
    #########################################
    #~~~~~~~~~~~~~~Convenience~~~~~~~~~~~~~~#
    #########################################
    
    def loadFromSettingFile(self, tag, folder=".", ID=0):
        
            data_dict = ReadData(tag, folder,"settingFile").dictionary
            
            #~~~~~~~~~~~~~~Save management~~~~~~~~~~~~~~#
            
            self.default_ID = ID
            self.default_tag = tag
            self.default_folder = folder
            
            #~~~~~~~~~~~~~~General~~~~~~~~~~~~~~#
            
            self.default_useGPU = data_dict["useGPU"]
            self.default_randomSeed = data_dict["randomSeed"]
                
            self.default_isMonochromatic = data_dict["isMonochromatic"]
            self.default_NoR = data_dict["NoR"]
                
            #~~~~~~~~~~~~~~Constants~~~~~~~~~~~~~~#
                
            self.default_M = data_dict["M"]
            self.default_rho = data_dict["rho"]
            self.default_c_p = data_dict["c_p"]
                
            #~~~~~~~~~~~~~~Domain~~~~~~~~~~~~~~#
                
            self.default_L = data_dict["L"]
            self.default_Nx = data_dict["Nx"]
            self.default_dx = 2*data_dict["L"]/data_dict["Nx"]
                
            self.default_xmax = data_dict["x_max"]
                
            self.default_tmax = data_dict["t_max"]
            self.default_Nt = data_dict["Nt"]
            self.default_dt = self.default_tmax/data_dict["Nt"]
                
                
            #~~~~~~~~~~~~~~Experiment config~~~~~~~~~~~~~~#
                
            self.default_depth = data_dict["depth"]
            self.default_cavity_r = data_dict["cavity_r"]
                
            self.default_mirror_positions = data_dict["mirror_positions"]
            self.default_mirror_directions = data_dict["mirror_directions"]
            self.default_mirror_count = len(self.default_mirror_positions)
                
            self.default_point_source_positions = data_dict["point_source_positions"]
            self.default_source_count = len(self.default_point_source_positions)+1
                
            #~~~~~~~~~~~~~~Event parameters~~~~~~~~~~~~~~#
              
            self.default_Awavemin = data_dict["Awave_min"]             
            self.default_Awavemax = data_dict["Awave_max"]
                
            self.default_fmin = data_dict["f_min"]
            self.default_fmax = data_dict["f_max"]
            self.default_fmono = data_dict["f_mono"]
                
            self.default_sigmafmin = data_dict["sigma_f_min"]
            self.default_sigmafmax = data_dict["sigma_f_max"]
                
            self.default_anisotropy = data_dict["anisotropy"]
            
    
    
    def generateDataset(self,
            
        #~~~~~~~~~~~~~~Load and save management~~~~~~~~~~~~~~#
        tag = None,
        folder = None,
    
        ID = None, 
        saveas = None, 
    
        #~~~~~~~~~~~~~~General~~~~~~~~~~~~~~#
    
        useGPU = None, 
        NoR = None,
    
        #~~~~~~~~~~~~~~Dataset Parameters~~~~~~~~~~~~~~#
                                       
        state = None, 
        NoS = None,
    
        freq = None,
        SNR = None,
        p = None,
        c_ratio = None,
    
        #~~~~~~~~~~~~~~Window Parameters~~~~~~~~~~~~~~#
    
        NoW = None,
        NoT = None,
        NoE = None,
    
        time_window_multiplier = None,
        twindow = None,
        randomlyPlaced = None,

        mirror_ID = None
        ):
            
            #~~~~~~~~~~~~~~Save management~~~~~~~~~~~~~~#
            ID = ID if not ID==None else self.default_ID
            folder = folder if not folder==None else self.default_folder
            if tag!=None:
                self.loadFromSettingFile(tag, folder, ID)
            else:
                tag = self.default_tag
                
            saveas = saveas if not saveas==None else self.default_saveas
            
            #~~~~~~~~~~~~~~General~~~~~~~~~~~~~~#
            
            useGPU = useGPU if not useGPU==None else self.default_useGPU
            NoR = NoR if not NoR==None else self.default_NoR
            #randomSeed = randomSeed if not randomSeed==None else self.default_randomSeed
            
            #~~~~~~~~~~~~~~Dataset Parameters~~~~~~~~~~~~~~#
            
            state = state if not state==None else self.default_state
            NoS = NoS if not NoS==None else self.default_NoS
        
            freq = freq if not freq==None else self.default_freq
            SNR = SNR if not SNR==None else self.default_SNR
            p = p if not p==None else self.default_p
            c_ratio = c_ratio if not c_ratio==None else self.default_c_ratio
        
            #~~~~~~~~~~~~~~Window Parameters~~~~~~~~~~~~~~#
        
            NoW = NoW if not NoW==None else self.default_NoW
            NoT = NoT if not NoT==None else self.default_NoT
            NoE = NoE if not NoE==None else self.default_NoE
        
            time_window_multiplier = time_window_multiplier if not time_window_multiplier==None else self.default_time_window_multiplier
            twindow = twindow if not twindow==None else self.default_twindow
            randomlyPlaced = randomlyPlaced if not randomlyPlaced==None else self.default_randomlyPlaced

            mirror_ID = mirror_ID if not mirror_ID == None else 0
            
            #####################################
            #~~~~~~~~~~~~~~Loading~~~~~~~~~~~~~~#
            #####################################
        
        
            #~~~~~~~~~~~~~~Read settings file~~~~~~~~~~~~~~#
        
                    
            data=ReadData(tag, folder)
            if useGPU:
                device=torch.device("cuda")
            else:
                device=torch.device("cpu")
        
        
            #~~~~~~~~~~~~~~Read parameters and constants~~~~~~~~~~~~~~#
        
            if data.dictionary["randomSeed"]=="None":
                randomSeed=None
            else:
                randomSeed=int(data.dictionary["randomSeed"])
                np.random.seed(randomSeed+1)
                
            isMonochromatic=data.dictionary["isMonochromatic"]
        
            NoR=min(int(data.dictionary["NoR"]),NoR)        #Number of runs/realizations
            NoT=min(int(data.dictionary["NoR"])-3,NoT)      #Number of runs without update of WF
        
            #constants
            pi=torch.pi
            G=co.gravitational_constant
        
            M=data.dictionary["M"]
            rho=data.dictionary["rho"]
        
            numerical_cavern_factor=-4*pi/3*G*M*rho         #force from shift of cavern per total density
        
            c_p=data.dictionary["c_p"]                      #sound velocity in rock  #6000 m/s
            c_s=c_p*c_ratio
        
        
            L=data.dictionary["L"]
        
            tmax=data.dictionary["t_max"]                   #time of simulation
            Nt=int(data.dictionary["Nt"])
            dt=tmax/Nt                                      #temporal stepwidth
        
            if twindow==None or twindow<0:
                Ntwindow=int(Nt*time_window_multiplier)
            else:
                Ntwindow=int(Nt*twindow/tmax)
        
        
            cavity_r=data.dictionary["cavity_r"]
        
            mirror_positions=data.dictionary["mirror_positions"]
            mirror_directions=data.dictionary["mirror_directions"]
            mirror_count=len(mirror_positions)
        
        
            try:
                point_source_positions=torch.tensor(np.array([[0,0,1e2]]+list(data.dictionary["point_source_positions"])))
                sources=torch.tensor(np.load(folder+"/wave_event_data_source_"+tag+".npy", mmap_mode='r')[:NoR].copy(), device=device)
            except:
                point_source_positions=torch.tensor([[2.2,-1e-3,1e5]])
                sources=torch.zeros(NoR,device=device)
        
            time=torch.tensor(np.linspace(0,tmax,Nt,endpoint=False), device=device)
        
        
            #~~~~~~~~~~~~~~Load data set~~~~~~~~~~~~~~#
              
            #mirror forces
            all_bulk_forces=np.zeros((NoR,mirror_count,Nt))
            for mirror in range(mirror_count):
                all_bulk_forces[:,mirror]=np.load(folder+"/wave_event_result_force_"+str(mirror)+"_"+tag+".npy", mmap_mode='r')[:NoR].copy()
            all_bulk_forces=torch.tensor(all_bulk_forces,device=device)
        
            #wave events
            all_polar_angles=torch.tensor(np.load(folder+"/wave_event_data_polar_angle_"+tag+".npy", mmap_mode='r')[:NoR].copy(), device=device)
            all_azimuthal_angles=torch.tensor(np.load(folder+"/wave_event_data_azimuthal_angle_"+tag+".npy", mmap_mode='r')[:NoR].copy(), device=device)
        
            all_x0s=torch.tensor(np.load(folder+"/wave_event_data_x0_"+tag+".npy", mmap_mode='r')[:NoR].copy(), device=device)
            all_t0s=torch.tensor(np.load(folder+"/wave_event_data_t0_"+tag+".npy", mmap_mode='r')[:NoR].copy(), device=device)
        
            all_As=torch.tensor(np.load(folder+"/wave_event_data_A_"+tag+".npy", mmap_mode='r')[:NoR].copy(), device=device)
            all_phases=torch.tensor(np.load(folder+"/wave_event_data_phase_"+tag+".npy", mmap_mode='r')[:NoR].copy(), device=device)
            all_fs=torch.tensor(np.load(folder+"/wave_event_data_f0_"+tag+".npy", mmap_mode='r')[:NoR].copy(), device=device)
            all_sigmafs=torch.tensor(np.load(folder+"/wave_event_data_sigmaf_"+tag+".npy", mmap_mode='r')[:NoR].copy(), device=device)
        
            #P and S
            all_s_polarization=torch.tensor(np.load(folder+"/wave_event_data_s_polarization_"+tag+".npy", mmap_mode='r')[:NoR].copy(), device=device)
        
            all_is_s=np.random.random(NoR)>p
            all_is_s=torch.tensor(all_is_s, device=device)
            all_cs=c_p*(all_is_s==False)+c_s*all_is_s
        
            #other preparations
            all_sin_polar=torch.sin(all_polar_angles)
            all_cos_polar=torch.cos(all_polar_angles)
            all_sin_azimuthal=torch.sin(all_azimuthal_angles)
            all_cos_azimuthal=torch.cos(all_azimuthal_angles)
            all_sin_s_polarization=torch.sin(all_s_polarization)
            all_cos_s_polarization=torch.cos(all_s_polarization)
        
            all_forces=torch.zeros((NoR,mirror_count,Nt), device=device)
            all_seismometer_data=torch.zeros((NoR,NoS,3,Nt), device=device)

            
            def gaussian_wave_packet(x,t,x0,t0,A,exp_const,sin_const,c,phase=0):
                    
                    diff = (x - x0) / c - (t - t0)
                    exp_term = torch.exp(exp_const * diff**2)
                    sin_term = torch.sin(sin_const * diff + phase)
                    
                    wave = A * exp_term * sin_term
                    return wave
            
            
            #~~~~~~~~~~~~~~Analytical displacement functions~~~~~~~~~~~~~~#
            
            def gaussian_wave_packet_displacement(x,t,x0,t0,f0,sigmaf,c,A,phase):
                
                    diff = (x - x0) / c - (t - t0)
                    
                    VF = 1/(math.sqrt(2*pi)*sigmaf)*1/2 * A * c_p * torch.exp(-1j * phase - f0**2 / (2 * sigmaf**2))
                    
                    if torch.all(phase==0):
                        wave = VF * torch.imag(sp.erf((2*pi * sigmaf**2 * diff + 1j * f0) / (math.sqrt(2) * sigmaf)))
                    else:
                        wave = -VF/2 * 1j * (sp.erf((2*pi * sigmaf**2 * diff + 1j * f0) / (math.sqrt(2) * sigmaf)) - np.exp(2 * 1j * phase) * sp.erf((2*pi * sigmaf**2 * diff - 1j * f0) / (math.sqrt(2) * sigmaf)))
                    return torch.real(wave)
            
            def monochromatic_wave_displacement(x,t,x0,t0,f0,c,A,phase):
                
                    diff = (x - x0) / c - (t - t0)
                    
                    wave=A*c_p/2/pi/f0*torch.cos(2*pi*f0*diff+phase)
                    return wave

            def precalculateForce(): 
                
                    #get local displacement (at each mirror)
                    pos=torch.tensor(mirror_positions, device=device).reshape(1,mirror_count,3)
                    di=torch.tensor(mirror_directions, device=device).reshape(1,mirror_count,3)
                    projectedMirrorPosition =pos[:,:,0]*(all_cos_polar*all_sin_azimuthal).reshape(NoR,1)
                    projectedMirrorPosition+=pos[:,:,1]*(all_sin_polar*all_sin_azimuthal).reshape(NoR,1)
                    projectedMirrorPosition+=pos[:,:,2]*(all_cos_azimuthal).reshape(NoR,1)
                    
                    point_source_distance=torch.sqrt((point_source_positions[sources,0].reshape(NoR,1)-pos[:,:,0])**2+(point_source_positions[sources,1].reshape(NoR,1)-pos[:,:,1])**2+(point_source_positions[sources,2].reshape(NoR,1)-pos[:,:,2])**2)
                    projectedMirrorPosition=projectedMirrorPosition*(sources==0).reshape(NoR,1)+point_source_distance*(sources!=0).reshape(NoR,1)
                    
                    if isMonochromatic:
                        absoluteDisplacement=monochromatic_wave_displacement(projectedMirrorPosition.reshape(NoR,mirror_count,1), time.reshape(1,1,Nt), all_x0s.reshape(NoR,1,1), all_t0s.reshape(NoR,1,1), all_fs.reshape(NoR,1,1), all_cs.reshape(NoR,1,1), all_As.reshape(NoR,1,1), all_phases.reshape(NoR,1,1))
                    else:
                        absoluteDisplacement=gaussian_wave_packet_displacement(projectedMirrorPosition.reshape(NoR,mirror_count,1), time.reshape(1,1,Nt), all_x0s.reshape(NoR,1,1), all_t0s.reshape(NoR,1,1), all_fs.reshape(NoR,1,1), all_sigmafs.reshape(NoR,1,1), all_cs.reshape(NoR,1,1), all_As.reshape(NoR,1,1), all_phases.reshape(NoR,1,1))
                    
                    point_source_vector=(pos-point_source_positions[sources].reshape(NoR,1,3))
                    point_source_polar_angle=torch.arctan2(point_source_vector[:,:,1],point_source_vector[:,:,0])
                    point_source_azimuthal_angle=torch.arccos(point_source_vector[:,:,2]/torch.sqrt(point_source_vector[:,:,0]**2+point_source_vector[:,:,1]**2+point_source_vector[:,:,2]**2))
                    cos_polar=all_cos_polar.reshape(NoR,1)*(sources==0).reshape(NoR,1)+torch.cos(point_source_polar_angle)*(sources!=0).reshape(NoR,1)
                    sin_polar=all_sin_polar.reshape(NoR,1)*(sources==0).reshape(NoR,1)+torch.sin(point_source_polar_angle)*(sources!=0).reshape(NoR,1)
                    cos_azi=all_cos_azimuthal.reshape(NoR,1)*(sources==0).reshape(NoR,1)+torch.cos(point_source_azimuthal_angle)*(sources!=0).reshape(NoR,1)
                    sin_azi=all_sin_azimuthal.reshape(NoR,1)*(sources==0).reshape(NoR,1)+torch.sin(point_source_azimuthal_angle)*(sources!=0).reshape(NoR,1)
                    
                    #cavern acceleration parallel to local displacement
                    all_p_cavern_forces =di[:,:,0]*cos_polar*sin_azi
                    all_p_cavern_forces+=di[:,:,1]*sin_polar*sin_azi
                    all_p_cavern_forces+=di[:,:,2]*cos_azi
                    all_p_cavern_forces=all_p_cavern_forces.reshape(NoR,mirror_count,1)*absoluteDisplacement*numerical_cavern_factor
                    
                    #cavern acceleration perpendicular to local displacement
                    all_s_cavern_forces =di[:,:,0]*(-sin_polar*all_sin_s_polarization.reshape(NoR,1)+cos_polar*cos_azi*all_cos_s_polarization.reshape(NoR,1))
                    all_s_cavern_forces+=di[:,:,1]*(cos_polar*all_sin_s_polarization.reshape(NoR,1)+sin_polar*cos_azi*all_cos_s_polarization.reshape(NoR,1))
                    all_s_cavern_forces+=di[:,:,2]*(-sin_azi*all_cos_s_polarization.reshape(NoR,1))
                    all_s_cavern_forces=all_s_cavern_forces.reshape(NoR,mirror_count,1)*absoluteDisplacement*numerical_cavern_factor
                    
                    #add P- and S-contributions
                    all_forces=(all_bulk_forces+all_p_cavern_forces)*(all_is_s==False).reshape(NoR,1,1) + all_s_cavern_forces*(all_is_s).reshape(NoR,1,1)
                    
                    return all_forces
            
            
            #extract displacement at seismometer positions
            def getDisplacement(seismometer_positions):
                    
                    #get total displacement at each seismometer
                    pos=torch.tensor(np.array(seismometer_positions), device=device).reshape(1,NoS,3)
                    projectedSeismometerPosition =pos[:,:,0]*(all_cos_polar*all_sin_azimuthal).reshape(NoR,1)
                    projectedSeismometerPosition+=pos[:,:,1]*(all_sin_polar*all_sin_azimuthal).reshape(NoR,1)
                    projectedSeismometerPosition+=pos[:,:,2]*(all_cos_azimuthal).reshape(NoR,1)
                    
                    point_source_distance=torch.sqrt((point_source_positions[sources,0].reshape(NoR,1)-pos[:,:,0])**2+(point_source_positions[sources,1].reshape(NoR,1)-pos[:,:,1])**2+(point_source_positions[sources,2].reshape(NoR,1)-pos[:,:,2])**2)
                    projectedSeismometerPosition=projectedSeismometerPosition*(sources==0).reshape(NoR,1)+point_source_distance*(sources!=0).reshape(NoR,1)
                        
                    if isMonochromatic:
                        absoluteDisplacement=monochromatic_wave_displacement(projectedSeismometerPosition.reshape(NoR,NoS,1), time.reshape(1,1,Nt), all_x0s.reshape(NoR,1,1), all_t0s.reshape(NoR,1,1), all_fs.reshape(NoR,1,1), all_cs.reshape(NoR,1,1), all_As.reshape(NoR,1,1), all_phases.reshape(NoR,1,1))
                    else:
                        absoluteDisplacement=gaussian_wave_packet_displacement(projectedSeismometerPosition.reshape(NoR,NoS,1), time.reshape(1,1,Nt), all_x0s.reshape(NoR,1,1), all_t0s.reshape(NoR,1,1), all_fs.reshape(NoR,1,1), all_sigmafs.reshape(NoR,1,1), all_cs.reshape(NoR,1,1), all_As.reshape(NoR,1,1), all_phases.reshape(NoR,1,1))
                    
                    point_source_vector=(pos-point_source_positions[sources].reshape(NoR,1,3))
                    point_source_polar_angle=torch.arctan2(point_source_vector[:,:,1],point_source_vector[:,:,0])
                    point_source_azimuthal_angle=torch.arccos(point_source_vector[:,:,2]/torch.sqrt(point_source_vector[:,:,0]**2+point_source_vector[:,:,1]**2+point_source_vector[:,:,2]**2))
                    cos_polar=all_cos_polar.reshape(NoR,1)*(sources==0).reshape(NoR,1)+torch.cos(point_source_polar_angle)*(sources!=0).reshape(NoR,1)
                    sin_polar=all_sin_polar.reshape(NoR,1)*(sources==0).reshape(NoR,1)+torch.sin(point_source_polar_angle)*(sources!=0).reshape(NoR,1)
                    cos_azi=all_cos_azimuthal.reshape(NoR,1)*(sources==0).reshape(NoR,1)+torch.cos(point_source_azimuthal_angle)*(sources!=0).reshape(NoR,1)
                    sin_azi=all_sin_azimuthal.reshape(NoR,1)*(sources==0).reshape(NoR,1)+torch.sin(point_source_azimuthal_angle)*(sources!=0).reshape(NoR,1)
                    
                    #project onto 3 axes
                    all_p_displacements=torch.zeros((NoR,NoS,3,Nt), device=device)
                    all_p_displacements[:,:,0,:]+=(cos_polar*sin_azi).reshape(NoR,NoS,1)
                    all_p_displacements[:,:,1,:]+=(sin_polar*sin_azi).reshape(NoR,NoS,1)
                    all_p_displacements[:,:,2,:]+=(cos_azi).reshape(NoR,NoS,1)
                    all_p_displacements=all_p_displacements*absoluteDisplacement.reshape(NoR,NoS,1,Nt)
                    
                    all_s_displacements=torch.zeros((NoR,NoS,3,Nt), device=device)
                    all_s_displacements[:,:,0,:]+=(-sin_polar*all_sin_s_polarization.reshape(NoR,1)+cos_polar*cos_azi*all_cos_s_polarization.reshape(NoR,1)).reshape(NoR,NoS,1)
                    all_s_displacements[:,:,1,:]+=(cos_polar*all_sin_s_polarization.reshape(NoR,1)+sin_polar*cos_azi*all_cos_s_polarization.reshape(NoR,1)).reshape(NoR,NoS,1)
                    all_s_displacements[:,:,2,:]+=(-sin_azi*all_cos_s_polarization.reshape(NoR,1)).reshape(NoR,NoS,1)
                    all_s_displacements=all_s_displacements*absoluteDisplacement.reshape(NoR,NoS,1,Nt)
                    
                    all_displacements=all_p_displacements*(all_is_s==False).reshape(NoR,1,1,1) + all_s_displacements*(all_is_s).reshape(NoR,1,1,1)
                    
                    return all_displacements
                
            
            all_forces=precalculateForce()
            all_seismometer_data=getDisplacement(state)
            
            return all_seismometer_data.reshape(NoR,NoS*3,Nt), all_forces[:,mirror_ID]
                
                


"""

#####################################
#~~~~~~~~~~~~~~Windows~~~~~~~~~~~~~~#
#####################################


class Window:

    
#~~~~~~~~~~~~~~Window properties~~~~~~~~~~~~~~#

    ID=0
    Ntwindow=Nt
    NoE=NoE
    seismometer_positions=state
    NoS=NoS
    
    forces=torch.zeros((mirror_count,Ntwindow))
    displacements=torch.zeros((NoS,3,Ntwindow))
    
    NxPlot=100
    
    forcecolor=["red","blue","green","orange","pink","lightblue","lightgreen","brown","magenta"]
    
    x2d=torch.zeros((NxPlot,NxPlot))
    y2d=torch.zeros((NxPlot,NxPlot))
    density_fluctuations=torch.zeros((NxPlot,NxPlot))
    
    startTime=torch.zeros(NoE)
    windowR=torch.linspace(0,NoE-1,NoE)
    time=torch.linspace(0,Ntwindow*dt-1,Ntwindow)
    
    randomlyPlaced=True
    
    def __init__(self, windowID, Ntwindow=Nt, NoE=NoE, seismometer_positions=state, NoS=NoS, randomlyPlaced=True):
        self.ID=windowID
        self.Ntwindow=Ntwindow
        self.NoE=NoE
        self.seismometer_positions=torch.tensor(np.array(seismometer_positions)).reshape(NoS,3)
        self.NoS=NoS
        self.time=np.linspace(0,(self.Ntwindow-1)*dt,self.Ntwindow)
        
        if self.ID<NoR//self.NoE:
            self.windowR=torch.linspace(self.ID*self.NoE,self.ID*self.NoE+self.NoE-1,self.NoE).int()
        else:
            self.windowR=np.random.randint(0,NoR,self.NoE)
        
        self.randomlyPlaced=randomlyPlaced
            
        
        self.forces=torch.zeros((mirror_count,Ntwindow))
        self.displacements=torch.zeros((NoS,3,Ntwindow))
        
        if randomlyPlaced:
            self.createRandomTimeWindow()
        else:
            self.createStaticTimeWindow()
        
        
#~~~~~~~~~~~~~~Build window~~~~~~~~~~~~~~#
        
    def createRandomTimeWindow(self):
        startIndex = np.random.randint(-Nt,self.Ntwindow,self.NoE)
        self.startTime = startIndex*dt
        for n in range(self.NoE):
            forces, displacements = all_forces[self.windowR[n]], all_seismometer_data[self.windowR[n]]
            #forces,displacements = getForceAndDisplacement(self.windowR[n], self.seismometer_positions)
            fromWindowindex=max(0,startIndex[n])
            toWindowindex=min(self.Ntwindow,startIndex[n]+Nt)
            fromRunIndex=max(0,-startIndex[n])
            toRunIndex=min(Nt,self.Ntwindow-startIndex[n])
            self.forces[:,fromWindowindex:toWindowindex]+=forces[:,fromRunIndex:toRunIndex]
            self.displacements[:,:,fromWindowindex:toWindowindex]+=displacements[:,:,fromRunIndex:toRunIndex]
        self.addNoise()
        
        
    def createStaticTimeWindow(self):
        self.startTime = np.zeros(self.NoE)*dt
        for n in range(self.NoE):
            forces, displacements = all_forces[self.windowR[n]], all_seismometer_data[self.windowR[n]]
            #forces,displacements = getForceAndDisplacement(self.windowR[n], self.seismometer_positions)
            self.forces[:,:Nt]+=forces[:,:Ntwindow] 
            self.displacements[:,:,:Nt]+=displacements[:,:,:Ntwindow]
        self.addNoise()
            
            
    def addNoise(self):
        if isMonochromatic: 
            #sigma=np.sqrt(torch.mean(plane_wave_displacement_analytic(0, 0, 0, 0, all_fs[self.windowR], all_sigmafs[self.windowR], all_cs[self.windowR], all_As[self.windowR])**2))/SNR*np.sqrt(self.Ntwindow)
            #sigma=np.sqrt(np.mean(np.max(np.sqrt(np.sum(np.array(self.displacements)**2,axis=1)),axis=-1)**2))/SNR*np.sqrt(self.Ntwindow)
            #maximum=float(-plane_wave_displacement_analytic(0, 0, 0, 0, all_fs[self.windowR], all_sigmafs[self.windowR], all_cs[self.windowR], all_As[self.windowR]))
            sigma=float(torch.sqrt(torch.mean((all_As[self.windowR]*all_cs[self.windowR]/2/pi/all_fs[self.windowR]/SNR*math.sqrt(self.Ntwindow)/math.sqrt(3)/2)**2)))
            #sigma=np.sqrt(np.mean(np.max(np.array(self.displacements),axis=-1)**2))/SNR*np.sqrt(self.Ntwindow)/2
            #sigma=np.sqrt(np.mean(maximum**2))/SNR*np.sqrt(self.Ntwindow)/2/np.sqrt(3)
        else:
            sigma=math.sqrt(np.mean(np.max(np.array(self.displacements),axis=-1)**2))/SNR*math.sqrt(self.Ntwindow)/2
            #sigma=np.sqrt(torch.mean(gauss_wave_packet_displacement_analytic(0, 0, 0, 0, all_fs[self.windowR], all_sigmafs[self.windowR], all_cs[self.windowR], all_As[self.windowR])**2))/SNR*np.sqrt(self.Ntwindow)/2
        self.displacements+=torch.tensor(np.random.normal(self.displacements*0,sigma))
    
    
#~~~~~~~~~~~~~~Window visualization~~~~~~~~~~~~~~#
    
    def calculateVisualWindow(self,Lzoom=L,NxPlot=100):
        self.NxPlot=NxPlot
        dx=2*Lzoom/self.NxPlot
        x=torch.linspace(-Lzoom+dx/2,Lzoom-dx/2,self.NxPlot)
        y=torch.linspace(-Lzoom+dx/2,Lzoom-dx/2,self.NxPlot)
        xyz=torch.meshgrid(x,y,indexing="ij")
        self.x2d=xyz[1]
        self.y2d=xyz[0]
        
        
    def calculateDensityFluctuation(self, timestep=0):
        self.density_fluctuations=torch.zeros((self.NxPlot,self.NxPlot))
        for n in range(self.NoE):
            R=self.windowR[n]
            if sources[R]==0:
                kx2D=all_cos_polar[R]*all_sin_azimuthal[R]*self.x2d+all_sin_polar[R]*all_sin_azimuthal[R]*self.y2d
            else:
                kx2D=torch.sqrt((self.x2d-point_source_positions[sources[R]][0])**2+(self.y2d-point_source_positions[sources[R]][1])**2)
            self.density_fluctuations+=gaussian_wave_packet(kx2D, timestep*dt, all_x0s[R], self.startTime[n], all_As[R], -2*pi**2*all_sigmafs[R]**2, 2*pi*all_fs[R], all_cs[R], all_phases[R])
    
    
    def vizualizeWindow(self, animate=True, timestep=0, Lzoom=1000, NxPlot=100):
        #nice 2D-image+animation
        self.calculateVisualWindow(Lzoom,NxPlot)
        self.calculateDensityFluctuation()
        dis_scale=1/torch.max(self.displacements)*Lzoom/30
        
        fullfig=plt.figure(figsize=(15,15))
        title=fullfig.suptitle("Density Fluctuations, Force and Seismometer Data",fontsize=16,y=0.95)
        
        #density fluctuation plot
        ax1=plt.subplot(2,6,(1,3))
        plt.title(r"density fluctuations in $(x,y,z=0)$")
        im=plt.imshow(np.array(self.density_fluctuations)[::-1,:],extent=[-Lzoom,Lzoom,-Lzoom,Lzoom],label=r"$\delta\rho$")
        plt.colorbar(ax=ax1,label=r"$\delta\rho/\rho$")
        plt.clim(-torch.max(all_As[self.windowR]),torch.max(all_As[self.windowR]))
        plt.xlabel(r"$x$ [m]")
        plt.ylabel(r"$y$ [m]")
        vec=[]
        cav=[]
        for mirror in range(mirror_count):
            pos=mirror_positions[mirror]
            di=mirror_directions[mirror]
            if float(max(torch.abs(self.forces[mirror])))>0:
                vec.append(plt.quiver(pos[0],pos[1],self.forces[mirror,timestep]*di[0],self.forces[mirror,timestep]*di[1],color=self.forcecolor[mirror],angles='xy', scale_units='xy',scale=float(max(torch.abs(self.forces.flatten()))/Lzoom)))
            cav.append(plt.Circle((pos[0],pos[1]),cavity_r*(Lzoom/100),fill=True,edgecolor="k",linewidth=1.5))
            ax1.add_patch(cav[mirror])
            plt.text(pos[0],pos[1],str(mirror),fontsize=13.5,horizontalalignment='center',verticalalignment='center')
        
        scat=plt.scatter(self.seismometer_positions[:,0],self.seismometer_positions[:,1],marker="d",c="white",s=150)
        stexts=[]
        for s in range(self.NoS):
            stexts.append(plt.text(self.seismometer_positions[s,0],self.seismometer_positions[s,1],str(s),fontsize=13.5,horizontalalignment='center',verticalalignment='center'))
        
        #force plot
        ax2=plt.subplot(2,6,(4,6))
        forceplot=[]
        for mirror in range(mirror_count):
            forceplot.append(ax2.plot(self.time,self.forces[mirror],label="◯"+str(mirror),color=self.forcecolor[mirror])[0])
        #estimateplot=ax2.plot(time,estimate,label="estimate")[0]
        #diffplot=ax2.plot(time,force-estimate,label="difference",color="red")[0]
        plt.xlim(0,np.max(self.time))
        if float(torch.max(torch.abs(self.forces)))>0:
            plt.ylim(-torch.max(np.abs(self.forces)),torch.max(np.abs(self.forces)))
        plt.title(r"force on mirror")
        plt.ylabel(r"force [N]")
        plt.xlabel(r"time [s]")
        ax2.yaxis.set_label_position("right")
        ax2.yaxis.tick_right()
        ax2.legend()
        
        #displacement plots
        axarray=[[],[],[]]
        lines=[[],[],[]]
        plotspace=2
        titles=[r"$x$-displacement [m]",r"$y$-displacement [m]",r"$z$-displacement [m]"]
        max_dim=3
        NoSplot=min(NoS,5)
        for s in range(NoSplot):
            for dim in range(max_dim):
                axarray[dim].append(plt.subplot(2*NoSplot+plotspace,6,(NoSplot*6+(int(6/max_dim)*dim+1)+(s+plotspace)*6,NoSplot*6+(int(6/max_dim)*dim+2)+(s+plotspace)*6)))
                color=[0,0,0]
                color[dim]=0.2+0.8/NoSplot*s
                lines[dim].append(axarray[dim][s].plot(self.time,self.displacements[s][dim],color=color)[0])
                plt.xlim(0,max(self.time))
                plt.ylim(-torch.max(torch.abs(self.displacements)),torch.max(torch.abs(self.displacements)))
                if dim==0:
                    plt.ylabel(r"◊"+str(s))
                if dim==1:
                    axarray[dim][s].set_yticklabels(())
                    axarray[dim][s].set_yticks(())
                if s==0:
                    plt.title(titles[dim])
                if s==NoSplot-1:
                    plt.xlabel(r"time [s]")
                else:        
                    axarray[dim][s].set_xticklabels(())
                    axarray[dim][s].tick_params(axis="x",direction="in")
                if dim==max_dim-1:
                    axarray[max_dim-1][s].yaxis.set_label_position("right")
                    axarray[max_dim-1][s].yaxis.tick_right()
                    
        plt.subplots_adjust(hspace=0)
        
        #animation
        stepw=2
        def update_full(i):
            i=stepw*i
            t=i*dt
            self.calculateDensityFluctuation(i)
            title.set_text(r"Density Fluctuations, Force and Seismometer Data at $t=$"+str(round(t,3)))
            im.set_array(np.array(self.density_fluctuations)[::-1,:])
            scat.set_offsets(np.array([np.array(self.seismometer_positions[:,0]+self.displacements[:,0,i]*dis_scale),np.array(self.seismometer_positions[:,1]+self.displacements[:,1,i]*dis_scale)]).T)
            for s in range(NoS):
                stexts[s].set_position((np.array(self.seismometer_positions[s,0]+self.displacements[s,0,i]*dis_scale),np.array(self.seismometer_positions[s,1]+self.displacements[s,1,i]*dis_scale)))
            for mirror in range(mirror_count):
                di=mirror_directions[mirror]
                if float(max(torch.abs(self.forces[mirror])))>0:
                    vec[mirror].set_UVC(self.forces[mirror][i]*di[0],self.forces[mirror][i]*di[1])
                forceplot[mirror].set_xdata(self.time[:i+1])
                forceplot[mirror].set_ydata(self.forces[mirror][:i+1])
            #estimateplot.set_xdata(time[:i+1])
            #estimateplot.set_ydata(estimate[:i+1])
            #diffplot.set_xdata(time[:i+1])
            #diffplot.set_ydata(force[:i+1]-estimate[:i+1])
            for s in range(NoSplot):
                for dim in range(max_dim):
                    lines[dim][s].set_xdata(self.time[:i+1])
                    lines[dim][s].set_ydata(self.displacements[s][dim][:i+1])
        
        if animate:
            anima=ani.FuncAnimation(fig=fullfig, func=update_full, frames=int((int(Ntwindow)+1)/stepw),interval=max(1,int(5/stepw)))
            anima.save("fullanimation3D"+str(tag)+"_"+str(self.ID)+".gif")
        else:
            update_full(timestep//stepw)
            plt.savefig("windowAtTimeStep"+str(timestep)+str(tag)+"_"+str(self.ID)+".svg")
            
            
    def visualizeForce(self, mirrors="all"):
        
        plt.figure()
        plt.title(r"mirror forces")
        plt.ylabel(r"force $F_M(t)$ [N]")
        plt.xlabel(r"time $t$ [s]")

        if type(mirrors)==int:
            plt.plot(self.time,self.forces[mirrors],label="◯"+str(mirrors),color=self.forcecolor[mirrors])
        elif type(mirrors)==list:
            for mirror in mirrors:
                plt.plot(self.time,self.forces[mirror],label="◯"+str(mirror),color=self.forcecolor[mirror])
        else:
            for mirror in range(mirror_count):
                plt.plot(self.time,self.forces[mirror],label="◯"+str(mirror),color=self.forcecolor[mirror])
                #estimateplot=ax2.plot(time,estimate,label="estimate")[0]
                #diffplot=ax2.plot(time,force-estimate,label="difference",color="red")[0]
        
        plt.legend()
        
        
    def visualizeDisplacement(self,seismometers="all"):
        
        plt.figure()
        plt.title("seismometer displacements")
        
        plt.xlabel(r"time $t$ [s]")
        plt.ylabel(r"displacement $\vec\ xi$ [m]")
        
        direction=[r"$x$",r"$y$",r"$z$"]
        max_dim=3
        
        if type(seismometers)==int:
            for dim in range(max_dim):
                color=[0,0,0]
                color[dim]=0.2+0.8/NoS*seismometers
                plt.plot(self.time,self.displacements[seismometers][dim],color=color,label=r"◊"+str(seismometers)+" "+str(direction[dim]))
        elif type(seismometers)==list:
            for s in seismometers:
                for dim in range(max_dim):
                    color=[0,0,0]
                    color[dim]=0.2+0.8/NoS*s
                    plt.plot(self.time,self.displacements[s][dim],color=color,label=r"◊"+str(s)+" "+str(direction[dim]))
        else:
            for s in range(NoS):
                for dim in range(max_dim):
                    color=[0,0,0]
                    color[dim]=0.2+0.8/NoS*s
                    plt.plot(self.time,self.displacements[s][dim],color=color,label=r"◊"+str(s)+" "+str(direction[dim]))
                    
        plt.legend()
"""
