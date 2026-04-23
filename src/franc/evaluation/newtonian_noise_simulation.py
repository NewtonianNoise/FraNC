# -*- coding: utf-8 -*-
# pylint: disable=too-many-lines,invalid-name,too-many-arguments,too-many-positional-arguments,too-many-instance-attributes,too-few-public-methods, too-many-locals,too-many-statements, no-member, too-many-branches, singleton-comparison
"""
Created on Tue Jan 27 12:45:15 2026

@author: schillings
"""

import math
import time as systime
import os
import random as rd
import ast

import numpy as np
import matplotlib.pyplot as plt
import scipy.constants as co
import scipy.special as sp

import torch

# plotting defaults
plt.rc("legend", fontsize=15)
plt.rc("axes", labelsize=16, titlesize=16)
plt.rc("xtick", labelsize=13)
plt.rc("ytick", labelsize=13)
plt.rc("figure", figsize=(10, 9))

total_start_time = systime.time()


pi = torch.pi
G = co.gravitational_constant


#########################################
# ~~~~~~~~~~~~~~File Reader~~~~~~~~~~~~~~#
#########################################


class ReadData:
    """Imports an output file from the Newtonian noise simulation and stores it as a dictionary
        The output file is typically named folder + "/" + fileType + tag + ".txt"

    :param tag: the tag used for generating the eventSet or dataSet
    :param folder: the path where the file is
    :param fileType: identifier of the file
    """

    tag = ""
    folder = ""
    fileType = ""

    dictionary: dict = {}

    def __init__(self, tag, folder, fileType="settingFile"):
        self.tag = tag
        self.folder = folder
        self.fileType = fileType

        dataFile = np.loadtxt(
            folder + "/" + fileType + tag + ".txt",
            dtype=str,
            delimiter="ö",
            comments="//",
        )

        for line in dataFile:
            key = line.split(" = ")[0]
            value = line.split(" = ")[1]
            if value in ["True", "False"]:
                value = value == "True"
            elif "np.array" in value or "torch.tensor" in value:
                value = np.array(ast.literal_eval(value.split("(")[1].split(")")[0]))
            elif "min" in value:
                key = key[1:]
                value = float(value.split(" min")[0])
            else:
                try:
                    value = float(value)
                    if value.is_integer():
                        value = int(value)
                except ValueError:
                    pass
            self.dictionary.update({key: value})


#############################################
# ~~~~~~~~~~~~~~Event Generator~~~~~~~~~~~~~~#
#############################################


class NewtonianNoiseDataGenerator:
    """Manages and generates Newtonian noise event- and dataSets.
    A dataSet is an abitrary combination of the bare eventSet.
    It generates 3D displacement witness channels and a Newtonian-noise force target channel.

    # ~~~~~~~~~~~~~~Save management~~~~~~~~~~~~~~#
    :param ID=5: int(argv[1])
    :param tag="X": dataSet identifier
    :param folder="testset/dataset": dataSet path
    :param saveas="testset/resultFile": Identifier for all savefiles produced
    # ~~~~~~~~~~~~~~General~~~~~~~~~~~~~~#
    :param useGPU=False: Set True if you have and want to use GPU-resources
    :param randomSeed=1: If None, use no seed
    :param isMonochromatic=False: Toggles between monochromatic and Gaussian plane wave packets
    :param NoR=10: Number of runs/realizations/wave events
    # ~~~~~~~~~~~~~~Constants~~~~~~~~~~~~~~#
    :param M=211: Mirror mass of LF-ET in kg
    :param rho=3000: Density of rock in kg/m³
    :param c_p=6000: Sound velocity of rock #6000 m/s
    # ~~~~~~~~~~~~~~Domain~~~~~~~~~~~~~~#
    :param L=6000: Length of simulation box in m
    :param Nx=200: Number of spatial steps for force calculation, choose even number
    :param xmax=6000: Distance of wave starting point from 0
    :param tmax=None: Time of simulation in s
    :param Nt=200: Number of time steps
    # ~~~~~~~~~~~~~~Experiment config~~~~~~~~~~~~~~#
    :param depth=6000: Upper domain cutoff (=L for full space)
    :param cavity_r=5: Radius of spherical cavern in m
    :param mirror_positions=[[0, 0, 0]]: Array of mirror position [x,y,z] in m
    :param mirror_directions=[[1, 0, 0]]: Array of mirror free moving axis unit vectors
    :param point_source_positions=[[-1000, -1000, 0]]: Array of point source position [x,y,z] in m
    # ~~~~~~~~~~~~~~Event parameters~~~~~~~~~~~~~~#
    :param Awavemin=1e-11: Minimum amplitude of delta_rho/rho
    :param Awavemax=1e-11: Maximum amplitude of delta_rho/rho
    :param fmin=1: Minimum frequency of seismic wave in Hz
    :param fmax=10: Maximum frequency of seismic wave in Hz
    :param fmono=1: The frequency of all monochromatic plane waves in Hz
    :param sigmafmin=0.5: Minimum width of frequency of Gaussian wave-packet in Hz
    :param sigmafmax=1: Maximum width of frequency of Gaussian wave-packet in Hz
    :param anisotropy="none": "none" for isotropy,
                              "quad" for more waves from above,
                              "left" for only waves from -x,
                              "p000"-"p100" for 0%-100% point source contribution
    # ~~~~~~~~~~~~~~Dataset Parameters~~~~~~~~~~~~~~#
    :param state=[
        [400, 350, 0],
        [-250, 250, 0],
        [200, -250, 0],
        [-100, -100, 0],
    ]: Seismometer positions
    :param NoS=4: Number of Seismometers
    :param freq=10: Frequency of the Wiener filter in Hz
    :param SNR=1e10: SNR as defined in earlier optimization attempts
    :param p=1: Ratio of P- and S-waves
    :param c_ratio=2/3: Ratio c_s/c_p
    # ~~~~~~~~~~~~~~Window Parameters~~~~~~~~~~~~~~#
    :param NoW=10: Number of total time windows
    :param NoT=1: Number of runs without update of WF (test)
    :param NoE=1: Number of wave events per time window
    :param time_window_multiplier=1: Time length of a time window to be evaluated in units of t_max
    :param twindow=None: Overwrites window length in seconds
    :param randomlyPlaced=False: Toggles if events are randomly shifted in time inside the window

    """

    ###########################################################
    # ~~~~~~~~~~~~~~Predefined Parameter Settings~~~~~~~~~~~~~~#
    ###########################################################

    single_mirror_position = [[0, 0, 0]]
    single_mirror_direction = [[1, 0, 0]]

    four_mirror_corner_position = [
        [64.12, 0, 0],
        [536.35, 0, 0],
        [64.12 * 0.5, 64.12 * math.sqrt(3) / 2, 0],
        [536.35 * 0.5, 536.35 * math.sqrt(3) / 2, 0],
    ]
    four_mirror_corner_direction = [
        [1, 0, 0],
        [1, 0, 0],
        [0.5, math.sqrt(3) / 2, 0],
        [0.5, math.sqrt(3) / 2, 0],
    ]

    ##############################################
    # ~~~~~~~~~~~~~~Setting defaults~~~~~~~~~~~~~~#
    ##############################################

    def __init__(
        self,
        # ~~~~~~~~~~~~~~Save management~~~~~~~~~~~~~~#
        ID=5,  # int(argv[1])
        tag="X",  # Dataset identifier
        folder="testset/dataset",  # Dataset path
        saveas="testset/resultFile",  # Identifier for all savefiles produced
        # ~~~~~~~~~~~~~~General~~~~~~~~~~~~~~#
        useGPU=False,  # Set True if you have and want to use GPU-resources
        randomSeed=1,  # If None, use no seed
        isMonochromatic=False,  # Toggles between monochromatic and Gaussian plane wave packets
        NoR=10,  # Number of runs/realizations/wave events
        # ~~~~~~~~~~~~~~Constants~~~~~~~~~~~~~~#
        M=211,  # Mirror mass of LF-ET in kg
        rho=3000,  # Density of rock in kg/m³
        c_p=6000,  # Sound velocity of rock #6000 m/s
        # ~~~~~~~~~~~~~~Domain~~~~~~~~~~~~~~#
        L=6000,  # Length of simulation box in m
        Nx=200,  # Number of spatial steps for force calculation, choose even number
        xmax=6000,  # Distance of wave starting point from 0
        tmax=None,  # Time of simulation in s
        Nt=200,  # Number of time steps
        # ~~~~~~~~~~~~~~Experiment config~~~~~~~~~~~~~~#
        depth=6000,  # Upper domain cutoff (=L for full space)
        cavity_r=5,  # Radius of spherical cavern in m
        mirror_positions=None,  # Array of mirror position [x,y,z] in m
        mirror_directions=None,  # Array of mirror free moving axis unit vectors
        point_source_positions=None,  # Array of point source position [x,y,z] in m
        # ~~~~~~~~~~~~~~Event parameters~~~~~~~~~~~~~~#
        Awavemin=1e-11,
        Awavemax=1e-11,  # Basically delta_rho/rho
        fmin=1,
        fmax=10,  # frequency of seismic wave in Hz
        fmono=1,  # The frequency of all monochromatic plane waves in Hz
        sigmafmin=0.5,
        sigmafmax=1,  # width of frequency of Gaussian wave-packet in Hz
        anisotropy="none",  # "none" for isotropy,
        # "quad" for more waves from above,
        # "left" for only waves from -x,
        # "p000"-"p100" for 0%-100% point source contribution
        # ~~~~~~~~~~~~~~Dataset Parameters~~~~~~~~~~~~~~#
        state=None,  # Seismometer positions
        NoS=4,  # Number of Seismometers
        freq=10,  # Frequency of the Wiener filter in Hz
        SNR=1e10,  # SNR as defined in earlier optimization attempts
        p=1,  # Ratio of P- and S-waves
        c_ratio=2 / 3,  # Ratio c_s/c_p
        # ~~~~~~~~~~~~~~Window Parameters~~~~~~~~~~~~~~#
        NoW=10,  # Number of total time windows
        NoT=1,  # Number of runs without update of WF (test)
        NoE=1,  # Number of wave events per time window
        time_window_multiplier=1,  # Time length of a time window to be evaluated in units of t_max
        twindow=None,  # Overwrites window length in seconds
        randomlyPlaced=False,  # Toggles if events are randomly shifted in time inside the window
    ):
        # ~~~~~~~~~~~~~~Save management~~~~~~~~~~~~~~#

        self.default_ID = ID
        self.default_tag = tag
        self.default_folder = folder
        self.default_saveas = saveas

        # ~~~~~~~~~~~~~~General~~~~~~~~~~~~~~#

        self.default_useGPU = useGPU
        self.default_randomSeed = randomSeed
        self.default_isMonochromatic = isMonochromatic
        self.default_NoR = NoR

        # ~~~~~~~~~~~~~~Constants~~~~~~~~~~~~~~#

        self.default_M = M
        self.default_rho = rho
        self.default_c_p = c_p

        # ~~~~~~~~~~~~~~Domain~~~~~~~~~~~~~~#

        self.default_L = L
        self.default_Nx = Nx
        self.default_dx = (
            2 * L / Nx
        )  # Spacial stepwidth in m, should be <c_P/10/max(f_0)

        self.default_xmax = xmax

        self.default_tmax = tmax or 2 * xmax / c_p
        self.default_Nt = Nt
        self.default_dt = self.default_tmax / Nt  # Temporal stepwidth in s

        # ~~~~~~~~~~~~~~Experiment config~~~~~~~~~~~~~~#

        self.default_depth = depth
        self.default_cavity_r = cavity_r

        if not mirror_positions is None:
            self.default_mirror_positions = mirror_positions
        else:
            self.default_mirror_positions = self.single_mirror_position
        if not mirror_directions is None:
            self.default_mirror_directions = mirror_directions
        else:
            self.default_mirror_directions = self.single_mirror_direction
        self.default_mirror_count = len(self.default_mirror_positions)

        if not point_source_positions is None:
            self.default_point_source_positions = point_source_positions
        else:
            self.default_point_source_positions = [[-1000, -1000, 0]]
        self.default_source_count = len(self.default_point_source_positions) + 1

        # ~~~~~~~~~~~~~~Event parameters~~~~~~~~~~~~~~#

        self.default_Awavemin = Awavemin
        self.default_Awavemax = Awavemax

        self.default_fmin = fmin
        self.default_fmax = fmax
        self.default_fmono = fmono

        self.default_sigmafmin = sigmafmin
        self.default_sigmafmax = sigmafmax

        self.default_anisotropy = anisotropy

        # ~~~~~~~~~~~~~~Dataset Parameters~~~~~~~~~~~~~~#

        self.default_state = (
            state
            if not state is None
            else [
                [400, 350, 0],
                [-250, 250, 0],
                [200, -250, 0],
                [-100, -100, 0],
            ]
        )

        self.default_NoS = NoS or len(state)
        self.default_freq = freq
        self.default_SNR = SNR
        self.default_p = p
        self.default_c_ratio = c_ratio

        # ~~~~~~~~~~~~~~Window Parameters~~~~~~~~~~~~~~#

        self.default_NoW = NoW
        self.default_NoT = NoT
        self.default_NoE = NoE

        self.default_time_window_multiplier = time_window_multiplier
        self.default_twindow = twindow

        self.default_randomlyPlaced = randomlyPlaced

    ##############################################
    # ~~~~~~~~~~~~~~Event Generation~~~~~~~~~~~~~~#
    ##############################################

    def generateEventSet(
        self,
        tag,
        NoR=None,
        # ~~~~~~~~~~~~~~Save management~~~~~~~~~~~~~~#
        ID=None,
        folder=None,
        # ~~~~~~~~~~~~~~General~~~~~~~~~~~~~~#
        useGPU=None,
        randomSeed=None,
        isMonochromatic=None,
        # ~~~~~~~~~~~~~~Constants~~~~~~~~~~~~~~#
        M=None,
        rho=None,
        c_p=None,
        # ~~~~~~~~~~~~~~Domain~~~~~~~~~~~~~~#
        L=None,
        Nx=None,
        xmax=None,
        tmax=None,
        Nt=None,
        # ~~~~~~~~~~~~~~Experiment config~~~~~~~~~~~~~~#
        depth=None,
        cavity_r=None,
        mirror_positions=None,
        mirror_directions=None,
        point_source_positions=None,
        # ~~~~~~~~~~~~~~Event parameters~~~~~~~~~~~~~~#
        Awavemin=None,
        Awavemax=None,
        fmin=None,
        fmax=None,
        fmono=None,
        sigmafmin=None,
        sigmafmax=None,
        anisotropy=None,
    ):
        """generates an eventSet.
        events are a set of wave parameters and time series of bulk Newtonian noise force
        given an experimental setup. They are saved to quickly access and generate different
        datasets from an eventSet.

            # ~~~~~~~~~~~~~~Save management~~~~~~~~~~~~~~#
        :param tag: eventSet identifier
        :param ID: (optional) running number
        :param tag: (optional) eventSet identifier
        :param folder: (optional) eventSet path
        # ~~~~~~~~~~~~~~General~~~~~~~~~~~~~~#
        :param useGPU: (optional) Set True if you have and want to use GPU-resources
        :param randomSeed: (optional) If None, use no seed
        :param isMonochromatic: (optional) Toggles between monochromatic and Gaussian wave packets
        :param NoR: (optional) Number of runs/realizations/wave events
        # ~~~~~~~~~~~~~~Constants~~~~~~~~~~~~~~#
        :param M: (optional) Mirror mass of LF-ET in kg
        :param rho: (optional) Density of rock in kg/m³
        :param c_p: (optional) Sound velocity of rock #6000 m/s
        # ~~~~~~~~~~~~~~Domain~~~~~~~~~~~~~~#
        :param L: (optional) Length of simulation box in m
        :param Nx: (optional) Number of spatial steps for force calculation, choose even number
        :param xmax: (optional) Distance of wave starting point from 0
        :param tmax: (optional) Time of simulation in s
        :param Nt: (optional) Number of time steps
        # ~~~~~~~~~~~~~~Experiment config~~~~~~~~~~~~~~#
        :param depth: (optional) Upper domain cutoff (=L for full space)
        :param cavity_r: (optional) Radius of spherical cavern in m
        :param mirror_positions: (optional) Array of mirror position [x,y,z] in m
        :param mirror_directions: (optional) Array of mirror free moving axis unit vectors
        :param point_source_positions: (optional) Array of point source position [x,y,z] in m
        # ~~~~~~~~~~~~~~Event parameters~~~~~~~~~~~~~~#
        :param Awavemin: (optional) Minimum amplitude of delta_rho/rho
        :param Awavemax: (optional) Maximum amplitude of delta_rho/rho
        :param fmin: (optional) Minimum frequency of seismic wave in Hz
        :param fmax: (optional) Maximum frequency of seismic wave in Hz
        :param fmono: (optional) The frequency of all monochromatic plane waves in Hz
        :param sigmafmin: (optional) Minimum width of frequency of Gaussian wave-packet in Hz
        :param sigmafmax: (optional) Maximum width of frequency of Gaussian wave-packet in Hz
        :param anisotropy: (optional) "none" for isotropy,
                                "quad" for more waves from above,
                                "left" for only waves from -x,
                                "p000"-"p100" for 0%-100% point source contribution
        """
        NoR = NoR if not NoR is None else self.default_NoR

        # ~~~~~~~~~~~~~~Save management~~~~~~~~~~~~~~#

        ID = ID if not ID is None else self.default_ID
        folder = folder if not folder is None else self.default_folder

        # ~~~~~~~~~~~~~~General~~~~~~~~~~~~~~#

        useGPU = useGPU if not useGPU is None else self.default_useGPU
        randomSeed = randomSeed if not randomSeed is None else self.default_randomSeed
        isMonochromatic = (
            isMonochromatic
            if not isMonochromatic is None
            else self.default_isMonochromatic
        )

        # ~~~~~~~~~~~~~~Constants~~~~~~~~~~~~~~#

        M = M if not M is None else self.default_M
        rho = rho if not rho is None else self.default_rho
        c_p = c_p if not c_p is None else self.default_c_p

        # ~~~~~~~~~~~~~~Domain~~~~~~~~~~~~~~#

        L = L if not L is None else self.default_L
        Nx = Nx if not Nx is None else self.default_Nx
        dx = 2 * L / Nx

        xmax = xmax if not xmax is None else self.default_xmax

        tmax = tmax if not tmax is None else 2 * xmax / c_p
        Nt = Nt if not Nt is None else self.default_Nt

        # ~~~~~~~~~~~~~~Experiment config~~~~~~~~~~~~~~#

        depth = depth if not depth is None else self.default_depth
        cavity_r = cavity_r if not cavity_r is None else self.default_cavity_r

        mirror_positions = (
            mirror_positions
            if not mirror_positions is None
            else self.default_mirror_positions
        )
        mirror_directions = (
            mirror_directions
            if not mirror_directions is None
            else self.default_mirror_directions
        )
        mirror_count = len(mirror_positions)

        point_source_positions = (
            point_source_positions
            if not point_source_positions is None
            else self.default_point_source_positions
        )
        source_count = len(point_source_positions) + 1

        # ~~~~~~~~~~~~~~Event parameters~~~~~~~~~~~~~~#

        Awavemin = Awavemin if not Awavemin is None else self.default_Awavemin
        Awavemax = Awavemax if not Awavemax is None else self.default_Awavemax

        fmin = fmin if not fmin is None else self.default_fmin
        fmax = fmax if not fmax is None else self.default_fmax
        fmono = fmono if not fmono is None else self.default_fmono

        sigmafmin = sigmafmin if not sigmafmin is None else self.default_sigmafmin
        sigmafmax = sigmafmax if not sigmafmax is None else self.default_sigmafmax

        anisotropy = anisotropy if not anisotropy is None else self.default_anisotropy

        # ~~~~~~~~~~~~~~Some Setting Checks~~~~~~~~~~~~~~#

        if not os.path.exists(folder):
            os.makedirs(folder)
        if os.path.exists(folder + "/settingFile" + tag + ".txt"):
            raise NameError("better not overwrite your data!")

        if randomSeed in [None, "none"]:
            rd.seed(randomSeed)
            np.random.seed(randomSeed)

        if useGPU:
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")

        # ~~~~~~~~~~~~~~Wave event parameter generation~~~~~~~~~~~~~~#

        # wave direction
        polar_angles = torch.tensor(np.random.random(NoR) * 2 * pi, device=device)
        azimuthal_angles = torch.tensor(
            np.arccos(2 * np.random.random(NoR) - 1), device=device
        )
        sources = torch.zeros(NoR, device=device, dtype=torch.int32)

        if anisotropy == "quad":
            azimuthal_angles = torch.tensor(
                np.arccos(2 * np.random.random(NoR) ** 2 - 1), device=device
            )
        elif anisotropy == "left":
            polar_angles = torch.tensor(
                np.random.random(NoR) * pi - pi / 2, device=device
            )
        elif "p" in anisotropy:
            point_probability = float(anisotropy.split("p")[1]) / 100
            sources = torch.tensor(
                np.random.choice(
                    np.arange(0, source_count),
                    NoR,
                    p=[1 - point_probability]
                    + [point_probability / (source_count - 1)] * (source_count - 1),
                ),
                dtype=torch.int32,
            )

        # packet properties
        As = torch.tensor(
            np.random.random(NoR) * (Awavemax - Awavemin) + Awavemin, device=device
        )
        phases = torch.zeros(NoR, device=device)
        x0s = torch.ones(NoR, device=device) * (-xmax)
        t0s = torch.zeros(NoR, device=device)

        if isMonochromatic:
            fs = torch.ones(NoR, device=device) * fmono
            sigmafs = torch.zeros(NoR, device=device)
        else:
            fs = torch.tensor(
                np.random.random(NoR) * (fmax - fmin) + fmin, device=device
            )
            sigmafs = torch.tensor(
                np.random.random(NoR) * (sigmafmax - sigmafmin) + sigmafmin,
                device=device,
            )

        # S-wave only
        s_polarisations = np.random.random(NoR) * 2 * pi

        # precalculation
        exp_const = -2 * pi**2 * sigmafs**2
        sin_const = 2 * pi * fs

        force_const = rho * G * M * dx**3

        sin_polar = torch.sin(polar_angles)
        cos_polar = torch.cos(polar_angles)
        sin_azi = torch.sin(azimuthal_angles)
        cos_azi = torch.cos(azimuthal_angles)

        # ~~~~~~~~~~~~~~Domain preparation~~~~~~~~~~~~~~#

        # time and space
        time = torch.linspace(0, tmax, Nt + 1, device=device)[:-1]
        x = torch.linspace(-L + dx / 2, L - dx / 2, Nx, device=device)
        y = torch.linspace(-L + dx / 2, L - dx / 2, Nx, device=device)
        z = torch.linspace(-L + dx / 2, L - dx / 2, Nx, device=device)
        xyz = torch.meshgrid(x, y, z, indexing="ij")
        x3d = xyz[1]
        y3d = xyz[0]
        z3d = xyz[2]

        # integration constants from mirror geometry
        r3d = torch.sqrt(x3d**2 + y3d**2 + z3d**2) + 1e-20
        cavity_kernel = r3d < L
        cavity_kernel *= z3d < depth

        r3ds = []
        geo_facts = torch.zeros((mirror_count, Nx, Nx, Nx), device=device)
        for mirror in range(mirror_count):
            pos = mirror_positions[mirror]
            di = mirror_directions[mirror]
            r3ds.append(
                torch.sqrt(
                    (x3d - pos[0]) ** 2 + (y3d - pos[1]) ** 2 + (z3d - pos[2]) ** 2
                )
                + 1e-20
            )
            cavity_kernel *= r3ds[mirror] > cavity_r
            geo_facts[mirror] = (
                (x3d - pos[0]) * di[0] + (y3d - pos[1]) * di[1] + (z3d - pos[2]) * di[2]
            ) / r3ds[mirror] ** 3
        for mirror in range(mirror_count):
            if useGPU:
                geo_facts[mirror].to(device=device)
            geo_facts[mirror] *= cavity_kernel

        # ~~~~~~~~~~~~~~Function definitions~~~~~~~~~~~~~~#

        def gaussian_wave_packet(x, t, x0, t0, A, exp_const, sin_const, phase=0):

            diff = (x - x0) / c_p - (t - t0)
            exp_term = torch.exp(exp_const * diff**2)
            sin_term = torch.sin(sin_const * diff + phase)

            wave = A * exp_term * sin_term
            return wave

        def calc_force(drho, mirror):
            F = force_const * torch.sum(geo_facts[mirror] * drho)
            return F

        # ~~~~~~~~~~~~~~Calculation of Newtonian noise~~~~~~~~~~~~~~#

        forces = torch.zeros((NoR, mirror_count, Nt), device=device)

        for R in range(NoR):

            # preparation
            if sources[R] == 0:
                kx3D = (
                    cos_polar[R] * sin_azi[R] * x3d
                    + sin_polar[R] * sin_azi[R] * y3d
                    + cos_azi[R] * z3d
                )
            else:
                kx3D = torch.sqrt(
                    (x3d - point_source_positions[sources[R] - 1][0]) ** 2
                    + (y3d - point_source_positions[sources[R] - 1][1]) ** 2
                    + (z3d - point_source_positions[sources[R] - 1][2]) ** 2
                )

            if useGPU:
                kx3D.to(device=device)

            # force calculation
            for i, t in enumerate(time):
                density_fluctuations = gaussian_wave_packet(
                    kx3D,
                    t,
                    x0s[R],
                    t0s[R],
                    As[R],
                    exp_const[R],
                    sin_const[R],
                    phases[R],
                )
                if useGPU:
                    density_fluctuations.to(device=device)
                for mirror in range(mirror_count):
                    forces[R][mirror][i] = calc_force(density_fluctuations, mirror)

        # ~~~~~~~~~~~~~~Write settings file~~~~~~~~~~~~~~#

        if not os.path.exists(folder + "/settingFile" + tag + ".txt"):
            with open(
                folder + "/settingFile" + tag + ".txt", "a+", encoding="utf8"
            ) as f:
                f.write("NoR = " + str(NoR) + "\n")
                f.write("isMonochromatic = " + str(isMonochromatic) + "\n")
                f.write("randomSeed = " + str(randomSeed) + "\n")
                f.write("M = " + str(M) + "\n")
                f.write("rho = " + str(rho) + "\n")
                f.write("c_p = " + str(c_p) + "\n")
                f.write("L = " + str(L) + "\n")
                f.write("x_max = " + str(xmax) + "\n")
                f.write("t_max = " + str(tmax) + "\n")
                f.write("Nx = " + str(Nx) + "\n")
                f.write("Nt = " + str(Nt) + "\n")
                f.write("depth = " + str(depth) + "\n")
                f.write("cavity_r = " + str(cavity_r) + "\n")
                f.write(
                    "mirror_positions = np.array("
                    + str(np.array(mirror_positions).tolist())
                    + ")\n"
                )
                f.write(
                    "mirror_directions = np.array("
                    + str(np.array(mirror_directions).tolist())
                    + ")\n"
                )
                f.write(
                    "point_source_positions = np.array("
                    + str(np.array(point_source_positions).tolist())
                    + ")\n"
                )
                f.write("Awave_max = " + str(Awavemax) + "\n")
                f.write("Awave_min = " + str(Awavemin) + "\n")
                f.write("f_min = " + str(fmin) + "\n")
                f.write("f_max = " + str(fmax) + "\n")
                f.write("f_mono = " + str(fmono) + "\n")
                f.write("sigma_f_min = " + str(sigmafmin) + "\n")
                f.write("sigma_f_max = " + str(sigmafmax) + "\n")
                f.write("anisotropy = " + str(anisotropy) + "\n")
                f.write("useGPU = " + str(useGPU) + "\n")

                f.write(
                    "#runtime = "
                    + str(np.round((systime.time() - total_start_time) / 60, 2))
                    + " min\n"
                )

            # ~~~~~~~~~~~~~~Write dataset~~~~~~~~~~~~~~#

            CPUdevice = torch.device("cpu")

            # mirror forces
            forces = np.array(forces.to(device=CPUdevice))
            for mirror in range(mirror_count):
                np.save(
                    folder
                    + "/wave_event_result_force_"
                    + str(mirror)
                    + "_"
                    + tag
                    + ".npy",
                    forces[:, mirror],
                )

            # wave events
            np.save(
                folder + "/wave_event_data_polar_angle_" + tag + ".npy",
                polar_angles.to(device=CPUdevice),
            )
            np.save(
                folder + "/wave_event_data_azimuthal_angle_" + tag + ".npy",
                azimuthal_angles.to(device=CPUdevice),
            )
            np.save(
                folder + "/wave_event_data_A_" + tag + ".npy", As.to(device=CPUdevice)
            )
            np.save(
                folder + "/wave_event_data_phase_" + tag + ".npy",
                phases.to(device=CPUdevice),
            )
            np.save(
                folder + "/wave_event_data_x0_" + tag + ".npy", x0s.to(device=CPUdevice)
            )
            np.save(
                folder + "/wave_event_data_t0_" + tag + ".npy", t0s.to(device=CPUdevice)
            )
            np.save(
                folder + "/wave_event_data_f0_" + tag + ".npy", fs.to(device=CPUdevice)
            )
            np.save(
                folder + "/wave_event_data_sigmaf_" + tag + ".npy",
                sigmafs.to(device=CPUdevice),
            )
            np.save(
                folder + "/wave_event_data_s_polarization_" + tag + ".npy",
                s_polarisations,
            )
            np.save(folder + "/wave_event_data_source_" + tag + ".npy", sources)

            total_time = (systime.time() - total_start_time) / 60
            print(
                "eventSet with tag "
                + tag
                + " generated.\n#total time: "
                + str(total_time)
                + " min"
            )

    #########################################
    # ~~~~~~~~~~~~~~Convenience~~~~~~~~~~~~~~#
    #########################################

    def loadFromSettingFile(self, tag, folder=".", ID=0):
        """Load default parameters from an existing eventSet settingFile.
        The settingFile should be named folder + "/" + "settingFile" + tag + ".txt"

        :param tag: the tag used for generating the eventSet
        :param folder=".": the path where the file is
        :param ID=0: identifier number
        """

        data_dict = ReadData(tag, folder, "settingFile").dictionary

        # ~~~~~~~~~~~~~~Save management~~~~~~~~~~~~~~#

        self.default_ID = ID
        self.default_tag = tag
        self.default_folder = folder

        # ~~~~~~~~~~~~~~General~~~~~~~~~~~~~~#

        self.default_useGPU = data_dict["useGPU"]
        self.default_randomSeed = data_dict["randomSeed"]

        self.default_isMonochromatic = data_dict["isMonochromatic"]
        self.default_NoR = data_dict["NoR"]

        # ~~~~~~~~~~~~~~Constants~~~~~~~~~~~~~~#

        self.default_M = data_dict["M"]
        self.default_rho = data_dict["rho"]
        self.default_c_p = data_dict["c_p"]

        # ~~~~~~~~~~~~~~Domain~~~~~~~~~~~~~~#

        self.default_L = data_dict["L"]
        self.default_Nx = data_dict["Nx"]
        self.default_dx = 2 * data_dict["L"] / data_dict["Nx"]

        self.default_xmax = data_dict["x_max"]

        self.default_tmax = data_dict["t_max"]
        self.default_Nt = data_dict["Nt"]
        self.default_dt = self.default_tmax / data_dict["Nt"]

        # ~~~~~~~~~~~~~~Experiment config~~~~~~~~~~~~~~#

        self.default_depth = data_dict["depth"]
        self.default_cavity_r = data_dict["cavity_r"]

        self.default_mirror_positions = data_dict["mirror_positions"]
        self.default_mirror_directions = data_dict["mirror_directions"]
        self.default_mirror_count = len(self.default_mirror_positions)

        self.default_point_source_positions = data_dict["point_source_positions"]
        self.default_source_count = len(self.default_point_source_positions) + 1

        # ~~~~~~~~~~~~~~Event parameters~~~~~~~~~~~~~~#

        self.default_Awavemin = data_dict["Awave_min"]
        self.default_Awavemax = data_dict["Awave_max"]

        self.default_fmin = data_dict["f_min"]
        self.default_fmax = data_dict["f_max"]
        self.default_fmono = data_dict["f_mono"]

        self.default_sigmafmin = data_dict["sigma_f_min"]
        self.default_sigmafmax = data_dict["sigma_f_max"]

        self.default_anisotropy = data_dict["anisotropy"]

    def generateDataset(
        self,
        # ~~~~~~~~~~~~~~Load and save management~~~~~~~~~~~~~~#
        tag=None,
        folder=None,
        ID=None,
        saveas=None,
        # ~~~~~~~~~~~~~~General~~~~~~~~~~~~~~#
        useGPU=None,
        NoR=None,
        # ~~~~~~~~~~~~~~Dataset Parameters~~~~~~~~~~~~~~#
        state=None,
        NoS=None,
        freq=None,
        SNR=None,
        p=None,
        c_ratio=None,
        # ~~~~~~~~~~~~~~Window Parameters~~~~~~~~~~~~~~#
        NoW=None,
        NoT=None,
        NoE=None,
        time_window_multiplier=None,
        twindow=None,
        randomlyPlaced=None,
        mirror_ID=None,
    ):
        """Generate a dataSet from an eventFile and a set of parameters.
        Returns the time series of the Newtonian noise force from one of the
        mirrors and the 3*NoS witness channel time series from the seismometer array.


        # ~~~~~~~~~~~~~~Save management~~~~~~~~~~~~~~#
        :param tag="X": dataSet identifier
        :param folder="testset/dataset": dataSet path
        :param ID=5: int(argv[1])
        :param saveas="testset/resultFile": Identifier for all savefiles produced
        # ~~~~~~~~~~~~~~General~~~~~~~~~~~~~~#
        :param useGPU=False: Set True if you have and want to use GPU-resources
        :param NoR=10: Number of wave events from the eventSet that will be included
        # ~~~~~~~~~~~~~~Dataset Parameters~~~~~~~~~~~~~~#
        :param state=[
            [400, 350, 0],
            [-250, 250, 0],
            [200, -250, 0],
            [-100, -100, 0],
        ]: Seismometer positions
        :param NoS=4: Number of Seismometers
        :param freq=10: Frequency of the Wiener filter in Hz
        :param SNR=1e10: SNR as defined in earlier optimization attempts
        :param p=1: Ratio of P- and S-waves
        :param c_ratio=2/3: Ratio c_s/c_p
        # ~~~~~~~~~~~~~~Window Parameters~~~~~~~~~~~~~~#
        :param NoW=10: Number of total time windows
        :param NoT=1: Number of runs without update of WF (test)
        :param NoE=1: Number of wave events per time window
        :param time_window_multiplier=1: Time length of a window to be evaluated in units of t_max
        :param twindow=None: Overwrites window length in seconds
        :param randomlyPlaced=False: Toggles if events are randomly shifted in time inside a window

        :return: time series of witness channels (3D seismometer displacement)
        :return: time series of target channel (Newtonian-noise force of one test mass)
        """

        # ~~~~~~~~~~~~~~Save management~~~~~~~~~~~~~~#
        ID = ID if not ID is None else self.default_ID
        folder = folder if not folder is None else self.default_folder
        if not tag is None:
            self.loadFromSettingFile(tag, folder, ID)
        else:
            tag = self.default_tag

        saveas = saveas if not saveas is None else self.default_saveas

        # ~~~~~~~~~~~~~~General~~~~~~~~~~~~~~#

        useGPU = useGPU if not useGPU is None else self.default_useGPU
        NoR = NoR if not NoR is None else self.default_NoR
        # randomSeed = randomSeed if not randomSeed==None else self.default_randomSeed

        # ~~~~~~~~~~~~~~Dataset Parameters~~~~~~~~~~~~~~#

        state = state if not state is None else self.default_state
        NoS = NoS if not NoS is None else self.default_NoS

        freq = freq if not freq is None else self.default_freq
        SNR = SNR if not SNR is None else self.default_SNR
        p = p if not p is None else self.default_p
        c_ratio = c_ratio if not c_ratio is None else self.default_c_ratio

        # ~~~~~~~~~~~~~~Window Parameters~~~~~~~~~~~~~~#

        NoW = NoW if not NoW is None else self.default_NoW
        NoT = NoT if not NoT is None else self.default_NoT
        NoE = NoE if not NoE is None else self.default_NoE

        time_window_multiplier = (
            time_window_multiplier
            if not time_window_multiplier is None
            else self.default_time_window_multiplier
        )
        twindow = twindow if not twindow is None else self.default_twindow
        randomlyPlaced = (
            randomlyPlaced
            if not randomlyPlaced is None
            else self.default_randomlyPlaced
        )

        mirror_ID = mirror_ID if not mirror_ID is None else 0

        #####################################
        # ~~~~~~~~~~~~~~Loading~~~~~~~~~~~~~~#
        #####################################

        # ~~~~~~~~~~~~~~Read settings file~~~~~~~~~~~~~~#

        data = ReadData(tag, folder)
        if useGPU:
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")

        # ~~~~~~~~~~~~~~Read parameters and constants~~~~~~~~~~~~~~#

        if data.dictionary["randomSeed"] == "None":
            randomSeed = None
        else:
            randomSeed = int(data.dictionary["randomSeed"])
            np.random.seed(randomSeed + 1)

        isMonochromatic = data.dictionary["isMonochromatic"]

        NoR = min(int(data.dictionary["NoR"]), NoR)  # Number of runs/realizations
        NoT = min(
            int(data.dictionary["NoR"]) - 3, NoT
        )  # Number of runs without update of WF

        # constants
        M = data.dictionary["M"]
        rho = data.dictionary["rho"]

        numerical_cavern_factor = (
            -4 * pi / 3 * G * M * rho
        )  # force from shift of cavern per total density

        c_p = data.dictionary["c_p"]  # sound velocity in rock  #6000 m/s
        c_s = c_p * c_ratio

        tmax = data.dictionary["t_max"]  # time of simulation
        Nt = int(data.dictionary["Nt"])

        mirror_positions = data.dictionary["mirror_positions"]
        mirror_directions = data.dictionary["mirror_directions"]
        mirror_count = len(mirror_positions)

        try:
            point_source_positions = torch.tensor(
                np.array(
                    [[0, 0, 1e2]] + list(data.dictionary["point_source_positions"])
                )
            )
            sources = torch.tensor(
                np.load(
                    folder + "/wave_event_data_source_" + tag + ".npy", mmap_mode="r"
                )[:NoR].copy(),
                device=device,
            )
        except FileNotFoundError:
            point_source_positions = torch.tensor([[2.2, -1e-3, 1e5]])
            sources = torch.zeros(NoR, device=device)

        time = torch.tensor(np.linspace(0, tmax, Nt, endpoint=False), device=device)

        # ~~~~~~~~~~~~~~Load data set~~~~~~~~~~~~~~#

        # mirror forces
        all_bulk_forces = np.zeros((NoR, mirror_count, Nt))
        for mirror in range(mirror_count):
            all_bulk_forces[:, mirror] = np.load(
                folder + "/wave_event_result_force_" + str(mirror) + "_" + tag + ".npy",
                mmap_mode="r",
            )[:NoR].copy()
        all_bulk_forces = torch.tensor(all_bulk_forces, device=device)

        # wave events
        all_polar_angles = torch.tensor(
            np.load(
                folder + "/wave_event_data_polar_angle_" + tag + ".npy", mmap_mode="r"
            )[:NoR].copy(),
            device=device,
        )
        all_azimuthal_angles = torch.tensor(
            np.load(
                folder + "/wave_event_data_azimuthal_angle_" + tag + ".npy",
                mmap_mode="r",
            )[:NoR].copy(),
            device=device,
        )

        all_x0s = torch.tensor(
            np.load(folder + "/wave_event_data_x0_" + tag + ".npy", mmap_mode="r")[
                :NoR
            ].copy(),
            device=device,
        )
        all_t0s = torch.tensor(
            np.load(folder + "/wave_event_data_t0_" + tag + ".npy", mmap_mode="r")[
                :NoR
            ].copy(),
            device=device,
        )

        all_As = torch.tensor(
            np.load(folder + "/wave_event_data_A_" + tag + ".npy", mmap_mode="r")[
                :NoR
            ].copy(),
            device=device,
        )
        all_phases = torch.tensor(
            np.load(folder + "/wave_event_data_phase_" + tag + ".npy", mmap_mode="r")[
                :NoR
            ].copy(),
            device=device,
        )
        all_fs = torch.tensor(
            np.load(folder + "/wave_event_data_f0_" + tag + ".npy", mmap_mode="r")[
                :NoR
            ].copy(),
            device=device,
        )
        all_sigmafs = torch.tensor(
            np.load(folder + "/wave_event_data_sigmaf_" + tag + ".npy", mmap_mode="r")[
                :NoR
            ].copy(),
            device=device,
        )

        # P and S
        all_s_polarization = torch.tensor(
            np.load(
                folder + "/wave_event_data_s_polarization_" + tag + ".npy",
                mmap_mode="r",
            )[:NoR].copy(),
            device=device,
        )

        all_is_s = np.random.random(NoR) > p
        all_is_s = torch.tensor(all_is_s, device=device)
        all_cs = c_p * (all_is_s == False) + c_s * all_is_s

        # other preparations
        all_sin_polar = torch.sin(all_polar_angles)
        all_cos_polar = torch.cos(all_polar_angles)
        all_sin_azimuthal = torch.sin(all_azimuthal_angles)
        all_cos_azimuthal = torch.cos(all_azimuthal_angles)
        all_sin_s_polarization = torch.sin(all_s_polarization)
        all_cos_s_polarization = torch.cos(all_s_polarization)

        all_forces = torch.zeros((NoR, mirror_count, Nt), device=device)
        all_seismometer_data = torch.zeros((NoR, NoS, 3, Nt), device=device)

        # ~~~~~~~~~~~~~~Analytical displacement functions~~~~~~~~~~~~~~#

        def gaussian_wave_packet_displacement(x, t, x0, t0, f0, sigmaf, c, A, phase):

            diff = (x - x0) / c - (t - t0)

            VF = (
                1
                / (math.sqrt(2 * pi) * sigmaf)
                * 1
                / 2
                * A
                * c_p
                * torch.exp(-1j * phase - f0**2 / (2 * sigmaf**2))
            )

            if torch.all(phase == 0):
                wave = VF * torch.imag(
                    sp.erf(
                        (2 * pi * sigmaf**2 * diff + 1j * f0) / (math.sqrt(2) * sigmaf)
                    )
                )
            else:
                wave = (
                    -VF
                    / 2
                    * 1j
                    * (
                        sp.erf(
                            (2 * pi * sigmaf**2 * diff + 1j * f0)
                            / (math.sqrt(2) * sigmaf)
                        )
                        - np.exp(2 * 1j * phase)
                        * sp.erf(
                            (2 * pi * sigmaf**2 * diff - 1j * f0)
                            / (math.sqrt(2) * sigmaf)
                        )
                    )
                )
            return torch.real(wave)

        def monochromatic_wave_displacement(x, t, x0, t0, f0, c, A, phase):

            diff = (x - x0) / c - (t - t0)

            wave = A * c_p / 2 / pi / f0 * torch.cos(2 * pi * f0 * diff + phase)
            return wave

        def precalculateForce():

            # get local displacement (at each mirror)
            pos = torch.tensor(mirror_positions, device=device).reshape(
                1, mirror_count, 3
            )
            di = torch.tensor(mirror_directions, device=device).reshape(
                1, mirror_count, 3
            )
            projectedMirrorPosition = pos[:, :, 0] * (
                all_cos_polar * all_sin_azimuthal
            ).reshape(NoR, 1)
            projectedMirrorPosition += pos[:, :, 1] * (
                all_sin_polar * all_sin_azimuthal
            ).reshape(NoR, 1)
            projectedMirrorPosition += pos[:, :, 2] * (all_cos_azimuthal).reshape(
                NoR, 1
            )

            point_source_distance = torch.sqrt(
                (point_source_positions[sources, 0].reshape(NoR, 1) - pos[:, :, 0]) ** 2
                + (point_source_positions[sources, 1].reshape(NoR, 1) - pos[:, :, 1])
                ** 2
                + (point_source_positions[sources, 2].reshape(NoR, 1) - pos[:, :, 2])
                ** 2
            )
            projectedMirrorPosition = projectedMirrorPosition * (sources == 0).reshape(
                NoR, 1
            ) + point_source_distance * (sources != 0).reshape(NoR, 1)

            if isMonochromatic:
                absoluteDisplacement = monochromatic_wave_displacement(
                    projectedMirrorPosition.reshape(NoR, mirror_count, 1),
                    time.reshape(1, 1, Nt),
                    all_x0s.reshape(NoR, 1, 1),
                    all_t0s.reshape(NoR, 1, 1),
                    all_fs.reshape(NoR, 1, 1),
                    all_cs.reshape(NoR, 1, 1),
                    all_As.reshape(NoR, 1, 1),
                    all_phases.reshape(NoR, 1, 1),
                )
            else:
                absoluteDisplacement = gaussian_wave_packet_displacement(
                    projectedMirrorPosition.reshape(NoR, mirror_count, 1),
                    time.reshape(1, 1, Nt),
                    all_x0s.reshape(NoR, 1, 1),
                    all_t0s.reshape(NoR, 1, 1),
                    all_fs.reshape(NoR, 1, 1),
                    all_sigmafs.reshape(NoR, 1, 1),
                    all_cs.reshape(NoR, 1, 1),
                    all_As.reshape(NoR, 1, 1),
                    all_phases.reshape(NoR, 1, 1),
                )

            point_source_vector = pos - point_source_positions[sources].reshape(
                NoR, 1, 3
            )
            point_source_polar_angle = torch.arctan2(
                point_source_vector[:, :, 1], point_source_vector[:, :, 0]
            )
            point_source_azimuthal_angle = torch.arccos(
                point_source_vector[:, :, 2]
                / torch.sqrt(
                    point_source_vector[:, :, 0] ** 2
                    + point_source_vector[:, :, 1] ** 2
                    + point_source_vector[:, :, 2] ** 2
                )
            )
            cos_polar = all_cos_polar.reshape(NoR, 1) * (sources == 0).reshape(
                NoR, 1
            ) + torch.cos(point_source_polar_angle) * (sources != 0).reshape(NoR, 1)
            sin_polar = all_sin_polar.reshape(NoR, 1) * (sources == 0).reshape(
                NoR, 1
            ) + torch.sin(point_source_polar_angle) * (sources != 0).reshape(NoR, 1)
            cos_azi = all_cos_azimuthal.reshape(NoR, 1) * (sources == 0).reshape(
                NoR, 1
            ) + torch.cos(point_source_azimuthal_angle) * (sources != 0).reshape(NoR, 1)
            sin_azi = all_sin_azimuthal.reshape(NoR, 1) * (sources == 0).reshape(
                NoR, 1
            ) + torch.sin(point_source_azimuthal_angle) * (sources != 0).reshape(NoR, 1)

            # cavern acceleration parallel to local displacement
            all_p_cavern_forces = di[:, :, 0] * cos_polar * sin_azi
            all_p_cavern_forces += di[:, :, 1] * sin_polar * sin_azi
            all_p_cavern_forces += di[:, :, 2] * cos_azi
            all_p_cavern_forces = (
                all_p_cavern_forces.reshape(NoR, mirror_count, 1)
                * absoluteDisplacement
                * numerical_cavern_factor
            )

            # cavern acceleration perpendicular to local displacement
            all_s_cavern_forces = di[:, :, 0] * (
                -sin_polar * all_sin_s_polarization.reshape(NoR, 1)
                + cos_polar * cos_azi * all_cos_s_polarization.reshape(NoR, 1)
            )
            all_s_cavern_forces += di[:, :, 1] * (
                cos_polar * all_sin_s_polarization.reshape(NoR, 1)
                + sin_polar * cos_azi * all_cos_s_polarization.reshape(NoR, 1)
            )
            all_s_cavern_forces += di[:, :, 2] * (
                -sin_azi * all_cos_s_polarization.reshape(NoR, 1)
            )
            all_s_cavern_forces = (
                all_s_cavern_forces.reshape(NoR, mirror_count, 1)
                * absoluteDisplacement
                * numerical_cavern_factor
            )

            # add P- and S-contributions
            all_forces = (all_bulk_forces + all_p_cavern_forces) * (
                all_is_s == False
            ).reshape(NoR, 1, 1) + all_s_cavern_forces * (all_is_s).reshape(NoR, 1, 1)

            return all_forces

        # extract displacement at seismometer positions
        def getDisplacement(seismometer_positions):

            # get total displacement at each seismometer
            pos = torch.tensor(np.array(seismometer_positions), device=device).reshape(
                1, NoS, 3
            )
            projectedSeismometerPosition = pos[:, :, 0] * (
                all_cos_polar * all_sin_azimuthal
            ).reshape(NoR, 1)
            projectedSeismometerPosition += pos[:, :, 1] * (
                all_sin_polar * all_sin_azimuthal
            ).reshape(NoR, 1)
            projectedSeismometerPosition += pos[:, :, 2] * (all_cos_azimuthal).reshape(
                NoR, 1
            )

            point_source_distance = torch.sqrt(
                (point_source_positions[sources, 0].reshape(NoR, 1) - pos[:, :, 0]) ** 2
                + (point_source_positions[sources, 1].reshape(NoR, 1) - pos[:, :, 1])
                ** 2
                + (point_source_positions[sources, 2].reshape(NoR, 1) - pos[:, :, 2])
                ** 2
            )
            projectedSeismometerPosition = projectedSeismometerPosition * (
                sources == 0
            ).reshape(NoR, 1) + point_source_distance * (sources != 0).reshape(NoR, 1)

            if isMonochromatic:
                absoluteDisplacement = monochromatic_wave_displacement(
                    projectedSeismometerPosition.reshape(NoR, NoS, 1),
                    time.reshape(1, 1, Nt),
                    all_x0s.reshape(NoR, 1, 1),
                    all_t0s.reshape(NoR, 1, 1),
                    all_fs.reshape(NoR, 1, 1),
                    all_cs.reshape(NoR, 1, 1),
                    all_As.reshape(NoR, 1, 1),
                    all_phases.reshape(NoR, 1, 1),
                )
            else:
                absoluteDisplacement = gaussian_wave_packet_displacement(
                    projectedSeismometerPosition.reshape(NoR, NoS, 1),
                    time.reshape(1, 1, Nt),
                    all_x0s.reshape(NoR, 1, 1),
                    all_t0s.reshape(NoR, 1, 1),
                    all_fs.reshape(NoR, 1, 1),
                    all_sigmafs.reshape(NoR, 1, 1),
                    all_cs.reshape(NoR, 1, 1),
                    all_As.reshape(NoR, 1, 1),
                    all_phases.reshape(NoR, 1, 1),
                )

            point_source_vector = pos - point_source_positions[sources].reshape(
                NoR, 1, 3
            )
            point_source_polar_angle = torch.arctan2(
                point_source_vector[:, :, 1], point_source_vector[:, :, 0]
            )
            point_source_azimuthal_angle = torch.arccos(
                point_source_vector[:, :, 2]
                / torch.sqrt(
                    point_source_vector[:, :, 0] ** 2
                    + point_source_vector[:, :, 1] ** 2
                    + point_source_vector[:, :, 2] ** 2
                )
            )
            cos_polar = all_cos_polar.reshape(NoR, 1) * (sources == 0).reshape(
                NoR, 1
            ) + torch.cos(point_source_polar_angle) * (sources != 0).reshape(NoR, 1)
            sin_polar = all_sin_polar.reshape(NoR, 1) * (sources == 0).reshape(
                NoR, 1
            ) + torch.sin(point_source_polar_angle) * (sources != 0).reshape(NoR, 1)
            cos_azi = all_cos_azimuthal.reshape(NoR, 1) * (sources == 0).reshape(
                NoR, 1
            ) + torch.cos(point_source_azimuthal_angle) * (sources != 0).reshape(NoR, 1)
            sin_azi = all_sin_azimuthal.reshape(NoR, 1) * (sources == 0).reshape(
                NoR, 1
            ) + torch.sin(point_source_azimuthal_angle) * (sources != 0).reshape(NoR, 1)

            # project onto 3 axes
            all_p_displacements = torch.zeros((NoR, NoS, 3, Nt), device=device)
            all_p_displacements[:, :, 0, :] += (cos_polar * sin_azi).reshape(
                NoR, NoS, 1
            )
            all_p_displacements[:, :, 1, :] += (sin_polar * sin_azi).reshape(
                NoR, NoS, 1
            )
            all_p_displacements[:, :, 2, :] += (cos_azi).reshape(NoR, NoS, 1)
            all_p_displacements = all_p_displacements * absoluteDisplacement.reshape(
                NoR, NoS, 1, Nt
            )

            all_s_displacements = torch.zeros((NoR, NoS, 3, Nt), device=device)
            all_s_displacements[:, :, 0, :] += (
                -sin_polar * all_sin_s_polarization.reshape(NoR, 1)
                + cos_polar * cos_azi * all_cos_s_polarization.reshape(NoR, 1)
            ).reshape(NoR, NoS, 1)
            all_s_displacements[:, :, 1, :] += (
                cos_polar * all_sin_s_polarization.reshape(NoR, 1)
                + sin_polar * cos_azi * all_cos_s_polarization.reshape(NoR, 1)
            ).reshape(NoR, NoS, 1)
            all_s_displacements[:, :, 2, :] += (
                -sin_azi * all_cos_s_polarization.reshape(NoR, 1)
            ).reshape(NoR, NoS, 1)
            all_s_displacements = all_s_displacements * absoluteDisplacement.reshape(
                NoR, NoS, 1, Nt
            )

            all_displacements = all_p_displacements * (all_is_s == False).reshape(
                NoR, 1, 1, 1
            ) + all_s_displacements * (all_is_s).reshape(NoR, 1, 1, 1)

            return all_displacements

        all_forces = precalculateForce()
        all_seismometer_data = getDisplacement(state)

        return all_seismometer_data.reshape(NoR, NoS * 3, Nt), all_forces[:, mirror_ID]

    def deleteEventSet(self, tag, folder=None):
        """Deletes an existing eventSet. Deletes all files containing the tag in the folder.

        :param tag: The tag used for generating the eventSet
        :param folder=None: The path where the files are located
        """

        if folder is None:
            folder = self.default_folder

        found_file = False

        files = os.listdir(folder)
        for file in files:
            if "_" + tag + ".npy" in file:
                os.remove(folder + "/" + file)
            if "settingFile" + tag + ".txt" in file:
                os.remove(folder + "/" + file)
                found_file = True

        if found_file:
            print("eventSet with tag " + tag + " deleted from folder " + folder + ".")
        else:
            print("No eventSet with tag " + tag + " found in folder " + folder + ".")
