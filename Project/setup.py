# --------------------------------------------------------------------------------------
# Setup file for processing AD, MCI and HC subjects (MMSE classification)
#
# By Gustavo Patow
#
# --------------------------------------------------------------------------------------
import numpy as np
import scipy.io as sio
import os
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 16})

from ADHopf_Ritter import dataLoader
import pandas as pd

# --------------------------------------------------------------------------
#  Begin setup...
# --------------------------------------------------------------------------
# import WholeBrain.Models.DynamicMeanField as DMF
# import DMF_AD
# neuronalModel = DMF_AD

import WholeBrain.Models.supHopf as Hopf
Hopf.initialValueX = Hopf.initialValueY = 0.1
# neuronalModel = Hopf

import WholeBrain.Integrators.EulerMaruyama as scheme
scheme.neuronalModel = Hopf
# scheme.clamping = False
import WholeBrain.Integrators.Integrator as integrator
integrator.integrationScheme = scheme
integrator.neuronalModel = Hopf
integrator.verbose = False

# import Integrators.EulerMaruyama
# integrator = Integrators.EulerMaruyama
# integrator.neuronalModel = neuronalModel
# integrator.verbose = False
# Integration parms...
dt = 5e-5
Tmax = 197 #20
# ds = 1e-4
# Tmaxneuronal = int((tmax+dt))

# import Utils.BOLD.BOLDHemModel_Stephan2007 as Stephan2007
# import Utils.simulate_SimAndBOLD as simulateBOLD
# simulateBOLD.integrator = integrator
# simulateBOLD.BOLDModel = Stephan2007
# simulateBOLD.TR = 3.
# from Utils.FIC import BalanceFIC as BalanceFIC
#
# BalanceFIC.integrator = integrator

import WholeBrain.Utils.simulate_SimOnly as simulateBOLD
simulateBOLD.warmUp = True
simulateBOLD.warmUpFactor = 606./2000.
simulateBOLD.integrator = integrator

import WholeBrain.Observables.phFCD as phFCD
import WholeBrain.Optimizers.ParmSweep as ParmSweep
ParmSweep.simulateBOLD = simulateBOLD
ParmSweep.integrator = integrator
ParmSweep.verbose = True

import WholeBrain.Observables.filteredPowerSpectralDensity as filtPowSpectr
import WholeBrain.Observables.BOLDFilters as BOLDFilters
# NARROW LOW BANDPASS
BOLDFilters.flp = .01      # lowpass frequency of filter
BOLDFilters.fhi = 0.09      # highpass
BOLDFilters.TR = 3.

# import WholeBrain.Optimizers.ParmSweep as ParmSeep

# ParmSeep.simulateBOLD = simulateBOLD
# ParmSeep.integrator = integrator
# --------------------------------------------------------------------------
#  End setup...
# --------------------------------------------------------------------------


# ===================== Normalize a SC matrix
normalizationFactor = 0.2
avgHuman66 = 0.0035127188987848714
areasHuman66 = 66
maxNodeInput66 = 0.7275543904602363

# Add the following line to store the number of nodes
numNodes = 0


def correctSC(SC):
    global numNodes  # Declare numNodes as a global variable
    N = SC.shape[0]
    logMatrix = np.log(SC + 1)
    maxNodeInput = np.max(np.sum(logMatrix, axis=0))
    finalMatrix = logMatrix * maxNodeInput66 / maxNodeInput

    # Update the value of numNodes
    numNodes = N

    return finalMatrix


# ==========================================================================
# Important config options: filenames
# ==========================================================================
#base_folder = "../../Data_Raw/from_Ritter"
#save_folder = "../../Data_Produced/ABeta_and_Tau"

base_folder = "../Dades_Gus/"
save_folder = "../Data_Produced/ABeta_and_Tau/"

x_path = base_folder
save_path = "/Users/joancarrerasdz/Desktop/CARPETES/UdG/TFG/Data_Produced/ADHopf_Ritter/"

# --------------------------------------------------
# Classify subject information into {HC, MCI, AD}
# --------------------------------------------------
subjectsDir = [os.path.basename(f.path) for f in os.scandir(base_folder+"connectomes/") if f.is_dir()]
classification = dataLoader.checkClassifications(subjectsDir)
HCSubjects = [s for s in classification if classification[s] == 'HC']
ADSubjects = [s for s in classification if classification[s] == 'AD']
MCISubjects = [s for s in classification if classification[s] == 'MCI']
subjects = HCSubjects + MCISubjects + ADSubjects
print(f"We have {len(HCSubjects)} HC, {len(MCISubjects)} MCI and {len(ADSubjects)} AD \n")
# print("HCSubjects:", HCSubjects)
# print("ADSubjects: ", ADSubjects)
# print("MCISubjects: ", MCISubjects)

dataSetLabels = ['HC', 'MCI', 'AD']

# Create a DataFrame with subjects and their labels
df = pd.DataFrame({'Subjects': subjects, 'Labels': [classification[s] for s in subjects]})

# Save the DataFrame to a CSV file
#df.to_csv(os.path.join(save_folder, 'subject_classification.csv'), index=False)


mode = 'heterogeneous'  # homogeneous/heterogeneous/shuffled
x = 'abeta' #abeta
conditionToStudy = 'MCI' # HC / AD

# --------------------------------------------------
# ------------ load or compute the AvgHC SC matrix
# --------------------------------------------------
avgSCPath = '../Data_Produced/avgSC.mat'
if os.path.exists(avgSCPath):
    avgSC = sio.loadmat(avgSCPath)["avgSC"]
else:
    avgSC = dataLoader.computeAvgSC_HC_Matrix(classification, base_folder+"connectomes/")
    avgSC = dataLoader.correctSC(avgSC)
    sio.savemat(avgSCPath, {"avgSC": avgSC})
N = avgSC.shape[0]
Hopf.setParms({"SC": avgSC})
Hopf.couplingOp.setParms(avgSC)


# --------------------------------------------------
# Load tauBurden using loadSubjectData
# --------------------------------------------------

# tauBurden = []

#for subject in subjects:
#    SCnorm, abeta_burden, tau_burden, fullSeries = dataLoader.loadSubjectData(subject,Tmax)
    #tauBurden.append(tau_burden)

    # Convert tauBurden to a NumPy array
    # tauBurden = np.array(tauBurden)

# if mode == 'homogeneous':
#     avgTau = np.average(tau_burden)
#     tauBurden = np.ones(numNodes) * avgTau
#     print('Using homogeneous tau for optimization')
# elif mode == 'shuffled':
#     np.random.shuffle(tau_burden)
#     print('Using shuffled tau for optimization')


def loadXBurden(x, condition):
    """
    Load and normalize burden data (x) based on the specified condition.

    Parameters:
    - x (str): Type of burden data ('tau' or 'abeta').
    - condition (str): Condition for normalization ('hc', 'mci', or 'ad').

    Returns:
    - numpy array: Normalized burden data.
    """

    # Load subject classifications
    subjectsDir = [os.path.basename(f.path) for f in os.scandir(base_folder+"connectomes/") if f.is_dir()]
    classification = dataLoader.checkClassifications(subjectsDir)

    # Select subjects based on the specified condition
    subjects = [subject for subject in classification.keys() if classification[subject] == condition]

    # Check if the list of subjects is not empty
    if not subjects:
        print(f"No subjects found for condition '{condition}'")
        return None

    # Load and normalize burden data for each subject
    x_burden = None
    for subject in subjects:
        if x == 'tau':
            subject_x = dataLoader.loadBurden(subject, 'Tau', base_folder, normalize=True)
        elif x == 'abeta':
            subject_x = dataLoader.loadBurden(subject, 'Amyloid', base_folder, normalize=True)
        else:
            print(f"Invalid value for 'x': {x}. Use 'tau' or 'abeta'.")
            return None

        if x_burden is None:
            x_burden = subject_x
        else:
            x_burden += subject_x

    # Normalize the aggregated burden data
    x_burden /= len(subjects)

    return x_burden


tauBurden = loadXBurden(x, conditionToStudy)


def loadAndProcessCohort(cohort):
    print("Cohort: " + cohort)
    all_fMRI = dataLoader.load_fullCohort_fMRI(classification, base_folder, Tmax, cohort=cohort)

    if cohort=='HC':
        Timepoints = all_HC_fMRI_1[list(all_HC_fMRI_1.keys())[0]].shape[1]
    elif cohort=='MCI':
        Timepoints = all_MCI_fMRI_1[list(all_MCI_fMRI_1.keys())[0]].shape[1]
    elif cohort=='AD':
        Timepoints = all_AD_fMRI_1[list(all_AD_fMRI_1.keys())[0]].shape[1]

    dataLoader.limit_forcedTmax = Timepoints
    for subj in all_fMRI:
        all_fMRI[subj] = dataLoader.cutTimeSeriesIfNeeded(all_fMRI[subj])
    return all_fMRI

# Load fMRI data using load_fullCohort_fMRI with Tmax

all_HC_fMRI_1 = dataLoader.load_fullCohort_fMRI(classification, base_folder, Tmax, cohort='HC')
all_MCI_fMRI_1 = dataLoader.load_fullCohort_fMRI(classification, base_folder, Tmax, cohort='MCI')
all_AD_fMRI_1 = dataLoader.load_fullCohort_fMRI(classification, base_folder, Tmax, cohort='AD')

all_HC_fMRI = loadAndProcessCohort('HC')
all_MCI_fMRI = loadAndProcessCohort('MCI')
all_AD_fMRI = loadAndProcessCohort('AD')
print('loaded')

'''
# Load fMRI data using load_fullCohort_fMRI with Tmax
all_HC_fMRI = dataLoader.load_fullCohort_fMRI(classification, base_folder, Tmax, cohort='HC')
all_MCI_fMRI = dataLoader.load_fullCohort_fMRI(classification, base_folder, Tmax, cohort='MCI')
all_AD_fMRI = dataLoader.load_fullCohort_fMRI(classification, base_folder, Tmax, cohort='AD')

Timepoints = all_HC_fMRI[list(all_HC_fMRI.keys())[0]].shape[1]
dataLoader.limit_forcedTmax = Timepoints
for subj in all_HC_fMRI:
    all_HC_fMRI[subj] = dataLoader.cutTimeSeriesIfNeeded(all_HC_fMRI[subj])
'''
# ------------------------------------------------
# Configure and compute Simulation
# ------------------------------------------------
# distanceSettings = {'FC': (FC, False), 'swFCD': (swFCD, True), 'phFCD': (phFCD, True)}
selectedObservable = 'phFCD'
distanceSettings = {'phFCD': (phFCD, True)}

simulateBOLD.TR = BOLDFilters.TR  # Recording interval: 1 sample every 3 seconds
simulateBOLD.dt = 0.1 * simulateBOLD.TR / 2.
simulateBOLD.Tmax = Tmax  # This is the length, in seconds
simulateBOLD.dtt = simulateBOLD.TR  # We are not using milliseconds
simulateBOLD.t_min = 10 * simulateBOLD.TR
# simulateBOLD.recomputeTmaxneuronal() <- do not update Tmaxneuronal this way!
# simulateBOLD.warmUpFactor = 6.
simulateBOLD.Tmaxneuronal = (Tmax-1) * simulateBOLD.TR + 30
integrator.ds = simulateBOLD.TR  # record every TR millisecond

baseline_ts = np.zeros((len(all_HC_fMRI), N, Tmax))
for n, subj in enumerate(all_HC_fMRI):
    baseline_ts[n] = all_HC_fMRI[subj]

base_a_value = -0.02
Hopf.setParms({'a': base_a_value})
# Hopf.beta = 0.01
f_diff = filtPowSpectr.filtPowSpetraMultipleSubjects(baseline_ts, TR=BOLDFilters.TR)  # baseline_group[0].reshape((1,52,193))
f_diff[np.where(f_diff == 0)] = np.mean(f_diff[np.where(f_diff != 0)])  # f_diff(find(f_diff==0))=mean(f_diff(find(f_diff~=0)))
# Hopf.omega = repmat(2*pi*f_diff',1,2);     # f_diff is the frequency power
Hopf.setParms({'omega': 2 * np.pi * f_diff})


print("ADHopf Setup done!")

# ================================================================================================================
# ================================================================================================================
# ================================================================================================================EOF

