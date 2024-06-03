# --------------------------------------------------------------------------------------
#  Hopf code: fitting
#  By Gustavo Patow
#  Based on the code by Xenia Koblebeva
#
# This code must be run AFTER setting setup.py and getting the result of prepro.py
# --------------------------------------------------------------------------------------
import sys
import matplotlib.pyplot as plt

from setup import *

import WholeBrain.Utils.Plotting.plotFitting as plotFitting


# =====================================================================
# =====================================================================
#                      Single Subject Pipeline
# =====================================================================
# =====================================================================
def fittingPipeline(all_fMRI,
                    burden,
                    distanceSettings,  # This is a dictionary of {name: (distance module, apply filters bool)}
                    rangeW,
                    mode):
    print("\n\n###################################################################")
    print("# Fitting with ParmSweep")
    print("###################################################################\n")
    # Now, optimize all we (G) values: determine optimal G to work with

    HopfParms = [{'a': base_a_value + parW * burden} for parW in rangeW]
    fitting = ParmSweep.distanceForAll_Parms(all_fMRI,
                                            rangeW, HopfParms,
                                            NumSimSubjects=5,  #len(all_fMRI)
                                            observablesToUse=distanceSettings,
                                            parmLabel='scaling',
                                            outFilePath=outFilePath,
                                            fileNameSuffix=mode)

    optimal = {sd: distanceSettings[sd][0].findMinMax(fitting[sd]) for sd in distanceSettings}
    return optimal


# =====================================================================
# =====================================================================
#                            main
# =====================================================================
# =====================================================================
def processRangeValues(argv):
    import getopt
    try:
        opts, args = getopt.getopt(argv,'',["tauStart=","tauEnd=","tauStep="])
    except getopt.GetoptError:
        print('Prepro.py --tauStart <tauStartValue> --tauEnd <tauEndValue> --tauStep <tauStepValue>')
        sys.exit(2)
    tauStart = -0.25; tauEnd = 0.25; tauStep = 0.025  # -1:0.025:1.025
    for opt, arg in opts:
        if opt == '-h':
            print('Prepro.py -tauStart <tauStartValue> -tauEnd <tauEndValue> -tauStep <tauStepValue>')
            sys.exit()
        elif opt in ("--tauStart"):
            tauStart = float(arg)
        elif opt in ("--tauEnd"):
            tauEnd = float(arg)
        elif opt in ("--tauStep"):
            tauStep = float(arg)
    print(f'Input values are: tauStart={tauStart}, tauEnd={tauEnd}, tauStep={tauStep}')
    return tauStart, tauEnd, tauStep


def applyMode(burden, mode):
    if mode == 'homogeneous':
        avgTau = np.average(burden)
        burden = np.ones(len(burden)) * avgTau
        print('Using homogeneous burden for optimization')
    elif mode == 'shuffled':
        np.random.shuffle(burden)  # shufflling in place
        print('Using shuffled burden for optimization')
    return burden

if mode == 'heterogeneous':
    file_mode = 'Heterogeneous'
elif mode == 'homogeneous':
    file_mode = 'Homogeneous'
elif mode == 'shuffled':
    file_mode = 'Shuffled'

visualizeAll = True
subjectMode = 'AD'
modality = 'Tau'
outFilePath = save_path + modality + '/' + subjectMode + '/' + file_mode

optimal_results = []
# Mostrar el vector con los resultados óptimos de todos los pacientes
heterogeneous_results = []
homogeneous_results = []
shuffled_results = []
# --------------------------------------------------------------------------------------
#  Hopf code: fitting
#  By Gustavo Patow
#  Based on the code by Xenia Koblebeva
#
# This code must be run AFTER setting setup.py and getting the result of prepro.py
# --------------------------------------------------------------------------------------
import sys
import matplotlib.pyplot as plt

from setup import *

import WholeBrain.Utils.Plotting.plotFitting as plotFitting


# =====================================================================
# =====================================================================
#                      Single Subject Pipeline
# =====================================================================
# =====================================================================
def fittingPipeline(all_fMRI,
                    burden,
                    distanceSettings,  # This is a dictionary of {name: (distance module, apply filters bool)}
                    rangeW,
                    mode):
    print("\n\n###################################################################")
    print("# Fitting with ParmSweep")
    print("###################################################################\n")
    # Now, optimize all we (G) values: determine optimal G to work with

    HopfParms = [{'a': base_a_value + parW * burden} for parW in rangeW]
    fitting = ParmSweep.distanceForAll_Parms(all_fMRI,
                                            rangeW, HopfParms,
                                            NumSimSubjects=5,  #len(all_fMRI)
                                            observablesToUse=distanceSettings,
                                            parmLabel='scaling',
                                            outFilePath=outFilePath,
                                            fileNameSuffix=mode)

    optimal = {sd: distanceSettings[sd][0].findMinMax(fitting[sd]) for sd in distanceSettings}
    return optimal


# =====================================================================
# =====================================================================
#                            main
# =====================================================================
# =====================================================================
def processRangeValues(argv):
    import getopt
    try:
        opts, args = getopt.getopt(argv,'',["tauStart=","tauEnd=","tauStep="])
    except getopt.GetoptError:
        print('Prepro.py --tauStart <tauStartValue> --tauEnd <tauEndValue> --tauStep <tauStepValue>')
        sys.exit(2)
    tauStart = -0.25; tauEnd = 0.25; tauStep = 0.025  # -1:0.025:1.025
    for opt, arg in opts:
        if opt == '-h':
            print('Prepro.py -tauStart <tauStartValue> -tauEnd <tauEndValue> -tauStep <tauStepValue>')
            sys.exit()
        elif opt in ("--tauStart"):
            tauStart = float(arg)
        elif opt in ("--tauEnd"):
            tauEnd = float(arg)
        elif opt in ("--tauStep"):
            tauStep = float(arg)
    print(f'Input values are: tauStart={tauStart}, tauEnd={tauEnd}, tauStep={tauStep}')
    return tauStart, tauEnd, tauStep


def applyMode(burden, mode):
    if mode == 'homogeneous':
        avgTau = np.average(burden)
        burden = np.ones(len(burden)) * avgTau
        print('Using homogeneous burden for optimization')
    elif mode == 'shuffled':
        np.random.shuffle(burden)  # shufflling in place
        print('Using shuffled burden for optimization')
    return burden

if mode == 'heterogeneous':
    file_mode = 'Heterogeneous'
elif mode == 'homogeneous':
    file_mode = 'Homogeneous'
elif mode == 'shuffled':
    file_mode = 'Shuffled'

visualizeAll = True
subjectMode = 'AD'
modality = 'Tau'
outFilePath = save_path + modality + '/' + subjectMode + '/' + file_mode

optimal_results = []
# Mostrar el vector con los resultados óptimos de todos los pacientes
heterogeneous_results = []
homogeneous_results = []
shuffled_results = []

if __name__ == '__main__':
    #import WholeBrain.Utils.decorators as decorators
    #decorators.forceCompute = True
    # Bias = -0.01: +0.01 (steps : 0.0025)
    # Scaling in [-1, 1.05] with steps 0.025
    print("\n\n########################################")
    print(f"Processing: {subjectMode}")
    #print(f"(To folder: {outFilePath})")
    print("########################################\n\n")

    # Definir el directorio base
    base_dir = os.path.join(save_path, modality, subjectMode)

    for i, (key, value) in enumerate(all_AD_fMRI.items(),1):
        # Crear el directorio para el paciente actual
        patient_dir = os.path.join(base_dir, f"Patient_{i}")
        os.makedirs(patient_dir, exist_ok=True)

        mode_dir = os.path.join(patient_dir, mode)
        os.makedirs(mode_dir, exist_ok=True)

        # Cargar los datos específicos del paciente y procesarlos para cada modo
        tauStart, tauEnd, tauStep = processRangeValues(sys.argv[1:])
        pacient = {key: value}
        optimal_dict = {}

        modes = ["heterogeneous", "homogeneous", "shuffled"]

        plt.rcParams.update({'font.size': 22})

        # Set G to 1.2, obtained in the prepro stage
        Hopf.setParms({'we': 7.75}) #7.75

        # Dentro de la carpeta del paciente, crear subdirectorios para cada modo
        for mode in modes:
            print(f"\n\n########################################")
            print(f"Processing: {mode}")
            print("########################################\n\n")
            # outFilePath = save_path + subjectName + '/' + mode

            if mode == 'heterogeneous':
                file_mode = 'Heterogeneous'
            elif mode == 'homogeneous':
                file_mode = 'Homogeneous'
            elif mode == 'shuffled':
                file_mode = 'Shuffled'

            outFilePath = os.path.join(save_path, modality, subjectMode, f"Patient_{i}", file_mode)
            print(f"(To folder: {outFilePath})")
            print("########################################\n\n")

            # tauStart = 0.5
            tauWs = np.arange(tauStart, tauEnd + tauStep, tauStep)

            #burden = dataLoader.loadBurden(key, modality, base_folder)
            #SCnorm, abeta_burden, tau_burden, fullSeries = dataLoader.loadSubjectData(key,200)
            abeta_burden = dataLoader.loadBurden(key, "Amyloid", base_folder, normalize=True)
            tau_burden = dataLoader.loadBurden(key, "Tau", base_folder, normalize=True)

            if modality == 'Tau':
                burden = tau_burden
            elif modality == 'Amyloid':
                burden = abeta_burden
            else:
                raise(Exception("Unrecognized modality"))

            burden_2 = applyMode(burden, mode)

            optimal = fittingPipeline(pacient,
                                      burden_2,
                                      distanceSettings,
                                      tauWs,
                                      mode='_' + mode)

            # Almacenar los resultados óptimos para el modo actual
            optimal_dict[mode] = optimal

            # Guardar los resultados en archivos específicos para cada modo
            with open(f'optimal_phFCD_index_{mode}.txt', 'w') as f:
                f.write(str(optimal['phFCD'][1]))

            # =======  Only for quick load'n plot test...
            plotFitting.loadAndPlot(outFilePath + '/fitting_scaling{}.mat', distanceSettings,
                                    weName='scaling',  # WEs=tauWs,
                                    empFilePath=outFilePath + '/fNeuro_emp' + '_' + mode + '.mat')

        print(f"Last info: Optimal in the CONSIDERED INTERVAL only: {tauStart}, {tauEnd}, {tauStep} (not in the whole set of results!!!)")
        print("".join(f" - Optimal {k}({optimal[k][1]}->{tauWs[optimal[k][1]]})={optimal[k][0]}\n" for k in optimal))

        # Agregar los resultados óptimos de este paciente a la lista
        optimal_results.append(optimal_dict)

        # Mostrar resumen de resultados para cada paciente
        print(f"Optimal results for Patient {i}: {optimal_dict}")

    # Volver al directorio base
    os.chdir(base_dir)

    print("Optimal results for all patients:")
    for i, result in enumerate(optimal_results, 1):
        print(f"Patient {i}: {result}")
        heterogeneous_results.append(result['heterogeneous']['phFCD'][0])
        homogeneous_results.append(result['homogeneous']['phFCD'][0])
        shuffled_results.append(result['shuffled']['phFCD'][0])

    print('Heterogeneous results: ', heterogeneous_results)
    print('Homogeneous results: ', homogeneous_results)
    print('Shuffled results: ', shuffled_results)

# ================================================================================================================
# ================================================================================================================
# ================================================================================================================EOF
if __name__ == '__main__':
    #import WholeBrain.Utils.decorators as decorators
    #decorators.forceCompute = True
    # Bias = -0.01: +0.01 (steps : 0.0025)
    # Scaling in [-1, 1.05] with steps 0.025
    print("\n\n########################################")
    print(f"Processing: {subjectMode}")
    #print(f"(To folder: {outFilePath})")
    print("########################################\n\n")

    # Definir el directorio base
    base_dir = os.path.join(save_path, modality, subjectMode)

    # Definir listas para almacenar los resultados óptimos de cada paciente
    #optimal_results = []

    for i, (key, value) in enumerate(all_AD_fMRI.items(),1):
        # Crear el directorio para el paciente actual
        patient_dir = os.path.join(base_dir, f"Patient_{i}")
        os.makedirs(patient_dir, exist_ok=True)

        mode_dir = os.path.join(patient_dir, mode)
        os.makedirs(mode_dir, exist_ok=True)

        # Cambiar la configuración de directorio actual
        ##os.chdir(mode_dir)

        # Cargar los datos específicos del paciente y procesarlos para cada modo
        tauStart, tauEnd, tauStep = processRangeValues(sys.argv[1:])
        pacient = {key: value}
        optimal_dict = {}

        modes = ["heterogeneous", "homogeneous", "shuffled"]
        # Diccionarios para almacenar resultados
        # tauWs_dict = {}
        # optimal_dict = {}
        # optimal_phFCD_index_dict = {}

        plt.rcParams.update({'font.size': 22})

        # Set G to 1.2, obtained in the prepro stage
        Hopf.setParms({'we': 7.75}) #7.75

        # Dentro de la carpeta del paciente, crear subdirectorios para cada modo
        for mode in modes:
            print(f"\n\n########################################")
            print(f"Processing: {mode}")
            print("########################################\n\n")
            # outFilePath = save_path + subjectName + '/' + mode

            if mode == 'heterogeneous':
                file_mode = 'Heterogeneous'
            elif mode == 'homogeneous':
                file_mode = 'Homogeneous'
            elif mode == 'shuffled':
                file_mode = 'Shuffled'

            outFilePath = os.path.join(save_path, modality, subjectMode, f"Patient_{i}", file_mode)
            print(f"(To folder: {outFilePath})")
            print("########################################\n\n")

            # tauStart = 0.5
            tauWs = np.arange(tauStart, tauEnd + tauStep, tauStep)

            #burden = dataLoader.loadBurden(key, modality, base_folder)
            #SCnorm, abeta_burden, tau_burden, fullSeries = dataLoader.loadSubjectData(key,200)
            abeta_burden = dataLoader.loadBurden(key, "Amyloid", base_folder, normalize=True)
            tau_burden = dataLoader.loadBurden(key, "Tau", base_folder, normalize=True)

            if modality == 'Tau':
                burden = tau_burden
            elif modality == 'Amyloid':
                burden = abeta_burden
            else:
                raise(Exception("Unrecognized modality"))

            burden_2 = applyMode(burden, mode)

            optimal = fittingPipeline(pacient,
                                      burden_2,
                                      distanceSettings,
                                      tauWs,
                                      mode='_' + mode)

            # Almacenar los resultados óptimos para el modo actual
            optimal_dict[mode] = optimal

            # Guardar los resultados en archivos específicos para cada modo
            with open(f'optimal_phFCD_index_{mode}.txt', 'w') as f:
                f.write(str(optimal['phFCD'][1]))

            # =======  Only for quick load'n plot test...
            plotFitting.loadAndPlot(outFilePath + '/fitting_scaling{}.mat', distanceSettings,
                                    weName='scaling',  # WEs=tauWs,
                                    empFilePath=outFilePath + '/fNeuro_emp' + '_' + mode + '.mat')

        print(f"Last info: Optimal in the CONSIDERED INTERVAL only: {tauStart}, {tauEnd}, {tauStep} (not in the whole set of results!!!)")
        print("".join(f" - Optimal {k}({optimal[k][1]}->{tauWs[optimal[k][1]]})={optimal[k][0]}\n" for k in optimal))

        # Agregar los resultados óptimos de este paciente a la lista
        optimal_results.append(optimal_dict)

        # Mostrar resumen de resultados para cada paciente
        print(f"Optimal results for Patient {i}: {optimal_dict}")

    # Volver al directorio base
    os.chdir(base_dir)
    ''''# Mostrar resumen de resultados
    #print(f"tauWs_: {tauWs_dict}")
    print(f"optimal: {optimal_dict}")
    #print(f"optimal_phFCD_index: {optimal_phFCD_index_dict}")'''

    print("Optimal results for all patients:")
    for i, result in enumerate(optimal_results, 1):
        print(f"Patient {i}: {result}")
        heterogeneous_results.append(result['heterogeneous']['phFCD'][0])
        homogeneous_results.append(result['homogeneous']['phFCD'][0])
        shuffled_results.append(result['shuffled']['phFCD'][0])

    print('Heterogeneous results: ', heterogeneous_results)
    print('Homogeneous results: ', homogeneous_results)
    print('Shuffled results: ', shuffled_results)

# ================================================================================================================
# ================================================================================================================
# ================================================================================================================EOF