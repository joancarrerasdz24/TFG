import matplotlib.pyplot as plt
import sys

import numpy as np

from WholeBrain.Optimizers import ParmSweep
from setup import *

import WholeBrain.Utils.Plotting.plotFitting as plotFitting

def preprocessingPipeline(all_HC_fMRI, avgSCMatrix, #, abeta,
                          distanceSettings,  # This is a dictionary of {name: (distance module, apply filters bool)}
                          wes):
    #Fet per Joan: import WholeBrain.Optimizers.ParmSweep as ParmSweep
    print("\n\n###################################################################")
    print("# Compute ParmSeep")
    print("###################################################################\n")
    # Now, optimize all we (G) values: determine optimal G to work with
    balancedParms = [{'we': we} for we in wes]

    fitting = ParmSweep.distanceForAll_Parms(all_HC_fMRI, wes, balancedParms, NumSimSubjects=20,  #20, #len(all_fMRI),  #10,
                                            observablesToUse=distanceSettings,
                                            parmLabel='we',
                                            outFilePath=outFilePath)

    '''fitting = ParmSweep.distanceForAll_Parms(all_fMRI, wes, balancedParms, NumSimSubjects=10,
                                             # 20, #len(all_fMRI),  #10,
                                             observablesToUse=distanceSettings,
                                             parmLabel='we',
                                             outFilePath=outFilePath)'''

    optimal = {}
    for sd in distanceSettings:
        optim = distanceSettings[sd][0].findMinMax(fitting[sd])
        optimal[sd] = (optim[0], optim[1], balancedParms[optim[1]])
    return optimal

def processRangeValues(argv):
    import getopt
    try:
        opts, args = getopt.getopt(argv,'',["wStart=","wEnd=","wStep="])
    except getopt.GetoptError:
        print('Prepro.py --wStart <wStartValue> --wEnd <wEndValue> --wStep <wStepValue>')
        sys.exit(2)
    wStart = 0.; wEnd = 12.0; wStep = 0.05 #0.2
    for opt, arg in opts:
        if opt == '-h':
            print('Prepro.py -wStart <wStartValue> -wEnd <wEndValue> -wStep <wStepValue>')
            sys.exit()
        elif opt in ("--wStart"):
            wStart = float(arg)
        elif opt in ("--wEnd"):
            wEnd = float(arg)
        elif opt in ("--wStep"):
            wStep = float(arg)
    print(f'Input values are: wStart={wStart}, wEnd={wEnd}, wStep={wStep}')
    return wStart, wEnd, wStep

visualizeAll = True
subjectName = 'AvgHC'
outFilePath = save_path + subjectName

if __name__ == '__main__':
    wStart, wEnd, wStep = processRangeValues(sys.argv[1:])
    plt.rcParams.update({'font.size': 22})
    # ----------- Plot whatever results we have collected ------------
    # quite useful to peep at intermediate results
    # G_optim.loadAndPlot(outFilePath='Data_Produced/AD/'+subjectName+'-temp', distanceSettings=distanceSettings)

    wes = np.arange(wStart, wEnd + wStep, wStep)
    optimal = preprocessingPipeline(all_HC_fMRI, avgSC,
                                    distanceSettings,
                                    wes)
    # optimal = preprocessingPipeline(all_HC_fMRI,
    #                                     distanceSettings,
    #                                     mode,
    #                                     wes)
    # =======  Only for quick load'n plot test...
    plotFitting.loadAndPlot(outFilePath + '/fitting_we{}.mat', distanceSettings,
                            WEs=wes, weName='we',
                            empFilePath=outFilePath+'/fNeuro_emp.mat')

    print (f"Last info: Optimal in the CONSIDERED INTERVAL only: {wStart}, {wEnd}, {wStep} (not in the whole set of results!!!)")
    print("".join(f" - Optimal {k}({optimal[k][2]})={optimal[k][0]}\n" for k in optimal))