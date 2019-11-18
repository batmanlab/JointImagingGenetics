#!/bin/env python

import os
import subprocess
import pandas as pd
from basis.utilities import execute
import numpy as np


execFn='/home/batmanghelich/Projects/dppRegression/build/bin/evaluateUSTemp2'
execHelperFn='/home/batmanghelich/Projects/dppRegression/build/bin/sgeLaunchMATLABScript'
#suprVoxelRoot='/scratch/users/batmanghelich/COPDGene/work/kayhan/Supervoxels/monoSLIC/'
slurmLauncher='/home/batmanghelich/Projects/dppRegression/src/scripts/slurmLauncher.sh'
logFileRoot='/scratch/users/batmanghelich/DPP/logFiles/'

#export MCR_DIR=/data/vision/polina/shared_software/MCR/v717/
#export scripLauncher=/data/vision/polina/users/kayhan/code/Projects/genomicImaging-build/bin/sgeLaunchScript
#export matlabLauncher=/data/vision/polina/users/kayhan/code/Projects/genomicImaging-build/bin/runMatlabExec
#export methodList='fixFrmDPPRegression'
numRepeat=50    # different draw from the training set
numSensors=range(20,250,10)   # different initialization
inFnRoot='/scratch/users/batmanghelich/DPP/USTemp/USTemp-ICML-data/'
outFnRoot='/scratch/users/batmanghelich/DPP/USTemp/USTemp-withIntercept-Results1/'
#outFnRoot='/scratch/users/batmanghelich/DPP/USTemp/USTemp-noIntercept-Results1/'
useIntercept=1
JOBNAME = 'USTemp-withIntercept'

randSeedList = np.random.randint(1000,9000,numRepeat)

for  r, randSeed  in zip(range(1,numRepeat+1) , randSeedList ):
    for numObs in numSensors:
      fn='USTemp_numSens%(numObs)d_numRep%(rep)d.mat'%{'rep':r, 'numObs':numObs }
      inFn=inFnRoot + fn
      outFn= outFnRoot + 'out_Obs%(numObs)d_Rep%(rep)d.mat'%{'rep':r, 'numObs':numObs }
      execCmd='%(execHelperFn)s   %(execFn)s    %(inFn)s   %(outFn)s   %(randSeed)d  %(useIntercept)d '%{'execHelperFn':execHelperFn, 'execFn':execFn,  'inFn':inFn, 'outFn':outFn, 'useIntercept':useIntercept, 'randSeed': randSeed}    
      execCmd = execCmd.split()      


      stdOut = logFileRoot + '%(jobname)s_Rep%(r)d_numObs%(numObs)d.stdout'%{'jobname':JOBNAME,  'r':r, 'numObs':numObs} 
      stdErr = logFileRoot + '%(jobname)s_Rep%(r)d_numObs%(numObs)d.stderr'%{'jobname':JOBNAME,  'r':r, 'numObs':numObs} 
      slurmCmd = 'sbatch -n2 -o  %(stdOut)s  -e %(stdErr)s  --job-name=%(jobname)s   %(slurmScript)s  '%{'stdOut':stdOut, 'stdErr': stdErr, 'jobname':JOBNAME, 'slurmScript': slurmLauncher } 
      slurmCmd = slurmCmd.split()

      cmd = slurmCmd + execCmd 
      print "cmd : ", cmd
      #p = subprocess.Popen(cmd , stdout=subprocess.PIPE)
      execute(cmd, simulate=False)








