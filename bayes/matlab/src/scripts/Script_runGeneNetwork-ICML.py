#!/bin/env python

import os
import subprocess
import pandas as pd
from basis.utilities import execute



execFn='/home/batmanghelich/Projects/dppRegression/build/bin/evaluateUCIClassification'
execHelperFn='/home/batmanghelich/Projects/dppRegression/build/bin/sgeLaunchMATLABScript'
#suprVoxelRoot='/scratch/users/batmanghelich/COPDGene/work/kayhan/Supervoxels/monoSLIC/'
slurmLauncher='/home/batmanghelich/Projects/dppRegression/src/scripts/slurmLauncher.sh'
logFileRoot='/scratch/users/batmanghelich/DPP/logFiles/'
JOBNAME = 'geneNetwork1500'

#export MCR_DIR=/data/vision/polina/shared_software/MCR/v717/
#export scripLauncher=/data/vision/polina/users/kayhan/code/Projects/genomicImaging-build/bin/sgeLaunchScript
#export matlabLauncher=/data/vision/polina/users/kayhan/code/Projects/genomicImaging-build/bin/runMatlabExec
#export methodList='fixFrmDPPRegression'
numRepeat=100    # different draw from the training set
numInit=10   # different initialization
inFnRoot='/scratch/users/batmanghelich/DPP/geneNetwork/Exp3MoreReplicateData/'
outFnRoot='/scratch/users/batmanghelich/DPP/geneNetwork/geneNetwork1-ICML/'
methodsList='DPP,DPPDPP,SpikeSlab,OMP,DPPMF'
useIntercept=1
continueFlag = 0

#datasetName='low1500pVal'
datasetName='low2000pVal'
for r  in range(1,numRepeat+1):
    for i in range(1,numInit+1):
      fn='%(datasetName)s_Rep%(r)d_Data_itr500_HubFeatures.mat'%{'datasetName':datasetName, 'r':r }
      jobName='NetData_HubFeatures_%(datasetName)s_Rep%(r)d_init%(i)d'%{'datasetName':datasetName, 'r':r, 'i':i}
      inFn=inFnRoot + fn
      outFn= outFnRoot + 'out_%(datasetName)s_Rep%(r)d_Init%(i)d_itr500.mat'%{'datasetName':datasetName, 'r':r, 'i':i}
      execCmd='%(execHelperFn)s   %(execFn)s    %(inFn)s   %(outFn)s   %(datasetName)s   %(r)d   %(methodsList)s  %(continueFlag)d '%{'execHelperFn':execHelperFn, 'execFn':execFn, 'outFn':outFn, 'r':r, 'continueFlag':continueFlag, 'inFn':inFn, 'outFn':outFn, 'r': r, 'datasetName':datasetName, 'methodsList':methodsList}    
      execCmd = execCmd.split()      


      stdOut = logFileRoot + '%(jobname)s_%(datasetName)s_Rep%(r)d_iter%(i)d.stdout'%{'jobname':JOBNAME, 'datasetName':datasetName, 'r':r, 'i':i} 
      stdErr = logFileRoot + '%(jobname)s_%(datasetName)s_Rep%(r)d_iter%(i)d.stderr'%{'jobname':JOBNAME, 'datasetName':datasetName, 'r':r, 'i':i} 
      slurmCmd = 'sbatch -n2 -o  %(stdOut)s  -e %(stdErr)s  --job-name=%(jobname)s   %(slurmScript)s  '%{'stdOut':stdOut, 'stdErr': stdErr, 'jobname':JOBNAME, 'slurmScript': slurmLauncher } 
      slurmCmd = slurmCmd.split()

      cmd = slurmCmd + execCmd 
      print "cmd : ", cmd
      #p = subprocess.Popen(cmd , stdout=subprocess.PIPE)
      execute(cmd, simulate=False)








