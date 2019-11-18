#! /usr/bin/env python

import time
import numpy as np
import os
from basis.utilities import execute


MCR_DIR='/home/batmanghelich/installed/MATLAB_Compiler_Runtime/v83/'
slurmLauncher='/home/batmanghelich/Projects/dppRegression/src/scripts/slurmLauncher.sh'   
matlabLauncher='/home/batmanghelich/Projects/dppRegression/build/bin/runMatlabExec'
execFile='/home/batmanghelich/Projects/dppRegression/build/bin/deployEndoPhenVB'
inFileRoots='/scratch/users/batmanghelich/ADNI/work/MATLAB_Data/covRemovedMatFiles/'    
outRoot='/scratch/users/batmanghelich/ADNI/work/Journal/ADNI_NatureList_covRemoved/'  
logFileRoot='/scratch/users/batmanghelich/ADNI/work/logFiles/'

seedNumbers = np.array([9711, 3975, 8501,  717, 3503, 3836, 1754, 8559, 1228, 2194])
csvFile=outRoot +  '/BFResultsFileList.csv'
matFn = 'natureImput-ChrAll_covRemoved'


#step='BF'
#step='normalize'
step='fxvb'

numInterpPhen = 94    
simulation = False
numRepeats=20
fxVbRepeats = 20    # number of internal repeats in the fxvb step  
jobRunTime='2-00:00'


if (step=='BF'):
   for colNum  in range(1,numInterpPhen+1):
      jobName='SPSLB_col%d_%s'%(colNum,matFn)
      inFn = inFileRoots + '/'  + matFn + '_Data94.mat'
      outFn = outRoot + '%(matFn)s_out_Col%(colNum)d.mat'%{'matFn':matFn,'colNum':colNum}

      #outFn=outRoot + '/' + matFn + '/SPSLB_out_Col%d.mat'%colNum
      execCmd = '%(matlabLauncher)s   %(MCR_DIR)s  %(execFile)s  step %(step)s  inputMat   %(inFn)s  colNum  %(colNum)d   outFile  %(outFn)s'\
                %{'matlabLauncher':matlabLauncher,'MCR_DIR':MCR_DIR,'execFile':execFile,'step':step,'inFn':inFn,'colNum':colNum,'outFn':outFn}
      execCmd = execCmd.split()

      stdOut = logFileRoot + '%(jobname)s.stdout'%{'jobname':jobName}
      stdErr = logFileRoot + '%(jobname)s.stderr'%{'jobname':jobName}
      slurmCmd = 'sbatch  -o  %(stdOut)s  -e %(stdErr)s  --job-name=%(jobname)s   -t %(jobRunTime)s  %(slurmScript)s  '%{'stdOut':stdOut, 'stdErr': stdErr, 'jobname':jobName, 'slurmScript': slurmLauncher, 'jobRunTime':jobRunTime  } 
      slurmCmd = slurmCmd.split()

      cmd = slurmCmd + execCmd 
      execute(cmd,simulate=simulation)  
      time.sleep(1)


elif (step=='normalize'):    
    fid = open(csvFile,'wt')
    fid.writelines('colNum,matFn\n')

    for colNum  in range(1,numInterpPhen+1):
      jobName='SPSLB_norm_col%d_%s'%(colNum,matFn)
      inFn = outRoot + '%(matFn)s_out_Col%(colNum)d.mat'%{'matFn':matFn,'colNum':colNum}
      outFn = outRoot + '%(matFn)s_out_Col%(colNum)d.mat'%{'matFn':matFn,'colNum':colNum}

      execCmd = '%(matlabLauncher)s   %(MCR_DIR)s  %(execFile)s  step %(step)s  inFile   %(inFn)s   outFile  %(outFn)s'\
                %{'matlabLauncher':matlabLauncher,'MCR_DIR':MCR_DIR,'execFile':execFile,'step':step,'inFn':inFn, 'outFn':outFn}
      execCmd = execCmd.split()

      stdOut = logFileRoot + '%(jobname)s.stdout'%{'jobname':jobName}
      stdErr = logFileRoot + '%(jobname)s.stderr'%{'jobname':jobName}
      slurmCmd = 'sbatch  -o  %(stdOut)s  -e %(stdErr)s  --job-name=%(jobname)s   %(slurmScript)s  '%{'stdOut':stdOut, 'stdErr': stdErr, 'jobname':jobName, 'slurmScript': slurmLauncher } 
      slurmCmd = slurmCmd.split()

      cmd = slurmCmd + execCmd 
      execute(cmd,simulate=simulation) 
      
      fid.writelines('%d,%s\n'%(colNum, outFn))
               
      #time.sleep(1)

    fid.close()


elif (step=='fxvb'):
  for r,sn in enumerate(seedNumbers):
      outFile=outRoot + '/fxVb_out_Rep%d.mat'%r

      jobName='fxVb_seed%d_%s'%(sn,matFn)

      inFn = inFileRoots + '/'  + matFn + '_Data94.mat'
      outFn = outRoot + '/fxVb_out_%(matFn)s_Seed%(sn)d.mat'%{'matFn':matFn,'sn':sn}


      execCmd = '%(matlabLauncher)s   %(MCR_DIR)s  %(execFile)s  step %(step)s  inputMat   %(inFn)s  csvFile  %(csvFile)s   randomSeed   %(randomSeed)d  outFile  %(outFn)s  numRepeats  %(fxVbRepeats)d   '\
                %{'matlabLauncher':matlabLauncher,'MCR_DIR':MCR_DIR,'execFile':execFile,'step':step,'inFn':inFn,'outFn':outFn, 'randomSeed':sn, 'csvFile':csvFile, 'fxVbRepeats':fxVbRepeats}
      execCmd = execCmd.split()

      stdOut = logFileRoot + '%(jobname)s.stdout'%{'jobname':jobName}
      stdErr = logFileRoot + '%(jobname)s.stderr'%{'jobname':jobName}
      slurmCmd = 'sbatch  -o  %(stdOut)s  -e %(stdErr)s  --job-name=%(jobname)s  -t %(jobRunTime)s   %(slurmScript)s  '%{'stdOut':stdOut, 'stdErr': stdErr, 'jobname':jobName, 'slurmScript': slurmLauncher, 'jobRunTime':jobRunTime  } 
      slurmCmd = slurmCmd.split()

      cmd = slurmCmd + execCmd 
      execute(cmd,simulate=simulation)  
      #time.sleep(1)










