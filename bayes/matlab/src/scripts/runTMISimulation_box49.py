#! /usr/bin/env python

import time
import numpy as np
import os
from basis.utilities import execute


MCR_DIR='/home/batmanghelich/installed/MATLAB_Compiler_Runtime/v83/'
slurmLauncher='/home/batmanghelich/Projects/dppRegression/src/scripts/slurmLauncher.sh'   
matlabLauncher='/home/batmanghelich/Projects/dppRegression/build/bin/runMatlabExec'
execFile='/home/batmanghelich/Projects/dppRegression/build/bin/deployEndoPhenVB'
inFileRoots='/scratch/users/batmanghelich/ADNI/simData/2014_09_28-box/'    # this value change to the each of the .MAT files
outRoot='/scratch/users/batmanghelich/ADNI/simData/2014_09_28-box_results1/'  # ALL OF THE .MAT FILES WILL BE SAVED HERE
logFileRoot='/scratch/users/batmanghelich/ADNI/simData/logFiles/'
noiseLevels = np.array([0.153, 0.194, 0.247, 0.314, 0.399, 0.507, 0.645, 0.819, 1.041, 11.468, 1.324, 14.577,\
               1.683, 18.529, 2.139, 23.425, 2.719, 29.651, 3.456, 4.393, 5.584, 7.098])
noiseLevels.sort()
oddRatioLevels = np.array([1.125])
numRepeats = 50
Repeats = range(1,numRepeats+1)
seedNumbers = np.array([9711, 3975, 8501,  717, 3503, 3836, 1754, 8559, 1228, 2194])

# OK, this is a large experiment let's focus on a subset
selNoiseLevels = noiseLevels[::2]
selRepeats = Repeats[:20]
fxVbRepeats = 1    # number of internal repeats in the fxvb step  


maxIter = 200     # maximum number of iteration for fxvb step

####    NOTE #####
# There are three steps, you should run them in order
#step='BF'      # STEP 1
#step='normalize'     # STEP 2
#step='csvGen'      # STEP 2.5, this is not a real step. It should be part of STEP 2.5
step='fxvb'      # STEP 3

numInterpPhen = 49    # You have a 7x7 grid correct?
simulation = False

if (step=='BF'):
  for  nl in  selNoiseLevels:
     for od in oddRatioLevels:
         for r in selRepeats:
            # make a folder for the experiment and if it does not exists make one
            matFn = 'sim_%(nl).3f_%(r)d_%(od).3f_mult'%{'nl':nl, 'r':r, 'od':od}
            if not(os.path.exists( outRoot + matFn )):
                  os.mkdir( outRoot + matFn )
            #else:
            #    continue

            for colNum  in range(1,numInterpPhen+1):
                jobName='SPSLB_col%d_%s'%(colNum,matFn)
                inFn = inFileRoots + '/'  + matFn + '.mat'
                outFn = outRoot + '/%(matFn)s/%(matFn)s_out_Col%(colNum)d.mat'%{'matFn':matFn,'colNum':colNum}

                #outFn=outRoot + '/' + matFn + '/SPSLB_out_Col%d.mat'%colNum
                execCmd = '%(matlabLauncher)s   %(MCR_DIR)s  %(execFile)s  step %(step)s  inputMat   %(inFn)s  colNum  %(colNum)d   outFile  %(outFn)s'\
                          %{'matlabLauncher':matlabLauncher,'MCR_DIR':MCR_DIR,'execFile':execFile,'step':step,'inFn':inFn,'colNum':colNum,'outFn':outFn}
                execCmd = execCmd.split()

                stdOut = logFileRoot + '%(jobname)s.stdout'%{'jobname':jobName}
                stdErr = logFileRoot + '%(jobname)s.stderr'%{'jobname':jobName}
                slurmCmd = 'sbatch  -o  %(stdOut)s  -e %(stdErr)s  --job-name=%(jobname)s   %(slurmScript)s  '%{'stdOut':stdOut, 'stdErr': stdErr, 'jobname':jobName, 'slurmScript': slurmLauncher } 
                slurmCmd = slurmCmd.split()

                cmd = slurmCmd + execCmd 
                execute(cmd,simulate=simulation)  
                time.sleep(1)

elif (step=='normalize'):
  for  nl in selNoiseLevels:
     for od in oddRatioLevels:
         for r in selRepeats:
            # make a folder for the experiment and if it does not exists make one
            matFn = 'sim_%(nl).3f_%(r)d_%(od).3f_mult'%{'nl':nl, 'r':r, 'od':od}
            csvFn = outRoot + '/%(matFn)s/%(matFn)s_BFResultsFileList.csv'%{'matFn':matFn}
            if os.path.exists(csvFn):
                continue  
            
            fid = open(csvFn,'wt')
            fid.writelines('colNum,matFn\n')
 
            for colNum  in range(1,numInterpPhen+1):
                jobName='SPSLB_norm_col%d_%s'%(colNum,matFn)
                inFn = outRoot + '/%(matFn)s/%(matFn)s_out_Col%(colNum)d.mat'%{'matFn':matFn,'colNum':colNum}
                outFn = outRoot + '/%(matFn)s/%(matFn)s_out_Col%(colNum)d.mat'%{'matFn':matFn,'colNum':colNum}

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
                	 
                time.sleep(1)

            fid.close()





#  fid = open(csvFile,'wt')
#  fid.writelines('colNum,matFn\n') 
#  for colNum  in range(1,numInterpPhen+1):
#     jobName='SPSLB_%d'%colNum
#     matFn=outRoot + '/SPSLB_out_Col%d.mat'%colNum
#     fid.writelines('%d,%s\n'%(colNum, matFn))	
#     cmdLine='qsub   -l mem_free=15G  -N %(jobName)s  %(scripLauncher)s   %(matlabLauncher)s   %(MCR_DIR)s  %(execFile)s  step %(step)s  inFile   %(matFn)s  outFile  %(matFn)s'\
#             %{'jobName':jobName,'scripLauncher':scripLauncher,'matlabLauncher':matlabLauncher,'MCR_DIR':MCR_DIR,'execFile':execFile,'step':step,'matFn':matFn}
#     basis.execute(cmdLine.split(),simulate=False)  
#     time.spleep(1)
#  fid.close()	

#elif (step=='csvGen'):   # Remember, there should be one csv file per experiment. In other words, for each 49 .mat files, there should be one .csv file in the folder where those 49 mat files reside.
#  fid = open(csvFile,'wt')
#  fid.writelines('colNum,matFn\n') 
#  for colNum  in range(1,numInterpPhen):
#     matFn=outRoot + '/SPSLB_out_Col%d.mat'%colNum
#     fid.writelines('%d,%s\n'%(colNum, matFn))	
#  fid.close()	


elif (step=='fxvb'):
  for  nl in selNoiseLevels:
     for od in oddRatioLevels:
         for r in selRepeats:
            # make a folder for the experiment and if it does not exists make one
            matFn = 'sim_%(nl).3f_%(r)d_%(od).3f_mult'%{'nl':nl, 'r':r, 'od':od}
            csvFn = outRoot + '/%(matFn)s/%(matFn)s_BFResultsFileList.csv'%{'matFn':matFn}


            for sn in seedNumbers:
                jobName='fxVb_seed%d_%s'%(sn,matFn)

                inFn = inFileRoots + '/'  + matFn + '.mat'
                outFn = outRoot + '/%(matFn)s/fxVb_out_%(matFn)s_Seed%(sn)d.mat'%{'matFn':matFn,'sn':sn}

                if os.path.exists(outFn):
                  continue

                execCmd = '%(matlabLauncher)s   %(MCR_DIR)s  %(execFile)s  step %(step)s  inputMat   %(inFn)s  csvFile  %(csvFile)s   randomSeed   %(randomSeed)d  outFile  %(outFn)s  numRepeats  %(fxVbRepeats)d   maxIter   %(maxIter)d'\
                          %{'matlabLauncher':matlabLauncher,'MCR_DIR':MCR_DIR,'execFile':execFile,'step':step,'inFn':inFn,'outFn':outFn, 'randomSeed':sn, 'csvFile':csvFn, 'maxIter':maxIter, 'fxVbRepeats':fxVbRepeats}
                execCmd = execCmd.split()

                stdOut = logFileRoot + '%(jobname)s.stdout'%{'jobname':jobName}
                stdErr = logFileRoot + '%(jobname)s.stderr'%{'jobname':jobName}
                slurmCmd = 'sbatch  -o  %(stdOut)s  -e %(stdErr)s  --job-name=%(jobname)s   %(slurmScript)s  '%{'stdOut':stdOut, 'stdErr': stdErr, 'jobname':jobName, 'slurmScript': slurmLauncher } 
                slurmCmd = slurmCmd.split()

                cmd = slurmCmd + execCmd 
                execute(cmd,simulate=simulation)  
                time.sleep(1)






#    numRepeats=20
#    for r in range(numRepeats):
#        jobName='fxVb_Rep%d'%r
#        outFile=outRoot + '/fxVb_out_Rep%d.mat'%r
#        cmdLine='qsub   -l mem_free=15G  -N %(jobName)s  %(scripLauncher)s   %(matlabLauncher)s   %(MCR_DIR)s  %(execFile)s  step %(step)s  csvFile   %(csvFile)s  inputMat  %(inputMat)s   randomSeed   %(randomSeed)d   outFile  %(outFile)s'\
#             %{'jobName':jobName,'scripLauncher':scripLauncher, 'matlabLauncher':matlabLauncher, 'MCR_DIR':MCR_DIR, 'execFile':execFile, 'step':step, 'csvFile':csvFile, 'inputMat':inFn, 'outFile':outFile, 'randomSeed':r}
#        basis.execute(cmdLine.split(),simulate=False)  
#        time.sleep(1)








