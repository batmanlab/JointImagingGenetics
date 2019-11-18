#! /usr/bin/env python

import basis
import time

MCR_DIR='/data/vision/polina/shared_software/MCR/v717/'
scripLauncher='/data/vision/polina/users/kayhan/code/Projects/genomicImaging-build/bin/sgeLaunchScript'
matlabLauncher='/data/vision/polina/users/kayhan/code/Projects/genomicImaging-build/bin/runMatlabExec'
execFile='/data/vision/polina/users/kayhan/code/Projects/dppRegression-build/bin/deployEndoPhenVB'
inFn='/data/vision/polina/projects/ADNI/work/kayhan/Journal/MATLAB_Data/natureImput-ChrAll_Data94.mat'
outRoot='/data/vision/polina/users/kayhan/Experiments/DPP/geneticResults/SpikeSlabResults_NatureList0'
csvFile='/data/vision/polina/users/kayhan/Experiments/DPP/geneticResults/SpikeSlabResults_NatureList0/BFResultsFileList.csv'
#step='BF'
#step='normalize'
step='fxvb'


if (step=='BF'):
  for colNum  in range(1,95):
    jobName='SPSLB_%d'%colNum
    outFn=outRoot + '/SPSLB_out_Col%d.mat'%colNum
    cmdLine='qsub   -l mem_free=15G  -N %(jobName)s  %(scripLauncher)s   %(matlabLauncher)s   %(MCR_DIR)s  %(execFile)s  step %(step)s  inputMat   %(inFn)s  colNum  %(colNum)d   outFile  %(outFn)s'\
            %{'jobName':jobName,'scripLauncher':scripLauncher,'matlabLauncher':matlabLauncher,'MCR_DIR':MCR_DIR,'execFile':execFile,'step':step,'inFn':inFn,'colNum':colNum,'outFn':outFn}
    basis.execute(cmdLine.split(),simulate=False)  
    time.sleep(1)
elif (step=='normalize'):
  fid = open(csvFile,'wt')
  fid.writelines('colNum,matFn\n')   
  for colNum  in range(1,95):
     jobName='SPSLB_%d'%colNum
     matFn=outRoot + '/SPSLB_out_Col%d.mat'%colNum
     fid.writelines('%d,%s\n'%(colNum, matFn))
     cmdLine='qsub   -l mem_free=15G  -N %(jobName)s  %(scripLauncher)s   %(matlabLauncher)s   %(MCR_DIR)s  %(execFile)s  step %(step)s  inFile   %(matFn)s  outFile  %(matFn)s'\
             %{'jobName':jobName,'scripLauncher':scripLauncher,'matlabLauncher':matlabLauncher,'MCR_DIR':MCR_DIR,'execFile':execFile,'step':step,'matFn':matFn}
     basis.execute(cmdLine.split(),simulate=False)  
     time.sleep(1)
  fid.close()


elif (step=='fxvb'):
    numRepeats=20
    for r in range(numRepeats):
        jobName='fxVb_Rep%d'%r
        outFile=outRoot + '/fxVb_out_Rep%d.mat'%r
        cmdLine='qsub   -l mem_free=15G  -N %(jobName)s  %(scripLauncher)s   %(matlabLauncher)s   %(MCR_DIR)s  %(execFile)s  step %(step)s  csvFile   %(csvFile)s  inputMat  %(inputMat)s   randomSeed   %(randomSeed)d   outFile  %(outFile)s'\
             %{'jobName':jobName,'scripLauncher':scripLauncher, 'matlabLauncher':matlabLauncher, 'MCR_DIR':MCR_DIR, 'execFile':execFile, 'step':step, 'csvFile':csvFile, 'inputMat':inFn, 'outFile':outFile, 'randomSeed':r}
        basis.execute(cmdLine.split(),simulate=False)  
        time.sleep(1)






