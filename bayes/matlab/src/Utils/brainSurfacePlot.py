#!/usr/bin/env python
"""
** NOTE for KAYHAN ** DON'T USE ANACONDA, IT WORKS ONLY WITH ENTHOUGHT. MAKE SURE TO RUN AS :   ipython --pylab qt

** NOTE if you want to run it with anaconda, use : ipython --pylab=gtk3

** NOTE freeSurfer script to set environment variable should be sourced before running this script. If you don't have it in your bashrc file, run the following:
         export FREESURFER_HOME=/data/vision/polina/shared_software/freesurfer_v5.1.0
         export FSL_DIR=/data/vision/polina/shared_software/fsl
         source $FREESURFER_HOME/SetUpFreeSurfer.sh

         VIRTUAL_ENV_DISABLE_PROMPT=1 source /afs/csail.mit.edu/u/k/kayhan/home_NFS/Enthought/Canopy_64bit/User/bin/activate
        

==================
Display ROI Values
==================

Here we demonstrate how to take the results of an ROI analysis performed within
each region of some parcellation and display those values on the surface to
quickly summarize the analysis.

"""
print __doc__

import os
import numpy as np
import nibabel as nib
from surfer import Brain
import scipy.io

import argparse

def  drawBrainSurface(args):
     import pandas as pd
     subject_id = args.subject_id
     hemi = args.hemi
     surface = args.surface
     
     """
     Read in the Buckner resting state network annotation. (This requires a
     relatively recent version of Freesurfer, or it can be downloaded separately).
     """
     aparc_file = os.path.join(os.environ["SUBJECTS_DIR"],
                          subject_id, "label",
                          hemi + ".aparc.annot")
     labels, ctab, names = nib.freesurfer.read_annot(aparc_file)
     
     """
     Make a random vector of scalar data corresponding to a value for each region in
     the parcellation.
     """
     matData = pd.read_csv(args.input)
     ePhenLabel = matData['ePhenLabel'].as_matrix()
     values = matData['value'].as_matrix()
     
     if hemi=="lh":
        prefixStr = 'Left-'
        prefixStrLen = len(prefixStr)
     else:
        prefixStr = 'Right-'
        prefixStrLen = len(prefixStr)
        
     # match the name of values with ROI's 
     idx = []
     roi_data = np.zeros((len(names),))   # fill in with default
     
     for i in range(0,len(ePhenLabel)):
        if (str(ePhenLabel[i]))[prefixStrLen:]  in names:
            #print "name of structure on MATLAB side :" , ePhenLabel[i]
            #print "name of structure on freesurfer side :", names[ names.index( (str(ePhenLabel[i]))[prefixStrLen:]  )  ]
            #print "---------------------------"
            idx.append(i)
            roi_data[ names.index( (str(ePhenLabel[i]))[prefixStrLen:]  )  ] =  values[ i ]
       


     if (args.interval==[np.inf,-np.inf]):
        minVal = roi_data.min()
        maxVal = roi_data.max()
     else:
        minVal = args.interval[0]
        maxVal = args.interval[1]
     #minVal = 0
     #maxVal = 1    

     """
     Make a vector containing the data point at each vertex.
     """
     vtx_data = roi_data[labels]

     """
     Display these values on the brain. Use a sequential colormap (assuming
     these data move from low to high values), and add an alpha channel so the
     underlying anatomy is visible.
     """
     """
     Bring up the visualization.
     """
     brain = Brain(subject_id, hemi, surface,
              config_opts=dict(background="white"))
              
     brain.add_data(vtx_data, minVal, maxVal, colormap=args.colormap, alpha=args.alpha)

     brain.save_image(args.output)

     brain.show_view(view=args.view)
     brain.save_image(args.output)

     
       

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='This code is to visualize a map on freesurfer brain')

    parser.add_argument('-m', '--hemi', 
                        default='lh',
                        choices=['lh','rh'],
                        help='which hemisphere to visualize')
    
    parser.add_argument('-j', '--subject_id', 
                        type=str,
                        default='fsaverage',
                        help='what to visualize. Which subject_id in setting(?!)')
                        
    parser.add_argument('-s', '--surface', 
                        type=str,
                        default='inflated',
                        help='what kind of surface to visualize')
                        
                        
    parser.add_argument('-i','--input', 
                        type=str,
                        required=True,
                        help="""input csv file. Each row is one ROI. It should contain the following columns: 
                                    value :  Value to be visualized on the brain
                                    ePhenLabel : Specifying matching name for each""")        

    parser.add_argument('-w','--view',
                        type=str,
                        choices=['medial','lateral','caudal','rostral','dorsal','ventral','frontal','parietal'],
                        default='medial',
                        help='Specify the view for visualization')
                                    
    parser.add_argument('-o','--output', 
                        type=str,
                        required=True,
                        help='output png file. ' ) 
                                    
                                    
    parser.add_argument('-a','--alpha',
                        type=np.float,
                        default=0.8,
                        help='Value of alpha-blending')                          
                        

    parser.add_argument('-c','--colormap',
                        type=str,
                        default='YlOrRd',
                        help='Colormap')                        


    parser.add_argument('-t','--interval',
                        type=np.float,
                        default=[np.inf,-np.inf],
                        nargs=2,
                        help="Specify the interval of values. If you don't, the program finds it automatically. ")
                        
    args = parser.parse_args()
    
    print args
    
    drawBrainSurface(args)
