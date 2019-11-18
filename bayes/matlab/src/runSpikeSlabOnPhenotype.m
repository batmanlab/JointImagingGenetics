% this is a simple function to compute run carbonetto code on all structures
function   runSpikeSlabOnPhenotype(inFile,colNum,outFile)
    load(inFile,'I','G')
    
    if isdeployed()
        colNum = str2num(colNum) ;
    end
    
    %LARGE_SCLAE_SPIKESLAB_PATH = '/data/vision/polina/shared_software/MATLAB_Toolboxes/Carbonetto_VBS/MATLAB' ;
    %addpath(LARGE_SCLAE_SPIKESLAB_PATH)
    %sigma  = linspace(0.2,1,5); sigma = sigma' ;
    sigma = ( 0.2:0.08:1 )' ;
    sa     = (0.025:0.025:0.4)';
    log10q = (-5:0.25:-3)';
    [sigma sa log10q] = ndgrid(sigma,sa,log10q);

    y1 = I(:,colNum) ; 
   

    % Right Hippocampus
    a = 0.02 ;
    b = 1 ;
    c = 0.02 ;
    [w,alpha, mu , lnZ] = varsimbvs(G,y1,sigma,sa,log10q,a,b,c);
    lnZ_weighted = lnZ(:)'*w(:) ;    



    % save the results
    save(outFile,'w','alpha', 'mu' , 'lnZ_weighted') ;
end
