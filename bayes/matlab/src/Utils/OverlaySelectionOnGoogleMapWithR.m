% this function converts the sampleImg (the first output of sampleDPP) to
% csv file that later to be read by 'R' and can be used for visualization.
% Inputs:
%         BoxCoord:  It is a cell each element of which is a structure
%       holding latitude and longitude of the data in addition to the
%       deptht of the tree.
%         geoRaneg : geographical rane (lonMax, lonMin,latMin,latMax)
function   OverlaySelectionOnGoogleMapWithR(BoxCoord)
           fn = [tempname '.csv'] 
           fid = fopen(fn,'wt') ;
           fprintf(fid,'lon , lat , val \n') ;
           fclose(fid) ;
           for sCnt=1:length(BoxCoord)
               dlmwrite( fn , [BoxCoord{sCnt}.lon(:), BoxCoord{sCnt}.lat(:), BoxCoord{sCnt}.depth(:)],'-append') ;
           end
           
end