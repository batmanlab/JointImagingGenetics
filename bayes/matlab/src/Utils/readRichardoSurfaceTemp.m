% this function reads the data I got from Richardo
function [dataFrame] = readRichardoSurfaceTemp(dataRoot)
     currPath = pwd ;
     cd(dataRoot)
     % read all WOD-slice data
     
     dataFrame = struct('Year',cell(1,12), 'Month',cell(1,12),...
                        'LatitudeCode',cell(1,12),'LongitudeCode',cell(1,12),...
                        'Latitude',cell(1,12),'Longitude',cell(1,12),...
                        'SST',cell(1,12),'Type',cell(1,12)) ;
     cd('wod_slices')
     for cnt=1:12
         fn = sprintf('WOD2001qc%03d.txt',cnt) ;
         format = '%f  %f  %f  %f %f  %f  %f  %f ' ;
         fid = fopen(fn) ;
         C = textscan(fid, format) ; 
         dataFrame(cnt) = struct('Year',C{1},'Month',C{2},...
                           'LatitudeCode',C{3},'LongitudeCode',C{4},...
                           'Latitude',C{5},'Longitude',C{6},...
                           'SST',C{7},'Type',C{7}) ;
         fclose(fid) ;
     end
     cd('..')
     
     % read all mask information to see which and exclude 
     
     cd(currPath)
end