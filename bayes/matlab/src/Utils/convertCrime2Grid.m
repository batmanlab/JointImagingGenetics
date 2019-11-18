% this function simply converts the crime data provided by Carin to grid
% format
%  Input:
%       fn: points to the place crime3yr_data.mat is stored
% Output:
%    eachCrimeData: cell containing crime rates categorized by crime type
%    overalCrimeDate : cell containing overall crimes commited in the are
function [eachCrimeData,overalCrimeDate] = convertCrime2Grid(fn)
    %load ../data/crimeData/crime3yr_data.mat
    load(fn)
    numCrimeTypes = size(x{1},2) ;
    numMonths = length(x) ;
    eachCrimeData = cell(numCrimeTypes,1) ;    % there are 17 types of crimes
    overalCrimeDate = cell(numMonths,1) ;
    
    xSz = length(min(s{1}(:,1)):0.002:max(s{1}(:,1))) ;
    ySz = length(min(s{1}(:,2)):0.002:max(s{1}(:,2))) ;
    overalCrimeDate(:) = {zeros(ySz,xSz)} ;
    
    idx1 = round( (s{1}(:,1)-min(s{1}(:,1)))/(0.002)+1 ) ;
    idx2 = round( (s{1}(:,2)-min(s{1}(:,2)))/(0.002)+1 ) ;
    
    for crmCnt=1:numCrimeTypes      % it iterates over different types of crime
        eachCrimeData{crmCnt} = cell(numMonths,1) ;
        for mntCnt=1:numMonths
            eachCrimeData{crmCnt}{mntCnt} = zeros(ySz,xSz) ;
            eachCrimeData{crmCnt}{mntCnt}(sub2ind(size(eachCrimeData{crmCnt}{mntCnt}),idx2,idx1)) = ...
                     x{mntCnt}(:,crmCnt) ;
                 
            overalCrimeDate{mntCnt} = overalCrimeDate{mntCnt} + ...
                                     eachCrimeData{crmCnt}{mntCnt} ;    
        end
    end
    
    
end
