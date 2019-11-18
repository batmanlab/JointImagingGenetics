%% BLOCK 1 : setting up experiment: where things are and which libraries/paths to add and formats
close all
clear all
clc

lhCortexThicknessFn = '/om/user/ysa/kayhandata/adni/ADNI_lh_cortex_thick.csv' ; 
rhCortexThicknessFn = '/om/user/ysa/kayhandata/adni/ADNI_rh_cortex_thick.csv' ; 
subCorticalVolFn = '/om/user/ysa/kayhandata/adni/ADNI_aseg_vols.csv' ; 
classLabelFn = '/om/user/ysa/kayhandata/adni/ADNI_diagnosis.txt' ;
apoeVariantFn = '/om/user/ysa/kayhandata/adni/APOE.csv' ;
mertMatFn = '/om/user/ysa/kayhandata/adni/adni_thickness_data_for_kayhan.mat' ;



% format of the csv files
cortexThickStringFormat = '%s ' ;
for cnt=1:34   % 34 is the number of elements in this set
    cortexThickStringFormat = [cortexThickStringFormat ' %f '] ;
end

subCorticalVolStringFormat = '%s ' ;
for cnt=1:26   % 26 is the number of elements in this set
    subCorticalVolStringFormat = [subCorticalVolStringFormat ' %f '] ;
end


% header does not include the word "Subject"
lhCortexThickStringHeader = {'bankssts','caudalanteriorcingulate','caudalmiddlefrontal','cuneus',...
    'entorhinal','fusiform','inferiorparietal','inferiortemporal','isthmuscingulate','lateraloccipital',...
    'lateralorbitofrontal','lingual','medialorbitofrontal','middletemporal','parahippocampal','paracentral',...
    'parsopercularis','parsorbitalis','parstriangularis','pericalcarine','postcentral','posteriorcingulate',...
    'precentral','precuneus','rostralanteriorcingulate','rostralmiddlefrontal','superiorfrontal','superiorparietal',...
    'superiortemporal','supramarginal','frontalpole','temporalpole','transversetemporal','insula'} ;
for cnt=1:length(lhCortexThickStringHeader), lhCortexThickStringHeader{cnt} = ['Left-' lhCortexThickStringHeader{cnt}] ; end 


rhCortexThickStringHeader = {'bankssts','caudalanteriorcingulate','caudalmiddlefrontal','cuneus','entorhinal','fusiform',...
    'inferiorparietal','inferiortemporal','isthmuscingulate','lateraloccipital','lateralorbitofrontal','lingual','medialorbitofrontal',...
    'middletemporal','parahippocampal','paracentral','parsopercularis','parsorbitalis','parstriangularis','pericalcarine',...
    'postcentral','posteriorcingulate','precentral','precuneus','rostralanteriorcingulate','rostralmiddlefrontal','superiorfrontal',...
    'superiorparietal','superiortemporal','supramarginal','frontalpole','temporalpole','transversetemporal','insula'} ;
for cnt=1:length(rhCortexThickStringHeader), rhCortexThickStringHeader{cnt} = ['Right-' rhCortexThickStringHeader{cnt}] ; end 


subCorticalVolStringHeader = {'Left-Cerebral-White-Matter','Left-Cerebral-Cortex','Left-Lateral-Ventricle','Left-Inf-Lat-Vent',...
                   'Left-Cerebellum-White-Matter','Left-Cerebellum-Cortex','Left-Thalamus-Proper','Left-Caudate',...
                   'Left-Putamen','Left-Pallidum','3rd-Ventricle','4th-Ventricle','Left-Hippocampus','Left-Amygdala',...
                   'Right-Cerebral-White-Matter','Right-Cerebral-Cortex','Right-Lateral-Ventricle','Right-Inf-Lat-Vent',...
                   'Right-Cerebellum-White-Matter','Right-Cerebellum-Cortex','Right-Thalamus-Proper','Right-Caudate',...
                   'Right-Putamen','Right-Pallidum','Right-Hippocampus','ICV'} ;


fprintf('** Sanity check: Are the order of the structures in the left and right hemisphere tables the same ? %d \n',...
    isequal(rhCortexThickStringHeader  ,lhCortexThickStringHeader)) ;



%% BLOCK 2: Reading data :

% 0. Read the class label and only keep subjects stayed Normal (AD) both at
% the base and current time for training
%  1:control, 2: mci and 3: AD 
fid = fopen(classLabelFn,'rt') ;
M = textscan(fid, '%s %s %d %d','Delimiter',',','CollectOutput',1,'HeaderLines',1) ;
fclose(fid) ;

diagnosisLblID = {} ;
diagnosisLbl = [] ;
for cnt=1:length(M{1})
    if  ( M{2}(cnt,1) == M{2}(cnt,2) )     % if the status of a patient is consistent between base and current visit  
        if ( M{2}(cnt,1) == 1 )
            diagnosisLblID = [diagnosisLblID; M{1}(cnt,1) ] ;
            diagnosisLbl = [diagnosisLbl; 0] ;
        elseif  ( M{2}(cnt,1) == 3 )
            diagnosisLblID = [diagnosisLblID; M{1}(cnt,1) ] ;
            diagnosisLbl = [diagnosisLbl; 1] ;
        end
    end
end

%   1.  Read the thickness file:
%   PROJECT_SOURCE/data/thickness/ADNI_[l/r]h_cortex_thick.csv   *OR*  /data/vision/polina/projects/ADNI/data/processing/thickness_data
% read the left hemisphere data
fid = fopen(lhCortexThicknessFn,'rt') ;
M = textscan(fid, cortexThickStringFormat,'Delimiter',',','CollectOutput',1,'HeaderLines',1) ;
fclose(fid) ;

lhIDs = M{1} ;
lhCortexThicknessMatrix = M{2} ;
lhCortexThicknessDiag = [] ;  % diagnosis of sujects
idx = [] ;  % subjects to remove: either they are MCI or we don't have class label for
for cnt=1:length(lhIDs)
    if ~ismember(strtrim(lhIDs{cnt}),diagnosisLblID) % lhIDs{cnt} iterates over each ID, returns the index of the IDs in lhIDs that are not in diagnosisLblID
        idx = [idx cnt] ; 
    else
        [~,~,ii] = intersect(lhIDs{cnt},diagnosisLblID) ; % else return the index
        lhCortexThicknessDiag = [lhCortexThicknessDiag ; diagnosisLbl(ii) ] ; 
    end
end
lhIDs(idx) = [] ;
lhCortexThicknessMatrix(idx,:) = [] ;

% read the right hemisphere data
fid = fopen(rhCortexThicknessFn,'rt') ;
M = textscan(fid, cortexThickStringFormat,'Delimiter',',','CollectOutput',1,'HeaderLines',1) ;
fclose(fid) ;

rhIDs = M{1} ;
rhCortexThicknessMatrix = M{2} ;
rhCortexThicknessDiag = [] ;  % diagnosis of sujects
idx = [] ;  % subjects to remove: either they are MCI or we don't have class label for
for cnt=1:length(rhIDs)
    if ~ismember(strtrim(rhIDs{cnt}),diagnosisLblID)
        idx = [idx cnt] ;
    else
        [~,~,ii] = intersect(rhIDs{cnt},diagnosisLblID) ;
        rhCortexThicknessDiag = [rhCortexThicknessDiag ; diagnosisLbl(ii) ] ;
    end
end
rhIDs(idx) = [] ;
rhCortexThicknessMatrix(idx,:) = [] ;

%  2. read the subcortical volume
fid = fopen(subCorticalVolFn,'rt') ;
M = textscan(fid, subCorticalVolStringFormat,'Delimiter',',','CollectOutput',1,'HeaderLines',1) ;
fclose(fid) ;

subCorticalVolIDs = M{1} ;
subCorticalVolMatrix = M{2} ;
subCorticalVolDiag = [] ;  % diagnosis of sujects
idx = [] ;  % subjects to remove: either they are MCI or we don't have class label for
for cnt=1:length(subCorticalVolIDs)
    if ~ismember(strtrim(subCorticalVolIDs{cnt}),diagnosisLblID)
        idx = [idx cnt] ;
    else
        [~,~,ii] = intersect(subCorticalVolIDs{cnt},diagnosisLblID) ;
        subCorticalVolDiag = [subCorticalVolDiag ; diagnosisLbl(ii) ] ;
    end
end
subCorticalVolIDs(idx) = [] ;
subCorticalVolMatrix(idx,:) = [] ;


%  3. Read APOE data, it is located at:
%  /data/vision/polina/projects/ADNI/data/processing/thickness_data/APOE.csv
fid = fopen(apoeVariantFn,'rt') ;
M = textscan(fid, '%s %s %f %f','Delimiter',',','CollectOutput',1,'HeaderLines',1) ;
fclose(fid) ;

apoeIDs = M{1}(:,1) ;
apoeMatrix = M{2} ;
idx = [] ;  % subjects to remove: either they are MCI or we don't have class label for
for cnt=1:length(apoeIDs)
    if ~ismember(strtrim(apoeIDs{cnt}),diagnosisLblID)
        idx = [idx cnt] ;
    else
        [~,~,ii] = intersect(apoeIDs{cnt},diagnosisLblID) ;
    end
end
apoeIDs(idx) = [] ;
apoeMatrix(idx,:) = [] ;


% extract the covariates
covList = {'age','educ','sex','handedness'} ;
mertData = load(mertMatFn) ;
covCell = {} ;
covData = zeros(length(diagnosisLblID),length(covList)) ;

% firt make sure you can find all subjects
tmp_diagnosisLblID_flag = zeros(length(diagnosisLblID),1) ;

for  cnt=1:length(diagnosisLblID)
    for jj=1:length(mertData.adni_data.sid_cell)
        if mertData.adni_data.sid_cell{jj} == ['S_' diagnosisLblID{cnt}]
            tmp_diagnosisLblID_flag(cnt) = 1 ;
        end
    end   
end
assert(all(tmp_diagnosisLblID_flag),'we couldn''t find some subject in the covariate list') ;
clear  tmp_diagnosisLblID  tmp_diagnosisLblID_flag
for  cnt=1:length(diagnosisLblID)
    for jj=1:length(mertData.adni_data.sid_cell)
        if mertData.adni_data.sid_cell{jj} == ['S_' diagnosisLblID{cnt}]
            break
        end
    end
    
    for covCnt=1:length(covList)
        covData(cnt,covCnt) = mertData.adni_data.(covList{covCnt})(jj)  ;
    end
end



% make endoPhenMatrix and normalize accordin to the normal class
endoPhenLabel = [ lhCortexThickStringHeader  rhCortexThickStringHeader  subCorticalVolStringHeader] ;
endoPhenMatrix = [ lhCortexThicknessMatrix   rhCortexThicknessMatrix  subCorticalVolMatrix] ;
endoPhenMatrix = bsxfun(@minus, endoPhenMatrix, mean(endoPhenMatrix(diagnosisLbl==0,:))) ;
endoPhenMatrix = bsxfun(@times, endoPhenMatrix, 1./std(endoPhenMatrix(diagnosisLbl==0,:))) ;


fprintf('Sanity check: order of IDs the same in two thickness tables is the same? %d \n', isequal(lhIDs,rhIDs)) ;
fprintf('Sanity check: class label for right and left and right hemis. is the same ? %d \n', all(rhCortexThicknessDiag==lhCortexThicknessDiag) ) ;
fprintf('Sanity check: order of IDs the same in two thickness tables is the same? %d \n', isequal(lhIDs,subCorticalVolIDs)) ;
fprintf('Sanity check: order of IDs APOE csv file and one of thickness tables is the same? %d \n', isequal(lhIDs,apoeIDs)) ;



%%  BLOCK 3: Reading genotype data from New QC-ed data (Chr19),  
fprintf('Working on the new Genotyped data ... \n \n'); 
% read subjInfo
addpath('~/code/Projects/genomicImaging/src/Utils/')
fn = '/data/vision/polina/projects/ADNI/work/kayhan/Journal/Genotype/ADNI_lastQCb37_Chr19.fam'  ; 
[subInfoCell_newQC_Chr19, subInfo_newQC_Chr19] = readFAM(fn) ;
subInfo2_newQC_Chr19 = repmat(struct('idx',[],'familyID',[], 'individualID', [] ,  'paternalID',[], 'maternalID', [], 'sex', [], 'phenotype',[]) ,...
                  length(subCorticalVolIDs),1) ;

% read snpInfo
fn = '/data/vision/polina/projects/ADNI/work/kayhan/Journal/Genotype/ADNI_lastQCb37_Chr19.bim'  ; 
[snpInfoCell_newQC_Chr19, snpInfo_newQC_Chr19] = readBIM(fn) ;

%  read the binary file
fn =  '/data/vision/polina/projects/ADNI/work/kayhan/Journal/Genotype/ADNI_lastQCb37_Chr19.h5'  ; 
genotype_newQC_Chr19 = hdf5read(fn,'genotype') ;
genotype2_newQC_Chr19 = zeros(length(subCorticalVolIDs),size(genotype_newQC_Chr19,2),'single') ;


% clean up the subject not in the list
missingIdx_newQC = [] ;
for cnt1=1:size(genotype2_newQC_Chr19)
    for cnt2=1:length(subInfo_newQC_Chr19)
        if  isequal(subCorticalVolIDs{cnt1},  subInfo_newQC_Chr19(cnt2).individualID(end-3:end) )
            genotype2_newQC_Chr19(cnt1,:) = single(genotype_newQC_Chr19(cnt2,:)) ;
            subInfo2_newQC_Chr19(cnt1) = subInfo_newQC_Chr19(cnt2) ;
            break
        end
    end
    % make sure that it finds the subject
    %assert( ~isempty(subInfo2(cnt1).idx) , ['subject ' subCorticalVolIDs{cnt1} ' not found !! '] ) ; 
    if isempty(subInfo2_newQC_Chr19(cnt1).idx)
        missingIdx_newQC = [missingIdx_newQC ; cnt1] ;
        warning(['subject ' subCorticalVolIDs{cnt1} ' with subject label ' num2str(diagnosisLbl(cnt1)) ' not found !! ']) ;
    end
end

%%  BLOCK 4 : Reading genotype data from Old QC-ed data (Chr19),  
fprintf('Working on the old Genotyped data ... \n \n'); 
% read subjInfo
addpath('~/code/Projects/genomicImaging/src/Utils/')
fn = '/data/vision/polina/projects/ADNI/work/kayhan/Journal/Genotype/adni_bed2_ceu_flipped_clean_Chr19.fam'  ; 
[subInfoCell_oldQC_Chr19, subInfo_oldQC_Chr19] = readFAM(fn) ;
subInfo2_oldQC_Chr19 = repmat(struct('idx',[],'familyID',[], 'individualID', [] ,  'paternalID',[], 'maternalID', [], 'sex', [], 'phenotype',[]) ,...
                  length(subCorticalVolIDs),1) ;

% read snpInfo
fn = '/data/vision/polina/projects/ADNI/work/kayhan/Journal/Genotype/adni_bed2_ceu_flipped_clean_Chr19.bim'  ; 
[snpInfoCell_oldQC_Chr19, snpInfo_oldQC_Chr19] = readBIM(fn) ;

%  read the binary file
fn =  '/data/vision/polina/projects/ADNI/work/kayhan/Journal/Genotype/adni_bed2_ceu_flipped_clean_Chr19.h5'  ; 
genotype_oldQC_Chr19 = hdf5read(fn,'genotype') ;
genotype2_oldQC_Chr19 = zeros(length(subCorticalVolIDs),size(genotype_oldQC_Chr19,2),'single') ;


% clean up the subject not in the list
missingIdx_oldQC = [] ;
for cnt1=1:size(genotype2_oldQC_Chr19)
    for cnt2=1:length(subInfo_oldQC_Chr19)
        if  isequal(subCorticalVolIDs{cnt1},  subInfo_oldQC_Chr19(cnt2).individualID(end-3:end) )
            genotype2_oldQC_Chr19(cnt1,:) = single(genotype_oldQC_Chr19(cnt2,:)) ;
            subInfo2_oldQC_Chr19(cnt1) = subInfo_oldQC_Chr19(cnt2) ;
            break
        end
    end
    % make sure that it finds the subject
    %assert( ~isempty(subInfo2(cnt1).idx) , ['subject ' subCorticalVolIDs{cnt1} ' not found !! '] ) ; 
    if isempty(subInfo2_oldQC_Chr19(cnt1).idx)
        missingIdx_oldQC = [missingIdx_oldQC ; cnt1] ;
        warning(['subject ' subCorticalVolIDs{cnt1} ' with subject label ' num2str(diagnosisLbl(cnt1)) ' not found !! ']) ;
    end
end

%%  BLOCK 4.1: Reading genotype data from nature imputed list 
fprintf('Working on the Nature imputed list data ... \n \n'); 
% read subjInfo
addpath('/om/user/ysa/kayhandata/adni/')
fn = '/om/user/ysa/kayhandata/adni/nature.snp_imputedList_onlyADvsCN.fam';
[subInfoCell_natureImput_ChrAll, subInfo_natureImput_ChrAll] = readFAM(fn) ;
subInfo2_natureImput_ChrAll = repmat(struct('idx',[],'familyID',[], 'individualID', [] ,  'paternalID',[], 'maternalID', [], 'sex', [], 'phenotype',[]) ,...
                  length(subCorticalVolIDs),1) ;

% read snpInfo
fn = '/om/user/ysa/kayhandata/adni/nature.snp_imputedList_onlyADvsCN.bim'  ;
[snpInfoCell_natureImput_ChrAll, snpInfo_natureImput_ChrAll] = readBIM(fn) ;

%  read the binary file
fn =  '/om/user/ysa/kayhandata/adni/nature.snp_imputedList_onlyADvsCN.h5'  ; 
genotype_natureImput_ChrAll = hdf5read(fn,'genotype') ;
genotype2_natureImput_ChrAll = zeros(length(subCorticalVolIDs),size(genotype_natureImput_ChrAll,2),'single') ;


% clean up the subject not in the list
missingIdx_natureImput = [] ;
for cnt1=1:size(genotype2_natureImput_ChrAll)
    for cnt2=1:length(subInfo_natureImput_ChrAll)
        if  isequal(subCorticalVolIDs{cnt1},  subInfo_natureImput_ChrAll(cnt2).individualID(end-3:end) )
            genotype2_natureImput_ChrAll(cnt1,:) = single(genotype_natureImput_ChrAll(cnt2,:)) ;
            subInfo2_natureImput_ChrAll(cnt1) = subInfo_natureImput_ChrAll(cnt2) ;
            break
        end
    end
    % make sure that it finds the subject
    %assert( ~isempty(subInfo2(cnt1).idx) , ['subject ' subCorticalVolIDs{cnt1} ' not found !! '] ) ; 
    if isempty(subInfo2_natureImput_ChrAll(cnt1).idx)
        missingIdx_natureImput = [missingIdx_natureImput ; cnt1] ;
        warning(['subject ' subCorticalVolIDs{cnt1} ' with subject label ' num2str(diagnosisLbl(cnt1)) ' not found !! ']) ;
    end
end



%%  BLOCK 4.2: Reading genotype data from unimputed list of SNPs with low p-value
fprintf('Working on the low p-value of unimputed data ... \n \n'); 
% read subjInfo
addpath('~/code/Projects/genomicImaging/src/Utils/')
fn = '/om/user/ysa/kayhandata/adni/lowPVal.snp_orgList_onlyADvsCN.fam'  ; 
[subInfoCell_unimputeLowPVal_ChrAll, subInfo_unimputeLowPVal_ChrAll] = readFAM(fn) ;
subInfo2_unimputeLowPVal_ChrAll = repmat(struct('idx',[],'familyID',[], 'individualID', [] ,  'paternalID',[], 'maternalID', [], 'sex', [], 'phenotype',[]) ,...
                  length(subCorticalVolIDs),1) ;

% read snpInfo
fn = '/om/user/ysa/kayhandata/adni/lowPVal.snp_orgList_onlyADvsCN.bim'  ;  
[snpInfoCell_unimputeLowPVal_ChrAll, snpInfo_unimputeLowPVal_ChrAll] = readBIM(fn) ;

%  read the binary file
fn =  '/om/user/ysa/kayhandata/adni/lowPVal.snp_orgList_onlyADvsCN.h5'  ; 
genotype_unimputeLowPVal_ChrAll = hdf5read(fn,'genotype') ;
genotype2_unimputeLowPVal_ChrAll = zeros(length(subCorticalVolIDs),size(genotype_unimputeLowPVal_ChrAll,2),'single') ;


% clean up the subject not in the list
missingIdx_unimputeLowPVal = [] ;
for cnt1=1:size(genotype2_unimputeLowPVal_ChrAll)
    for cnt2=1:length(subInfo_unimputeLowPVal_ChrAll)
        if  isequal(subCorticalVolIDs{cnt1},  subInfo_unimputeLowPVal_ChrAll(cnt2).individualID(end-3:end) )
            genotype2_unimputeLowPVal_ChrAll(cnt1,:) = single(genotype_unimputeLowPVal_ChrAll(cnt2,:)) ;
            subInfo2_unimputeLowPVal_ChrAll(cnt1) = subInfo_unimputeLowPVal_ChrAll(cnt2) ;
            break
        end
    end
    % make sure that it finds the subject
    %assert( ~isempty(subInfo2(cnt1).idx) , ['subject ' subCorticalVolIDs{cnt1} ' not found !! '] ) ; 
    if isempty(subInfo2_unimputeLowPVal_ChrAll(cnt1).idx)
        missingIdx_unimputeLowPVal = [missingIdx_unimputeLowPVal ; cnt1] ;
        warning(['subject ' subCorticalVolIDs{cnt1} ' with subject label ' num2str(diagnosisLbl(cnt1)) ' not found !! ']) ;
    end
end



%%  BLOCK 4.3: Reading genotype data from independent SNPS which found in the imputed list
fprintf('Working on the independent list of SNPs ... \n \n'); 
% read subjInfo
addpath('~/code/Projects/genomicImaging/src/Utils/')
fn = '/om/user/ysa/kayhandata/adni/independent.snp_imputedList_onlyADvsCN.fam'  ; 
[subInfoCell_independentSNPs_ChrAll, subInfo_independentSNPs_ChrAll] = readFAM(fn) ;
subInfo2_independentSNPs_ChrAll = repmat(struct('idx',[],'familyID',[], 'individualID', [] ,  'paternalID',[], 'maternalID', [], 'sex', [], 'phenotype',[]) ,...
                  length(subCorticalVolIDs),1) ;

% read snpInfo
fn = '/om/user/ysa/kayhandata/adni/independent.snp_imputedList_onlyADvsCN.bim'  ;  
[snpInfoCell_independentSNPs_ChrAll, snpInfo_independentSNPs_ChrAll] = readBIM(fn) ;

%  read the binary file
fn =  '/om/user/ysa/kayhandata/adni/independent.snp_imputedList_onlyADvsCN.h5'  ; 
genotype_independentSNPs_ChrAll = hdf5read(fn,'genotype') ;
genotype2_independentSNPs_ChrAll = zeros(length(subCorticalVolIDs),size(genotype_independentSNPs_ChrAll,2),'single') ;


% clean up the subject not in the list
missingIdx_independentSNPs = [] ;
for cnt1=1:size(genotype2_independentSNPs_ChrAll)
    for cnt2=1:length(subInfo_independentSNPs_ChrAll)
        if  isequal(subCorticalVolIDs{cnt1},  subInfo_independentSNPs_ChrAll(cnt2).individualID(end-3:end) )
            genotype2_independentSNPs_ChrAll(cnt1,:) = single(genotype_independentSNPs_ChrAll(cnt2,:)) ;
            subInfo2_independentSNPs_ChrAll(cnt1) = subInfo_independentSNPs_ChrAll(cnt2) ;
            break
        end
    end
    % make sure that it finds the subject
    %assert( ~isempty(subInfo2(cnt1).idx) , ['subject ' subCorticalVolIDs{cnt1} ' not found !! '] ) ; 
    if isempty(subInfo2_independentSNPs_ChrAll(cnt1).idx)
        missingIdx_independentSNPs = [missingIdx_independentSNPs ; cnt1] ;
        warning(['subject ' subCorticalVolIDs{cnt1} ' with subject label ' num2str(diagnosisLbl(cnt1)) ' not found !! ']) ;
    end
end

%%  BLOCK 4.4: Reading genotype data from a larger list from the nature imputed list 
fprintf('Working on the Nature imputed list data ... \n \n'); 
% read subjInfo
addpath('~/code/Projects/genomicImaging/src/Utils/')
fn = '/om/user/ysa/kayhandata/adni/nature.snp_imputedList_onlyADvsCN.fam'  ; 
[subInfoCell_natureImput_largeList_ChrAll, subInfo_natureImput_largeList_ChrAll] = readFAM(fn) ;
subInfo2_natureImput_largeList_ChrAll = repmat(struct('idx',[],...
                                            'familyID',[],...
                                            'individualID',[],...
                                            'paternalID',[],...
                                            'maternalID', [],...
                                            'sex', [],...
                                            'phenotype',[]) ,...
                                            length(subCorticalVolIDs),1) ;

% read snpInfo
fn = '/om/user/ysa/kayhandata/adni/nature.snp_imputedList_onlyADvsCN.bim'  ;  
[snpInfoCell_natureImput_largeList_ChrAll, snpInfo_natureImput_largeList_ChrAll] = readBIM(fn) ;

%  read the binary file
fn =  '/om/user/ysa/kayhandata/adni/nature.snp_imputedList_onlyADvsCN.h5'  ; 
genotype_natureImput_largeList_ChrAll = hdf5read(fn,'genotype') ;
genotype2_natureImput_largeList_ChrAll = zeros(length(subCorticalVolIDs),size(genotype_natureImput_largeList_ChrAll,2),'single') ;


% clean up the subject not in the list
missingIdx_natureImput_largeList = [] ;
for cnt1=1:size(genotype2_natureImput_largeList_ChrAll)
    for cnt2=1:length(subInfo_natureImput_largeList_ChrAll)
        if  isequal(subCorticalVolIDs{cnt1},  subInfo_natureImput_largeList_ChrAll(cnt2).individualID(end-3:end) )
            genotype2_natureImput_largeList_ChrAll(cnt1,:) = single(genotype_natureImput_largeList_ChrAll(cnt2,:)) ;
            subInfo2_natureImput_largeList_ChrAll(cnt1) = subInfo_natureImput_largeList_ChrAll(cnt2) ;
            break
        end
    end
    % make sure that it finds the subject
    %assert( ~isempty(subInfo2(cnt1).idx) , ['subject ' subCorticalVolIDs{cnt1} ' not found !! '] ) ; 
    if isempty(subInfo2_natureImput_largeList_ChrAll(cnt1).idx)
        missingIdx_natureImput_largeList = [missingIdx_natureImput_largeList ; cnt1] ;
        warning(['subject ' subCorticalVolIDs{cnt1} ' with subject label ' num2str(diagnosisLbl(cnt1)) ' not found !! ']) ;
    end
end



%% BLOCK 5: Save the data into several .MAT files  *WARNING* IT IS TIME CONSUMING!
%endoPhenMatrix_rHipp = endoPhenMatrix(:,93) ;
%endoPhenMatrix_lHipp = endoPhenMatrix(:,81) ;
%endoPhenMatrix_bothHipp = endoPhenMatrix(:,[81 93]) ;
%endoPhenMatrix_topStruct = endoPhenMatrix(:,[5    15    39    69    70    71    72    73    74    75    76    77    78    79    80    81  82    83    84    85    86    87    88    89    90    91    92    93    94]) ;

% rank imaging features according to their relevance to Y and make different set
[IDX,Z ] = rankfeatures(endoPhenMatrix',diagnosisLbl) ;

MATLABPATH = '/data/vision/polina/projects/ADNI/work/kayhan/Journal/MATLAB_Data' ;
prefixFileNameList = {'oldQC-Chr19',...
                      'newQC-Chr19',...
                      'natureImput-ChrAll',...
                      'unimputedLowPVal-ChrAll',...
                      'independentSNPs-ChrAll',...
                      'natureImput-largeList-ChrAll'} ;
                  
missingIdxList = {'missingIdx_oldQC',...
                  'missingIdx_newQC',...
                  'missingIdx_natureImput',...
                  'missingIdx_unimputeLowPVal',...
                  'missingIdx_independentSNPs',...
                  'missingIdx_natureImput_largeList'} ;
              
genotypeList = {'genotype2_oldQC_Chr19',...
                'genotype2_newQC_Chr19',...
                'genotype2_natureImput_ChrAll',...
                'genotype2_unimputeLowPVal_ChrAll',...
                'genotype2_independentSNPs_ChrAll',...
                'genotype2_natureImput_largeList_ChrAll'} ;
            
subInfoList = {'subInfo2_oldQC_Chr19',...
               'subInfo2_newQC_Chr19',...
               'subInfo2_natureImput_ChrAll',...
               'subInfo2_unimputeLowPVal_ChrAll',...
               'subInfo2_independentSNPs_ChrAll',...
               'subInfo2_natureImput_largeList_ChrAll'} ;
           
snpInfoList = {'snpInfo_oldQC_Chr19',...
               'snpInfo_newQC_Chr19',...
               'snpInfo_natureImput_ChrAll',...
               'snpInfo_unimputeLowPVal_ChrAll',...
               'snpInfo_independentSNPs_ChrAll',...
               'snpInfo_natureImput_largeList_ChrAll'} ;
%endoPhenList = {'endoPhenMatrix','endoPhenMatrix_rHipp','endoPhenMatrix_lHipp','endoPhenMatrix_bothHipp','endoPhenMatrix_topStruct'} ;

for eCnt=94  %1:length(IDX)
    eCnt
    for cnt=1:length(prefixFileNameList)
        missingIdx = eval(missingIdxList{cnt}) ;
        genotype = eval(genotypeList{cnt}) ;
        subInfo = eval(subInfoList{cnt}) ;
        snpInfo = eval(snpInfoList{cnt}) ;
        %I = eval(endoPhenList{eCnt}) ;  % endophenotype
        I = endoPhenMatrix(:,IDX(1:eCnt)) ;  % endophenotype
        ePhenLabel = endoPhenLabel(IDX(1:eCnt)) ;
        
        
        missingIDs = diagnosisLblID(missingIdx) ;
        subInfo(missingIdx) = [] ;
        snpInfo(end+1).snpID = 'e3' ;
        snpInfo(end).chrNum = 19 ;
        snpInfo(end+1).snpID = 'e4' ;
        snpInfo(end).chrNum = 19 ;
        
        y = diagnosisLbl ;
        y(missingIdx) = [] ;
        
        CovDATA = covData ;
        CovDATA(missingIdx,:) = [] ;
        
        APOE = apoeMatrix ;
        APOE(missingIdx,:) = [] ;
        G = genotype ;
        G(missingIdx,:) = [] ;
        G = [double(G) APOE] ;
        % getting rid of missing values (=3)
        h =  @(x)mean(x(x~=3)) ;
        for jj=1:size(G,2)
            G( G(:,jj)==3 ,jj) = h(G(:,jj)) ;
        end
        % centering genetic data
        G = bsxfun(@minus,mean(G),G) ;
        
        
        I(missingIdx,:) = [] ;
        % centering I with respect to the normal subjects
        mu = mean(I(y==0,:)) ;
        s = std(I(y==0,:)) ;
        I = bsxfun(@minus, I,mu) ;
        I = bsxfun(@times, I, 1./s) ;
        
        fn = [prefixFileNameList{cnt} '_Data' num2str(eCnt) '.mat' ] ;
        save([MATLABPATH '/' fn],'I','G','y','missingIDs','APOE','snpInfo','subInfo','ePhenLabel','CovDATA','covList') ;
        clear I G y missingIdx   genotype   subInfo  snpInfo  APOE  ePhenLabel CovDATA
    end
end


%% BLOCK 5.1: Save the data into several .MAT files WITHOUT centeing the data *WARNING* IT IS TIME CONSUMING!
%endoPhenMatrix_rHipp = endoPhenMatrix(:,93) ;
%endoPhenMatrix_lHipp = endoPhenMatrix(:,81) ;
%endoPhenMatrix_bothHipp = endoPhenMatrix(:,[81 93]) ;
%endoPhenMatrix_topStruct = endoPhenMatrix(:,[5    15    39    69    70    71    72    73    74    75    76    77    78    79    80    81  82    83    84    85    86    87    88    89    90    91    92    93    94]) ;

% rank imaging features according to their relevance to Y and make different set
[IDX,Z ] = rankfeatures(endoPhenMatrix',diagnosisLbl) ;

MATLABPATH = '/data/vision/polina/projects/ADNI/work/kayhan/Journal/MATLAB_Data' ;
prefixFileNameList = {'oldQC-Chr19',...
                      'newQC-Chr19',...
                      'natureImput-ChrAll',...
                      'unimputedLowPVal-ChrAll',...
                      'independentSNPs-ChrAll',...
                      'natureImput-largeList-ChrAll'} ;
                  
missingIdxList = {'missingIdx_oldQC',...
                  'missingIdx_newQC',...
                  'missingIdx_natureImput',...
                  'missingIdx_unimputeLowPVal',...
                  'missingIdx_independentSNPs',...
                  'missingIdx_natureImput_largeList'} ;
              
genotypeList = {'genotype2_oldQC_Chr19',...
                'genotype2_newQC_Chr19',...
                'genotype2_natureImput_ChrAll',...
                'genotype2_unimputeLowPVal_ChrAll',...
                'genotype2_independentSNPs_ChrAll',...
                'genotype2_natureImput_largeList_ChrAll'} ;
            
subInfoList = {'subInfo2_oldQC_Chr19',...
               'subInfo2_newQC_Chr19',...
               'subInfo2_natureImput_ChrAll',...
               'subInfo2_unimputeLowPVal_ChrAll',...
               'subInfo2_independentSNPs_ChrAll',...
               'subInfo2_natureImput_largeList_ChrAll'} ;
           
snpInfoList = {'snpInfo_oldQC_Chr19',...
               'snpInfo_newQC_Chr19',...
               'snpInfo_natureImput_ChrAll',...
               'snpInfo_unimputeLowPVal_ChrAll',...
               'snpInfo_independentSNPs_ChrAll',...
               'snpInfo_natureImput_largeList_ChrAll'} ;
%endoPhenList = {'endoPhenMatrix','endoPhenMatrix_rHipp','endoPhenMatrix_lHipp','endoPhenMatrix_bothHipp','endoPhenMatrix_topStruct'} ;

for eCnt=94  %1:length(IDX)
    eCnt
    for cnt=1:length(prefixFileNameList)
        missingIdx = eval(missingIdxList{cnt}) ;
        genotype = eval(genotypeList{cnt}) ;
        subInfo = eval(subInfoList{cnt}) ;
        snpInfo = eval(snpInfoList{cnt}) ;
        %I = eval(endoPhenList{eCnt}) ;  % endophenotype
        I = endoPhenMatrix(:,IDX(1:eCnt)) ;  % endophenotype
        ePhenLabel = endoPhenLabel(IDX(1:eCnt)) ;
        
        
        missingIDs = diagnosisLblID(missingIdx) ;
        subInfo(missingIdx) = [] ;
        snpInfo(end+1).snpID = 'e3' ;
        snpInfo(end).chrNum = 19 ;
        snpInfo(end+1).snpID = 'e4' ;
        snpInfo(end).chrNum = 19 ;
        
        y = diagnosisLbl ;
        y(missingIdx) = [] ;
        
        CovDATA = covData ;
        CovDATA(missingIdx,:) = [] ;
        
        APOE = apoeMatrix ;
        APOE(missingIdx,:) = [] ;
        G = genotype ;
        G(missingIdx,:) = [] ;
        G = [double(G) APOE] ;
        % getting rid of missing values (=3)
        h =  @(x)mean(x(x~=3)) ;
        for jj=1:size(G,2)
            G( G(:,jj)==3 ,jj) = h(G(:,jj)) ;
        end
        % centering genetic data
        G = bsxfun(@minus,mean(G),G) ;
        
        
        I(missingIdx,:) = [] ;
        % centering I with respect to the normal subjects
        %mu = mean(I(y==0,:)) ;
        %s = std(I(y==0,:)) ;
        %I = bsxfun(@minus, I,mu) ;
        %I = bsxfun(@times, I, 1./s) ;
        
        fn = [prefixFileNameList{cnt} '_NoNormalCenter_Data' num2str(eCnt) '.mat' ] ;
        save([MATLABPATH '/' fn],'I','G','y','missingIDs','APOE','snpInfo','subInfo','ePhenLabel','CovDATA','covList') ;
        clear I G y missingIdx   genotype   subInfo  snpInfo  APOE  ePhenLabel CovDATA
    end
end

%% BLOCK 5.2: Remove the effect of the covariates on the non-Centered data and Save the data into several .MAT files
% *WARNING* IT IS TIME CONSUMING!
%endoPhenMatrix_rHipp = endoPhenMatrix(:,93) ;
%endoPhenMatrix_lHipp = endoPhenMatrix(:,81) ;
%endoPhenMatrix_bothHipp = endoPhenMatrix(:,[81 93]) ;
%endoPhenMatrix_topStruct = endoPhenMatrix(:,[5    15    39    69    70    71    72    73    74    75    76    77    78    79    80    81  82    83    84    85    86    87    88    89    90    91    92    93    94]) ;
%}

% rank imaging features according to their relevance to Y and make different set
[IDX,Z ] = rankfeatures(endoPhenMatrix',diagnosisLbl) ;

MATLABPATH = '/data/vision/polina/projects/ADNI/work/kayhan/Journal/MATLAB_Data/covRemovedMatFiles' ;
prefixFileNameList = {'oldQC-Chr19',...
                      'newQC-Chr19',...
                      'natureImput-ChrAll',...
                      'unimputedLowPVal-ChrAll',...
                      'independentSNPs-ChrAll',...
                      'natureImput-largeList-ChrAll'} ;
                  
missingIdxList = {'missingIdx_oldQC',...
                  'missingIdx_newQC',...
                  'missingIdx_natureImput',...
                  'missingIdx_unimputeLowPVal',...
                  'missingIdx_independentSNPs',...
                  'missingIdx_natureImput_largeList'} ;
              
genotypeList = {'genotype2_oldQC_Chr19',...
                'genotype2_newQC_Chr19',...
                'genotype2_natureImput_ChrAll',...
                'genotype2_unimputeLowPVal_ChrAll',...
                'genotype2_independentSNPs_ChrAll',...
                'genotype2_natureImput_largeList_ChrAll'} ;
            
subInfoList = {'subInfo2_oldQC_Chr19',...
               'subInfo2_newQC_Chr19',...
               'subInfo2_natureImput_ChrAll',...
               'subInfo2_unimputeLowPVal_ChrAll',...
               'subInfo2_independentSNPs_ChrAll',...
               'subInfo2_natureImput_largeList_ChrAll'} ;
           
snpInfoList = {'snpInfo_oldQC_Chr19',...
               'snpInfo_newQC_Chr19',...
               'snpInfo_natureImput_ChrAll',...
               'snpInfo_unimputeLowPVal_ChrAll',...
               'snpInfo_independentSNPs_ChrAll',...
               'snpInfo_natureImput_largeList_ChrAll'} ;
%endoPhenList = {'endoPhenMatrix','endoPhenMatrix_rHipp','endoPhenMatrix_lHipp','endoPhenMatrix_bothHipp','endoPhenMatrix_topStruct'} ;

for eCnt=94  %1:length(IDX)
    eCnt
    for cnt=1:length(prefixFileNameList)
        missingIdx = eval(missingIdxList{cnt}) ;
        genotype = eval(genotypeList{cnt}) ;
        subInfo = eval(subInfoList{cnt}) ;
        snpInfo = eval(snpInfoList{cnt}) ;
        %I = eval(endoPhenList{eCnt}) ;  % endophenotype
        I = endoPhenMatrix(:,IDX(1:eCnt)) ;  % endophenotype
        ePhenLabel = endoPhenLabel(IDX(1:eCnt)) ;
        
        
        missingIDs = diagnosisLblID(missingIdx) ;
        subInfo(missingIdx) = [] ;
        snpInfo(end+1).snpID = 'e3' ;
        snpInfo(end).chrNum = 19 ;
        snpInfo(end+1).snpID = 'e4' ;
        snpInfo(end).chrNum = 19 ;
        
        y = diagnosisLbl ;
        y(missingIdx) = [] ;
        
        CovDATA = covData ;
        CovDATA(missingIdx,:) = [] ;
        
        APOE = apoeMatrix ;
        APOE(missingIdx,:) = [] ;
        G = genotype ;
        G(missingIdx,:) = [] ;
        G = [double(G) APOE] ;
        % getting rid of missing values (=3)
        h =  @(x)mean(x(x~=3)) ;
        for jj=1:size(G,2)
            G( G(:,jj)==3 ,jj) = h(G(:,jj)) ;
        end
        % centering genetic data
        G = bsxfun(@minus,mean(G),G) ;
        
        
        I(missingIdx,:) = [] ;
        % remove the effect of the covariates
        for  jj=1:size(I,2)
            glmModel = glmfit(CovDATA(y==0,:), ...
                              I(y==0,jj),...
                              'normal','link','identity') ;
            I(:,jj) = I(:,jj) - ...
                      glmval(glmModel, CovDATA , 'identity' ) ;   
                  
        end
        mu = mean(I(y==0,:)) ;
        s = std(I(y==0,:)) ;
        I = bsxfun(@minus, I,mu) ;
        I = bsxfun(@times, I, 1./s) ;
        
        fn = [prefixFileNameList{cnt} '_covRemoved_Data' num2str(eCnt) '.mat' ] ;
        save([MATLABPATH '/' fn],'I','G','y','missingIDs','APOE','snpInfo','subInfo','ePhenLabel','CovDATA','covList') ;
        clear I G y missingIdx   genotype   subInfo  snpInfo  APOE  ePhenLabel CovDATA
    end
end



%% BLOCK 6: on the normal group find the brain structures with highest genetic influence  *WARNING* IT IS TIME CONSUMING!
% these are the steps we need to perform
%   1. first run lasso for the whole regularization path using each
%   strucutre as the dependent variable and the genotype as the regressor.
%   Repeat this step for new and old QC.
%   2. Now, we have identified genotype influencing normal sub-population,
%   remove the effect genotype (based on normal population) and compute the
%   residual which should be used an endophenotype.

if exist('/data/vision/polina/projects/ADNI/work/kayhan/Journal/MATLAB_Data/workspaceAfterLasso.mat','file')   % this step is very time consuming, it the file exist, just load the workspace
    load('/data/vision/polina/projects/ADNI/work/kayhan/Journal/MATLAB_Data/workspaceAfterLasso.mat')
else
    % ---- new QC
    missingIdx = missingIdx_newQC ;
    
    y = diagnosisLbl ;
    y(missingIdx) = [] ;
    
    
    APOE = apoeMatrix ;
    APOE(missingIdx,:) = [] ;
    G = genotype2_newQC_Chr19 ;
    G(missingIdx,:) = [] ;
    G = [double(G) APOE] ;
    % getting rid of missing values (=3)
    h =  @(x)mean(x(x~=3)) ;
    for jj=1:size(G,2)
        G( G(:,jj)==3 ,jj) = h(G(:,jj)) ;
    end
    % centering genetic data
    G = bsxfun(@minus,mean(G),G) ;
    
    I = endoPhenMatrix ;
    I(missingIdx,:) = [] ;
    % centering I with respect to the normal subjects
    mu = mean(I(y==0,:)) ;
    s = std(I(y==0,:)) ;
    I = bsxfun(@minus, I,mu) ;
    I = bsxfun(@times, I, 1./s) ;
    
    B_newQC = zeros(size(G,2),size(I,2)) ;
    intercept_newQC = zeros(1,size(I,2)) ;
    DF_newQC = zeros(1,size(I,2)) ;
    for cnt=1:size(I,2)
        [B,FitInfo] = lasso(G(y==0,:), I(y==0,cnt),'CV',10) ;
        B_newQC(:,cnt) = B(:,FitInfo.IndexMinMSE) ;
        intercept_newQC(cnt) = FitInfo.Intercept(FitInfo.IndexMinMSE) ;
        DF_newQC(cnt) = FitInfo.DF(FitInfo.IndexMinMSE) ;
        if mod(cnt,floor(size(I,2)/10))==0
            fprintf('.') ;
        end
    end
    
    
    % -- old QC
    missingIdx = missingIdx_oldQC ;
    
    y = diagnosisLbl ;
    y(missingIdx) = [] ;
    
    
    APOE = apoeMatrix ;
    APOE(missingIdx,:) = [] ;
    G = genotype2_oldQC_Chr19 ;
    G(missingIdx,:) = [] ;
    G = [double(G) APOE] ;
    % getting rid of missing values (=3)
    h =  @(x)mean(x(x~=3)) ;
    for jj=1:size(G,2)
        G( G(:,jj)==3 ,jj) = h(G(:,jj)) ;
    end
    % centering genetic data
    G = bsxfun(@minus,mean(G),G) ;
    
    I = endoPhenMatrix ;
    I(missingIdx,:) = [] ;
    % centering I with respect to the normal subjects
    mu = mean(I(y==0,:)) ;
    s = std(I(y==0,:)) ;
    I = bsxfun(@minus, I,mu) ;
    I = bsxfun(@times, I, 1./s) ;
    
    B_oldQC = zeros(size(G,2),size(I,2)) ;
    intercept_oldQC = zeros(1,size(I,2)) ;
    DF_oldQC = zeros(1,size(I,2)) ;
    for cnt=1:size(I,2)
        [B,FitInfo] = lasso(G(y==0,:), I(y==0,cnt),'CV',10) ;
        B_oldQC(:,cnt) = B(:,FitInfo.IndexMinMSE) ;
        intercept_oldQC(cnt) = FitInfo.Intercept(FitInfo.IndexMinMSE) ;
        DF_oldQC(cnt) = FitInfo.DF(FitInfo.IndexMinMSE) ;
        if mod(cnt,floor(size(I,2)/10))==0
            fprintf('.') ;
        end
    end
end


% --- newQC : compute the residual endo-phenotype
endoPhenMatrix_residual_oldQC = zeros(size(endoPhenMatrix)) ;
missingIdx = missingIdx_oldQC ;
    
y = diagnosisLbl ;
y(missingIdx) = [] ;
    
    
APOE = apoeMatrix ;
APOE(missingIdx,:) = [] ;
G = genotype2_oldQC_Chr19 ;
G(missingIdx,:) = [] ;
G = [double(G) APOE] ;
% getting rid of missing values (=3)
h =  @(x)mean(x(x~=3)) ;
for jj=1:size(G,2)
    G( G(:,jj)==3 ,jj) = h(G(:,jj)) ;
end
% centering genetic data
G = bsxfun(@minus,mean(G),G) ;
    
I = endoPhenMatrix ;
I(missingIdx,:) = [] ;
% centering I with respect to the normal subjects
mu = mean(I(y==0,:)) ;
s = std(I(y==0,:)) ;
I = bsxfun(@minus, I,mu) ;
I = bsxfun(@times, I, 1./s) ;


endoPhenMatrix_residual_oldQC = I - G*B_oldQC ;


% --- oldQC : compute the residual endo-phenotype
endoPhenMatrix_residual_newQC = zeros(size(endoPhenMatrix)) ;
missingIdx = missingIdx_newQC ;

y = diagnosisLbl ;
y(missingIdx) = [] ;


APOE = apoeMatrix ;
APOE(missingIdx,:) = [] ;
G = genotype2_newQC_Chr19 ;
G(missingIdx,:) = [] ;
G = [double(G) APOE] ;
% getting rid of missing values (=3)
h =  @(x)mean(x(x~=3)) ;
for jj=1:size(G,2)
    G( G(:,jj)==3 ,jj) = h(G(:,jj)) ;
end
% centering genetic data
G = bsxfun(@minus,mean(G),G) ;

I = endoPhenMatrix ;
I(missingIdx,:) = [] ;
% centering I with respect to the normal subjects
mu = mean(I(y==0,:)) ;
s = std(I(y==0,:)) ;
I = bsxfun(@minus, I,mu) ;
I = bsxfun(@times, I, 1./s) ;


endoPhenMatrix_residual_newQC = I - G*B_newQC ;


% rank imaging features according to their relevance to Y and make different set
[IDX,Z ] = rankfeatures(endoPhenMatrix',diagnosisLbl) ;

MATLABPATH = '/data/vision/polina/projects/ADNI/work/kayhan/Journal/MATLAB_Data' ;
prefixFileNameList = {'oldQC-Chr19','newQC-Chr19'} ;
missingIdxList = {'missingIdx_oldQC','missingIdx_newQC'} ;
genotypeList = {'genotype2_oldQC_Chr19','genotype2_newQC_Chr19'} ;
phenotypeList = {'endoPhenMatrix_residual_oldQC','endoPhenMatrix_residual_newQC'} ;
subInfoList = {'subInfo2_oldQC_Chr19','subInfo2_newQC_Chr19'} ;
snpInfoList = {'snpInfo_oldQC_Chr19','snpInfo_newQC_Chr19'} ;

for eCnt=1:length(IDX)
    eCnt
    for cnt=1:length(prefixFileNameList)
        missingIdx = eval(missingIdxList{cnt}) ;
        genotype = eval(genotypeList{cnt}) ;
        subInfo = eval(subInfoList{cnt}) ;
        snpInfo = eval(snpInfoList{cnt}) ;
        I = eval(phenotypeList{cnt}) ;  % endophenotype
        I = I(:,IDX(1:eCnt)) ;  % endophenotype
        ePhenLabel = endoPhenLabel(IDX(1:eCnt)) ;
        
        
        missingIDs = diagnosisLblID(missingIdx) ;
        subInfo(missingIdx) = [] ;
        snpInfo(end+1).snpID = 'e3' ;
        snpInfo(end).chrNum = 19 ;
        snpInfo(end+1).snpID = 'e4' ;
        snpInfo(end).chrNum = 19 ;
        
        y = diagnosisLbl ;
        y(missingIdx) = [] ;
        
        
        APOE = apoeMatrix ;
        APOE(missingIdx,:) = [] ;
        G = genotype ;
        G(missingIdx,:) = [] ;
        G = [double(G) APOE] ;
        % getting rid of missing values (=3)
        h =  @(x)mean(x(x~=3)) ;
        for jj=1:size(G,2)
            G( G(:,jj)==3 ,jj) = h(G(:,jj)) ;
        end
        % centering genetic data
        G = bsxfun(@minus,mean(G),G) ;
        
               
        fn = [prefixFileNameList{cnt} '_Data_residAfterLasso_' num2str(eCnt) '.mat' ] ;
        save([MATLABPATH '/' fn],'I','G','y','missingIDs','APOE','snpInfo','subInfo','ePhenLabel','covData','covList') ;
        clear I G y missingIdx   genotype   subInfo  snpInfo  APOE  ePhenLabel
    end
end

%% BLOCK 7
% --- get rid of all subject not in the phenotype list
%missingIdx = missingIdx_oldQC ;
%genotype2 = genotype2_oldQC_Chr19 ;
%subInfo2 = subInfo2_oldQC_Chr19 ;
%snpInfo = snpInfo_oldQC_Chr19 ;
Y = diagnosisLbl ;
missingIdx = missingIdx_newQC ;
genotype2 = genotype2_newQC_Chr19 ;
subInfo2 = subInfo2_newQC_Chr19 ;
snpInfo = snpInfo_newQC_Chr19 ;


missingIDs = diagnosisLblID(missingIdx) ;
% removing missing labels from diagnosis label
diagnosisLbl(missingIdx) = [] ;
diagnosisLblID(missingIdx) = [] ;
Y(missingIdx) = [] ;

% removing missing subjects from genotype
apoeIDs(missingIdx) = [] ;
apoeMatrix(missingIdx,:) = [] ;
genotype2(missingIdx,:) = [] ;
% removing missing subjects from phenotype data
subCorticalVolDiag(missingIdx) = [] ;
subCorticalVolIDs(missingIdx) = [] ;
subCorticalVolMatrix(missingIdx,:) = [] ;

rhIDs(missingIdx) = [] ;
rhCortexThicknessDiag(missingIdx) = [] ;
rhCortexThicknessMatrix(missingIdx,:) = [] ;

lhIDs(missingIdx) = [] ;
lhCortexThicknessDiag(missingIdx) = [] ;
lhCortexThicknessMatrix(missingIdx,:) = [] ;

endoPhenMatrix(missingIdx,:) = [] ;

% removing missing subjects from data structure
subInfo2(missingIdx) = [] ;
subInfo = subInfo2 ;
genotype = genotype2 ;
genotype = [double(genotype) apoeMatrix] ;
snpInfo(end+1).snpID = 'e3' ;
snpInfo(end).chrNum = 19 ;
snpInfo(end+1).snpID = 'e4' ;
snpInfo(end).chrNum = 19 ;
clear subInfo2 genotype2;


%% BLOCK 8: association test

h =  @(x)mean(x(x~=3)) ;   
for cnt=1:size(genotype,2)
    genotype( genotype(:,cnt)==3 ,cnt) = h(genotype(:,cnt)) ; 
end
genotype = bsxfun(@minus,mean(genotype),genotype) ;

% association test
g = double(genotype) ;
test1 = zeros(size(g,2),1) ;
test2 = zeros(size(g,2),1) ;
test3 = zeros(size(g,2),1) ;
for cnt=1:size(g,2)
    n = 13 ;
    [~,~,stats] = glmfit(g(:,cnt), subCorticalVolMatrix(:,n) ) ;
    test1(cnt) = -log10(stats.p(2)) ;
    
    n = 25 ;
    [~,~,stats] = glmfit(g(:,cnt), subCorticalVolMatrix(:,n) ) ;
    test2(cnt) = -log10(stats.p(2)) ;
    
    [~,~,stats] = glmfit(g(:,cnt), Y , 'binomial') ;
    test3(cnt) = -log10(stats.p(2)) ;
    
    if mod(cnt,floor(size(g,2)/10))==0
        fprintf('.') ;
    end
end
fprintf('\n') ;

figure
subplot(1,3,1)
n = 13 ;
plot(test1,'.')
title([ 'Association with respect to ' subCorticalVolStringHeader{n}])


subplot(1,3,2)
n = 25 ;
plot(test2,'.')
title([ 'Association with respect to ' subCorticalVolStringHeader{n}])

subplot(1,3,3)
plot(test3,'.')
title('Association with respect to class label (Y)')


%% BLOCK 9: do some plotting to make sure that data is read in correctly
figure; 

n = 10 ;
subplot(3,2,1)
hold on

x = lhCortexThicknessMatrix(:,n) ;
[nCount,xout] = hist(x,50) ;
bar(xout,nCount/sum(nCount))

mu1 = mean(x(lhCortexThicknessDiag==1)) ;
var1 = var(x(lhCortexThicknessDiag==1)) ;
p1 = pdf('Normal', linspace(min(x),max(x),100), mu1, sqrt(var1)) ;
plot(linspace(min(x),max(x),100),p1/sum(p1),'--r')

mu2 = mean(x(lhCortexThicknessDiag==0)) ;
var2 = var(x(lhCortexThicknessDiag==0)) ;
p2 = pdf('Normal', linspace(min(x),max(x),100), mu2, sqrt(var2)) ;
plot(linspace(min(x),max(x),100),p2/sum(p2),'--g')

title([ 'Thickness hist of (lh) ' lhCortexThickStringHeader{n}])


%----------
n = 30 ;
subplot(3,2,2)
hold on

x = lhCortexThicknessMatrix(:,n) ;
[nCount,xout] = hist(x,50) ;
bar(xout,nCount/sum(nCount))

mu1 = mean(x(lhCortexThicknessDiag==1)) ;
var1 = var(x(lhCortexThicknessDiag==1)) ;
p1 = pdf('Normal', linspace(min(x),max(x),100), mu1, sqrt(var1)) ;
plot(linspace(min(x),max(x),100),p1/sum(p1),'--r')

mu2 = mean(x(lhCortexThicknessDiag==0)) ;
var2 = var(x(lhCortexThicknessDiag==0)) ;
p2 = pdf('Normal', linspace(min(x),max(x),100), mu2, sqrt(var2)) ;
plot(linspace(min(x),max(x),100),p2/sum(p2),'--g')

title([ 'Thickness hist of (lh) ' lhCortexThickStringHeader{n}])


%----------
n = 10 ;
subplot(3,2,3)
hold on

x = rhCortexThicknessMatrix(:,n) ;
[nCount,xout] = hist(x,50) ;
bar(xout,nCount/sum(nCount))

mu1 = mean(x(rhCortexThicknessDiag==1)) ;
var1 = var(x(rhCortexThicknessDiag==1)) ;
p1 = pdf('Normal', linspace(min(x),max(x),100), mu1, sqrt(var1)) ;
plot(linspace(min(x),max(x),100),p1/sum(p1),'--r')

mu2 = mean(x(rhCortexThicknessDiag==0)) ;
var2 = var(x(rhCortexThicknessDiag==0)) ;
p2 = pdf('Normal', linspace(min(x),max(x),100), mu2, sqrt(var2)) ;
plot(linspace(min(x),max(x),100),p2/sum(p2),'--g')

title([ 'Thickness hist of (rh) ' rhCortexThickStringHeader{n}])

%----------
n = 30 ;
subplot(3,2,4)
hold on

x = rhCortexThicknessMatrix(:,n) ;
[nCount,xout] = hist(x,50) ;
bar(xout,nCount/sum(nCount))

mu1 = mean(x(rhCortexThicknessDiag==1)) ;
var1 = var(x(rhCortexThicknessDiag==1)) ;
p1 = pdf('Normal', linspace(min(x),max(x),100), mu1, sqrt(var1)) ;
plot(linspace(min(x),max(x),100),p1/sum(p1),'--r')

mu2 = mean(x(rhCortexThicknessDiag==0)) ;
var2 = var(x(rhCortexThicknessDiag==0)) ;
p2 = pdf('Normal', linspace(min(x),max(x),100), mu2, sqrt(var2)) ;
plot(linspace(min(x),max(x),100),p2/sum(p2),'--g')

title([ 'Thickness hist of (rh) ' rhCortexThickStringHeader{n}])


%----------
n = 8 ;
subplot(3,2,5)
hold on

x = subCorticalVolMatrix(:,n) ;
[nCount,xout] = hist(x,50) ;
bar(xout,nCount/sum(nCount))

mu1 = mean(x(subCorticalVolDiag==1)) ;
var1 = var(x(subCorticalVolDiag==1)) ;
p1 = pdf('Normal', linspace(min(x),max(x),100), mu1, sqrt(var1)) ;
plot(linspace(min(x),max(x),100),p1/sum(p1),'--r')

mu2 = mean(x(subCorticalVolDiag==0)) ;
var2 = var(x(subCorticalVolDiag==0)) ;
p2 = pdf('Normal', linspace(min(x),max(x),100), mu2, sqrt(var2)) ;
plot(linspace(min(x),max(x),100),p2/sum(p2),'--g')

title([ 'Hist of  volume' subCorticalVolStringHeader{n}])


%----------
n = 25 ;
subplot(3,2,6)
hold on

x = subCorticalVolMatrix(:,n) ;
[nCount,xout] = hist(x,50) ;
bar(xout,nCount/sum(nCount))

mu1 = mean(x(subCorticalVolDiag==1)) ;
var1 = var(x(subCorticalVolDiag==1)) ;
p1 = pdf('Normal', linspace(min(x),max(x),100), mu1, sqrt(var1)) ;
plot(linspace(min(x),max(x),100),p1/sum(p1),'--r')

mu2 = mean(x(subCorticalVolDiag==0)) ;
var2 = var(x(subCorticalVolDiag==0)) ;
p2 = pdf('Normal', linspace(min(x),max(x),100), mu2, sqrt(var2)) ;
plot(linspace(min(x),max(x),100),p2/sum(p2),'--g')

title([ 'Hist of volume ' subCorticalVolStringHeader{n}])

%-----
figure
idx = randint(1,6,[1 length(endoPhenLabel)]) ;
cnt = 1 ;
for n=idx
    subplot(3,2,cnt)
    hold on

    x = endoPhenMatrix(:,n) ;
    [nCount,xout] = hist(x,50) ;
    bar(xout,nCount/sum(nCount))

    mu1 = mean(x(Y==1)) ;
    var1 = var(x(Y==1)) ;
    p1 = pdf('Normal', linspace(min(x),max(x),100), mu1, sqrt(var1)) ;
    plot(linspace(min(x),max(x),100),p1/sum(p1),'--r')

    mu2 = mean(x(Y==0)) ;
    var2 = var(x(Y==0)) ;
    p2 = pdf('Normal', linspace(min(x),max(x),100), mu2, sqrt(var2)) ;
    plot(linspace(min(x),max(x),100),p2/sum(p2),'--g')

    title([ 'Hist ' endoPhenLabel{n} ' after normalization'])
    cnt = cnt + 1;
end

%   To make sue that normal subjects are normaly distributed, choose
%   only normal subjects and plot the histogram of random subcortical volumes or thickness. We want to make sure that our assumption about normal distribution of the regions is more/less valid. You plot as many as histogram of the features as you can. It is alo a good practice to plot them along with a p-value test for normality (search for lillietest in MATLAB).
%----- test of Gaussianity for the normal subjects
figure
pValNormal = zeros(size(endoPhenMatrix,2),1) ;
for cnt=1:size(endoPhenMatrix,2)
    [~,pValNormal(cnt)] = lillietest(endoPhenMatrix(Y==0,cnt)) ;
end

bar(-log10(pValNormal))
xlabel('features for regions ')
ylabel('- log p')
title(['Lillie Test for normality for features (sig. Level ' num2str(-log10(0.05/length(pValNormal))) ')'])

%----- histogram of APOE variants
figure;

[~, ~, stats_p3] =  glmfit(apoeMatrix(:,1),double(Y),'binomial') ;
[~, ~, stats_p4] =  glmfit(apoeMatrix(:,2),double(Y),'binomial') ;

subplot(2,2,1)
hist(double(apoeMatrix(Y==0,:)))
legend('e3','e4')
title('Histogram of APOE for normal population')

subplot(2,2,2)
hist(double(apoeMatrix(Y==1,:)))
legend('e3','e4')
title('Histogram of APOE for AD population')

subplot(2,2,3)
[n,xout] = hist(double(apoeMatrix(Y==1,1))) ;
bar(xout,n)
hold on
[n,xout] = hist(double(apoeMatrix(Y==0,1))) ;
bar(xout+0.2,n,'r')
title([ 'Distribution of APOE-e3 for Normal/AD (-log10 p =  ' num2str(-log10(stats_p3.p(2)))   ')']  )
legend('AD','Normal')

subplot(2,2,4)
[n,xout] = hist(double(apoeMatrix(Y==1,2))) ;
bar(xout,n)
hold on
[n,xout] = hist(double(apoeMatrix(Y==0,2))) ;
bar(xout+0.2,n,'r')
title(['Distribution of APOE-e4 for Normal/AD (-log10 p =  ' num2str(-log10(stats_p4.p(2)))   ')'])
legend('AD','Normal')




%% BLOCK 10: Is there any discriminative signal in the image?
%  
%  1. Use sparse logistic regression. You can use LIBSVM. Here is how you can use it:
%         Find it at : /data/vision/polina/shared_software/MATLAB_Toolboxes/liblinear-1.94/matlab
%         ** This is compiled with MATLAB 2010a **
%  2. " -s " option for the LIBLINEAR should be:
%	            6 -- L1-regularized logistic regression
%         (Use default c)     

%  3. Use it once without " -v " option (Cross-Validation) and check the output model to see if the obvious regions get picked up.

%  4. Use it one with the " -v  5 " to have an overall evaluation of the classification results.
% We have done something similar this before. Take a look at : SOURCE/example/testSpikeSlabVarBayes_old.m (lines  33-37 )


LIBLINEARPATH = '/data/vision/polina/shared_software/MATLAB_Toolboxes/liblinear-1.94/matlab' ;
addpath(LIBLINEARPATH)
cRange = 10.^[-7:0.2:2] ;
CVValues = zeros(length(cRange),1) ;    % cross-validation values
nnzValues = zeros(length(cRange),1) ;    % number of non-zero values
YY = Y ;
YY(Y==0) =  2 ;
fprintf('performing cross validation using endo phenotype ') ;
for cnt=1:length(cRange)
    c = cRange(cnt) ;
    optStr = sprintf('-s 6 -c %.8f -B 1 ',c) ;
    CVValues(cnt) = train(sparse(YY), sparse(endoPhenMatrix), [optStr ' -v 10']) ;
    model = train(sparse(YY), sparse(endoPhenMatrix), optStr) ;
    nnzValues(cnt) = sum(model.w(1:end-1)~=0) ;
    if mod(cnt,floor(length(nnzValues)/10))==0
        fprintf('.') ;
    end
end
fprintf('Done \n') ;

figure
plot(nnzValues/size(endoPhenMatrix,2),CVValues,'LineWidth',2)
axis([0 1 0 90])
xlabel('Percentage of the non-zero W')
ylabel('Accuracy %')
title('Accuracy of prediction (cross-validation) with respect to number of non-zero coeff')
grid on
%% BLOCK 11: Is Spike-and-Slab able to detect imaging regions?
%  1. Use class label as y, and thickness and volume as features and run
%  Spike-and-Slab (Carbonetto's paper)
%  2. Check if posterior relevance (alpha) is roughly in the region we
%  expect. This is how you can run Carbonetto's work:
% we have actually an example of this in  SOURCE/example/testSpikeSlabVarBayes.m 

LARGE_SCLAE_SPIKESLAB_PATH = '/data/vision/polina/shared_software/MATLAB_Toolboxes/Carbonetto_VBS/MATLAB' ;
addpath(LARGE_SCLAE_SPIKESLAB_PATH)
sa     = 1;   % you can give a range of values if you would like to try different parameters
log10q = -0.2;  % you can give a range of values of you would liek to try different parameters
[sa log10q] = ndgrid(sa,log10q);
options.verbose = false ;
fprintf('Running variational Spike and Slab .... ') ;
X = endoPhenMatrix ;
X = bsxfun(@minus, X , mean(X)) ;
X = bsxfun(@times, X, 1./std(X)) ;
[LnZ,Alpha,Mu,S,Eta] = varbvsbin(X,double(Y),sa,log10q, options) ;
fprintf('Done \n') ;
fprintf('Here is the list of structures detected by  Spike-and-Slab using Y: \n') ;
idx = find(Alpha>0.5) ;
for cnt=1:length(idx)
    fprintf('%s \n ',endoPhenLabel{idx(cnt)}) ;
end


%% BLOCK 12 : Is Spike-and-Slab able to detect APOE using Y as class label

LARGE_SCLAE_SPIKESLAB_PATH = '/data/vision/polina/shared_software/MATLAB_Toolboxes/Carbonetto_VBS/MATLAB' ;
addpath(LARGE_SCLAE_SPIKESLAB_PATH)
sa     = 1;   % you can give a range of values if you would like to try different parameters
log10q = -2;  % you can give a range of values of you would liek to try different parameters
[sa log10q] = ndgrid(sa,log10q);
options.verbose = false ;
fprintf('Running variational Spike and Slab .... ') ;
X = genotype ;
X = bsxfun(@minus, X , mean(X)) ;
X = bsxfun(@times, X, 1./std(X)) ;
[LnZ,Alpha,Mu,S,Eta] = varbvsbin(X,double(Y),sa,log10q, options) ;
fprintf('Done \n') ;
fprintf('Here is the list of SNP detected by Spike-and-Slab using Y: \n') ;
idx = find(Alpha>0.5) ;
for cnt=1:length(idx)
    s = snpInfo(idx(cnt)) ;
    fprintf('%s \n ', s.snpID) ;
end


%% BLOCK 13 : Is Spike-and-Slab able to detect APOE using only hippocampus volume ?
%   1. Use un-imputed genetic data, add APOE to it.
%   2. Use the volume of hippocampus (right or left) as y and check if it get the APOE 

% this is the same previois one excdep that instead of varbvsbin, you need to call "varsimbvs" . 
% Note that this is not in our package and you need to call it from original Carbonetto's code.
% I put the code in the Matlab_Toolbox :  /data/vision/polina/shared_software/MATLAB_Toolboxes/Carbonetto_VBS/MATLAB 
LARGE_SCLAE_SPIKESLAB_PATH = '/data/vision/polina/shared_software/MATLAB_Toolboxes/Carbonetto_VBS/MATLAB' ;
addpath(LARGE_SCLAE_SPIKESLAB_PATH)
sigma  = ( 0.2:0.08:1 )';
sa     = (0.025:0.025:0.4)';
log10q = (-5:0.25:-3)';
[sigma sa log10q] = ndgrid(sigma,sa,log10q);
X = genotype ;
X = bsxfun(@minus, X , mean(X)) ;
X = bsxfun(@times, X, 1./std(X)) ;

y1 = endoPhenMatrix(:,93) ;   % Right Hippocampus
mu = mean(y1(Y==0)) ;
s = std(y1(Y==0)) ;
y1 = (y1 - mu)/s ;

y2 = endoPhenMatrix(:,81) ;   % Left Hippocampus
mu = mean(y2(Y==0)) ;
s = std(y2(Y==0)) ;
y2 = (y2 - mu)/s ;

% Right Hippocampus
a = 0.02 ;
b = 1 ;
c = 0.02 ;
[w alpha_rightHipp mu] = varsimbvs(X,y1,sigma,sa,log10q,a,b,c);
figure
plot(alpha_rightHipp,'.')
xlabel('SNPs from Chr 19')
ylabel('\alpha after important sampling (right Hippocampus as target)')

% Left Hippocampus
[w alpha_leftHipp mu] = varsimbvs(X,y2,sigma,sa,log10q,a,b,c);
figure
plot(alpha_leftHipp,'.')
xlabel('SNPs from Chr 19')
ylabel('\alpha after important sampling (left Hippocampus as target)')







