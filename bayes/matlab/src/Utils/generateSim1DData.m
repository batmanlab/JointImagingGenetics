% This function generates simulated data
%xMin = -5;
%xMax = 5;
%numBasisCenters = 20; % number of basis in each scale
%numSelectedBasisCenters = 10;  % total number of basis
%numGrids = 300;  % number of grids
%basisWidth = [0.2  0.4  0.8];   % standard deviation of the basis
%numObserv = 57;   % number of observation points
%coefStd = 1; %standard deviation of coefficients
%noiseStd = .2; %0.1;
%numTopEigen = 20; % number of selected eigen vector for low-rank L
function   generateSim1DData(fnList,numObservList)
           %% making simulated data ready
           xMin = -10;
           xMax = 10;
           numBasisCenters = 30; % number of basis in smallest scale
           numSelectedBasisCenters = 15;  % total number of basis
           numGrids = 300;  % number of grids
           numBasisLevels = 3 ;  % number of multi resolution basis
           coefStd = 1; %standard deviation of coefficients
           noiseStd = .2; %0.1;
           numTopEigen = 30; % number of selected eigen vector for low-rank L
           numRepeats = 100 ;
           
           %% intialize all required data structure
           allData =cell(length(numObservList),1) ;
           for obsCnt=1:length(numObservList)
               allData{obsCnt} = repmat(struct('yTrain',[],'XTrain',[],'selTrainIdx',[],'coordTrain',[],...
                                    'yTest',[],'XTest',[],'selTestIdx',[],'coordTest',[],...
                                    'basisMatrix', [], 'selBasisIdx', [],  'basisCenters',[], 'selBasisWidth', [] ,...
                                    'L',[],'Phi',[], 'xGrid', [], 'y', []),[numRepeats 1]) ;
           end

           
           %% first sample basis information for all repeats
           % sample the basis info for the first and copy it for the rest
           data  = allData{1} ;
           for repCnt=1:numRepeats               
               %[data(repCnt).yTrain, data(repCnt).XTrain, data(repCnt).selTrainIdx, data(repCnt).coordTrain,...
               %    data(repCnt).yTest, data(repCnt).XTest, data(repCnt).selTestIdx, data(repCnt).coordTest,...
               [data(repCnt).basisMatrix, data(repCnt).basisCenters, data(repCnt).selBasisIdx, data(repCnt).selBasisCoord , data(repCnt).selBasisWidth, ...
                   data(repCnt).L, data(repCnt).Phi, data(repCnt).xGrid, data(repCnt).y ] = ...
                   selectRandomBasis(xMin, xMax , ...
                   numBasisCenters, ...  % number of basis in each scale
                   numSelectedBasisCenters, ...  % total number of basis
                   numGrids,...  % number of grids
                   numBasisLevels, ...    % number of multi resolution basis
                   numTopEigen, coefStd) ;
           end
           allData{1} = data ;
           
           for obsCnt=2:length(numObservList)
               allData{obsCnt} = allData{1} ;
           end
           
           %% given basis information, sample observation points and train/test data
           for obsCnt=1:length(numObservList)
                numObserv = numObservList(obsCnt) ;
                data = allData{obsCnt} ;
                
                for repCnt=1:numRepeats
                    % observe the signal : pair location value(x,y)
                    selObservIdx = randperm(numGrids);    % selected Observation points;
                    selObservIdx = selObservIdx(1:numObserv);
                    selX = data(repCnt).xGrid(selObservIdx);  % selected coordinates
                    selY = data(repCnt).y(selObservIdx) + randn(length(selObservIdx),1)*noiseStd;
                    %figure(2); plot(xGrid,y); hold on; plot(selX,selY,'r*'); title('signal and observation')
                    
                    
                    % making data ready for output
                    data(repCnt).selTrainIdx = selObservIdx ;
                    data(repCnt).yTrain = selY ;
                    data(repCnt).XTrain = data(repCnt).basisMatrix(data(repCnt).selTrainIdx, :) ;
                    data(repCnt).coordTrain = selX ;
                    
                    
                    data(repCnt).selTestIdx = setdiff( 1:size(data(repCnt).basisMatrix,1), data(repCnt).selTrainIdx) ;   % unobserved points
                    data(repCnt).yTest = data(repCnt).y(data(repCnt).selTestIdx) ;
                    data(repCnt).XTest = data(repCnt).basisMatrix(data(repCnt).selTestIdx, :) ;
                    data(repCnt).coordTest = data(repCnt).xGrid( data(repCnt).selTestIdx);
                    
                end
                
                allData{obsCnt} = data ;
           end
           
           %% Now save all data into separate files
           for obsCnt=1:length(numObservList)
               data = allData{obsCnt} ;
               save(fnList{obsCnt},'data') ;
           end
end



%function  [yTrain,XTrain,selTrainIdx,coordTrain,...
%           yTest,XTest,selTestIdx,coordTest,...
function   [basisMatrix, basisCenters, selBasisIdx, selBasisCoord , selBasisWidth, ...
           L, Phi, xGrid, y ] = ...
           selectRandomBasis(xMin, xMax , ...
                      numBasisCenters, ...  % number of basis in each scale
                      numSelectedBasisCenters, ...  % total number of basis
                      numGrids,...  % number of grids
                      numBasisLevels, ...   % standard deviation of the basis
                      numTopEigen,...  % number of top eigen vectors selected from L  
                      coefStd )  % standard deviation of the coefficients
        
    % create basis matrix
    basisFcn = @(x,c,s)exp(-((x-c).^2)./(2*s.^2));
    xGrid = linspace(xMin, xMax,numGrids)';
    
    totalNumBasisCenters = 0 ;
    basisWidth = zeros(numBasisLevels,1) ;
    centers = cell(numBasisLevels,1) ;
    for sCnt=1:numBasisLevels
        if (sCnt==1)
            centers{sCnt} = linspace(xMin-xMin/5, xMax-xMax/5,numBasisCenters)';  
            basisWidth(sCnt) = abs(centers{sCnt}(1)-centers{sCnt}(2)) ;
        else
            centers{sCnt} = filter([1 1]/2,1,centers{sCnt-1}) ;
            centers{sCnt} = centers{sCnt}(2:end) ;
            basisWidth(sCnt) = basisWidth(sCnt-1)*2 ;
        end
        %centers{sCnt} = (xMin-xMin/5):basisWidth(sCnt):(xMax-xMax/5) ;
        totalNumBasisCenters = totalNumBasisCenters + length(centers{sCnt}) ;
    end
    basisMatrix = zeros(length(xGrid),totalNumBasisCenters);
    basisCenters = zeros(1,totalNumBasisCenters);
    cnt = 1;
    for sCnt=1:numBasisLevels
        s = basisWidth(sCnt) ;
        for cCnt=1:length(centers{sCnt})
            c = centers{sCnt}(cCnt) ;
            basisMatrix(:,cnt) = basisFcn(xGrid,c,s);
            basisCenters(cnt) = c ;
            cnt = cnt + 1;
        end
    end
    
    % create the signal
    %selCentersIdx = floor(linspace(1,numBasisCenters,numSelectedBasisCenters));  % select center of basis for the basis
    %selWidthIdx = randi(length(basisWidth) ,numSelectedBasisCenters,1);
    %selIdx = selCentersIdx + (selWidthIdx' - 1)*length(centers);  % get index of selected basis on the basisMatrix
    
    
    % this function center of basis and inhibits the simulation from choosing children of the basis
    function [selIdx,selCentersIdx,selWidth] = chooseBasisIdx()
        % make some required data structures
        basisIdx = cell(numBasisLevels,1) ;
        childrenIdx = cell(numBasisLevels,1) ;
        allCenters = [] ;
        allBasisIdx = [] ;
        allWidthsIdx = [] ;
        allRelIdx = [] ;
        allWidths = [] ;
        cnt = 0 ;
        for i=1:numBasisLevels
            basisIdx{i} = (cnt+1):(cnt+length(centers{i})) ;
            allCenters = [allCenters; centers{i}(:) ] ;
            allBasisIdx = [allBasisIdx; basisIdx{i}(:) ] ;
            allWidthsIdx = [allWidthsIdx; i*ones(length(basisIdx{i}),1)] ;
            allWidths = [allWidths; basisWidth(i)*ones(length(basisIdx{i}),1)] ;
            allRelIdx = [allRelIdx; (1:length(centers{i}))'] ;
            cnt = cnt + length(centers{i}) ;
        end
        for i=numBasisLevels:-1:1
            childrenIdx{i} = cell(length(centers{i}),1) ;
            for j=1:length(childrenIdx{i})
                childrenIdx{i}{j} = allBasisIdx(  centers{i}(j)-basisWidth(i) < allCenters  &...
                    centers{i}(j)+basisWidth(i) > allCenters  )  ;
            end
        end
        
        %
        selWidthIdx = [] ;
        selWidth = [] ;
        selCentersIdx = [] ;
        selIdx = [] ;
        selRelIdx = [] ;
        for i=1:numSelectedBasisCenters
            % choose a scale first
            if length(unique(allWidthsIdx))>1
                selWidthIdx(i) = randsample(unique(allWidthsIdx),1,1) ;
            else
                selWidthIdx(i) = unique(allWidthsIdx) ;
            end
            % choose center in that scale
            if length( allBasisIdx(allWidthsIdx == selWidthIdx(i)) )>1
                selIdx(i) = randsample( allBasisIdx(allWidthsIdx == selWidthIdx(i)),1,1) ;
            else
                selIdx(i) = allBasisIdx(allWidthsIdx == selWidthIdx(i)) ;
            end
            selCentersIdx(i) = allCenters( allBasisIdx == selIdx(i) & allWidthsIdx == selWidthIdx(i) ) ;
            selWidth(i) = allWidths( allBasisIdx == selIdx(i) & allWidthsIdx == selWidthIdx(i) ) ;
            selRelIdx(i) = allRelIdx( allBasisIdx == selIdx(i) & allWidthsIdx == selWidthIdx(i) ) ;
            % remove all centers in the that scale and finer one
            ii = find(ismember(allBasisIdx,childrenIdx{selWidthIdx(i)}{selRelIdx(i)} )) ;
            allBasisIdx( ii )  = [] ;
            allCenters( ii  )  = [] ;
            allWidthsIdx( ii )  = [] ;
            allRelIdx(ii) = []  ;
            allWidths(ii) = [] ; 
            if isempty(allBasisIdx)
                break ;
            end
        end
        
    end

    [selBasisIdx,selBasisCoord,selBasisWidth] = chooseBasisIdx() ;
    %figure(1); plot(basisMatrix(:,selBasisIdx)); title('selected basis')
    
    % create similarity features
    L =  basisMatrix'*basisMatrix ;
    [V,D] = eig(L) ;
    D = diag(D) ;
    [~,idx] = sort(D,'descend') ;
    D_cut = D(idx(1:numTopEigen)) ;
    V_cut = V(:,idx(1:numTopEigen)) ;
    Phi = V_cut*sqrt(diag(D_cut)) ; 
       
    
%     % observe the signal : pair location value(x,y)
    numSelectedBasisCenters = length(selBasisIdx) ;
    c = randn(numSelectedBasisCenters,1)*coefStd;
    y = basisMatrix(:,selBasisIdx)*c;
%     selObservIdx = randperm(numGrids);    % selected Observation points;
%     selObservIdx = selObservIdx(1:numObserv);
%     selX = xGrid(selObservIdx);  % selected coordinates
%     selY = y(selObservIdx) + randn(length(selObservIdx),1)*noiseStd;
%     %figure(2); plot(xGrid,y); hold on; plot(selX,selY,'r*'); title('signal and observation')
%     
%     
%     % making data ready for output
%     selTrainIdx = selObservIdx ;
%     yTrain = selY ;
%     XTrain = basisMatrix(selTrainIdx, :) ;
%     coordTrain = selX ;
% 
%     
%     selTestIdx = setdiff( 1:size(basisMatrix,1) ,selTrainIdx) ;   % unobserved points
%     yTest = y(selTestIdx) ;
%     XTest = basisMatrix(selTestIdx, :) ;
%     coordTest = xGrid(selTestIdx); 
%     
%     %selBasisCoord = basisCenters(selBasisIdx) ;
    
end


