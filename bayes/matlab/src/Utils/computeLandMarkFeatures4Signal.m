% this function computes coarses *features* that will be used later as a
% low-rank representation for similarity matrix.
%
% output is a feature cell with the following dimension:
%          (#spatialFeatures + #numMovingWindow)  x  #levels x  #numMovingWindow
%          where:
%             (#spatialFeatures + #numMovingWindow) : is total number of features
%             #Levels : different levels of box size
%             #numMovingWindow : number of moving window along the temporal (time) axis
function [FeatureCell,K] = computeLandMarkFeatures4Signal(img,options)
timeFeatureValue = options.timeFeatureValue ;
windowLength = options.windowLength ;
numTimePoints = length(img) ;
numMovWindows = numTimePoints - windowLength +  1 ;    % NOTICE : here Moving window is over TIME not the DOMAIN
landmarks = options.landmarks ;
numLandmarks = length(landmarks) ;
scaleInterval = options.scaleInterval ;
s = options.sigma ;
imgSize = size(img{1}) ;
method = options.method ;
K = [] ;

switch (method)
    case '4corners-1Cent',  % min distance between 4 corners plus the center of the moving box
        % extract overlapping features
        fprintf('extract overlapping features ... ') ;
        numLevels = scaleInterval(2) - scaleInterval(1) + 1 ;
        FeatureCell = cell(numLandmarks + numMovWindows,numLevels,numMovWindows) ;
        for winCnt=1:numMovWindows % counter for moving windows over time: begin
            for levCnt=scaleInterval(1):scaleInterval(2)
                xImg = getCenterBox(imgSize,2^levCnt) ;
                [X1, X2] = getCornerBox(imgSize,2^levCnt) ;
                for fCnt=1:size(FeatureCell,1)  % loop over features in the set (landmarks)
                    if (fCnt<=numLandmarks)  % some of the feature are related to time, check if this feature is related to the location or time
                        if (winCnt==1)   % check if winCnt=1 (ie it is the first time you see this)
                            % for location, compute the overlap ratio if winCnt=1,
                            
                            %FeatureCell{fCnt,levCnt,winCnt} = exp( -( (xImg - landmarks(fCnt,1)).^2 +  (yImg - landmarks(fCnt,2)).^2 ) / (s^2) ) ;
                            FeatureCell{fCnt,levCnt,winCnt} = exp(-min(cat(1,...
                                (X1 - landmarks(fCnt)).^2  , ...
                                (X2 - landmarks(fCnt)).^2  , ...
                                (xImg - landmarks(fCnt)).^2  ...
                                ),[],1)/ (s^2)) ;
                            FeatureCell{fCnt,levCnt,winCnt} = FeatureCell{fCnt,levCnt,winCnt}.*double(FeatureCell{fCnt,levCnt,winCnt} > 0.1) ;   % make it sparse
                            FeatureCell{fCnt,levCnt,winCnt} = sparse(FeatureCell{fCnt,levCnt,winCnt}) ;
                        else % otherwise it is just copying from the first time you compute it
                            FeatureCell{fCnt,levCnt,winCnt} = FeatureCell{fCnt,levCnt,1} ;
                        end
                    else % if this feature is time related,
                        %compute the ratio of overlap with other windows
                        if  any(ismember(winCnt:(winCnt + windowLength - 1),(fCnt - numLandmarks):(fCnt - numLandmarks + windowLength - 1)))
                            val = sum(ismember(winCnt:(winCnt + windowLength - 1),(fCnt - numLandmarks):(fCnt - numLandmarks + windowLength - 1)))/windowLength*timeFeatureValue ;
                        else
                            val = 0 ;
                        end
                        FeatureCell{fCnt,levCnt,winCnt} = val*ones(size(FeatureCell{1,levCnt,1})) ;
                    end
                end
            end
        end   % counter for moving windows over time: end
    case 'gaussKL-fullRank',  % using KL divergance between two Gaussian dist's whose std are equal to widths of the windows
        % notice that this method returns the the whole matrix not the low rank representation
        if (numMovWindows>1)
            error('this method is not implemented for the temporal case ! ') ;
        end
        fprintf('extract overlapping features ... ') ;
        numLevels = scaleInterval(2) - scaleInterval(1) + 1 ;
        FeatureCell = {} ;
        numItems = 0 ;
        levVector = [] ;
        xCent = [] ;
        for levCnt=scaleInterval(1):scaleInterval(2)
            xImg = getCenterBox(imgSize,2^levCnt) ;
            xCent = [xCent; xImg'] ;
            numItems = numItems +  length(xImg)  ;
            levVector = [levVector; ones(length(xImg),1) ] ;
        end
        Sigma = (2.^levVector).^2 ;
        D = zeros(numItems,numItems) ;
        for cnt1=1:numItems
            l1 = levVector(cnt1) ;
            x1 = xCent(cnt1) ;
            s1 = Sigma(cnt1) ;
            D(cnt1,:) = 0.5*( ((x1 - xCent).^2)./s1 - 1 + (s1./Sigma) - log(s1./Sigma) ) +  ...  % one side of KL
                        0.5*( ((x1 - xCent).^2)./Sigma - 1 + (Sigma./s1) -log(Sigma./s1) ) ;        % the other side     
            %for cnt2=cnt1+1:numItems
            %    l2 = levVector(cnt2) ;
            %    x2 = xCent(cnt2) ;
            %    D(cnt1,cnt2) = 0.5*(x1 - x2)^2/(2^l1) - 1 + 0.5*(2^(l1-l2))^2 +  ...  % one side of KL
            %                   0.5*(x1 - x2)^2/(2^l2) - 1 + 0.5*(2^(l2-l1))^2 ;        % the other side     
            %end
        end
        %D = D + D' ;
        K = exp(-D/s) ;
    case 'gaussKL-nystrom', % the same as "gaussKL-fullRank" except that nystrom methos is used to compute lowrank features
    otherwise
        error('this method for the features are not implemented') ;
end
fprintf('Done !\n') ;

end


% this function yields X and Y coordinates of center of the moving boxes for different scales
% X is the x-coordinate(columns) of the domain and Y is the y-coordinate (rows) of the domain
function x = getCenterBox(imgSize,bxSize)
         %x =  1-bxSize/2:1:imgSize(2)+bxSize/2 ;
         x = 1 - bxSize + bxSize/2:imgSize+bxSize/2-1 ;
end


% this function yields X and Y coordinates of the corners of the moving boxes for different scales
% X is the x-coordinate(columns) of the domain and Y is the y-coordinate (rows) of the domain
function [x1, x2] = getCornerBox(imgSize,bxSize)
         xc = 1 - bxSize + bxSize/2:imgSize+bxSize/2-1 ;
         x1 = xc + bxSize/2 ;
         
         x2 = xc - bxSize/2 ;
end













