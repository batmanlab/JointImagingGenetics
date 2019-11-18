% this function computes coarses *features* that will be used later as a
% low-rank representation for similarity matrix.
%
% output is a feature cell with the following dimension:
%          (#spatialFeatures + #numMovingWindow)  x  #levels x  #numMovingWindow
%          where:
%             (#spatialFeatures + #numMovingWindow) : is total number of features
%             #Levels : different levels of box size
%             #numMovingWindow : number of moving window along the temporal (time) axis
function FeatureCell = computeLandMarkFeatures4TemporalDPP(img,options)
         timeFeatureValue = options.timeFeatureValue ;
         windowLength = options.windowLength ;
         numTimePoints = length(img) ;
         numMovWindows = numTimePoints - windowLength +  1 ;    % NOTICE : here Moving window is over TIME not the DOMAIN
         landmarks = options.landmarks ;
         numLandmarks = size(landmarks,1) ;
         scaleInterval = options.scaleInterval ;
         s = options.sigma ;
         imgSize = size(img{1}) ;

         % extract overlapping features
         fprintf('extract overlapping features ... ') ;
         numLevels = scaleInterval(2) - scaleInterval(1) + 1 ;               
         FeatureCell = cell(numLandmarks + numMovWindows,numLevels,numMovWindows) ;     
         for winCnt=1:numMovWindows % counter for moving windows over time: begin
             for levCnt=scaleInterval(1):scaleInterval(2)
                 [xImg,yImg] = getXYCenterBox(imgSize,2^levCnt) ;
                 [X1, Y1, X2, Y2, X3, Y3, X4, Y4] = getXYCornerBox(imgSize,2^levCnt) ;
                 for fCnt=1:size(FeatureCell,1)  % loop over features in the set (landmarks)
                     if (fCnt<=numLandmarks)  % some of the feature are related to time, check if this feature is related to the location or time
                        if (winCnt==1)   % check if winCnt=1 (ie it is the first time you see this)
                            % for location, compute the overlap ratio if winCnt=1,
                            
                            %FeatureCell{fCnt,levCnt,winCnt} = exp( -( (xImg - landmarks(fCnt,1)).^2 +  (yImg - landmarks(fCnt,2)).^2 ) / (s^2) ) ;
                            FeatureCell{fCnt,levCnt,winCnt} = exp(-min(cat(3,...
                                                                  (X1 - landmarks(fCnt,1)).^2 +  (Y1 - landmarks(fCnt,2)).^2 , ...
                                                                  (X2 - landmarks(fCnt,1)).^2 +  (Y2 - landmarks(fCnt,2)).^2 , ...
                                                                  (X3 - landmarks(fCnt,1)).^2 +  (Y3 - landmarks(fCnt,2)).^2 , ...
                                                                  (X4 - landmarks(fCnt,1)).^2 +  (Y4 - landmarks(fCnt,2)).^2 , ...
                                                                  (xImg - landmarks(fCnt,1)).^2 +  (yImg - landmarks(fCnt,2)).^2 ...
                                                                  ),[],3)/ (s^2)) ;
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
         fprintf('Done !\n') ;
 
end


% this function yields X and Y coordinates of center of the moving boxes for different scales
% X is the x-coordinate(columns) of the domain and Y is the y-coordinate (rows) of the domain
function [X, Y] = getXYCenterBox(imgSize,bxSize)
         %x =  1-bxSize/2:1:imgSize(2)+bxSize/2 ;
         x = 1 - bxSize + bxSize/2:imgSize(2)+bxSize/2-1 ;
         %y =  1-bxSize/2:1:imgSize(1)+bxSize/2 ;
         y = 1 - bxSize + bxSize/2:imgSize(1)+bxSize/2-1 ;
         [X,Y] = meshgrid(x,y) ;
end


% this function yields X and Y coordinates of the corners of the moving boxes for different scales
% X is the x-coordinate(columns) of the domain and Y is the y-coordinate (rows) of the domain
function [X1, Y1, X2, Y2, X3, Y3, X4, Y4] = getXYCornerBox(imgSize,bxSize)
         xc = 1 - bxSize + bxSize/2:imgSize(2)+bxSize/2-1 ;
         yc = 1 - bxSize + bxSize/2:imgSize(1)+bxSize/2-1 ;
         x1 = xc + bxSize/2 ;
         y1 = yc + bxSize/2 ;
         
         x2 = xc + bxSize/2 ;
         y2 = yc - bxSize/2 ;
         
         x3 = xc - bxSize/2 ;
         y3 = yc + bxSize/2 ;
         
         x4 = xc - bxSize/2 ;
         y4 = yc - bxSize/2 ;
         
         [X1,Y1] = meshgrid(x1,y1) ;
         [X2,Y2] = meshgrid(x2,y2) ;
         [X3,Y3] = meshgrid(x3,y3) ;
         [X4,Y4] = meshgrid(x4,y4) ;
end













