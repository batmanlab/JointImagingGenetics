% this is a function to get similarity and quality values in a matrix
% format
% input:
%     FCell : 
%     logScore : 
%     scaleInterval :
%     cutNum :
%     featureIdx :
% output:
%     q : quality core (log-scale)
%     F : a matrix similarity features 
function  [q,F] = getSimQual(FCell,logScore,scaleInterval,featureIdx)
      q = [] ;
      F = [] ;
      if ~isempty(logScore)
        numElements = 0 ;
        for s=scaleInterval(1):scaleInterval(2)
            numElements = numElements + length(logScore(:)) ;
        end  
        for  s=scaleInterval(1):scaleInterval(2)
            q = [q; ...
              reshape(logScore{s}(:),length(logScore{s}(:)),1) ] ;
        end
      end
      
      if ~isempty(FCell)
        numElements = 0 ;
        for s=scaleInterval(1):scaleInterval(2)
            numElements = numElements + length(FCell{featureIdx(1),s}(:)) ;
        end
        F = zeros(numElements,length(featureIdx)) ;
        cnt = 1 ;
        for idx=featureIdx
            f = [] ;
            for s=scaleInterval(1):scaleInterval(2)
                f = [f; ...
                    reshape(FCell{idx,s}(:),length(FCell{idx,s}(:)),1)] ;
            end
            F(:,cnt) = f ;
            cnt = cnt + 1 ; 
        end
      end
end