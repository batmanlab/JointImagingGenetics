% this function approximates the a matrix a low-rank matrix
% U,D,V are outputs of svd or eig 
% spectRatio is the spectral ratio to be retained
function [cutU,cutD,cutV] = lowRankApproxMatrix(U,D,V,spectRatio)
        D = diag(D) ;
        [~,ii] = sort(-D) ;
        D = D(ii) ;
        U = U(:,ii) ;
        cutD = D(cumsum(D)/sum(D) < spectRatio) ;
        cutU = U(:,cumsum(D)/sum(D) < spectRatio)  ;
        cutV = V(:,cumsum(D)/sum(D) < spectRatio)  ;
end