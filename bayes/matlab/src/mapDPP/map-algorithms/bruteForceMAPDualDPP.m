% this function searches for MAP of DPP by sampling
function [minVal,mapIdx] = bruteForceMAPDualDPP(q,Phi,maxItr,k)
    mapIdx = [] ;
    if nargin  > 3
      minVal = -inf ;
    else
      minVal = 0 ;  % for the empty set
    end
    itr = 1 ;
    C = decompose_kernel(Phi'*diag(q.^2)*Phi) ;
    L = diag(q) * (Phi*Phi')* diag(q) ;
    while itr < maxItr
         if nargin >3
            S = sample_dual_dpp( diag(q)*Phi, C, k) ;
         else
            S = sample_dual_dpp( diag(q)*Phi, C) ;
         end 
         e = eig(L(S,S))  ;
         m = sum( log(e(e>0)) ) ;
         if minVal <  m 
            minVal = m ;
            mapIdx = S ;
            fprintf(' ** %i **',itr) ;
         end
         itr = itr +  1 ;
    end
    assert(length(unique(mapIdx))==length(mapIdx),'There are redundant elements in the set !!') ;
    fprintf('\n') ;
end
