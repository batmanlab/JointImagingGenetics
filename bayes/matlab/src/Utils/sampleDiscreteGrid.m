function [y] = sampleDiscreteGrid(p)
   % Returns a sample from a discrete probability mass function indexed by p which is provided as a grid
   gridSz = size(p) ;
   numStates = gridSz(end) ;
   gridSz = gridSz(1:end-1) ;
   numDim = length(gridSz) ;
   if ~((numDim==2) || (numDim==3) )
      error('this function is only implemented for 2D and 3D images !!!') ;
   end


   U = rand( gridSz );
   u = zeros( gridSz );
   y = zeros( gridSz );
   lockGrid = zeros( gridSz ) ;
  
   for statCnt=1:numStates
       if (numDim == 2)
          u  =  u  + p(:,:,statCnt)  ;
       elseif (numDim == 3)
          u =  u  +  p(:,:,:,statCnt) ;
       end
       % find variables which are not locked and satisfy the condition
       ind = find( (lockGrid==0) & ( u > U ) ) ;
       y(ind) = statCnt ;
       lockGrid = u > U  ;     % update the lockGrid variable  
   end

end
