% this function computes coarses *features* that will be used later as a
% low-rank representation for similarity matrix.
%
% output is a feature cell with the following dimension:
%          (#spatialFeatures + #numMovingWindow)  x  #levels x  #numMovingWindow
%          where:
%             (#spatialFeatures + #numMovingWindow) : is total number of features
%             #Levels : different levels of box size
%             #numMovingWindow : number of moving window along the temporal (time) axis
function FeatureCell = computeCoarseFeatures4TemporalDPP(img,options)
      timeFeatureValue = options.timeFeatureValue ;
      windowLength = options.windowLength ;
      numTimePoints = length(img) ;
      numMovWindows = numTimePoints - windowLength +  1 ;    % NOTICE : here Moving window is over TIME not the DOMAIN

       switch(options.DPPFeatureMode)  
           % coarse grid the domain into a chess board and construct features 
           % which are basically overlap of the moving domain with each 
           % element of the chess board. These feature can be computed via 
           % convolution.
           case 1     
               ImgSize = size(img{1}) ;
               scaleInterval = options.scaleInterval ;
               numSegments = options.numSegments ;
               Seg = 1:numSegments ;
               Seg = imresize(reshape(Seg,[sqrt(numSegments) sqrt(numSegments)])',ImgSize,'nearest') ;
               
               % extract overlapping features
               fprintf('extract overlapping features ... ') ;
               numLevels = scaleInterval(2) - scaleInterval(1) + 1 ;               
               FeatureCell = cell(numSegments + numMovWindows,numLevels,numMovWindows) ;     
               for  winCnt=1:numMovWindows  % counter for moving windows: begin 
                    for fCnt=1:size(FeatureCell,1)  % one counter for features in the set
                        for levCnt=scaleInterval(1):scaleInterval(2)
                            if (fCnt<=numSegments)  % check if this feature is related to the location or time
                                 if (winCnt==1)   % check if winCnt=1 (ie it is the first time you see this)
                                     % for location, compute the overlap ratio if winCnt=1,
                                     segMask = double(Seg==fCnt) ;   
                                     FeatureCell{fCnt,levCnt,winCnt} = fastComputeOverlapCoeff(segMask,2^levCnt) ;
                                     FeatureCell{fCnt,levCnt,winCnt} = sparse(FeatureCell{fCnt,levCnt,winCnt}) ;
                                 else % otherwise it is just copying from the first time you compute it
                                     FeatureCell{fCnt,levCnt,winCnt} = FeatureCell{fCnt,levCnt,1} ;
                                 end
                            else % if this feature is time related, 
                                 %compute the ratio of overlap with other windows
                                 if  any(ismember(winCnt:(winCnt + windowLength - 1),(fCnt - numSegments):(fCnt - numSegments + windowLength - 1)))
                                     val = sum(ismember(winCnt:(winCnt + windowLength - 1),(fCnt - numSegments):(fCnt - numSegments + windowLength - 1)))/windowLength*timeFeatureValue ;
                                 else
                                     val = 0 ;
                                 end
                                 FeatureCell{fCnt,levCnt,winCnt} = val*ones(size(FeatureCell{1,levCnt,1})) ;
                            end
                        end
                    end
               end  % counter for moving window: end
               fprintf('Done !\n') ;
 
           case 2     % segment the image into super pixels 
               error('You first have to figure out how to handle temporal segmentation...., fix this part of the code !!!') ;
               scaleInterval = options.scaleInterval ;
               I(:,:,1) = im2double(img);
               I(:,:,2) = im2double(img);
               I(:,:,3) = im2double(img);

               N = size(I,1);
               M = size(I,2);
               N_sp=200;
               N_sp2=1000;
               N_ev = options.numSegments ;  % number of eigenvectors
               % ncut parameters for superpixel computation
               diag_length = sqrt(N*N + M*M);
               par = imncut_sp;
               par.int=0;
               par.pb_ic=1;
               par.sig_pb_ic=0.05;
               par.sig_p=ceil(diag_length/50);
               par.verbose=0;
               par.nb_r=ceil(diag_length/60);
               par.rep = -0.005;  % stability?  or proximity?
               par.sample_rate=0.2;
               par.nv = N_ev;
               par.sp = N_sp;

               % Intervening contour using mfm-pb
               fprintf('running PB \n');
               [emag,ephase] = pbWrapper(I,par.pb_timing);
               emag = pbThicken(emag);
               par.pb_emag = emag;
               par.pb_ephase = ephase;
               clear emag ephase;

               st=clock;
               fprintf('Ncutting...');
               [Sp,Seg] = imncut_sp(I,par);
               fprintf(' took %.2f minutes\n',etime(clock,st)/60);

               I_sp = segImage(I,Sp);
               I_seg = segImage(I,Seg);

               % extract overlapping features
               fprintf('extract overlapping features ... ') ;
               numTimePoints = length(img) ;
               numLevels = scaleInterval(2) - scaleInterval(1) + 1 ;
               numSeg = length(unique(Seg(:))) ;
               FeatureCell = cell(numSeg + numTimePoints,numLevels) ;
               for cnt=1:numSeg
                  segMask = double(Seg==cnt) ;   
   
                  for levCnt=scaleInterval(1):scaleInterval(2)
                    FeatureCell{cnt,levCnt} = fastComputeOverlapCoeff(segMask,2^levCnt) ;
                    FeatureCell{cnt,levCnt} = sparse(FeatureCell{cnt,levCnt}) ;
                  end
               end
               for  cnt=numSegments+1:numSegments+numTimePoints
                  for  levCnt=scaleInterval(1):scaleInterval(2)
                       FeatureCell{cnt,levCnt} = cnt*ones(size(FeatureCell{1,levCnt})) ;    
                  end 
               end

               fprintf('Done !\n') ;

           otherwise
              error('computeCoarseFeatures4DPP : Unkown Mode for feature construction !!') ;
       end
end



% this function computes Haar Coefficents using integral image
%  [sc, err, notCovered_coef, notCovered_err] = FastComputeHaarCoeff(img,bxSize)
%  Inputs:
%   img : input image
%   bxSize : box size 
%  Output:
%   sc : scale coefficients
%   err: reconstruction error
%   notCovered_coef:   scale coefficient of not covered area
%   notCovered_err:    reconstruction err of not covered area
%   nrm :              area covered by the box
%   notCov_nrm :       are not covered by the box
%
%       p4         p6         p2
%       *----------*----------*
%       |          |          |
%       |          |          |
%       |          |p9        |
%     p8*----------*----------*p7
%       |          |          |
%       |          |          |
%       |          |          |
%      p3*----------*----------*p1
%                   p5 
function   [sc, err, notCovered_coef, notCovered_err, nrm, notCov_nrm] = fastComputeOverlapCoeff(observedImg,bxSize)
     
      img = observedImg ;
      img_int = cumsum(cumsum(padarray(img,[bxSize bxSize]),2)) ;
      img_sq_int = cumsum(cumsum(padarray(img.^2,[bxSize bxSize]),2)) ;    % integral of the square of image
      ones_int = cumsum(cumsum(padarray(ones(size(img)),[bxSize bxSize]),2)) ;    % integral of the square of image
      
      Sz0 = size(img) ;
      Sz1 = size(img_int) ;
      %Sz2 = max([size(img_int,1)-max(0,bxSize-1),size(img_int,2)-max(0,bxSize-1)],0) ;
      Sz2 = [Sz0(1) + bxSize - 1, Sz0(2) + bxSize - 1] ;
      sc = zeros(Sz2) ;                 % scale coefficients
      %d1 = zeros(Sz2) ;                 % detail 1 coefficients
      %d2 = zeros(Sz2) ;                 % detail 2 coefficients
      %d3 = zeros(Sz2) ;                 % detail 3 coefficients
      err = zeros(Sz2) ;                % reconstruction error with the scale coefficient
      nrm = zeros(Sz2) ;
      notCovered_coef = zeros(Sz2) ;       % it holds the coefficient of area not covered by the box
      notCovered_err  = zeros(Sz2) ;       % reconstruction error with the notCovered_coef
      img_Total = sum(img(:)) ;        % total sum of all elements of the image
      img_sq_Total = sum(img(:).^2) ; % total sum of all elements of the square of the image
      %[I,J] = meshgrid( Sz1(1):-1:Sz1(1)-Sz2(1)+1, Sz1(2):-1:Sz1(2)-Sz2(2)+1) ;
      [I,J] = meshgrid( Sz1(1)-1:-1:Sz1(1)-Sz2(1), Sz1(2)-1:-1:Sz1(2)-Sz2(2)) ;
      [I0,J0] = meshgrid( Sz2(1):-1:1, Sz2(2):-1:1) ;
      %img_int = [img_int  zeros(size(img_int,1),1)] ;   % extend it by one extra column. It will be used later in this function
      %img_sq_int = [img_sq_int  zeros(size(img_int,1),1)] ; 
      %ones_int = [ones_int zeros(size(img_int,1),1) ] ;
      
      ind0 = sub2ind(size(sc), I0(:), J0(:)) ;
      %% computing scale coefficients
      %ind1 = sub2ind([Sz1(1) Sz1(2)+1], I(:),J(:)) ;   % linear index for the corner on the lower-right
      ind1 = sub2ind([Sz1(1) Sz1(2)], I(:),J(:)) ;   % linear index for the corner on the lower-right
      I1 = I-bxSize ;
      J1 = J ;
      %J1(I1<1) = Sz1(2)+1 ;   % if the index is outside of an image, map it the last column, last row 
      %I1(I1<1) = Sz1(1) ;
      %ind2 = sub2ind([Sz1(1) Sz1(2)+1], I1(:), J1(:)) ;   % linear index for the corner on the upper-right
      ind2 = sub2ind([Sz1(1) Sz1(2)], I1(:), J1(:)) ;   % linear index for the corner on the upper-right 
      
      I1 = I ;
      J1 = J-bxSize ;
      %I1(J1<1) = Sz1(1) ;   % if the index is outside of an image, map it the last column, last row 
      %J1(J1<1) = Sz1(2)+1 ;
      %ind3 = sub2ind([Sz1(1) Sz1(2)+1], I1(:), J1(:)) ;   % linear index for the corner on the lower-left
      ind3 = sub2ind([Sz1(1) Sz1(2)], I1(:), J1(:)) ;   % linear index for the corner on the lower-left

      I1 = I-bxSize ;
      J1 = J-bxSize ;
      %J1( I1<1 ) = Sz1(2) + 1 ;   % if the index is outside of an image, map it the last column, last row
      %I1( I1<1 ) = Sz1(1)  ; 
      %I1( J1<1 ) = Sz1(1) ;
      %J1( J1<1 ) = Sz1(2) + 1 ;
      %ind4 = sub2ind([Sz1(1) Sz1(2)+1], I1(:), J1(:)) ;   % linear index for the corner on the lower-right
      ind4 = sub2ind([Sz1(1) Sz1(2)], I1(:), J1(:)) ;   % linear index for the corner on the lower-right

      sc(ind0) = img_int(ind1) + img_int(ind4) - img_int(ind2) - img_int(ind3) ;
      nrm(ind0) = ones_int(ind1) + ones_int(ind4) - ones_int(ind2) - ones_int(ind3) ;
      notCov_nrm = Sz0(1)*Sz0(2) - nrm ;
      sc = sc./nrm ;
      notCovered_coef =  (img_Total - sc.*nrm )./notCov_nrm ;
      notCovered_coef(isnan(notCovered_coef)) = 0 ;   % speciall case when whole image is covered
      notCovered_coef(isinf(notCovered_coef)) = 0 ;   % speciall case when whole image is covered
      %% compute error based on scale coefficient
      err_term1 = zeros(size(err)) ;
      err_term2 = zeros(size(err)) ;
      err_term3 = zeros(size(err)) ;
      err_term1(ind0) = img_sq_int(ind1) + img_sq_int(ind4) - img_sq_int(ind2) - img_sq_int(ind3) ;    % conv2(Img_pad.^2,MaskImg,'valid') 
      err_term2(ind0) = (sc(ind0).^2).*( ones_int(ind1) + ones_int(ind4) - ones_int(ind2) - ones_int(ind3)  )  ;     %(ScaleCoeff{levCnt}.^2).*conv2(ones(size(Img_pad)),MaskImg,'valid'i) 
      err_term3(ind0) = -2*(sc(ind0)).*( img_int(ind1) + img_int(ind4) - img_int(ind2) - img_int(ind3) ) ;        %-2*ScaleCoeff{levCnt}.*conv2(Img_pad,MaskImg,'valid')
      err = err_term1 + err_term2 + err_term3 ;      

      %% compute error based on notCovered_coef (i.e. area whic is not covered by the box)
      
      err_term2 = zeros(size(notCovered_err)) ;
      err_term3 = zeros(size(notCovered_err)) ;
      err_term2(ind0) = (notCovered_coef(ind0).^2).*( ones_int(ind1) + ones_int(ind4) - ones_int(ind2) - ones_int(ind3)  )  ;     %(ScaleCoeff{levCnt}.^2).*conv2(ones(size(Img_pad)),MaskImg,'valid'i) 
      err_term3(ind0) = -2*(notCovered_coef(ind0)).*( img_int(ind1) + img_int(ind4) - img_int(ind2) - img_int(ind3) ) ;        %-2*ScaleCoeff{levCnt}.*conv2(Img_pad,MaskImg,'valid')
      errTotal = img_sq_Total + (notCovered_coef.^2)*Sz0(1)*Sz0(2) - 2*notCovered_coef*img_Total    ;
      notCovered_err =  errTotal - ( err_term1 + err_term2 + err_term3 )  ;    
end












