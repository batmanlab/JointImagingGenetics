% this function implements fixed form variational bayese to compute the quality of the DPP
% X : regressors (M x N) : 
% Phi : similarity features (N x P)
% y : labels ( M x 1)
% if number of items and features are not the same a mapping should be provided.
% -----------------------
% N : number of regressors (Items)
% M : number of observations
% P : number of features (rank(L) <= P )
% Here is an example on how to use it:
%   opt.maxItr = 10000 ;
%   opt
function [eta] = fixedFormVB(y,X,Phi,opt,itemToFeatureMap)
   if  (nargin<5)   % no mapping is provided, so it should be identity map
      itemToFeatureMap = [] ;
   end
   % set up environment and Initialization
   maxItr = opt.maxIter ;
   numItems = size(Phi,1) ;
   
   if isfield(opt,'numDrawBeforeInv')
       numDrawBeforeInv = opt.numDrawBeforeInv ;    % number of random draw before applying regression
   else
       numDrawBeforeInv = 1 ;    % number of random draw before applying regression
   end
   
   
   if isfield(opt,'useIntercept')
       useIntercept = opt.useIntercept ;
   else
       useIntercept = false ;
   end
   
   if maxItr < numDrawBeforeInv
       error('number of max iterations should be bigger than 3*numItems !!') ;
   end
   
   if isfield(opt,'initialFcn')  % function is provided to initialize eta and C
       regWeight = 0.0 ;
       [eta,C] = opt.initialFcn() ;
       
   else   % we assume that it is a initialization for DPP
       
       if opt.normalizePhi
           q0 = sqrt( sum( Phi.*conj(Phi) ,2) ) ;
           Phi = bsxfun(@times,Phi,1./q0) ;
           Phi(q0==0,:) = 0 ;  % in case q0 = 0
       else
           q0 = ones(numItems,1) ;
       end
       L = diag(q0)*(Phi*Phi')*diag(q0) ;
       %regWeight = 0.001 ;
       regWeight = 0.0 ;
       if ~isfield(opt,'stopEta0')
           opt.stopEta0 = inf ;
       end
       if ~isfield(opt,'plotEta')
           opt.plotEta = true ;
       end
       
       
       %eta = [log(q0) ; -log(det(L+eye(numItems)))]  ;
       eta = [2*log(q0) ; 0]  ;
       K = computeKHelper(Phi, eta(1:end-1)) ;
       if opt.useMarginal4C    % should
           C = [K diag(K) ; diag(K)' 1];
       else
           C = [diag(diag(K))  zeros(numItems,1) ; zeros(1,numItems)  1];  C = real(C) ;
       end
   end
   
   g = C*eta ;
   C_bar = zeros(numItems+1,numItems+1) ;
   g_bar = zeros(numItems+1,1) ;
   w = 1/sqrt(maxItr) ; 

   % for loop
   %status = 0 ;
   itr = 0 ;
   %fprintf('iter : min(eta)  mean(eta)  median(eta)  max(eta)   eta0   sum(sum(SMatrix)==0) \n') ;
   fprintf('iter:  logData    logPrior    min(diag(L))    sum(diag(K))   \n  ') ;
   while 1
      % sample from approximate distribution
      if isfield(opt,'approxSampleFcn')    % if the user provided a function for sampling
          SMatrix = opt.approxSampleFcn(eta(1:end-1),opt.numBatchSamp) ;
          status = 0 ;
          
          %if sum(SMatrix)==0
          %    fprintf('skipping sample ...\n')
          %    continue
          %end
      else   % we assume it is DPP
          try
              [SMatrix,status] = sampleFromDPP(eta(1:end-1)/2, Phi, opt) ;
              if (status>0)
                  warning('we stop the algorithm here ... \n') ;
                  break
              end
          catch EXPRESSION
              EXPRESSION
              fprintf('Error : DPP sampling died, check eta \n') ;
              break ;
          end
      end

      % in case Phi ~= X, there should be a mapping between items and features
      if ~isempty(itemToFeatureMap)
          SMatrix2 = zeros(size(X,2), opt.numBatchSamp)  ;
          for cnt=1:opt.numBatchSamp
              idx = item2Feature(find(SMatrix(:,cnt)), itemToFeatureMap) ;
              SMatrix2(idx,cnt) = 1 ;
          end
          %SMatrix = SMatrix2 ;
          %clear SMatrix2
      else  % identity map
          SMatrix2 = SMatrix ;
      end

      % compute the join likelihood   
      if isfield(opt,'joinLikelihoodFcn')   % likelihood function is provided
          logData = opt.joinLikelihoodFcn(y, X, SMatrix2) ;
      else   % we assume it is regression likelihood
          if useIntercept
              [~,logData] = computeInclusionProbability(y, [X ones(size(X,1),1) ], [SMatrix2; ones(1,size(SMatrix2,2))], opt.inclCompOpt)  ;
          else
              [~,logData] = computeInclusionProbability(y, X, SMatrix2 , opt.inclCompOpt)  ;
          end
      end
      
      % compute prior
      if isfield(opt,'priorFcn')  % distribution of the prior is provided
          logPrior = opt.priorFcn(SMatrix)  ;
      else  % We assume Bernoulli
          logPrior = log(opt.priorInclusion)*sum(SMatrix)';
      end
      
      % compute the log of the base measure
      logBase = opt.logBaseFcn(Phi, SMatrix) ;
      
      logP = logData + logPrior - logBase;  
  
      % compute the gradient (g) 
      g_hat = [SMatrix ; ones(1,size(SMatrix,2))]*logP ;
      g_hat = 1/opt.numBatchSamp * g_hat ;

      % compute the design matrix (C)
      if opt.useMarginal4C    % should 
          K = computeKHelper(Phi, eta(1:end-1)) ;
          C_hat = [K diag(K) ; diag(K)' 1]; 
      else
         C_hat = 0 ;
         for ii=1:opt.numBatchSamp
             C_hat = C_hat + [SMatrix(:,ii);1]*[SMatrix(:,ii);1]' ;
         end
         C_hat = 1/opt.numBatchSamp * C_hat ;
      end
      

      % update statistics
      g = (1 - w)*g + w*g_hat ;
      C = (1 - w)*C + w*C_hat ;

      % update the parameter
      if (itr > numDrawBeforeInv)
        % I don't know if extra regularization fixes the problem or not but it is worth testing
        eta = (C + [regWeight*eye(numItems) zeros(numItems,1);zeros(1,numItems+1)]) \ g ; 
        %eta = C \ g ;
        %eta = lsqlin(C,g,[],[],[],[],[5*ones(numItems,1); -inf],inf*ones(numItems+1,1),eta)  ;
        assert(~any(isnan(eta)),'there are NaN in the eta, something went wrong !!') ;
      end
      
      assert(any(isnan(eta))==0,'NaN value in eta !!') ;
      if opt.plotEta
        plot(eta(1:end-1))
        drawnow
      end

      % compute accumulative statistics
      if (itr > opt.maxIter/2)
         g_bar = g_bar + g_hat ;
         C_bar = C_bar + C_hat ;
      end      

      % monitor progress
      if (itr > opt.maxIter)
         break ;
      end
      %fprintf('iter %d : %f  %f  %f  %f  %f  %d \n ',itr, min(eta), mean(eta), median(eta), max(eta), eta(end), sum(sum(SMatrix)==0) ) ;
      fprintf('iter   %d:  logData    logPrior    min(diag(L))    sum(diag(K))   \n  ') ;
      if ~isfield(opt,'approxSampleFcn')   % it default DPP
        q = exp(eta(1:end-1)/2) ;
        L = diag(q)*(Phi*Phi')*diag(q) ;
        K = computeKHelper(Phi, eta(1:end-1)) ;
      end
      fprintf('iter (%d) : ',itr ) ; 
      fprintf('** logData (%4.4f) ** ',logData) ;
      fprintf('** logPrior (%4.4f) ** ', logPrior ) ;
      if ~isfield(opt,'approxSampleFcn')   % it default DPP
        fprintf('** min_diag(L) (%4.4f) ** ', min(diag(L)) ) ;
        fprintf('** sum(diag(K)) (%4.4f) ** ', sum(diag(K)) ) ;
        fprintf('** min(K(:)) (%1.4f) ** \n', min(K(:))) ;
      else
          fprintf('\n') ;
      end
      
      itr = itr + 1 ;
      
  
   end
   % compute final output
   if (status==0)
     etaOld = eta ;
     try 
        eta = (C_bar + [regWeight*eye(numItems) zeros(numItems,1);zeros(1,numItems+1)]) \g_bar ;
        if any(isnan(eta))
            eta = pinv((C_bar/itr))*(g_bar/itr) ;
        end
        assert(~any(isnan(eta)),'there are NaN in the eta, something went wrong, switching to old eta !!') ;
     catch EXPRESSION
         warning('be careful !!! the last inversion failed ! I am reporting the latest eta !!!') ;
         eta = etaOld ;
     end
     %eta = C_bar\g_bar ;
     %eta = lsqlin(C_bar,g_bar,[],[],[],[],[5*ones(numItems,1); -inf],inf*ones(numItems+1,1),eta) ;
   else
       warning('!!!  DIDN''T REACH TO THE HALF OF ITERATION !!!') ;
   end
end



% this is a helper function to map items to the features
function   outIdx = item2Feature(inIdx, itemToFeatureMap) 
        outIdx = [] ;
        for idx = inIdx(:)'
            outIdx = union(outIdx,itemToFeatureMap(idx).idx) ;
        end  
end
 

% helper function to compute marginal matrix
function K = computeKHelper(Phi,eta)
    N = size(Phi,1) ;
    L = diag(exp(eta/2))*Phi ;
    L = L*L' ;
    K = (L + eye(N))\L ;
end

% sample from DPP
function [SMatrix, status] = sampleFromDPP(eta, Phi, opt)
      % update the dual kernel according to eta
      status = 0 ;
      try
          q = exp(eta) ;
          C = decompose_kernel(Phi'*diag(q.^2)*Phi) ;
      catch expr
          %if any(isinf(q.^2))     % q is unusually large
          %    warning('eta is unusally out of bound, the sampling code is trying to handle it !!!') ;
          %    [U,S,~] = svd(Phi'*diag(q)) ;
          %    C.V = real(U) ;
          %    C.D = S.^2 ;
          %else
          %    expr
          %    error('Error : there is an error in sampling that I cannot fix !!!') ;
          %end
          status = 1 ;
          return
      end
      
      % sample from DPP
      numItems = size(Phi,1) ;
      S = cell(opt.numBatchSamp,1) ;
      SMatrix = zeros(numItems, opt.numBatchSamp) ;

      if any(C.D<=0)
          warning('at least one of the eigen values turn negative which is a sign numerical instability') ;
          idx = find(C.D<=0 | imag(C.D)~=0) ;
          C.D(idx) = [] ;
          C.V(:,idx) = [] ;
          %status = 1 ;
          %return
      end

      
      for cnt=1:opt.numBatchSamp
          if opt.useKDpp
             S{cnt} = sample_dual_dpp( diag(q)*Phi, C, opt.expCard) ;
          else
             S{cnt} = sample_dual_dpp( diag(q)*Phi, C) ;
          end
          
          SMatrix(S{cnt},cnt) = 1 ;
      end
      
      
 
end



