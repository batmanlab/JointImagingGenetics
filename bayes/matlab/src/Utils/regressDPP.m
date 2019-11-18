% this function implements the sampling-optimization algorthm in the report
% A : regressors (M x N) : 
% Phi : simil;arity features (P x N)
% y : labels ( M x 1)
% -----------------------
% N : number of regressors (Items)
% M : number of observations
% P : number of features (rank(L) <= P )
function [a,K,Phi, selIdx] = regressDPP(y,A,Phi,opt)
    [~,numItems] = size(A) ;
    %[U,S] = eig(Phi*Phi') ;
    %V_space = U(:,diag(S)<eps) ;  % find the Null-space
    %V_space = V_space*V_space' ;
    %C = decompose_kernel(Phi'*Phi);
    q0 = sqrt( sum(Phi.^2,2) ) ;
    Phi_nrm = bsxfun(@times,Phi,1./q0) ;
    L = diag(q0)*(Phi_nrm*Phi_nrm')*diag(q0) ;
    %[U, D] = eig(L);
    %B = U*diag(sqrt(diag(D)));
    C = decompose_kernel(Phi'*Phi);
    
    opt.L = L ;
    numSamples = opt.numSamples ;
    K = [] ;

    K0 = opt.L*pinv(eye(length(opt.L)) + opt.L) ;
    a0 = zeros(numSamples,1) ;
    Phi0 = Phi ;
    iter = 1 ;
    while 1
      % sample from DPP
      S = cell(numSamples,1) ;
      SMatrix = zeros(numItems,numSamples) ;
      if   isequal(opt.sampDist,'uniform')
          for cnt=1:numSamples
              tmpIdx = randperm(numItems) ;
              S{cnt} = tmpIdx(1:opt.numTopBasis) ;
              SMatrix(S{cnt},cnt) = 1 ;
          end
      elseif   isequal(opt.sampDist,'dpp')
          for cnt=1:numSamples
              if opt.useKDpp
                  S{cnt} = sample_dual_dpp(Phi,C,opt.expCard) ;
              else
                  S{cnt} = sample_dual_dpp(Phi,C) ;
              end
              SMatrix(S{cnt},cnt) = 1 ;
          end
      else
          error('unkown sampling distribution') ;
      end

      % compute emprirical estimate for the inclusion proability (a)
      %a = computeEmpiricalInclusion_George(y,A,SMatrix,opt) ;
      a = computeInclusionProbability(y, A, SMatrix, opt.inclCompOpt) ;
      
      switch (opt.method)
          case 'cvx',
              % solve the optimization ptoblem with respect to K
              cvx_begin
                variable K(numItems,numItems) symmetric
                minimize(trace(K*Phi_nrm))
                subject to
                    K == semidefinite(numItems)
                    (eye(numItems) - K) == semidefinite(numItems)
                    diag(K) == a
              cvx_end
              if ~isnan(cvx_optval)
                  [U,D] = eig(K) ;
                  [cutU,cutD,~] = lowRankApproxMatrix(U,D,U',0.99) ;
                  eigL = 1./(1 - cutD) ;
                  L = cutU*diag(eigL)*cutU' ;
                  Phi = cutU*diag(sqrt(eigL)) ;
                  C = decompose_kernel(Phi'*Phi);
                  a0 = a ;
                  Phi0 = Phi ;
                  opt.L = L ;
                  fprintf(['|| K_new - K_old || = ' num2str(norm(K - K0)) '\n']) ;
                  K0 = K ;
              else
                  if iter > 1
                      a = a0 ;
                      K = K0 ;
                      Phi = Phi0 ;
                  end
                  break ;
              end
              iter = iter + 1 ;
              if (iter > opt.maxIter)
                  break ;
              end
          case 'topProb',
              [~,jj] = sort(a);
              idx = zeros(size(a));
              idx(jj(end-opt.numTopBasis + 1:end)) = 1; 
              selIdx = find(idx) ;
              return
          case 'MAP',
              fitDPPOpt = [] ;
              fitDPPOpt.mode = 'basic' ;
              fitDPPOpt.maxItr = 100 ;
              fitDPPOpt.tol = 1e-4 ;

              m = a ;
              mIdx_significant = find(m>1e-4) ;
                
              [q, relErrHist ]  = fitDPPbyMarginals(q0,Phi_nrm,...
                                      m(mIdx_significant),1:length(mIdx_significant),...
                                      fitDPPOpt) ;
                                  
              mapDPP_idx = softmax(Phi_nrm(mIdx_significant,:)*Phi_nrm(mIdx_significant,:)',...
                                   log(q(mIdx_significant)), opt.qWeight) ;
              selIdx =  mIdx_significant(mapDPP_idx) ;
              return
          otherwise
              error('This method is not supported') ;
      end
      
      
      
    end
end

%function factorDPP_MCMC(y,x,model,opt)
%   while 1
%       % sample from prior Theta: check p8
%       sigma = 1./gamrnd(hyper.sigma.a,hyper.sigma.b,1) ;
%
%       % sample/optimize w
%       w =  mvnrnd(hyper.w.mu,hyper.w.sigma) ;
%
%       % sample DPP
%       S = sample_dual_dpp(B,C,k) ;
%       
%       % accept/reject
%       logp_t = logPosterior_S(S,y,x,w) ;
%           % update params
%       if iter > numSamples
%           break;
%       end
%   end
%end

% % computes the empirical probability of inclusion of a regressor (Kayhan way)
% function a = computeEmpiricalInclusion_Kayhan(y,A,SMatrix,opt)
%       numSamples = size(SMatrix,2) ;
%       numItems = size(A,2) ;
%       logP = zeros(numSamples,1) ;
%       a = zeros(numItems,1) ;
%       for cnt=1:numSamples  % looping over samples
%           logP(cnt) = logPosterior_S(find(SMatrix(:,cnt)),y,A,opt) ;
%       end
%       % 
%       for cnt=1:numItems
%           inSetIdx = find(SMatrix(cnt,:)==1) ;
%           notInSetIdx = find(SMatrix(cnt,:)==0) ;
%           if length(inSetIdx) < length(notInSetIdx)  % there are more zeros than ones
%               idx1 = inSetIdx ;
%               idx0 = randperm(length(notInSetIdx)) ;
%               idx2 = notInSetIdx(idx0(1:length(idx1))) ;   % randomly choose the same number of elements
%           else  % there are more ones that zeros
%               idx2 = notInSetIdx ;
%               idx0 = randperm(length(inSetIdx)) ;
%               idx1 = inSetIdx(idx0(1:length(idx2))) ;
%           end
%           %a(cnt) = exp(logsumexp(logP(SMatrix(cnt,:)==1),1) - logsumexp(logP(SMatrix(cnt,:)==0),1) ) ;
%           r = exp(logsumexp(logP(idx1),1) - logsumexp(logP(idx2),1) ) ;
%           a(cnt) = r/(1+r) ;
%       end
% end
% 
% 
% % computes the empirical probability of inclusion of a regressor (George way)
% function a = computeEmpiricalInclusion_George(y,A,SMatrix,opt)
%       numSamples = size(SMatrix,2) ;
%       numItems = size(A,2) ;
%       logP = zeros(numSamples,1) ;
%       a = zeros(numItems,1) ;
%       for cnt=1:numSamples  % looping over samples
%           logP(cnt) = logPosterior_Y(find(SMatrix(:,cnt)),y,A,opt) ;
%       end
%       P = exp(logP - logsumexp(logP,1)) ;
%       % 
%       for cnt=1:numItems
%           inSetIdx = SMatrix(cnt,:)==1 ;
%           a(cnt) = sum(P(inSetIdx)) ;
%       end
% end
% 
% 
% % computes log unnormalized posterior of S
% function val = logPosterior_S(S,y,B,opt) 
%     logY = logPosterior_Y(S,y,B,opt) ;
%     val = logY + log(det(L(S,S))) ;
% end
% 
% % computes log of unnormalized posterior of Y
% function val = logPosterior_Y(S,y,B,opt)
%     m = length(y) ;
%     Sigma_w = opt.Sigma_w*eye(length(S)) ;
%     sig = opt.sigma ;
%     M = B(:,S)*Sigma_w*B(:,S)' + sig*eye(m)  ;
%     val = -0.5*y'*(M\y) -0.5*log(det(M))  ;
% end
% 
% 
