% this function implements different methods to compute the inclusion
% probability
function  [probs, logP] = computeInclusionProbability(y, X, SMatrix, opt)
      switch(opt.method)
          case 'noPrior', %  no prior is assumed over Y (variance is given)
              [probs, logP] = estimate_inclusion_probs_noPrior(y, X,SMatrix,opt) ;
          case 'priorY', % inv-Gamma is assumed over variance of Y
              a_0 = opt.a_0 ;
              b_0 = opt.b_0 ;
              Lambda_0 = opt.Lambda_0 ;
              [probs, logP] = estimate_inclusion_probs_priorY(y, X, SMatrix, ...
                                               a_0, b_0, Lambda_0) ;
          otherwise
              error('this method is not implemented yet !!!') ;
      end
end



% computes the empirical probability of inclusion of a regressor (George way)
function [a, logP] = estimate_inclusion_probs_noPrior(y,A,SMatrix,opt)
      numSamples = size(SMatrix,2) ;
      numItems = size(A,2) ;
      logP = zeros(numSamples,1) ;
      a = zeros(numItems,1) ;
      for cnt=1:numSamples  % looping over samples
          logP(cnt) = logPosterior_Y(find(SMatrix(:,cnt)),y,A,opt) ;
      end
      P = exp(logP - logsumexp(logP,1)) ;
      % 
      for cnt=1:numItems
          inSetIdx = SMatrix(cnt,:)==1 ;
          a(cnt) = sum(P(inSetIdx)) ;
      end
end


% computes log unnormalized posterior of S
function val = logPosterior_S(S,y,B,opt) 
    logY = logPosterior_Y(S,y,B,opt) ;
    val = logY + log(det(L(S,S))) ;
end

% computes log of unnormalized posterior of Y
function val = logPosterior_Y(S,y,B,opt)
    m = length(y) ;
    Sigma_w = opt.Sigma_w*eye(length(S)) ;
    sig = opt.sigma ;
    M = B(:,S)*Sigma_w*B(:,S)' + sig*eye(m)  ;
    val = -0.5*y'*(M\y) -0.5*log(det(M))  ;
end


% estimate each dictionary element's inclusion probability using
% importance sampling
% ** NOTE **: in this case inverse-Gamma is assumed over Y
function [probs, logP] = estimate_inclusion_probs_priorY(y, X, SMatrix, ...
                                               a_0, b_0, Lambda_0)
    n = size(SMatrix, 2);
    r = size(X, 2);
    
    logP = zeros(n, 1);
    for idx = 1:n
        [~, ~, ~, ~, logP(idx)] = ...
            compute_weight_priorY(find(SMatrix(:, idx)), y, X, ...
                           a_0, b_0, Lambda_0);
    end
    
    weights = exp(logP - logsumexp(logP,1)) ; 
    %weights = weights / sum(weights);
    
    probs = zeros(r, 1);
    for idx = 1:r
        samples_containing_idx = SMatrix(idx, :) == 1;
        probs(idx) = sum(weights(samples_containing_idx));
    end
end

% computes importance sampling weight
% ** NOTE **: in this case inverse-Gamma is assumed over Y
function [Lambda_S, mu_S, a_S, b_S, weight] = ...
  compute_weight_priorY(S, y, X, a_0, b_0, Lambda_0)
    n = size(X,1) ;
    Lambda_S = Lambda_0(S, S) + X(:, S)'*X(:, S); % |S| by |S|
    mu_S     = Lambda_S \ (X(:, S)' * y);
    a_S      = a_0 + n/2;
    b_S      = b_0 + 1/2*(y'*y - mu_S'*Lambda_S*mu_S);
    
    e1 = eig(Lambda_0(S, S)) ;
    e2 = eig(Lambda_S) ;
    
    %weight = 0.5*( log(det(Lambda_0(S, S))) - log(det(Lambda_S) )  -n*log(2*pi) ) + ...
    %         log(b_0)*a_0 - log(b_S)*a_S + gammaln(a_S) - gammaln(a_0);
         
    weight = 0.5*( sum(log(e1)) - sum(log(e2))  -n*log(2*pi) ) + ...
             log(b_0)*a_0 - log(b_S)*a_S + gammaln(a_S) - gammaln(a_0);     
end
