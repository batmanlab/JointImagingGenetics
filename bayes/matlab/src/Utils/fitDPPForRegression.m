%
function  q = fitDPPForRegression(y,X,Phi,opt)
    r = size(X, 2); % number of dictionary elements/regressors
    q0 = sqrt(sum(Phi.^2,2)) ;
    Phi = bsxfun(@times,Phi,1./q0) ;
    q0 = ones(r,1) ;
    
    % default values for optional parameters
    if nargin<4
        n            = 1000 ;
        a_0          = .001;
        b_0          = .001;
        Lambda_0     = .001*eye(r);
        maxItr       = 10 ;
        tol          = 1e-2 ;
        opt.learnOpt = [] ;
        opt.learnOpt.mode = 'basic' ;
        opt.learnOpt.maxItr = 1000 ;
        opt.learnOpt.tol = 1e-3 ;
    else
        n            = opt.numSamples ;
        a_0          = opt.a_0 ;
        b_0          = opt.b_0;
        Lambda_0     = opt.Lambda_0 ;
        maxItr       = opt.maxItr ;
        tol          = opt.tol ;
    end
    
    q = q0 ; 
    itr = 1 ;
    while 1
        B = diag(q)*Phi ;
        C = decompose_kernel(B'*B);
       
        Gammabar_indices = 1:r;
        SMatrix = zeros(r, n); 
        % sample
        for idx = 1:n
            S{idx} = Gammabar_indices(sample_dual_dpp(B, C));
            SMatrix(S{idx}, idx) = 1;
        end
       
        
        % estimate item inclusion probabilities with importance sampling
        %probs = estimate_item_inclusion_probs(y, X, SMatrix, ...
        %                                      a_0, b_0, Lambda_0);
        options.method = 'priorY'  ;
        options.a_0 = a_0 ;
        options.b_0 = b_0 ;
        options.Lambda_0 = Lambda_0 ;
        probs = computeInclusionProbability(y, X, SMatrix, options) ;                                  
        
        % TODO: find out which atoms are not sampled enough and exclude
        % them from learning list
        mIdx = 1:r ;
        mIdx(sum(SMatrix,2)<10 | probs<1e-4) = [] ;
        
        % adjust DPP
        q0 =  q ;
        [q,relErrHist] = fitDPPbyMarginals(q0,Phi,probs,mIdx,opt.learnOpt) ;
        
        if (itr > maxItr) || (relerrL2(q,q0)< tol)
            break ;
        end
        itr = itr + 1 ;
                                          
    end
end

% 
% % estimate each dictionary element's inclusion probability using
% % importance sampling
% function probs = estimate_item_inclusion_probs(y, X, SMatrix, ...
%                                                a_0, b_0, Lambda_0)
%     n = size(SMatrix, 2);
%     r = size(X, 2);
%     
%     weights = zeros(n, 1);
%     for idx = 1:n
%         [~, ~, ~, ~, weights(idx)] = ...
%             compute_weight(find(SMatrix(:, idx)), y, X, ...
%                            a_0, b_0, Lambda_0);
%     end
%     weights = exp(weights - logsumexp(weights,1)) ; 
%     %weights = weights / sum(weights);
%     
%     probs = zeros(r, 1);
%     for idx = 1:r
%         samples_containing_idx = SMatrix(idx, :) == 1;
%         probs(idx) = sum(weights(samples_containing_idx));
%     end
% end
% 
% % computes importance sampling weight
% function [Lambda_S, mu_S, a_S, b_S, weight] = ...
%   compute_weight(S, y, X, a_0, b_0, Lambda_0)
%     Lambda_S = Lambda_0(S, S) + X(:, S)'*X(:, S); % |S| by |S|
%     mu_S     = Lambda_S \ (X(:, S)' * y);
%     a_S      = a_0 + 1/2;
%     b_S      = b_0 + 1/2*(y'*y - mu_S'*Lambda_S*mu_S);
%     
%     %weight = sqrt(det(Lambda_0(S, S)) / (det(Lambda_S) * 2*pi)) * ...
%     %         b_0^a_0 / b_S^a_S * gamma(a_S) / gamma(a_0);   % I THINK ONE
%     %         TERM IS MISSING IN HERE: SOMETHING THAT INVOLVES   exp(b_0 *
%     %         a_s) or something like this
%     weight = 0.5*( log(det(Lambda_0(S, S))) - log(det(Lambda_S) * 2*pi) ) + ...
%              log(b_0)*a_0 - log(b_S)*a_S + gammaln(a_S) - gammaln(a_0);
% end

% computes the KL-distance
function  d= relerrKL(p,q)
      d = sum(p .* log ((eps + p)./(eps + q))) + sum(q .* log ((eps + q)./(eps + p))) ;
end

% computes the L2-distance
function  d= relerrL2(p,q)
      d = norm(p - q)/norm(q) ;
end



