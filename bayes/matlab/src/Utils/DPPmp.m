% DPPmp.m
% Coded by George H. Chen (georgehc@csail.mit.edu)
%
% This function implements an OMP-like support recovery algorithm for
% Bayesian linear regression where there is a prior distribution over which
% dictionary elements (i.e., regressors) are non-zero. Effectively this
% prior distribution assigns a probability to every subset of dictionary
% elements. The code supports uniform and DPP priors over subsets.
%
% While its core functionality is support recovery, the algorithm also
% outputs an estimate of the regression coefficients (beta) given the
% estimated support (Gamma), a reconstruction (yfit) of the observed
% signal (y), and the residuals (y - yfit).
%
% For details, see the documentation: dpp-regression.pdf
%
% [yfit, residuals, beta_hat, Gamma] = DPPmp(y, X, Phi, k) returns
%   - yfit (d-by-1 column vector): a greedy approximation of y
%   - residuals (d-by-1 column vector): y - yfit
%   - beta_hat (r-by-1 column vector): an estimate of regression
%       coefficients
%   - Gamma (k-by-1 column vector): an estimate of the support of the
%       regression coefficients
%
% The inputs are
%   - y (d-by-1 column vector): the observed signal
%   - X (d-by-r matrix): the dictionary/regressors; each column is a
%       dictionary element
%   - Phi (r-by-p matrix): feature vectors for each of the r dictionary
%       elements; the inner product <Phi_i, Phi_j> gives the similarity
%       between dictionary elements i and j
%   - k (positive integer): number of nonzero entries we seek in the
%       support; the algorithm will keep growing the support (Gamma)
%       greedily until the support has length k
%
% Optional arguments can be specified in pairs; for example:
%
%   DPPmp(y, X, Phi, k, 'importance_sampling_num_samples', 5000)
%
% The optional arguments are
%   - 'importance_sampling_num_samples' (positive integer): number of
%       samples to use for importance sampling
%     (default value: 1000)
%   - 'a_0': inverse gamma shape parameter for the prior on noise variance
%       sigma^2
%     (default value: .001)
%   - 'b_0': inverse gamma scale parameter for the prior on noise variance
%       sigma^2
%     (default value: .001)
%   - 'Lambda_0': PSD matrix mulitplied by noise variance sigma^2
%       that is the covariance for the (non-zero) regression coefficients
%     (default value: .001*eye(r))
%   - 'subset_prior': prior to use for sampling subsets among dictionary
%       element indices 1,2,...,r; the valid options are 'dpp' and
%       'uniform'
%     (default value: 'dpp')
%   - 'p_0': for the uniform subset prior, this value is the (i.i.d.)
%       probability of including a dictionary element; in fact, only when
%       p_0 is .5 is the distribution over subsets actually uniform but we
%       allow changing p_0 away from .5 to change the expected size of the
%       sampled subset
%     (default value: .5)
%
% TODO: Right now Phi is a mandatory input even though it only makes sense
%   if the subset prior is a DPP. This should be fixed... Stylistically I
%   find it annoying to have to specify Phi as an optional parameter for
%   typical use of the function. Given the name of the function (DPPmp),
%   perhaps it makes sense to create a separate function focused on the
%   uniform case that doesn't use a DPP prior...
%
function [yfit, residuals, beta_hat, Gamma] = DPPmp(y, X, Phi, k, varargin)
    r = size(X, 2); % number of dictionary elements/regressors
    
    % default values for optional parameters
    n            = 1000;
    subset_prior = 'dpp';
    a_0          = .001;
    b_0          = .001;
    Lambda_0     = .001*eye(r);
    p_0          = k/r;
    computeInclusionProbMethod = 'priorY' ;
    
    for idx = 1:length(varargin)
        if strcmpi(varargin{idx}, 'importance_sampling_num_samples')
            n = varargin{idx+1};
        elseif strcmpi(varargin{idx}, 'subset_prior')
            subset_prior = varargin{idx+1};
        elseif strcmp(varargin{idx}, 'a_0')
            a_0 = varargin{idx+1};
        elseif strcmp(varargin{idx}, 'b_0')
            b_0 = varargin{idx+1};
        elseif strcmp(varargin{idx}, 'Lambda_0')
            Lambda_0 = varargin{idx+1};
        elseif strcmp(varargin{idx}, 'p_0')
            p_0 = varargin{idx+1};
        elseif strcmp(varargin{idx}, 'computeInclusionProbMethod')
            computeInclusionProbMethod = varargin{idx+1};
        end
    end
    
    L = Phi*Phi'; % r by r
    
    Gamma = [];
    
    while length(Gamma) < k
        fprintf('Running DPPmp iteration %d...\n', length(Gamma) + 1);
        
        S = cell(n, 1); % S{idx} will store a subset (as a list of indices)
        SMatrix = zeros(r, n); % SMatrix(i, j) says whether sample j (which
                               % is a subset of [r]) contains item i or not
        
        if strcmpi(subset_prior, 'uniform')
            for idx = 1:n
                S{idx} = find(rand(r, 1) < p_0);
                SMatrix(S{idx}, idx) = 1;
            end
        elseif strcmpi(subset_prior, 'dpp')
            % take the prior DPP and condition on Gamma being included in
            % the subset
            if ~isempty(Gamma)
                disp(Gamma)
                
                I_Gammabar = eye(r);
                Gammabar   = true(r, 1);
                for idx = Gamma
                  I_Gammabar(idx, idx) = 0;
                  Gammabar(idx)        = false;
                end
                
                Gammabar_indices = find(Gammabar);
                
                intermediate_matrix = inv(L + I_Gammabar);
                
                L_Gamma = inv(intermediate_matrix(Gammabar, Gammabar)) ...
                          - eye(r - length(Gamma));
            else
                Gammabar_indices = 1:r;
                L_Gamma = L;
            end
            
            % compute dual representation for sampling
            [U, D] = eig(L_Gamma);
            B = U*diag(sqrt(diag(D)));
            C = decompose_kernel(B'*B);
            
            % sample
            for idx = 1:n
                S{idx} = Gammabar_indices(sample_dual_dpp(B, C));
                S{idx} = [S{idx}; Gamma] ;
                SMatrix(S{idx}, idx) = 1;
            end
        else
            error('The specified prior on subsets is not supported.');
        end

        % estimate item inclusion probabilities with importance sampling
        %probs = estimate_item_inclusion_probs(y, X, SMatrix, ...
        %                                      a_0, b_0, Lambda_0);
        opt.method =  computeInclusionProbMethod ;
        opt.a_0 = a_0 ;
        opt.b_0 = b_0 ;
        opt.Lambda_0 = Lambda_0 ;
        probs = computeInclusionProbability(y, X, SMatrix, opt) ;
                                          
        
        probs(Gamma) = -inf ; % Gamma won't be selected again as max but will be added again
        [~, idx] = max(probs);
        Gamma = [Gamma; idx]; %#ok<AGROW>
    end
    
    beta_hat = zeros(r, 1);
    [~, beta_hat(Gamma), ~, ~, ~] = ...
        compute_weight(Gamma, y, X, a_0, b_0, Lambda_0);
    yfit = X(:, Gamma)*beta_hat(Gamma);
    residuals = y - yfit;
end

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

% computes importance sampling weight
function [Lambda_S, mu_S, a_S, b_S, weight] = ...
  compute_weight(S, y, X, a_0, b_0, Lambda_0)
    Lambda_S = Lambda_0(S, S) + X(:, S)'*X(:, S); % |S| by |S|
    mu_S     = Lambda_S \ (X(:, S)' * y);
    a_S      = a_0 + 1/2;
    b_S      = b_0 + 1/2*(y'*y - mu_S'*Lambda_S*mu_S);
    
    %weight = sqrt(det(Lambda_0(S, S)) / (det(Lambda_S) * 2*pi)) * ...
    %         b_0^a_0 / b_S^a_S * gamma(a_S) / gamma(a_0);   % I THINK ONE
    %         TERM IS MISSING IN HERE: SOMETHING THAT INVOLVES   exp(b_0 *
    %         a_s) or something like this
    weight = 0.5*( log(det(Lambda_0(S, S))) - log(det(Lambda_S) * 2*pi) ) + ...
             log(b_0)*a_0 - log(b_S)*a_S + gammaln(a_S) - gammaln(a_0);
end
