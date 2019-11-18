% this function normalize log-Factors along n'th axis
function out = logNormalize(A,n)
   pi_max = max(A, [], n);
   nrm = pi_max + log(sum(exp(bsxfun(@minus, A, pi_max)), n));
   out = bsxfun(@minus, A, nrm) ;
end