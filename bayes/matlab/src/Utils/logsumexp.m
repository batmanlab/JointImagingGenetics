% File: logsumexp.m
%

function out = logsumexp(A,n)

% LOGSUMEXP
% Computes log( sum( exp( ) ) ) of tensor A along n'th index
% If A is an M_1 x M_2 x M_3 matrix and n=2, then out is a M_1 x 1 x M_3 vector.

pi_max = max(A, [], n);
out = pi_max + log(sum(exp(bsxfun(@minus, A, pi_max)), n));
