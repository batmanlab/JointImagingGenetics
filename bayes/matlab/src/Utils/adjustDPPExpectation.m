% this function adjust the eigen-value od L-ensemble to have pre-specified
% expectation
function [D2,q] = adjustDPPExpectation(D1,E)
    fcn = @(x) sum(D1*exp(x)./(1 + D1.*exp(x))) - E ;
    q = exp(fzero(fcn,0)) ;
    D2 = D1*q ;
end