% this function fits a DPP according to set of givem marginals
% Input:
%     q0 : Initial quaity scores (N x 1 )
%     Phi : Similary features ( N x d )
%     m   : Marginals  (at most N x 1 )
%     mIdx : index of the marginals for m
%     opt : options, it should contain the following fields
%         .maxItr : maximum iteration
%         .tol : tolerance
%         .mode : mode of the optimizaiton
%         
% Output: 
%     q : 
function [q, relErrHist ]  = fitDPPbyMarginals(q0,Phi,m,mIdx,opt)
        relErrHist = [] ;
        switch (opt.mode)
            case 'basic'
                itr = 1 ;
                q = q0 ;
                d = size(Phi,2) ;
                %Lambda = pinv(Phi*Phi') ;
                
                while 1
                    fprintf('.') ;
                    M =  Phi * (( Phi'*diag(q0.^2)*Phi  + eye(d)) \ Phi')  ;
                    q(mIdx) = sqrt( m(mIdx)./ abs( diag(M(mIdx,mIdx)) ) ) ;
                    % adjust q to have the same expectation as m
                    L = diag(q) * (Phi*Phi') * diag(q) ;
                    S = eig(L(mIdx,mIdx)) ;
                    [~,alpha] = adjustDPPExpectation(real(S),sum(m)) ;
                    q(mIdx) = sqrt(alpha)*q(mIdx)  ;
                    % -- 
                    Kii = diag( (eye(size(L,1)) + L)\L ) ;
                    %--
                    if (itr > opt.maxItr) || (relerr(m(mIdx),Kii(mIdx))< opt.tol)
                       break ;
                    end
                    relErrHist = [relErrHist ; relerr(Kii(mIdx),m(mIdx))] ;
                    q0 = q ;
                    itr = itr + 1 ;
                end
             case 'fsolve'
                warning('I never manage to get it working !!') ;
                fcn = @(x) norm(m - getDiag(x,Phi) ) ; 
                x = fsolve(fcn, log(q0)) ;
                q = exp(x) ;
            otherwise
              error('not supported !!!') ;
        end
end

% computes the KL-distance
function  d= relerr(p,q)
      d = sum(p .* log ((eps + p)./(eps + q))) + sum(q .* log ((eps + q)./(eps + p))) ;
end


% function computing diagonal
function y = getDiag(x,Phi)
    N = length(x) ;
    L = diag( exp(x) )*(Phi*Phi')*diag( exp(x) ) ;
    K = eye(N) - inv(L + eye(N) );
    y = diag(K) ;
end

