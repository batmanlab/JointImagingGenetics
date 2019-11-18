% this is helper function to compute the marginal likelihood for
% classification
% Here is an example how to use it
%
%   MAKE SURE Y = +1/-1
%   covFcn = {@covLIN};   hyp.cov  = [] ;
%   meanFcn = {@meanZero};  hyp.mean = [];
%   likFcn = {'likLogistic'} ;
%   infFcn = {'infEP'} ;
%
%   fcn = @(y,X,S) computeGPLnZHelper(y,X,S,hyp, infFcn, meanFcn, covFcn, likFcn)
% 
% See usageClassification for more options
function lnZ = computeGPLnZHelper(y,X,SMatrix,hyp, infFcn, meanFcn, covFcn, likFcn)
    lnZ = zeros(size(SMatrix,2),1) ;
    
    for cnt=1:size(SMatrix,2)
        % set up parameters
        xtr = X(:,SMatrix(:,cnt)>0) ;
    
        % call gp
        lnZ(cnt) = -gp(hyp, infFcn, meanFcn, covFcn, likFcn, xtr, y) ;
    end
end

