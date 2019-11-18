% This function samples from DPP and solves the regression everytime
% It helps to give mean and standard deviation from from model
function  [MSE_sample] = sampleDPPRegression(N,log_q,yTrain,XTrain,yTest,XTest,Phi,opt)

       if opt.normalizePhi
            q0 = sqrt( sum(Phi.^2,2) ) ;
            Phi = bsxfun(@times,Phi,1./q0) ;
       else
            q0 = ones(numItems,1) ;
       end

       % sample from DPP
       opt.numBatchSamp = N ;
       [SMatrix,status] = sampleFromDPP(log_q(1:length(log_q)-1)/2, Phi, opt) ;
       SMatrix = SMatrix>0 ;
       
       % compute of the linear system
       MSE_sample = zeros(size(SMatrix,2),1) ;
       for cnt=1:size(SMatrix,2)
           if (sum(SMatrix(:,cnt))>0)
               A = XTrain(:,SMatrix(:,cnt)) ;
               b = yTrain ;
               cc = (A'*A + 0.01*eye(size(A,2))) \ (A'*b) ;
           
               MSE_sample(cnt) = mean((XTest(:,SMatrix(:,cnt))*cc - yTest).^2) ;
           else
               MSE_sample(cnt) = mean((yTest).^2) ;
           end
       end
       
end



% sample from DPP
function [SMatrix, status] = sampleFromDPP(eta, Phi, opt)
      % update the dual kernel according to eta
      q = exp(eta) ;
      C = decompose_kernel(Phi'*diag(q.^2)*Phi) ;
      status = 0 ;
      
      % sample from DPP
      numItems = size(Phi,1) ;
      S = cell(opt.numBatchSamp,1) ;
      SMatrix = zeros(numItems, opt.numBatchSamp) ;

      if any(C.D<0)
          warning('at least one of the eigen values turn negative which is a sign numerical instability') ;
          %status = 1 ;
          %return
      end

%       cnt = 1 ;
%       ii = 0 ;
%       while (sum(sum(SMatrix)==0) > 0) 
%           
%           if opt.useKDpp
%              S{cnt} = sample_dual_dpp( diag(q)*Phi, C, opt.expCard) ;
%           else
%              S{cnt} = sample_dual_dpp( diag(q)*Phi, C) ;
%           end
%           if ~isempty(S{cnt})
%             SMatrix(S{cnt},cnt) = 1 ;
%             cnt = cnt + 1 ;
%           else
%               fprintf('*') ;
%           end
%           ii = ii + 1 ;
%           if (ii > 10*opt.numBatchSamp)  % it stayed inside of the loop too much
%               warning('I couldn''t get enough samples, probability of the items are too low !!!') ;
%               status = 1 ;
%               break
%           end
%       end
      
      for cnt=1:opt.numBatchSamp
          if opt.useKDpp
             S{cnt} = sample_dual_dpp( diag(q)*Phi, C, opt.expCard) ;
          else
             S{cnt} = sample_dual_dpp( diag(q)*Phi, C) ;
          end
          
          SMatrix(S{cnt},cnt) = 1 ;
      end
      
 
end
