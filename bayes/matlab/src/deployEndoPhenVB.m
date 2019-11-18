% this function is an executable that runs several steps of Joint Imaging genetics
% Here is the list of arguments. Each step has its own arguments. Read the
% corresponding section in the code.
%       step   :  It is one of the following: BF, Normalize, fxVB
%          * BF:   Compute Bayes Factor. It may run for default, one, a range or list of values in the text 
%          * Normalize: It normalizes the weights. It may read one .mat file,
%               or series of mat files
%          * fxVB: perform fixed-formed variational Bayes inference (ie the
%          final step) to score the brain regions
function  deployEndoPhenVB(varargin)

    display('inside of the code !') ;
    
    [REG, PROP]=parseparams(varargin) ;
    args = struct(PROP{:}) ;
    display('Input arguments : ') ;
    args
    
    % perform the steps
    switch lower(args.step)
        case 'bf'
            fprintf('Computing Bayes Factor for following arguments ...\n')  ;
            
            % Input arguments are as follows:
            %  (*)  inputMat : [MANDATORY] .mat file is provided. The file should contain 'I' and 'G'
            %       variable
            %
            %  (*)  sigma, sa, logodd :  [OPTIONAL] if any of the 'sigma', 'sa', 'logodd' is not provided.
            %       It uses the default variables:
            %       Defaults are:
            %           sigma = ( 0.2:0.08:1 )' ;
            %           sa     = (0.025:0.025:0.4)';
            %           logodd = (-5:0.25:-3)';
            %
            %  (*) a,b,c : [OPTIONAL] Values of Hyper-Hyper parameters. If not defined, it
            %      switches back to the default parameters.
            %      Defaults are:
            %          a,b :  parameters of the beta distribution (0.02, 1)
            %           c  :  constant in proportion of variance explained
            %           (PVE). The default is 0.02
            %
            %   (*) colNum : [MANDATORY] column number for the entermediate phenotype
            %
            %   (*) outFile : [MANDATORY] .mat file saving output results 

            BFArgs = parseBFArguments(args) ;
            display('Arguments for Bayes Factor :') ;
            BFArgs
            %[w,alpha, mu , lnZ] = ...
            %    varsimbvs(BFArgs.G,...
            %              BFArgs.I(:,BFArgs.colNum),...
            %              BFArgs.sigma, ...
            %              BFArgs.sa, ...
            %              BFArgs.log10q,...
            %              BFArgs.a, BFArgs.b, BFArgs.c);
                          
            %fit=varbvs(G,Z,yI,colnames,[],struct('logodds',-5:0.25:-3)); 
            %yoel% : the way you ran it (above line) is equivalent of running BFArgs.logodd = -5:0.25:-3 and leaving the rest as is
                             
            fit = varbvs(BFArgs.G,...
                         ... %yoel% add Z here, this function should read that in somehow 
                         BFArgs.I(:,BFArgs.colNum),...
                         colnames,...
                         BFArgs) ;
            
            w = fit.w ;
            alpha = fit.PIP ;
            lnZ_weighted = fit.logw ;
            mu = fit.beta ; 
                      
            %save(args.outFile,'w','alpha', 'mu' , 'lnZ') ;       
            save(args.outFile,'w','alpha', 'mu' , 'lnZ_weighted') ;
            
        %case 'normalize'    
        %    fprintf('Normalizing the Bayes Factor scores ...\n')  ;
        %    
        %    % Here is the list of possible inputs:
        %    %  (*) inFile : [MADATORY]  input file. Either it is .mat file
        %    %       or list of MATLAB file covering all parameters.
        %    %
        %    %  (*) outFile : [MANDATORY] .mat file saving output results 
        %    
        %    normArgs = parseNormalizeArguments(args) ;
        %    lnZ_weighted = normArgs.lnZ(:)'*normArgs.w(:) ;
        %    
        %    if exist(args.outFile,'file') % if the file exists
        %        save(args.outFile,'lnZ_weighted','-append') ;
        %    else
        %        save(args.outFile,'lnZ_weighted') ;
        %    end
            
        case 'fxvb'
            fprintf('Running fixed-form variational Bayes ...\n')  ;
            
            % Input arguments are as follows
            % (*)  inFile: [MANDATORY]  a .csv file containing list of
            %      .mat files (output of the previous step) and
            %      corresponding columns number.
            %
            % (*)
            % (*) outFile : [MANDATORY]  .mat file saving output results
            %
            % Here is an example:
            %   deployEndoPhenVB('step','fxvb',...
            %                    'csvFile','/data/vision/polina/users/kayhan/Experiments/DPP/geneticResults/SpikeSlabResults_NatureList0/BFResultsFileList.csv',...
            %                    'inputMat','/data/vision/polina/projects/ADNI/work/kayhan/Journal/MATLAB_Data/natureImput-ChrAll_Data94.mat',...
            %                    'outFile','test.mat')
            
            fxVBArgs = parseFXVBArguments(args) 
            train_idx = fxVBArgs.train_idxs;          
            % prepairing data for the next step
            I = fxVBArgs.I(train_idx, :)
            Z = fxVBArgs.Z(train_idx, :)
            numEndoPhen = size(I ,2) ;
            % changed by yoel
            Y = double(fxVBArgs.y')(train_idx) ;
            
            Y(fxVBArgs.y'(train_idx)==0) = -1 ;
            
            % compute the lnZ of the null model
            mu = mean(I(Y==-1,:)) ;
            v = var(I(Y==-1,:)) ;
            Z1 = bsxfun(@minus, mu , I) ;
            Z2 =  bsxfun(@times, Z1.^2, 1./( 2*v ) ) ;
            nullLnZ =  -sum(Z2) - 0.5*log(2*pi)*size(I,1) ;
            nullLnZ = nullLnZ' ;
            
            % compute the difference
            lnZ_all = fxVBArgs.altLnZ - nullLnZ ;
            
            
            RandStream.setGlobalStream(RandStream('mt19937ar','seed', fxVBArgs.randomSeed ));
            options = [] ;
            options.logodds =  lnZ_all/log(10) ;   % I divide it by log(10) to bring it to the base-10 log
            options.logodds = options.logodds + linspace(-3,3,10) ;   % what I am doing here is this: I am allowing each region to wiggle  (-3,3) around it prior 
            options.alpha = fxVBArgs.alpha_all;
            fit = varbvs(I,...
                         Z,... 
                         Y,...
                         [],...
                         'binomial',...
                         options) ;
     
            save(args.outFile,'lnZOpt','etaOpt','lnZHist','etaHist') ; 
        otherwise
            fprintf('step : %s \n', args.step) ;
            error('This step is not defined !!!') ;
    end
end


% this is helper function to parse arguments for the BayesFactor step
function BFArgs = parseBFArguments(args) 

    if isfield(args,'inputMat')  % if the input .MAT file is provided
        BFArgs = load(args.inputMat,'I','G') ;
    else
        % read the I and G separately
        error('inputMat file is not provided. Other options are not implemented yet') ;
    end
    
    % sigma
    if isfield(args,'sigma')
        if isdeployed()
            BFArgs.sigma = colon(sscanf(args.sigma,'%f:%f:%f')) ;
        else
            BFArgs.sigma = args.sigma ;
        end
    else
        % default parameters
        BFArgs.sigma = ( 0.2:0.08:1 )' ;
    end
    
    % sa
    if isfield(args,'sa')
        if isdeployed()
            BFArgs.sa = colon(sscanf(args.sa,'%f:%f:%f')) ;
        else
            BFArgs.sa = args.sa ;
        end
    else
        % default parameters
        BFArgs.sa     = (0.025:0.025:0.4)';
    end
    
    % logodd
    if isfield(args,'logodd')
        if isdeployed()
            BFArgs.logodd = colon(sscanf(args.logodd,'%f:%f:%f')) ;
        else
            BFArgs.logodd = args.logodd ;
        end
    else
        % default parameters
        BFArgs.logodd = (-5:0.25:-3)';
    end
    [BFArgs.sigma , BFArgs.sa , BFArgs.logodd] = ...
        ndgrid(BFArgs.sigma, BFArgs.sa, BFArgs.logodd);
    
    % a
    if isfield(args,'a')
        if isdeployed()
            BFArgs.a = sscanf(args.a,'%f') ;
        else
            BFArgs.a = args.a ;
        end
    else
        BFArgs.a = 0.02 ;
    end
    
    % b
    if isfield(args,'b')
        if isdeployed()
            BFArgs.b = sscanf(args.b,'%f') ;
        else
            BFArgs.b = args.b ;
        end
    else
        BFArgs.b = 1 ;
    end
    
    % c
    if isfield(args,'c')
        if isdeployed()
            BFArgs.c = sscanf(args.c,'%f') ;
        else
            BFArgs.c = args.c ;
        end
    else
        BFArgs.c = 0.02 ;
    end
    
    % colNum, the number of phenotype for which BF has to be computed
    if isfield(args,'colNum')
        if isdeployed()
            BFArgs.colNum = sscanf(args.colNum,'%d') ;
        else
            BFArgs.colNum = args.colNum ;
        end
    else
        error('colNum is not provided') ;
    end
    
end

% This is a helper function to parse the argument for normalization step
function normArgs = parseNormalizeArguments(args) 
    if isfield(args,'inFile')  % if the input .MAT file is provided
        [~,~,ext] = fileparts(args.inFile) ;
        if isequal(ext,'.mat')
            normArgs = load(args.inFile,'lnZ','w') ;
        elseif  isequal(ext,'.txt')
            error('Not implemented yet! I don''t know how to put the lnZ files together !') ;
            fid = fopen(args.inFile,'rt') ;
            % ** OK ** THE NEXT LINE IS NOT CERTAIN. I AM NOT SURE IT IS A
            % GOOD IDEA !!!
            C = textscan(fid,'%d,%s','HeaderLines',1) ;  
            fclose(fid) ;
            % do some sanity check and make sure all parameters are covered
            
            % first sort the 
            [~,idx] = sort(C{1}) ;
            fnList = C{2}(idx) ;
            for i=1:length(fnList)
                % read the 
            end
        else
            error('inFile should be either .MAT file or .txt file containing list of .MAT files !') ;
        end  
    else
        % read the I and G separately
        error('inFile file is not provided !!! ') ;
    end
end



% This is a helper function to parse the argument for the fixed-form variatrional Bayese step
function fxVBArgs = parseFXVBArguments(args) 
% check the mandatory fields
assert(all(isfield(args,{'csvFile','inputMat'})),'Some of the mandatory fields are not provided !!') ;


if isfield(args,'inputMat')  % if the input .MAT file is provided
    fxVBArgs = load(args.inputMat,'I','G','y','Z') ;
else
    % read the I and G separately
    error('inputMat file is not provided. Other options are not implemented yet') ;
end

if isfield(args, 'cvIdx')
    estring = sprintf('train.train_%i', args.num_idx);
    train = load(args.cvIdx, 'train');
    train_idxs = eval(estring);
    fxVBArgs.train_idxs = train_idxs;
end 

[~,~,ext] = fileparts(args.csvFile) ;
if  isequal(ext,'.csv')
    fid = fopen(args.csvFile,'rt') ;
    C = textscan(fid,'%d,%s','HeaderLines',1) ;
    fclose(fid) ;
    % do some sanity check and make sure all parameters are covered
    
    % first sort the
    [~,idx] = sort(C{1}) ;
    fnList = C{2}(idx) ;
    colNumList = C{1}(idx) ;
    maxColNum = max(colNumList) ;
    
    % memory allocation
    res = load(fnList{1}) ;
    altLnZ = zeros(maxColNum,1) ;
    mu_all = zeros(maxColNum, length(res.mu)) ;
    alpha_all = zeros(maxColNum, length(res.alpha)) ;
    for cnt=1:maxColNum
        % read the
        res = load(fnList{cnt}) ;
        
        altLnZ(cnt) = res.lnZ_weighted ;
        mu_all(cnt,:) = res.beta ;
        alpha_all(cnt,:) = res.alpha ;
        clear res
    end
    
    fxVBArgs.altLnZ = altLnZ ;
    fxVBArgs.mu_all = mu_all ;
    fxVBArgs.alpha_all = alpha_all ;
else
    error('csvFile should be either .csv file containing list of .MAT files and corresponding column !') ;
end


% numRepeats
if isfield(args,'numRepeats')
    fxVBArgs.numRepeats = args.numRepeats ;
else
    fxVBArgs.numRepeats = 5 ;% use to be 20
end


% randomSeed
if isfield(args,'randomSeed')
    fxVBArgs.randomSeed = args.randomSeed ;
else
    fxVBArgs.randomSeed = 0 ;
end


% maxIter
if isfield(args,'maxIter')
    fxVBArgs.maxIter = args.maxIter ;
else
    fxVBArgs.maxIter = 200 ; % use to be 500
end

% numDrawBeforeInv
if isfield(args,'numDrawBeforeInv')
    fxVBArgs.numDrawBeforeInv = args.numDrawBeforeInv ;
else
    fxVBArgs.numDrawBeforeInv = 10 ;
end


%numBatchSamp
if isfield(args,'numBatchSamp')
    fxVBArgs.numBatchSamp = args.numBatchSamp ;
else
    fxVBArgs.numBatchSamp = 1 ;
end

%useMarginal4C
if isfield(args,'useMarginal4C')
    fxVBArgs.useMarginal4C = args.useMarginal4C ;
else
    fxVBArgs.useMarginal4C = false ;
end

% useKDpp
if isfield(args,'useKDpp')
    fxVBArgs.useKDpp = args.useKDpp ;
else
    fxVBArgs.useKDpp = [] ;
end

% endoPhenPrior: prior inclusion for endophenotype
if isfield(args,'endoPhenPrior')
       fxVBArgs.endoPhenPrior = args.endoPhenPrior ;
else
    fxVBArgs.endoPhenPrior = 10/94 ;   % 10 regions out of 94
end

% if deployed, we need to take care of string <-> numeric conversion
if isdeployed()
    argFieldNames = {'maxIter','numDrawBeforeInv',...
        'numBatchSamp','useMarginal4C','useKDpp','endoPhenPrior',...
        'randomSeed','numRepeats'} ;
    argFieldTypes = {'numeric','numeric','numeric','boolean','boolean','numeric',...
        'numeric','numeric'} ;
    
    for cnt=1:length(argFieldNames)
        if isfield(args,argFieldNames{cnt})
            if isequal(argFieldTypes{cnt},'boolean')
                if isequal(lower(fxVBArgs.(argFieldNames{cnt})),'true')
                    fxVBArgs.(argFieldNames{cnt}) = true ;
                elseif  isequal(lower(fxVBArgs.(argFieldNames{cnt})),'false')
                    fxVBArgs.(argFieldNames{cnt}) = false ;
                else
                    error([ argFieldNames{cnt} ': this value should be true or false !' ] ) ;
                end
            elseif isequal(argFieldTypes{cnt},'numeric')
                fxVBArgs.(argFieldNames{cnt}) = str2double(fxVBArgs.(argFieldNames{cnt})) ;
            end
        end
    end
end


end


%initializer helper function
function [eta,C]=initHelper(eta0,C0)
    eta = [eta0 ; 0]  ;
    C = [C0  zeros(size(C0,2),1) ; zeros(1,size(C0,1))  1];
end

% helper function for prior probability
function logP = priorHelper(p,SMatrix)
    logP = log(p)*sum(SMatrix)' ;
end


% helper function for the joinlikehood
function logData = joinLikeHelper(y,X,S,fcn)
    if sum(S)==0
        logData = -10000 ;
    else
        logData = fcn(y,X,S) ;
    end
    
    %if isnan(logData)
    %    logData = -200 ;
    %end
end

