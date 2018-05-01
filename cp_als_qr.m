function [P,Uinit,output] = cp_als_qr(X,R,varargin)
%CP_ALS_QR Compute a CP decomposition of any type of tensor using the QR-decomposition.
%
%   This code is based on the CP_ALS function in the MATLAB Tensor Toolbox
%   from Sandia Corporation, which can be found at
%   http://www.sandia.gov/~tgkolda/TensorToolbox.
%
%   P = CP_ALS_QR(X,R) computes an estimate of the best rank-R
%   CP model of a tensor X using an alternating least-squares
%   algorithm and the QR-decomposition.  The input X can be a 
%   tensor, sptensor, ktensor, or ttensor. The result P is a ktensor.
%
%   P = CP_ALS_QR(X,R,'param',value,...) specifies optional parameters and
%   values. Valid parameters and their default values are:
%      'tol' - Tolerance on difference in fit {1.0e-4}
%      'maxiters' - Maximum number of iterations {50}
%      'dimorder' - Order to loop through dimensions {1:ndims(A)}
%      'init' - Initial guess [{'random'}|'nvecs'|cell array]
%      'printitn' - Print fit every n iterations; 0 for no printing {1}
%      'implicit' - Apply Q0 implicity, default is explicit
%      'iskentsor' - Enter true if input is a ktensor, default is false
%
%   [P,U0] = CP_ALS_QR(...) also returns the initial guess.
%
%   [P,U0,out] = CP_ALS_QR(...) also returns additional output that contains
%   the input parameters.

% Copyright information:
% MATLAB Tensor Toolbox.
% Copyright 2015, Sandia Corporation.

% NOTE: Updated in various minor ways per work of Phan Anh Huy. See Anh
% Huy Phan, Petr Tichavsk?, Andrzej Cichocki, On Fast Computation of
% Gradients for CANDECOMP/PARAFAC Algorithms, arXiv:1204.1586, 2012.

% This is the MATLAB Tensor Toolbox by T. Kolda, B. Bader, and others.
% http://www.sandia.gov/~tgkolda/TensorToolbox.
% Copyright (2015) Sandia Corporation. Under the terms of Contract
% DE-AC04-94AL85000, there is a non-exclusive license for use of this
% work by or on behalf of the U.S. Government. Export of this data may
% require a license from the United States Government.
% The full license terms can be found in the file TT_LICENSE.txt

%% Extract number of dimensions and norm of X.
N = ndims(X);
normX = norm(X);

%% Set algorithm parameters from input or by using defaults
params = inputParser;
params.addParamValue('tol',1e-4,@isscalar);
params.addParamValue('maxiters',50,@(x) isscalar(x) & x > 0);
params.addParamValue('dimorder',1:N,@(x) isequal(sort(x),1:N));
params.addParamValue('init', 'random', @(x) (iscell(x) || ismember(x,{'random','nvecs'})));
params.addParamValue('printitn',1,@isscalar);
params.addParamValue('implicit', false, @islogical); %%% Add an option for applying Q0 implicitly. Default is explicit computation. %%%
params.addParamValue('isktensor',false,@islogical); %%% Enter true if input is a ktensor. Otherwise, value is false. %%%
params.parse(varargin{:});

%% Copy from params object
fitchangetol = params.Results.tol;
maxiters = params.Results.maxiters;
dimorder = params.Results.dimorder;
init = params.Results.init;
printitn = params.Results.printitn;
implicit = params.Results.implicit;
isktensor = params.Results.isktensor;

%% Error checking 

%% Set up and error checking on initial guess for U.
if iscell(init)
    Uinit = init;
    if numel(Uinit) ~= N
        error('OPTS.init does not have %d cells',N);
    end
    for n = dimorder(2:end);
        if ~isequal(size(Uinit{n}),[size(X,n) R])
            error('OPTS.init{%d} is the wrong size',n);
        end
    end
else
    % Observe that we don't need to calculate an initial guess for the
    % first index in dimorder because that will be solved for in the first
    % inner iteration.
    if strcmp(init,'random')
        Uinit = cell(N,1);
        for n = dimorder(2:end)
            Uinit{n} = rand(size(X,n),R);
        end
    elseif strcmp(init,'nvecs') || strcmp(init,'eigs') 
        Uinit = cell(N,1);
        for n = dimorder(2:end)
            Uinit{n} = nvecs(X,n,R);
        end
    else
        error('The selected initialization method is not supported');
    end
end

%% Set up for iterations - initializing U and the fit.
U = Uinit;
fit = 0;

if printitn>0
  fprintf('\nCP_ALS_QR:\n');
end

%% Main Loop: Iterate until convergence

%%% Changes for cp_als_qr start here: %%%

count = 1;

%%% Initialize a cell array Qs and Rs to hold decompositions of factor matrices. %%%
Qs = cell(N,1); %%% The Kronecker product of these tells us part of the Q of the Khatri-Rao product. %%%
Rs = cell(N,1); %%% The Khatri-Rao product of these tells us the rest of Q and the R of the Khatri-Rao product. %%%

%%% Compute economy-sized QR decomposition. %%%
for i = 1:N
    if ~isempty(U{i})
        [Qs{i}, Rs{i}] = qr(U{i},0); 
    end
end
   
for iter = 1:maxiters
        
    fitold = fit;
        
    % Iterate over all N modes of the tensor
    for n = dimorder(1:end)
       
        %%% Compute the QR of the Khatri-Rao product of Rs. %%%
        %%% This is where we need to utilize the structure of the product. %%%
        %%% First compute the Khatri-Rao product on all modes but n. %%%
        M = khatrirao(Rs{[1:n-1,n+1:N]},'r');
        
        if implicit == false %%% Compute the explicit QR factorization. %%%
            %%% This QR ignores the sparsity structure of M. %%%
            [Q0,R0] = qr(M,0);

            %%% Q = Kronecker product of non-empty matrices in Qs times Q0. %%%
            %%% R =  R0. %%%
        end

        %%% TTM on all modes but mode n. %%%
        Y = ttm(X,Qs,-n,'t');
        
        %%% Now multiply by Q0 on the right. %%%
        %%% There are four cases for this computation. %%%
        
        if implicit == false && isktensor == false 
            %%% CASE 1: Explicitly apply Q0 to any tensor other than a ktensor. %%%
            %%% We just use the explicit computation of Q0. %%%
            Z = tenmat(Y,n) * Q0;
                
            %%% Calculate updated factor matrix by backsolving with R0' and Z. %%%
            U{n} = double(Z) / R0';
        elseif implicit == false && isktensor == true 
            %%% CASE 2: Explicitly apply Q0 to a ktensor. %%%
            %%% Save all the factor matrices of Y in a cell array. %%%
            %%% Then, we can compute the Khatri Rao product in one line.%%%
            K = cell(N,1);
            
            for k = 1:N
                if k ~= n
                    K{k} = Y.U{k};
                end
            end
        
            %%% We just use the explicit computation of Q0. %%%
            Z = Y.U{n} * (khatrirao(K{[1:n-1,n+1:N]},'r')' * Q0);
                
            %%% Calculate updated factor matrix by backsolving with R0' and Z. %%%
            U{n} = double(Z) / R0';
        elseif implicit == true && isktensor == false 
            %%% CASE 3: Implicitly apply Q0 to any tensor other than a ktensor. %%%
            %%% We apply Q0 implicitly to take advantage of the sparsity of the khatri-rao product M
            S = sparse(M); %%% We know M is sparse, so save by converting it to a sparse matrix. %%%
            Yn = double(tenmat(Y,n));
            [C,R] = qr(S,Yn'); %%% Here, Q0' is applied implicitly to Yn. %%%
            U{n} = (R \ C)'; %%% Backsolve with R0 and take the transpose to get the nth factor matrix. %%%
        else
            %%% CASE 4: Implicitly apply Q0 to a ktensor. %%%
            %%% Save all the factor matrices of Y in a cell array. %%%
            %%% Then, we can compute the Khatri Rao product in one line.%%%
            K = cell(N,1);
            
            for k = 1:N
                if k ~= n
                    K{k} = Y.U{k};
                end
            end
            
            Yn = Y.U{n} * khatrirao(K{[1:n-1,n+1:N]},'r')'; %%% Matricize Y in the nth mode. %%%
            %%% We apply Q0 implicitly to take advantage of the sparsity of the khatri-rao product M. %%%
            S = sparse(M); %%% We know M is sparse, so save by converting it to a sparse matrix. %%%
            [C,R] = qr(S,Yn'); %%% Here, Q0' is applied implicitly to Yn. %%%
            U{n} = (R \ C)'; %%% Backsolve with R0 and take the transpose to get the nth factor matrix. %%%
        end
        
               
        % Normalize each vector to prevent singularities in coefmatrix
        if iter == 1
            lambda = sqrt(sum(U{n}.^2,1))'; %2-norm
        else
            lambda = max( max(abs(U{n}),[],1), 1 )'; %max-norm
        end 
        
        U{n} = bsxfun(@rdivide, U{n}, lambda');
        
        %%% Recompute QR factorization for updated factor matrix. %%%
        [Qs{n}, Rs{n}] = qr(U{n},0);
    end

    %%% Changes for cp_als_qr end here. %%%
        
    P = ktensor(lambda,U);
        
    if normX == 0
        fit = norm(P)^2 - 2 * innerprod(X,P);
    else
        normresidual = sqrt( normX^2 + norm(P)^2 - 2 * innerprod(X,P) );
        fit = 1 - (normresidual / normX); %fraction explained by model
        rel_err = normresidual / normX;  %%% Keep track of relative errors to add them to the output. %%%
        rel_err_vec(iter,:) = rel_err;
    end
    fitchange = abs(fitold - fit);
        
    % Check for convergence
    if (iter > 1) && (fitchange < fitchangetol)
        flag = 0;
    else
        flag = 1;
    end
        
    if isnan(fit) %%% If the fit is NaN, just stop the process. %%%
        break;
    end
    
    if (mod(iter,printitn)==0) || ((printitn>0) && (flag==0))
        fprintf(' Iter %2d: f = %e f-delta = %7.1e\n', iter, fit, fitchange);
    end
        
    % Check for convergence
    if (flag == 0)
        break;
    end        
end   


%% Clean up final result
% Arrange the final tensor so that the columns are normalized.
P = arrange(P);
% Fix the signs
P = fixsigns(P);

if printitn>0
    if normX == 0
        fit = norm(P)^2 - 2 * innerprod(X,P);
    else
        normresidual = sqrt( normX^2 + norm(P)^2 - 2 * innerprod(X,P) );
        fit = 1 - (normresidual / normX); %fraction explained by model
    end
  fprintf(' Final f = %e \n', fit);
end

output = struct;
output.params = params.Results;
output.iters = iter;
output.rel_err_vec = rel_err_vec; %%% Add a rel_err vector to output. %%%
