close all; clear; clc
% HW#4, Prob#1
%Order Selection for Gaussian Mixture Model (GMM) Fitting
R = 1000; %1000 Monte-Carlo runs
N = 100; % Samples in GMM model
mu{1} = [0;0];
sigma{1} = [1 0; 0 1];
mu{2} = [3;0];
sigma{2} = [1 0;0 1];
mu{3} = [0;2];
sigma{3} = [1 0;0 1];

beta_all = [0.75 0.85 0.90;0.20 0.10 0.05;0.05 0.05 0.05];
BIC_f = zeros(5,100,3);
BIC_eval = zeros(5,100,3);
for beta_indx = 1:3
    for r = 1:R
        beta = beta_all(:,beta_indx);
        [data, classIndex] = generateGaussianSamples(mu, sigma, N, beta);
        for M = 1:5
            GMModel = fitgmdist(data,M,'MaxIter',500,'covariancetype',...
                'diagonal','SharedCovariance', true,'RegularizationValue',10^-15);
            
            BIC_f(M,r,beta_indx) = GMModel.BIC;
            mu_est = GMModel.mu';
            sigma_est =  zeros(2,2,M);
            for indx = 1:M
            sigma_est(:,:,indx) = [GMModel.Sigma(1) 0; 0 GMModel.Sigma(2)];
            end
            beta_est = GMModel.ComponentProportion;
            BIC_eval(M,r,beta_indx) =  ModelOrderSelection(data',beta_est,mu_est,sigma_est,N,M);
        end
    end
    
    
    figure;parallelcoords(BIC_f(:,:,1)','quantile',0.05);
    figure;parallelcoords(BIC_f(:,:,2)','quantile',0.05);
    figure;parallelcoords(BIC_f(:,:,3)','quantile',0.05);
    
    [~,min_inx] = min(BIC_f(:,:,1));
    figure;histogram(min_inx);
    [~,min_inx] = min(BIC_f(:,:,2));
    figure;histogram(min_inx);
    [~,min_inx] = min(BIC_f(:,:,3));
    figure;histogram(min_inx);
    
end
%% Function for generating N gaussian samples
function [data, classIndex] = generateGaussianSamples(mu, sigma, nSamples, prior)


%  *   Name           | I/P | O/P | I/O | Purpose
%  *   ---------------+-----+-----+-----+----------------------------------
%  *   mu             |  X  |     |     | cell with the class dependent
%  *                  |     |     |     | d-dimensional mean vector
%  *                  |     |     |     |
%  *   sigma          |  X  |     |     | k-by-1 cell with the class
%  *                  |     |     |     | dependent d-by-d covariance matrix
%  *                  |     |     |     |
%  *   nSamples       |  X  |     |     | scalar indicating number of
%  *                  |     |     |     | samples to be generated
%  *                  |     |     |     |
%  *   prior          |  X  |     |     | prior - k-by-1 vector with
%  *                  |     |     |     | class dependent mean
%  *                  |     |     |     |
%  *   data           |     |  X  |     | nSamples-by-d array with the
%  *                  |     |     |     | simulated data distributed along the rows
%  *                  |     |     |     |
%  *   classIndex     |     |  X  |     | vector of length nSamples with
%  *                  |     |     |     | the class index for each datapoint
%  *                  |     |     |     |
%  *   ---------------+-----+-----+-----+----------------------------------


% Error checking
if sum(prior) ~= 1
    error('priors do no add to one');
end

% First, sample the class indexes. We can do this by generating uniformly
% distributed numbers from 0 to 1 and using thresholds based on the prior probabilities

classTempScalar = rand(nSamples, 1);
priorThresholds = cumsum([0; prior]);

nClass = numel(mu);

data = cell(nClass, 1);
classIndex = cell(nClass, 1);

for idxClass = 1:nClass
    nSamplesClass = nnz(classTempScalar>=priorThresholds(idxClass) & classTempScalar<priorThresholds(idxClass+1));
    
    % Generate samples according to class dependent parameters
    data{idxClass} = mvnrnd(mu{idxClass}, sigma{idxClass}, nSamplesClass);
    
    % Set class labels
    classIndex{idxClass} = ones(nSamplesClass,1) * (idxClass-1);
end

data = cell2mat(data);
classIndex = cell2mat(classIndex);
end


%% Function to fit a mixture of M gaussian pdfs using EM algorithm
function [BIC] = ModelOrderSelection(x,alpha, mu, Sigma,N,M)
y = evalGMM(x,alpha,mu,Sigma);
Neg2lnP = -2*sum(log(y));
BIC =  Neg2lnP + ((2*M)+(M-1+2))*log(N);
end

function gmm = evalGMM(x,alpha,mu,Sigma)
gmm = zeros(1,size(x,2));
for m = 1:length(alpha) % evaluate the GMM on the grid
    gmm = gmm + alpha(m)*evalGaussian(x,mu(:,m),Sigma(:,:,m));
end
end

function g = evalGaussian(x,mu,Sigma)
% Evaluates the Gaussian pdf N(mu,Sigma) at each coumn of X
[n,N] = size(x);
C = ((2*pi)^n * det(Sigma))^(-1/2);
E = -0.5*sum((x-repmat(mu,1,N)).*(inv(Sigma)*(x-repmat(mu,1,N))),1);
g = C*exp(E);
end