close all; clear; clc;

%% 
mu{1} = [0;0];
sigma{1} = [1 0; 0 1];
mu{2} = [3;0];
sigma{2} = [1 0;0 0.5];
mu{3} = [0;2];
sigma{3} = [0.5 0;0 1];
%% Generation of gaussian samples
beta = [0.4;0.3;0.3];
M = 3;
N = 100;
[data, classIndex] = generateGaussianSamples(mu, sigma, N, beta);
 %% GMM Fit         
 GMModel = fitgmdist(data,M,'MaxIter',500,'covariancetype',...
     'diagonal','RegularizationValue',10^-15);
mu_est = GMModel.mu';
sigma_est =  zeros(2,2,M);
            for indx = 1:M
            sigma_est(:,:,indx) = [GMModel.Sigma(1,1,indx) 0;...
                0 GMModel.Sigma(1,2,indx)];
            end
beta_est = GMModel.ComponentProportion;
[GMMlabels] = evaluatePosterierprob(data',beta_est,mu_est,sigma_est,M);

%% K-means clustering
Kmeans_labels = kmeans(data,3,'Distance','correlation');
%% Plotting
figure;scatter(data(:,1),data(:,2),24,'g+');
figure;gscatter(data(:,1),data(:,2),GMMlabels,'br','++');
figure;gscatter(data(:,1),data(:,2),Kmeans_labels,'br','++');
%% Functions Used
function [labels] = evaluatePosterierprob(x,beta,mu,Sigma,M)
for m = 1:M
    num = evalGaussian(x,mu(:,m),Sigma(:,:,m));
    num = num.*beta(m);
    den = evalGMM(x,beta,mu,Sigma);
    post_prb(m,:) = num./den; 
end
 [~,labels] = max(post_prb);
end
%% Function for generating N gaussian samples
function [data, classIndex] = generateGaussianSamples(mu, sigma, nSamples, prior)

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

function gmm = evalGMM(x,alpha,mu,Sigma)
gmm = zeros(1,size(x,2));
for m = 1:length(alpha) % evaluate the GMM on the grid
    gmm = gmm + alpha(m)*evalGaussian(x,mu(:,m),Sigma(:,:,m));
end
end
