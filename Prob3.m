close all; clear; clc;
%% Generation of Gaussian Samples
%% 
mu{1} = [0;0];
sigma{1} = [1 0; 0 1];
mu{2} = [3;0];
sigma{2} = [1 0;0 1];
mu{3} = [0;2];
sigma{3} = [1 0;0 1];
%% Generation of gaussian samples
beta = [0.5;0.25;0.25];
M = 3;
N = 100;
[data, classIndex] = generateGaussianSamples(mu, sigma, N, beta);
classIndex(classIndex > 0) = -1;
classIndex(classIndex == 0) = 1;
%% SVM Classifier
SVMModel = fitcsvm(data,classIndex,'Standardize',true,'KernelFunction','Gaussian',...
    'KernelScale','auto');
%% Optimize Hyper parameters
%% Modified SVM classifier
svmmod = fitcsvm(data,classIndex,'KernelFunction','rbf','OptimizeHyperparameters','auto',...
    'HyperparameterOptimizationOptions',struct('AcquisitionFunctionName',...
    'expected-improvement-plus'));

% Train a Gaussian kernel SVM with cross-validation
% to select hyperparameters that minimize probability 
% of error (i.e. maximize accuracy; 0-1 loss scenario)
c = cvpartition(200,'KFold',10);
%% Optimize the Fit
% To find a good fit, meaning one with a low cross-validation loss, set
% options to use Bayesian optimization. Use the same cross-validation
% partition |c| in all optimizations.
%
% For reproducibility, use the |'expected-improvement-plus'| acquisition
% function.
% opts = struct('Optimizer','bayesopt','ShowPlots',true,'CVPartition',c,...
%     'AcquisitionFunctionName','expected-improvement-plus');
% svmmod = fitcsvm(data',classIndex,'KernelFunction','rbf',...
%     'OptimizeHyperparameters','auto','HyperparameterOptimizationOptions',opts);

%% Test Samples
beta = [0.5;0.25;0.25];
M = 3;
N = 10000;
[data, classIndex] = generateGaussianSamples(mu, sigma, N, beta);

classIndex(classIndex > 0) = -1;
classIndex(classIndex == 0) = 1;
[label,score] = predict(svmmod,data);
figure;gscatter(data(:,1),data(:,2),classIndex,'br','++');
figure;gscatter(data(:,1),data(:,2),label,'br','++');
% svmmod = SVMModel;
%%
% Find the loss of the optimized model.
test_err = loss(svmmod,data,classIndex);
%%
% This loss is the same as the loss reported in the optimization output
% under "Observed objective function value".
%%
% Visualize the optimized classifier.
d = 0.02;
[x1Grid,x2Grid] = meshgrid(min(data(:,1)):d:max(data(:,1)),...
    min(data(:,2)):d:max(data(:,2)));
xGrid = [x1Grid(:),x2Grid(:)];
[~,scores] = predict(svmmod,xGrid);
figure;
h = nan(3,1); % Preallocation
dec = zeros(N,1);
dec(find(label == classIndex)) = 1;
h(1:2) = gscatter(data(:,1),data(:,2),dec,'rg','++');
hold on
% h(3) = gscatter(data(:,1),data(:,2),(label==classIndex),'gg','++');
% h(3) = plot(data(svmmod.IsSupportVector,1),...
%     data(svmmod.IsSupportVector,2),'r+');
contour(x1Grid,x2Grid,reshape(scores(:,2),size(x1Grid)),[0 0],'k');
legend({'Incorrect','Correct','SVM Boundary'},'Location','Southeast');
axis equal
hold off

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
