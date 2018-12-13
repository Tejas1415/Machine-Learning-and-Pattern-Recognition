function EMforGMM(N)
% Generates N samples from a specified GMM,
% then uses EM algorithm to estimate the parameters
% of a GMM that has the same number of components
% as the true GMM that generates the samples.

close all,
delta = 1e-5; % specifies tolerance for EM stopping criterion

% Generate samples from a 3-component GMM
alpha_true = [0.2,0.3,0.5];
mu_true = [-10 0 10;0 0 0];
Sigma_true(:,:,1) = [3 1;1 20];
Sigma_true(:,:,2) = [7 1;1 2];
Sigma_true(:,:,3) = [4 1;1 16];
x = randGMM(N,alpha_true,mu_true,Sigma_true);
[d,M] = size(mu_true); % determine dimensionality of samples and number of GMM components

% Initialize the GMM to randomly selected samples
alpha = ones(1,M)/M;
shuffledIndices = randperm(N);
mu = x(:,shuffledIndices(1:M)); % pick M random samples as initial mean estimates
[~,assignedCentroidLabels] = min(pdist2(mu',x'),[],1); % assign each sample to the nearest mean
for m = 1:M % use sample covariances of initial assignments as initial covariance estimates
    Sigma(:,:,m) = cov(x(:,find(assignedCentroidLabels==m))');
end
t = 0; displayProgress(t,x,alpha,mu,Sigma,NaN,delta);

Converged = 0; % Not converged at the beginning
while ~Converged
    for l = 1:M
        temp(l,:) = repmat(alpha(l),1,N).*evalGaussian(x,mu(:,l),Sigma(:,:,l));
    end
    plgivenx = temp./sum(temp,1);
    alphaNew = mean(plgivenx,2)';
    w = plgivenx./repmat(sum(plgivenx,2),1,N);
    muNew = x*w';
    for l = 1:M
        v = x-repmat(muNew(:,l),1,N);
        u = repmat(w(l,:),d,1).*v;
        SigmaNew(:,:,l) = u*v';
    end
    Dalpha = sum(abs(alphaNew-alpha));
    Dmu = sum(sum(abs(muNew-mu)));
    DSigma = sum(sum(abs(abs(SigmaNew-Sigma))));
    sumD = (Dalpha+Dmu+sum(squeeze(DSigma)));
    % Being lazy in sumD; included off-diagonal covariance entries twice...
    Converged = (sumD<delta); % Check if converged
    alpha = alphaNew; mu = muNew; Sigma = SigmaNew;
    t = t+1; displayProgress(t,x,alpha,mu,Sigma,sumD,delta);
end
keyboard,

%%%
function displayProgress(t,x,alpha,mu,Sigma,sumD,delta)
figure(1),
if size(x,1)==2
    subplot(1,3,1), cla, plot(x(1,:),x(2,:),'b.'); 
    xlabel('x_1'), ylabel('x_2'), title('Data and Estimated GMM Contours'),
    axis equal, hold on;
    rangex1 = [min(x(1,:)),max(x(1,:))];
    rangex2 = [min(x(2,:)),max(x(2,:))];
    [x1Grid,x2Grid,zGMM] = contourGMM(alpha,mu,Sigma,rangex1,rangex2);
    contour(x1Grid,x2Grid,zGMM); axis equal, 
end
logLikelihood = sum(log(evalGMM(x,alpha,mu,Sigma)));
subplot(1,3,2), plot(t,logLikelihood,'b.'); hold on,
xlabel('Iteration Index'), ylabel('Log-Likelihood of Data'),
title('Monitoring if the log-likelihood objective is going up...'),
if t >= 1
    subplot(1,3,3), 
    semilogy(t,sumD,'b.'), hold on,
    semilogy(t,delta,'r-'), 
    xlabel('Iteration Index'), ylabel('Absolute-sum-of-parameter-updates')
    title('Monitoring if stopping criterion has been met...'),
end
drawnow; pause(0.1),

%%%
function x = randGMM(N,alpha,mu,Sigma)
d = size(mu,1); % dimensionality of samples
cum_alpha = [0,cumsum(alpha)];
u = rand(1,N); x = zeros(d,N); labels = zeros(1,N);
for m = 1:length(alpha)
    ind = find(cum_alpha(m)<u & u<=cum_alpha(m+1)); 
    x(:,ind) = randGaussian(length(ind),mu(:,m),Sigma(:,:,m));
end

%%%
function x = randGaussian(N,mu,Sigma)
% Generates N samples from a Gaussian pdf with mean mu covariance Sigma
n = length(mu);
z =  randn(n,N);
A = Sigma^(1/2);
x = A*z + repmat(mu,1,N);

%%%
function [x1Grid,x2Grid,zGMM] = contourGMM(alpha,mu,Sigma,rangex1,rangex2)
x1Grid = linspace(floor(rangex1(1)),ceil(rangex1(2)),101);
x2Grid = linspace(floor(rangex2(1)),ceil(rangex2(2)),91);
[h,v] = meshgrid(x1Grid,x2Grid);
GMM = evalGMM([h(:)';v(:)'],alpha, mu, Sigma);
zGMM = reshape(GMM,91,101);
%figure(1), contour(horizontalGrid,verticalGrid,discriminantScoreGrid,[minDSGV*[0.9,0.6,0.3],0,[0.3,0.6,0.9]*maxDSGV]); % plot equilevel contours of the discriminant function 

%%%
function gmm = evalGMM(x,alpha,mu,Sigma)
gmm = zeros(1,size(x,2));
for m = 1:length(alpha) % evaluate the GMM on the grid
    gmm = gmm + alpha(m)*evalGaussian(x,mu(:,m),Sigma(:,:,m));
end

%%%
function g = evalGaussian(x,mu,Sigma)
% Evaluates the Gaussian pdf N(mu,Sigma) at each coumn of X
[n,N] = size(x);
C = ((2*pi)^n * det(Sigma))^(-1/2);
E = -0.5*sum((x-repmat(mu,1,N)).*(inv(Sigma)*(x-repmat(mu,1,N))),1);
g = C*exp(E);
