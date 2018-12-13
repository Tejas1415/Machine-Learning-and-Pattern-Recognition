%Homwwork42_1
%https://in.mathworks.com/help/stats/clustering-using-gaussian-mixture-models.html
%Important - When to Regularize
% Sometimes, during an EM iteration, a fitted covariance matrix can become ill conditioned,
% that is, the likelihood is escaping to infinity. This can happen if:
% 1. There are more predictors than data points.
% 2. You specify to fit with too many components.
% 3. Variables are highly correlated.

clc
clear all
close all

beta_true1 = [0.75,0.20,0.05];
beta_true2 = [0.80, 0.15, 0.05];
beta_true3 = [0.90, 0.05, 0.05];
mu_true(1,:) = [0, 0];
mu_true(2,:) = [3, 0];
mu_true(3,:) = [0, 2];
Sigma_true(:,:,1) = [1 0;0 1];
Sigma_true(:,:,2) = [1 0;0 0.5];
Sigma_true(:,:,3) = [0.5 0;0 1];

X1 = [mvnrnd(mu_true(1,:),Sigma_true(:,:,1),100*beta_true1(1)); mvnrnd(mu_true(2,:),Sigma_true(:,:,2),100*beta_true1(2)); mvnrnd(mu_true(3,:),Sigma_true(:,:,3),100*beta_true1(3))];
gm1 = fitgmdist(X1,3);

clusterX1 = cluster(gm1, X1);     % Gaussian clustering 
figure(1),
h1= scatter(X1(:,1), X1(:,2), clusterX1);   % plain data
figure(2), 
h2 = gscatter(X1(:,1), X1(:,2), clusterX1); title 'Clustered with GMM procedure'; % clustered GMM datax

% Kmeans clustering 
[grp,c] = kmeans(X1,3,'Distance','sqeuclidean');
figure(3);
plot(X1(:,1),X1(:,2),'k*','MarkerSize',5); title ' All Samples ';
figure(4);gscatter(X1(:,1),X1(:,2),grp,'brg','+++');

% Plotting with centroids for final Kmeans clusters

opts = statset('Display','final');
[idx,C] = kmeans(X1,3,'Distance','sqeuclidean','Replicates',5,'Options',opts);

figure(5);
plot(X1(idx==1,1),X1(idx==1,2),'r.','MarkerSize',12) % class 1
hold on
plot(X1(idx==2,1),X1(idx==2,2),'b.','MarkerSize',12) % class 2
plot(C(:,1),C(:,2),'kx','MarkerSize',15,'LineWidth',3)  % Plotting centroids
hold on
plot(X1(idx==3,1),X1(idx==3,2),'g.','MarkerSize',12) % class 3
legend('Cluster 1','Cluster 2', 'Centroids','Cluster 3','Location','NW')
title ('Cluster Assignments and Centroids');
hold off
