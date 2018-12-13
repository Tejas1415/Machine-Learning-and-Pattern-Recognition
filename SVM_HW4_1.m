% Homework4 SVM 
clc
clear all
close all

N=100;
beta_true1 = [0.5,0.25,0.25];
mu_true(1,:) = [0, 0];
mu_true(2,:) = [3, 0];
mu_true(3,:) = [0, 2];
Sigma_true(:,:,1) = [1 0;0 1];
Sigma_true(:,:,2) = [1 0;0 0.5];
Sigma_true(:,:,3) = [0.5 0;0 1];

X1 = [mvnrnd(mu_true(1,:),Sigma_true(:,:,1),N*beta_true1(1)); mvnrnd(mu_true(2,:),Sigma_true(:,:,2),N*beta_true1(2)); mvnrnd(mu_true(3,:),Sigma_true(:,:,3),N*beta_true1(3))];
labels = zeros(1,N);
Class1_start_index = N*beta_true1(1); % Number of class 0 samples 
for i= Class1_start_index: N
    labels[i] = 1;
end


