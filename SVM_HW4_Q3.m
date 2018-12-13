% Homework4 SVM 
clc
clear all
close all

%% Prepare training dataset.
N=100;
beta_true1 = [0.5,0.25,0.25];
mu_true(1,:) = [0, 0];
mu_true(2,:) = [3, 0];
mu_true(3,:) = [0, 2];
Sigma_true(:,:,1) = [1 0;0 1];
Sigma_true(:,:,2) = [1 0;0 0.5];
Sigma_true(:,:,3) = [0.5 0;0 1];

X1 = [mvnrnd(mu_true(1,:),Sigma_true(:,:,1),N*beta_true1(1)); mvnrnd(mu_true(2,:),Sigma_true(:,:,2),N*beta_true1(2)); mvnrnd(mu_true(3,:),Sigma_true(:,:,3),N*beta_true1(3))];
labels = ones(1,N);                   % The first component class labels = +1
Class1_start_index = N*beta_true1(1); % Number of class +1 samples 

for i= Class1_start_index: N
    labels(i) = -1;          % Second and Third component guassian samples are regarded as class -1.
end

%% Now create an SVM model to train the 100 samples generated.
% Guassian Kernel with less box constraint value gives considerably good
% results over the trained model.
% Guassian Kernel - since it is a TALL data than wide data.
SVM_Model = fitcsvm(X1,labels,'Standardize',true,'KernelFunction','Gaussian',...
    'KernelScale','auto','BoxConstraint',1);

% 10 fold cross validation of the trained SVM model.
Cross_Validation = crossval(SVM_Model, 'KFold', 10);  
% Calculating the K-fold loss of the trained model.
% Emperical probability of error
Loss_train = kfoldLoss(Cross_Validation, 'LossFun', 'ClassifError')


% Now generating test set
N_test = 10000;

X1_test = [mvnrnd(mu_true(1,:),Sigma_true(:,:,1),N_test*beta_true1(1));...
    mvnrnd(mu_true(2,:),Sigma_true(:,:,2),N_test*beta_true1(2));...
    mvnrnd(mu_true(3,:),Sigma_true(:,:,3),N_test*beta_true1(3))];

%% Class labels generation. Componenet 1 = +1, component2 and component3
% generated values = -1.
labels_test = ones(1,N_test);
Class1_start_index_test = N_test*beta_true1(1); % Number of class +1 samples in test set 

for i= Class1_start_index_test: N_test
    labels_test(i) = -1;          % Second and Third component guassian samples are regarded as class -1.
end
labels_test = transpose(labels_test);
predict_labels = predict(SVM_Model, X1_test);
% Loss_test = loss(SVM_Model,predict_labels, labels_test);

%% Now to calculate the emperical probability of error with the classifier
% Manually check to find the number of misclassified elements.
count =0;
for i=1:N_test
    if(labels_test(i) ~= predict_labels(i)) % checking if misclassified.
        count = count + 1;                  % count the number of misclassifications
    end
end
Loss_calculated = count/ N_test       %misclassified/ total num of test data points. 


%% Visualise the results obtained.
figure(1),
for i=1:N_test
    if(labels_test(i) == predict_labels(i))
        plot(X1_test(i,1), X1_test(i,2),'g.','MarkerSize',5);
        hold on
    else
         plot(X1_test(i,1), X1_test(i,2),'r.','MarkerSize',5);
         hold on
    end
end
d = 0.02;
[x1Grid,x2Grid] = meshgrid(min(X1_test(:,1)):d:max(X1_test(:,1)),min(X1_test(:,2)):d:max(X1_test(:,2)));
xGrid = [x1Grid(:),x2Grid(:)];
[~,scores] = predict(SVM_Model,xGrid);
%Plotting the Decision Boundry
contour(x1Grid,x2Grid,reshape(scores(:,2),size(x1Grid)),[0 0],'k'); 
legend({'Incorrect','Correct','SVM Decision Boundary'},'Location','Best');
axis equal
title 'SVM Classification over the test dataset'

hold off