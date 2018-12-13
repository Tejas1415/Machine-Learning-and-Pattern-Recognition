clc
clear all 
close all

n=2; % Dimension
N=10000; % Num of samples to be generated 
mu(:,1) = [-1; 0]; mu(:,2) = [1; 0]; % keep 2 columns for easier looping further
% Let sigma be the diagonal matrix with eigen values lambdai of covariance
% on the diagonal
Sigma(:,:,1) = [16 0;0 1]; Sigma(:,:,2) = [1 0;0 16];
p = [0.35, 0.65]; % Class priors for class 0 and class 1
label = rand(1,N) >= p(1); % creating labels for each sample generated
Nc = [length(find(label==0)),length(find(label==1))]; % number of samples from each class
x = zeros(n,N); 
for l = 0:1
    x(:,label==l) = randGaussian(Nc(l+1),mu(:,l+1),Sigma(:,:,l+1));
end
figure(2), clf,title('Fig 2');
plot(x(1,label==0),x(2,label==0),'o'), hold on,
plot(x(1,label==1),x(2,label==1),'+'), axis equal,
legend('class1', 'class2');

lambda = [0 1;1 0]; % loss values
gamma = (lambda(2,1)-lambda(1,1))/(lambda(1,2)-lambda(2,2)) * p(1)/p(2); %threshold
discriminantScore = log(evalGaussian(x,mu(:,2),Sigma(:,:,2)))-log(evalGaussian(x,mu(:,1),Sigma(:,:,1)));% - log(gamma);
decision = (discriminantScore >= log(gamma));

ind00 = find(decision==0 & label==0); p00 = length(ind00)/Nc(1); % probability of true negative
ind10 = find(decision==1 & label==0); p10 = length(ind10)/Nc(1); % probability of false positive
ind01 = find(decision==0 & label==1); p01 = length(ind01)/Nc(2); % probability of false negative
ind11 = find(decision==1 & label==1); p11 = length(ind11)/Nc(2); % probability of true positive
pe_min_expected_loss_classifier = [p10,p01]*Nc'/N % probability of error, empirically estimated

figure(1), % class 0 circle, class 1 +, correct green, incorrect red
plot(x(1,ind00),x(2,ind00),'og'); hold on,
plot(x(1,ind10),x(2,ind10),'or'); hold on,
plot(x(1,ind01),x(2,ind01),'+r'); hold on,
plot(x(1,ind11),x(2,ind11),'+g'); hold on,
axis equal,

% Till the above part, true positives and false negetives are printed. Now
% we try to print the gaussian decision boundries into the same figure.
horizontalGrid = linspace(floor(min(x(1,:))),ceil(max(x(1,:))),101);
verticalGrid = linspace(floor(min(x(2,:))),ceil(max(x(2,:))),91);
[h,v] = meshgrid(horizontalGrid,verticalGrid);
discriminantScoreGridValues = log(evalGaussian([h(:)';v(:)'],mu(:,2),Sigma(:,:,2)))-log(evalGaussian([h(:)';v(:)'],mu(:,1),Sigma(:,:,1))) - log(gamma);
minDSGV = min(discriminantScoreGridValues);
maxDSGV = max(discriminantScoreGridValues);
discriminantScoreGrid = reshape(discriminantScoreGridValues,91,101);
figure(1), contour(horizontalGrid,verticalGrid,discriminantScoreGrid,[minDSGV*[0.9,0.6,0.3],0,[0.3,0.6,0.9]*maxDSGV]); % plot equilevel contours of the discriminant function 
% including the contour at level 0 which is the decision boundary


Sb = (mu(:,1)-mu(:,2))*(mu(:,1)-mu(:,2))';
Sw = Sigma(:,:,1) + Sigma(:,:,2);
[V,D] = eig(inv(Sw)*Sb); % LDA solution satisfies alpha Sw w = Sb w; ie w is a generalized eigenvector of (Sw,Sb)
% equivalently alpha w  = inv(Sw) Sb w
[~,ind] = sort(diag(D),'descend');
wLDA = V(:,ind(1)); % Fisher LDA projection vector
yLDA = wLDA'*x; % All data projected on to the line spanned by wLDA
wLDA = sign(mean(yLDA(find(label==1)))-mean(yLDA(find(label==0))))*wLDA; % ensures class1 falls on the + side of the axis
yLDA = sign(mean(yLDA(find(label==1)))-mean(yLDA(find(label==0))))*yLDA; % flip yLDA accordingly

% Applying Threshold by Approxicating the means.
% calculating the ratio of the number of elements in class 1 and number of
% elements in class 2
ratio = Nc(2) / Nc(1);
Threshold = (mean(yLDA(find(label==0)))*ratio + mean(yLDA(find(label==1))))/2; 
% Avg of both means,with respect to their distribution count

figure(3), clf,

plot(yLDA(find(label==0)),zeros(1,Nc(1)),'o'), hold on,
plot(yLDA(find(label==1)),zeros(1,Nc(2)),'+'), hold on, axis equal
plot([Threshold Threshold], [-10 10])
tau = 0;
decisionLDA = (yLDA >= 0);

%Plotting ROC curve 
figure(4),           
plotroc(label, decision); title('ROC curve for Minimum Expected Loss Classifier'); hold on  % in the syntax plotroc(targets,outputs)
figure(5),
plotroc(label, decisionLDA); title('ROC curve for FisherLDA');


LDA00 = find(decisionLDA==0 & label==0); pLDA00 = length(LDA00)/Nc(1); % probability of true negative
LDA10 = find(decisionLDA==1 & label==0); pLDA10 = length(LDA10)/Nc(1); % probability of false positive
LDA01 = find(decisionLDA==0 & label==1); pLDA01 = length(LDA01)/Nc(2); % probability of false negative
LDA11 = find(decisionLDA==1 & label==1); pLDA11 = length(LDA11)/Nc(2); % probability of true positive
pe_LDA = [pLDA10,pLDA01]*Nc'/N % probability of error, empirically estimated


% For Min expected loss classifier since gamma value is 1, the MAP threshold is (p10,p11) 
figure(6),
Draw_ROC(decision, label);
hold on
plot(p10,p11,'*r');
hold on
Draw_ROC(decisionLDA, label);
hold on
plot(pLDA10, pLDA11, '*b');
leg = legend('Min expected loss classifier','0-1 loss Map','FisherLDA', 'Threshold for min p(error) of LDA');
set(leg, 'location', 'best');                    % Placing the legend in the best position 


% from min(yLDA) to max(yLDA) sort the matrix without altering their
% corresponding labels.
%yLDA_labels = [yLDA; label];
%sorted_yLDA = sortrows(transpose(yLDA_labels)); % Lexicographically sorts based on values in first column without altering the corresponding second column 
% Plotting the Threshold for FisherLDA
%class1=0; class2=0;
%for tau = min(yLDA):0.2:max(yLDA)
 %   for i= tau-0.5:0.1:tau+0.5
        
