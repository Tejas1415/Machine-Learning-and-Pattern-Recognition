clear all, close all,

L = 3; % Polynomial order
sigma = 1e0; % AWG noise std
N = 10; % number of samples

wTrue = [randn(L,1);1]; % true parameters [a,b,c,d]

Ng = 2*6*10+1; Ns = 1000;
Gamma = 10.^linspace(-2,3,Ng); % parameter of prior
ParameterSquaredError = zeros(Ns,Ng);
for g = 1:Ng
    gamma = Gamma(g);
    for s = 1:Ns
        % Generate data
        x = 2*(rand(1,N)-0.5); v = sigma*randn(1,N);
        % Form transposed van der Monde matrix with N 
        % values in x up to polynomial power L
        PsiX = zeros(L+1,N); PsiX(1,:) = ones(1,N);
        for m = 1:L, PsiX(m+1,:) = x.^m; end,
        y = wTrue'*PsiX + v;
        % Estimate parameters with MAP for specified prior
        R = PsiX*PsiX'; q = PsiX*y'; 
        wMAP = inv(R+(sigma/gamma)^2*eye(L+1))*q;
        ParameterSquaredError(s,g) = norm(wMAP-wTrue)^2;
    end
end
Y = prctile(ParameterSquaredError,[0,5,25,50,75,95,100],1);
figure(1), loglog(Gamma,Y), 
xlabel('Gamma'), ylabel('Parameter Squared Error Percentiles'),
title('Percentiles [0,5,25,50,75,95,100]% are shown...'),

