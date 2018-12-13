clc
clear all
close all

for i= 1:4
    xi = [1 0 -1 0]; yi= [0 1 0 -1]; % each iteration takes i reference points.
% Considering the four reference points to be (1,0) (0,1) (-1,0) (0, -1)

%Let the true position of the vehicle be at 0.5, 0.6
xt = 0.5;      % Arbitary points chosen inside the unit circle with center zero.
yt = 0.6;

x = -2:0.1:2; %Limits as provided in the question
y = -2:0.1:2;

mu = [0,0];
sigma = [0.3 0; 0 0.3];
[X Y] = meshgrid(x,y);
di = sqrt((xt-xi).^2 + (yt-yi).^2);      % Calculating the distance between the reference point and the true position
ni = (1/(2*3.14*0.3))*exp(-0.5 * ((X-0)^2) / (0.3)); % producing noise with the specified gaussian(0, 0.3) 
%Z  = di+ni;
Zmap = (0.5* 1/di(i).^2 - (X./0.3))*(0.3); % The solution that was obtained for the question 4 Part a.
Z = mvnpdf([Zmap(:), Zmap(:)], mu, sigma);
%Z = reshape(Zmap, size(X));
[X,Y,Zmap] = peaks;
figure(i), contour(X,Y,Zmap, 20), hold on
plot(xt,yt,'r+'); hold on
plot(xi, yi, 'bo'); hold on
%plot(Zmap,1,'r*');
end

