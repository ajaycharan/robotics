
clc
clear all
close all
% Load training data
trainData = xlsread('trainData.xlsx');
% State
x = trainData(:,1:2)';
% Observation
y = trainData(:,3:4)';
% Set up the parameters
thrsh = 0.01;
A = 1*eye(2);
Q = 1*eye(2);
C = 1*eye(2);
R = 1*eye(2);

pi1 = x(:,1);
V1 = 0.1*eye(2);
num_itr_max = 50;
A_initial = A;
Q_initial = Q;
C_initial = C;
R_initial = R;
pi1_initial = pi1;
V1_initial = V1;
T=length(y);
LL=[];
converged = 0;
prev_log_lik = -Inf;
itr_num = 0;
while(itr_num <= num_itr_max)
% Initializing
xtt = zeros(2,T);
xtt1 = zeros(2,T);
xtT = zeros(2,T);
Vtt = zeros(2,2*T);
Vtt1 = zeros(2,2*T);
Vtt1T = zeros(2,2*T);
VtT = zeros(2,2*T);
J = zeros(2,2*T);
% E-Step
for t=1:T
if(t==1)
% For first iteration, start with intitial values
xtt1(:,1) = pi1;
Vtt1(:,2*t-1:2*t) = V1;
else
% Run Kalman Filter Forward
xtt1(:,t) = A*xtt(:,t-1);
Vtt1(:,2*t-1:2*t) = A*Vtt(:,2*t-3:2*t-2)*A' + Q;
end
Kt = Vtt1(:,2*t-1:2*t)*C' * inv(C*Vtt1(:,2*t-1:2*t)*C' + R);
xtt(:,t) = xtt1(:,t) + Kt*(y(:,t) - C*xtt1(:,t));
Vtt(:,2*t-1:2*t) = Vtt1(:,2*t-1:2*t)-Kt*C*Vtt1(:,2*t-1:2*t);
end
KT = Kt;
xtT(:,T) = xtt(:,T);
VtT(:,2*T-1:2*T) = Vtt(:,2*T-1:2*T);
for t=T:-1:2
% J is the ratio of v's at t and t-1
J(:,2*t-3:2*t-2) = Vtt(:,2*t-3:2*t-2)*A'*inv(Vtt1(:,2*t-1:2*t));
% get x and v by rauch recursion (back ward recursion)
xtT(:,t-1) = xtt(:,t-1) + J(:,2*t-3:2*t-2)*(xtT(:,t)-A*xtt(:,t-1));VtT(:,2*t-3:2*t-2) = Vtt(:,2*t-3:2*t-2) + J(:,2*t-3:2*t-2)*(VtT(:,2*t-1:2*t)-Vtt1(:,2*t-1:2*t))*J(:,2*t-3:2*t-2)';
end
x_hat = xtT;
Pt = zeros(size(VtT));
for t=1:T
Pt(:,2*t-1:2*t) = VtT(:,2*t-1:2*t) + xtT(:,t)*xtT(:,t)';
end
% covariance between x and x at t-1
Vtt1T(:,2*T-1:2*T) = (1 - KT*C)*A*Vtt(:,2*T-3:2*T-2);
for t=T:-1:3
Vtt1T(:,2*t-3:2*t-2) = Vtt(:,2*t-3:2*t-2)*J(:,2*t-5:2*t-4)' + J(:,2*t-3:2*t-2)*(Vtt1T(:,2*t-1:2*t)- A*Vtt(:,2*t-3:2*t-2))*J(:,2*t-5:2*t-4)';
end
Ptt1 = zeros(size(VtT));
for t=2:T
Ptt1(:,2*t-1:2*t) = Vtt1T(:,2*t-1:2*t) + xtT(:,t)*xtT(:,t-1)';
end
% Compute log likelihoods
loglik_part1 =0;
for t=1:T
loglik_part1 = loglik_part1 - (y(:,t)-
C*x_hat(:,t))'*inv(R)*(y(:,t)-C*x_hat(:,t));
end
loglik_part1 = loglik_part1 - T*log(abs(det(R)));
loglik_part1 = loglik_part1/2;
loglik_part2 =0;
for t=2:T
loglik_part2 = loglik_part2 - (x_hat(:,t)-(A*x_hat(:,t-1)))'*inv(Q)*(x_hat(:,t)-(A*x_hat(:,t-1)));
end
loglik_part2 = loglik_part2 - (T-1)*log(abs(det(Q)));
loglik_part2 = loglik_part2/2;
loglik_part3 = - ((x_hat(1)-pi1)'*inv(V1)*(x_hat(1)-pi1))/2 -
log(abs(det(V1)))/2 - T*log(2*pi);
loglik = loglik_part1 + loglik_part2 + loglik_part3;
LL=[LL loglik];
% M-Step
num = 0;
den = 0;
for t=1:T
num = num + y(:,t)*x_hat(:,t)';
den = den + Pt(:,2*t-1:2*t);
end
C_new = num*inv(den);
R_new = 0;
for t=1:T
R_new = R_new + y(:,t)*y(:,t)' - C_new*x_hat(:,t)*y(:,t)';
end
R_new = R_new/T;
num = 0;
den = 0;
for t=2:T
num = num + Ptt1(:,2*t-1:2*t);
den = den + Pt(:,2*t-3:2*t-2);
end
A_new = num*inv(den);
part1 = 0;
part2 = 0;
for t=2:T
part1 = part1 + Pt(:,2*t-1:2*t);
part2 = part2 + Ptt1(:,2*t-1:2*t);
end
Q_new = (part1 - A_new*part2)/(T-1);
pi1_new = x_hat(:,1);
V1_new = Pt(:,1:2)-x_hat(:,1)*x_hat(:,1)';
pi1 = pi1_new;
V1 = V1_new;
A = A_new;
Q = Q_new;
C = C_new;
R = R_new;
% Check for convergence
if (abs(loglik - prev_log_lik) / ((abs(loglik) + abs(prev_log_lik) +
eps)/2)) < thrsh
break;
end
prev_log_lik = loglik;
itr_num = itr_num+1;
end
C = C+0.4*eye(2);
