close all
% Load Data
testData = xlsread('testData.xlsx');
Tlen = length(testData);
x_test = testData(:,1:2)';
y_test = testData(:,3:4)';
% Run motion model of learned filter
x_bel = A*x_test(:,1:Tlen-1);
% Run observation model of learned filter
y_bel = C*x_test;
% % Plotting
% figure(1)
% x_test1 = x_test(1,2:end);% x_bel1 = x_bel(1,:);
% x_test2 = x_test(2,2:end);
% y_bel2 = x_bel(2,:);
% plot(x_test1, x_test2, '-r');
% hold on;
% plot(x_bel1, x_bel2, '-b');
%
% figure(2)
% y_test1 = y_test(1,2:end);
% y_bel1 = y_bel(1,:);
% y_test2 = y_test(2,2:end);
% y_bel2 = y_bel(2,:);
% plot(y_test1, y_test2, '-r');
% hold on;
% plot(y_bel1, y_bel2, '-b');
figure(1)
x_test1 = x_test(1,2:end);
x_bel1 = x_bel(1,:);
subplot(1, 2, 1)
plot(x_test1,'-r')
title('States:');
hold on;
subplot(1, 2, 1)
plot(x_bel1,'-b')
legend('Actual','Predicted');
x_test2 = x_test(2,2:end);
x_bel2 = x_bel(2,:);
subplot(1, 2, 2)
plot(x_test2,'-r')
hold on;
subplot(1, 2, 2)
plot(x_bel2,'-b')
legend('Actual','Predicted');
hold off
figure(2)
y_test1 = y_test(1,2:end);
y_bel1 = y_bel(1,2:end);
subplot(1, 2, 1)
plot(y_test1,'-r')
title('Observations:');
hold on;
subplot(1, 2, 1)
plot(y_bel1,'-b')
legend('Actual','Predicted');
y_test2 = y_test(2,2:end);
y_belt2 = y_bel(2,2:end);
subplot(1, 2, 2)
plot(y_test2,'-r')
hold on;
subplot(1, 2, 2)
plot(y_belt2,'-b')
legend('Actual','Predicted');
