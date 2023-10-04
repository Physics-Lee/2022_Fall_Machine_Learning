x = load('D:\Public_repository\ml_2022_f\lab\lab4\Datasets\Aggregation.txt');
scatter(x(:,1),x(:,2))

figure(2)
y = load('D:\Public_repository\ml_2022_f\lab\lab4\Datasets\D31.txt');
scatter(y(:,1),y(:,2))

figure(3)
z = load('D:\Public_repository\ml_2022_f\lab\lab4\Datasets\R15.txt');
scatter(z(:,1),z(:,2))

% tic;
% my_matrix_1 = Euclid_Metric(x);
% toc;
% save('my_matrix_1.mat','my_matrix_1');
% 
% tic;
% my_matrix_2 = Euclid_Metric(y);
% toc;
% % save('my_matrix_2.mat','my_matrix_2');
% 
% tic;
% my_matrix_3 = Euclid_Metric(z);
% toc;
% save('my_matrix_3.mat','my_matrix_3');