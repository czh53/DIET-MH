%%  Model M-DOF
%   Model M-DOF simulate vertical tissue dynamics




%%
clear all;
close all;

%---number of DOF.
m = 200;
%% Asuume prior probablity funtion for theta 
N_refpoint = 200;
z = linspace(0,N_refpoint,N_refpoint);
k_health_low = 1;
k_health_high = 5; % ensure satruation
k_tumor_low = 4;
k_tumor_high = 10;


mu_health = (k_health_low + k_health_high)/2;
sigma_health = (k_health_high - mu_health)/3;

mu_tumor = (k_tumor_low + k_tumor_high)/2;
sigma_tumor = (k_tumor_high - mu_tumor)/3;

% generate the simulated healthy case
k_simu_health = mu_health*ones(N_refpoint,1) + randn(N_refpoint,1)*sigma_health ; 

% define tumor size
tumor_percent = 0.2;

% define tumor location
tumor_loc1 = 0.4;
tumor_loc2 = tumor_loc1  + tumor_percent;

% generate baseline elastance for cancerous breast
mu_z = (tumor_loc1 + tumor_loc2)/2*N_refpoint;
sigma_z = (tumor_percent*N_refpoint/2)/3;
k1 = mu_health * (2*pi)^0.5 * sigma_z;
k2 = mu_tumor * (2*pi)^0.5 * sigma_z - k1;

z_pdf = @(x,k1,k2,mu_z,sigma_z) 1/sigma_z/(2*pi)^0.5*(k1+k2*exp(-(x-mu_z).^2/2/sigma_z^2));
k_tumor_base = z_pdf(z,k1,k2,mu_z,sigma_z);

% generate the simulated tumor case
k_simu_tumor = k_tumor_base' + randn(N_refpoint,1)*sigma_health;

figure(1);
plot(z ,k_simu_health,'o','LineWidth',2)
xlabel('z(points)', 'FontSize',16);ylabel('elastance values', 'FontSize',16)
sgtitle(['healthy case--> ','mu=',num2str(sprintf('%0.1f',mu_health)),' & sigma=',num2str(sprintf('%0.3f',sigma_health))])
ylim([0 10])
grid on
figure(2);
plot(z ,k_simu_tumor,'o','LineWidth',2)
xlabel('z(points)', 'FontSize',16);ylabel('elastance values', 'FontSize',16)
sgtitle(['cancerous case--> ','mu=',num2str(sprintf('%0.1f',mu_tumor)),' & sigma=',num2str(sprintf('%0.3f',sigma_tumor))])
ylim([0 10])
grid on
%% get k trial samples
N_trial = 200; % This values has to be small enough to aviod product = 0 
RS_step = N_refpoint/N_trial;
k_RS_health = k_simu_health(1:RS_step:end);
k_RS_tumor = k_simu_tumor(1:RS_step:end);
k_values_health = k_RS_health(randperm(N_trial),:);
k_values_tumor = k_RS_tumor(randperm(N_trial),:);
k_trial_health = k_values_health;
k_trial_tumor = k_values_tumor;


figure(10);
histogram(k_trial_health,50)
histfit(k_trial_health,50,'normal')
xlabel('healthy elastance', 'FontSize',16);
ylabel('Number of elastance observations', 'FontSize',16);
xlim([0 10])
sgtitle(['mu',num2str(sprintf('%0.1f',mu_health)),' & sigma=',num2str(sprintf('%0.3f',sigma_health))])

figure(20);
histogram(k_trial_tumor,50)
histfit(k_trial_tumor,50,'normal')
pd_tumor = fitdist(k_trial_tumor,'Normal');
xlabel('cancerous elastance', 'FontSize',16);
ylabel('Number of elastance observations', 'FontSize',16);
xlim([0 10])
sgtitle(['mu',num2str(sprintf('%0.1f',pd_tumor.mu)),' & sigma=',num2str(sprintf('%0.3f',pd_tumor.sigma))])

%% get the "proposed" pdf for k and its parameters mu & sigma

k_prop_pdf = @(x,para) 1/para(2)/(2*pi)^0.5*exp(-(x-para(1)).^2/2/para(2)^2);
mu_min = 0.1;
mu_max = 10;
sigma_min = 0.01;
sigma_max = 5;
mu_prior = @(mu) unifpdf(mu,mu_min,mu_max);
sigma_prior = @(sigma) unifpdf(sigma,sigma_min,sigma_max);

y_est = k_prop_pdf(linspace(0,10,100),[3 1]);
y_mu = mu_prior(linspace(mu_min,mu_max,100));
y_sigma = sigma_prior(linspace(sigma_min,sigma_max,100));


%% get the Posterior pdf for health case
post_health = @(para) prod(k_prop_pdf(k_trial_health,para)) * mu_prior(para(1)) * sigma_prior(para(2)); 

%% get the Posterior pdf for tumor case
post_tumor = @(para) prod(k_prop_pdf(k_trial_tumor,para)) * mu_prior(para(1)) * sigma_prior(para(2)); 
            
%% visualize////NOTE, if the size of k_trial_health is too big, their product tends towards zero
% check for zero product

mu = linspace(mu_min,mu_max, 100);
sigma = linspace(sigma_min,sigma_max, 100);
simpost = zeros(100,100);
for i = 1:length(mu)
    for j = 1:length(sigma)
        simpost(i,j) = post_tumor([mu(i), sigma(j)]);
    end;
end;

figure(1000)
mesh(sigma,mu,simpost)
xlabel('sigma')
ylabel('mu')
zlabel('Posterior density')
view(-110,30)


%% MH algorithm 
init_para = [4 1];
nsamples = 10000;
delta1 = 0.1;
delta2 = 0.01;
% proppdf =  @(para,y) unifpdf(y(1)-para(1),-delta1,delta1) + unifpdf(y(2)-para(2),-delta2,delta2);
% proprnd = @(para) [(para(1)+ rand*2*delta1 - delta1) (para(2)+ rand*2*delta2 - delta2)];   
proppdf =  @(para,y) mvnpdf(y,para,eye(2));
proprnd = @(para) [(para(1)+ rand*2*delta1 - delta1) (para(2)+ rand*2*delta2 - delta2)];   
BM = 1000;
TM = 10;
%% healthy
[smpl_heal,accept_heal] = ...
    mhsample(init_para,nsamples,'pdf',post_health,'proppdf',proppdf,'proprnd',proprnd, 'burnin',BM,'thin', TM);

%%
figure(10000);
hist3(smpl_heal,[25,25]);
xlabel('mu')
xlim([2.7 3.7])
ylabel('sigma')
ylim([0.5 1.4])
zlabel('Posterior density')
view(-110,30)


figure(20000);
histogram(smpl_heal(:,1))
histfit(smpl_heal(:,1),50);
pd1_heal = fitdist(smpl_heal(:,1),'Normal');
xlabel('mu');
ylabel('density');
xlim([2.7 3.7])
sgtitle(['mu(true)=',num2str(sprintf('%0.1f',mu_health)),' & mu(est)=',num2str(sprintf('%0.1f',pd1_heal.mu))])

figure(30000);
histogram(smpl_heal(:,2))
histfit(smpl_heal(:,2),50);
pd2_heal = fitdist(smpl_heal(:,2),'Normal');
xlabel('sigma');
ylabel('density');
xlim([0.5 1.4])
sgtitle(['sigma(true)=',num2str(sprintf('%0.3f',sigma_health)),' & sigma(est)=',num2str(sprintf('%0.3f',pd2_heal.mu))])

%% tumor

[smpl_tumor,accept_tumor] = ...
    mhsample(init_para,nsamples,'pdf',post_tumor,'proppdf',proppdf,'proprnd',proprnd, 'burnin',BM,'thin', TM);
%%
figure(10001);
hist3(smpl_tumor,[25,25]);
xlabel('mu')
xlim([2.7 3.7])
ylabel('sigma')
ylim([0.5 1.4])
zlabel('Posterior density')
view(-110,30)


figure(20001);
histogram(smpl_tumor(:,1))
histfit(smpl_tumor(:,1),50);
pd1_tumor = fitdist(smpl_tumor(:,1),'Normal');
xlabel('mu');
ylabel('density');
xlim([2.7 3.7])
sgtitle(['mu(true)=',num2str(sprintf('%0.1f',pd_tumor.mu)),' & mu(est)=',num2str(sprintf('%0.1f',pd1_tumor.mu))])

figure(30001);
histogram(smpl_tumor(:,2))
histfit(smpl_tumor(:,2),50);
pd2_tumor = fitdist(smpl_tumor(:,2),'Normal');
xlabel('sigma');
ylabel('density');
xlim([0.5 1.4])
sgtitle(['sigma(true)=',num2str(sprintf('%0.3f',pd_tumor.sigma)),' & sigma(est)=',num2str(sprintf('%0.3f',pd2_tumor.mu))])

%%
figure(50000);
hist3(smpl_heal,[25,25]);
hold on 
hist3(smpl_tumor,[25,25],'EdgeColor','red');
xlabel('mu')
xlim([2.7 3.7])
ylabel('sigma')
ylim([0.5 1.4])
zlabel('Posterior density')
view(-110,30)
