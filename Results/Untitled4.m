close all;
clear;
clc;
format compact;
%% ������ȡ
%addpath('/redhdd/changhongli/SSA/');
folderPath='I:\������ȸ�㷨������ѡ������EEG����֢ʶ��\Results/'
dataName=dir([folderPath '*.mat']);
Curve=zeros(50,8);
for i=1:53
    i
    dataName(i).name
    load([folderPath dataName(i).name]);
    
    curve=curve(:,1:50)';
    Curve=Curve+ curve;
end

 plot(Curve/53,'DisplayName','curve');
 legend('BLDA+SSA ALL trials','BLDA +SSA Happy trials','BLDA+SSA Fear trials','BLDA+SSA Sad trials',...
        'BLDA+RSSA ALL trials','BLDA RSSA Happy trials','BLDA+RSSA Fear trials','BLDA+RSSA Sad trials');  