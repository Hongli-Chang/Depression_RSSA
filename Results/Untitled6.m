close all;
clear;
clc;
format compact;
%% 数据提取
%addpath('/redhdd/changhongli/SSA/');
folderPath='I:\基于麻雀算法的特征选择用于EEG抑郁症识别\Results/'
dataName=dir([folderPath '*.mat']);
Curve=zeros(50,8);
numFeaSub=zeros(8,128);
for i=1:53
    i
    numFea=[];
    dataName(i).name
    load([folderPath dataName(i).name]);
    for j=1:8
        chan=reshape(B(j,:),5,128);
        f=sum(chan);
      numFea=[numFea;f];
      
    end
   numFeaSub=numFeaSub+numFea; 
end
Mean = (numFeaSub/53)/5*100;
mean = Mean';
plot(mean,'DisplayName','mean');
legend('BLDA+SSA ALL trials','BLDA+SSA Happy trials','BLDA+SSA Fear trials','BLDA+SSA Sad trials',...
        'BLDA+RSSA ALL trials','BLDA RSSA Happy trials','BLDA+RSSA Fear trials','BLDA+RSSA Sad trials'); 
    