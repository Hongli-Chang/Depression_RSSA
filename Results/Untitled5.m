close all;
clear;
clc;
format compact;
%% ������ȡ
%addpath('/redhdd/changhongli/SSA/');
folderPath='H:\������ȸ�㷨������ѡ������EEG����֢ʶ��\Results/'
dataName=dir([folderPath '*.mat']);
Curve=zeros(50,8);
numFeaSub=[];
for i=1:53
    i
    numFea=[];
    dataName(i).name
    load([folderPath dataName(i).name]);
    for j=1:8
      f=size(find(B(j,:)==1),2);
      numFea=[numFea,f];
      
    end
   numFeaSub(i,:)=numFea; 
end
 m=mean(numFeaSub);
  s=std(numFeaSub);
%  plot(Curve/53,'DisplayName','curve');
%  legend('BLDA ALL trials','BLDA Happy trials','BLDA Fear trials','BLDA Sad trials',
%         'BLDA+SSA ALL trials','BLDA+SSA Happy trials','BLDA+SSA Fear trials','BLDA+SSA Sad trials',...
%         'BLDA+RSSA ALL trials','BLDA+RSSA Happy trials','BLDA+RSSA Fear trials','BLDA+RSSA Sad trials');  