close all;
clear;
clc;
format compact;
%% 数据提取
%addpath('/redhdd/changhongli/SSA/');
folderPath='H:\基于麻雀算法的特征选择用于EEG抑郁症识别\Results/'
dataName=dir([folderPath '*.mat']);
Acc=[];

for i=1:53
    i
   
    dataName(i).name
    load([folderPath dataName(i).name]);
    Acc=cat(1,Acc,acc);
    
end
testAcc=Acc(:,2:2:end);
%testAcc=testAcc(:,[1 5 9 2 6 10 3 7 11 4 8 12]);
m=mean(testAcc)*100;

s=std(testAcc)*100;
m1=reshape(m,4,3);
s1=reshape(s,4,3);
bar([1 4 7 10],m(1:4),'BarWidth',0.2);  
hold on;  
bar([2 5 8 11],m(5:8),'BarWidth',0.2);
bar([3 6 9 12],m(9:12),'BarWidth',0.2);
errorbar(m([1 5 9 2 6 10 3 7 11 4 8 12]),s([1 5 9 2 6 10 3 7 11 4 8 12]),'k','LineStyle','none'); 
