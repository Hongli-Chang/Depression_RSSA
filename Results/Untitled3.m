close all;
clear;
clc;
format compact;
%% 数据提取
%addpath('/redhdd/changhongli/SSA/');
folderPath='H:\基于麻雀算法的特征选择用于EEG抑郁症识别\Results/'
dataName=dir([folderPath '*.mat']);
Y_all=[];
Y_H=[];
Y_F=[];
Y_S=[];

Pre_all=[];
Pre_h=[];
Pre_f=[];
Pre_s=[];

Pre_S_all=[];
Pre_S_h=[];
Pre_S_f=[];
Pre_S_s=[];

Pre_R_all=[];
Pre_R_h=[];
Pre_R_f=[];
Pre_R_s=[];
S_H=[];
for i=1:53
    i
    dataName(i).name
    load([folderPath dataName(i).name]);
    
    Y_all = cat(1,Y_all,TEST_Y.test_Y');
    Y_H = cat(1,Y_H,TEST_Y.testH_Y');
    Y_F = cat(1,Y_F,TEST_Y.testF_Y');
    Y_S = cat(1,Y_S,TEST_Y.testS_Y');
    
    Pre_all=cat(1,Pre_all,PRE_TEST_Y.predict_labelTest_all');
    Pre_h=cat(1,Pre_h,PRE_TEST_Y.predict_labelTest_h');
    Pre_f=cat(1,Pre_f,PRE_TEST_Y.predict_labelTest_f');
    Pre_s=cat(1,Pre_s,PRE_TEST_Y.predict_labelTest_s');
   
    PRE_TEST_Y.S_predict_labelTest_H(PRE_TEST_Y.S_predict_labelTest_H>0)=1;
    PRE_TEST_Y.S_predict_labelTest_H(PRE_TEST_Y.S_predict_labelTest_H<0)=-1;
    S_accuracyTest_H=sum(PRE_TEST_Y.S_predict_labelTest_H==TEST_Y.testH_Y)/length(TEST_Y.testH_Y);
    S_H=[S_H;S_accuracyTest_H];
    Pre_S_all=cat(1,Pre_S_all,PRE_TEST_Y.S_predict_labelTest_all');
    Pre_S_h=cat(1,Pre_S_h,PRE_TEST_Y.S_predict_labelTest_H');
    Pre_S_f=cat(1,Pre_S_f,PRE_TEST_Y.S_predict_labelTest_F');
    Pre_S_s=cat(1,Pre_S_s,PRE_TEST_Y.S_predict_labelTest_S');

    Pre_R_all=cat(1,Pre_R_all,PRE_TEST_Y.R_S_predict_labelTest_all');
    Pre_R_h=cat(1,Pre_R_h,PRE_TEST_Y.R_S_predict_labelTest_H');
    Pre_R_f=cat(1,Pre_R_f,PRE_TEST_Y.R_S_predict_labelTest_F');
    Pre_R_s=cat(1,Pre_R_s,PRE_TEST_Y.R_S_predict_labelTest_S');    
end
Y_all(find(Y_all ==-1))=0;
Y_H(find(Y_H ==-1))=0;
Y_F(find(Y_F ==-1))=0;
Y_S(find(Y_S ==-1))=0;

Pre_all(find(Pre_all ==-1))=0;
Pre_h(find(Pre_h ==-1))=0;
Pre_f(find(Pre_f ==-1))=0;
Pre_s(find(Pre_s ==-1))=0;

Pre_S_all(find(Pre_S_all ==-1))=0;
Pre_S_h(find(Pre_S_h ==-1))=0;
Pre_S_f(find(Pre_S_f ==-1))=0;
Pre_S_s(find(Pre_S_s ==-1))=0;

Pre_R_all(find(Pre_R_all ==-1))=0;
Pre_R_h(find(Pre_R_h ==-1))=0;
Pre_R_f(find(Pre_R_f ==-1))=0;
Pre_R_s(find(Pre_R_s ==-1))=0;



save_path = ['H:\基于麻雀算法的特征选择用于EEG抑郁症识别\Results\','testResults.mat'];
    save(save_path, 'Y_all','Y_H','Y_F','Y_S',...
                    'Pre_all', 'Pre_h','Pre_f','Pre_s',...
                    'Pre_S_all','Pre_S_h','Pre_S_f','Pre_S_s',...
                    'Pre_R_all','Pre_R_h','Pre_R_f','Pre_R_s');