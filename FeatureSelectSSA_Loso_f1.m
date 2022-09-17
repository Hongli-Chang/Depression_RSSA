%% 基于麻雀搜索算法同步优化特征选择。

%% 清空环境变量
close all;
clear;
clc;
format compact;
%% 数据提取
folderPath='I:\基于麻雀算法的特征选择用于EEG抑郁症识别\128_ERP_fea\';
dataName=dir([folderPath '*.mat']);
Acc=[];
for i=1:53
    i
    acc=[]; B=[]; curve=[];TEST_Y=[];PRE_TEST_Y=[];
        trainNum=[1:53];
        dataName(i).name
        load([folderPath dataName(i).name]);
        Label(1,find(Label(1,:)==0))=-1;
        
        test_Y=Label(1,:);
        test_X=reshape(Fea,[5*128  size(Fea,3)]);
        
        hcue=find(Label(2,:)==0);
        testH_Y=Label(1,hcue);
        testH_X=test_X(:,hcue);
        
        fcue=find(Label(2,:)==1);
        testF_Y=Label(1,fcue);
        testF_X=test_X(:,fcue);
        
        scue=find(Label(2,:)==2);
        testS_Y=Label(1,scue);
        testS_X=test_X(:,scue);
        
        trainNum(i)=[];
        Train_X=[];Train_Y=[];
        for j=1:52
            j
            load([folderPath dataName(trainNum(j)).name]);
            Label(1,find(Label(1,:)==0))=-1;
            Train_X = cat(3,Train_X,Fea);
            Train_Y=cat(2,Train_Y,Label);
        end
        
       train_Y=Train_Y(1,:); 
       train_X=reshape(Train_X,[5*128  size(Train_X,3)]);
   
        hcue=find(Train_Y(2,:)==0);
        trainH_Y=Train_Y(1,hcue);
        trainH_X=train_X(:,hcue);
        
        fcue=find(Train_Y(2,:)==1);
        trainF_Y=Train_Y(1,fcue);
        trainF_X=train_X(:,fcue);
        
        scue=find(Train_Y(2,:)==2);
        trainS_Y=Train_Y(1,scue);
        trainS_X=train_X(:,scue);
    %%  麻雀参数设置
    % 定义优化参数的个数，在该场景中，优化参数的个数为数据集特征总数 。
    % 优化参数的个数 特征维度
    dim = size(train_X,1);  %信道维度
    % 优化参数的取值下限，[0,1],大于0.5为选择该特征，小于0.5为不选择该特征
    lb = 0;
    ub = 1;
    pop =100; %麻雀数量
    Max_iteration=50;%最大迭代次数   
    %目标函数
    %% 优化
    fobj_all = @(x) fun_BLDA_f1(x,train_Y,train_X,test_Y,test_X);    
   
    [Best_pos_all,~,curve_all]=SSA(pop,Max_iteration,lb,ub,dim,fobj_all);   
    B_all = Best_pos_all>0.5; 
    train_New = train_X(B_all,:);
    test_New = test_X(B_all,:);
    b=bayeslda(1);
    b=train(b,train_New,train_Y);
    S_predict_labelTrain_all=classify(b,train_New);S_predict_labelTrain_all(S_predict_labelTrain_all>0)=1;S_predict_labelTrain_all(S_predict_labelTrain_all<0)=-1;
    S_accuracyTrain_all=sum(S_predict_labelTrain_all==train_Y)/length(train_Y);
    S_predict_labelTest_all=classify(b,test_New);S_predict_labelTest_all(S_predict_labelTest_all>0)=1;S_predict_labelTest_all(S_predict_labelTest_all<0)=-1;
    S_accuracyTest_all=sum(S_predict_labelTest_all==test_Y)/length(test_Y);
    
    fobj_H = @(x) fun_BLDA_f1(x,trainH_Y,trainH_X,testH_Y,testH_X);    
   
    [Best_pos_H,~,curve_H]=SSA(pop,Max_iteration,lb,ub,dim,fobj_H);   
    B_H = Best_pos_H>0.5; 
    train_New = trainH_X(B_H,:);
    test_New = testH_X(B_H,:);
    b=bayeslda(1);
    b=train(b,train_New,trainH_Y);
    S_predict_labelTrain_H=classify(b,train_New);S_predict_labelTrain_H(S_predict_labelTrain_H>0)=1;S_predict_labelTrain_H(S_predict_labelTrain_H<0)=-1;
    S_accuracyTrain_H=sum(S_predict_labelTrain_H==trainH_Y)/length(trainH_Y);
    S_predict_labelTest_H=classify(b,test_New);S_predict_labelTest_H(S_predict_labelTest_H>0)=1;S_predict_labelTest_H(S_predict_labelTest_H<0)=-1;
    S_accuracyTest_H=sum(S_predict_labelTest_H==testH_Y)/length(testH_Y);
    
    fobj_F = @(x) fun_BLDA_f1(x,trainF_Y,trainF_X,testF_Y,testF_X);    
   
    [Best_pos_F,~,curve_F]=SSA(pop,Max_iteration,lb,ub,dim,fobj_F);   
    B_F = Best_pos_F>0.5; 
    train_New = trainF_X(B_F,:);
    test_New = testF_X(B_F,:);
    b=bayeslda(1);
    b=train(b,train_New,trainF_Y);
    S_predict_labelTrain_F=classify(b,train_New);S_predict_labelTrain_F(S_predict_labelTrain_F>0)=1;S_predict_labelTrain_F(S_predict_labelTrain_F<0)=-1;
    S_accuracyTrain_F=sum(S_predict_labelTrain_F==trainF_Y)/length(trainF_Y);
    S_predict_labelTest_F=classify(b,test_New);S_predict_labelTest_F(S_predict_labelTest_F>0)=1;S_predict_labelTest_F(S_predict_labelTest_F<0)=-1;
    S_accuracyTest_F=sum(S_predict_labelTest_F==testF_Y)/length(testF_Y);
    
    fobj_S = @(x) fun_BLDA_f1(x,trainS_Y,trainS_X,testS_Y,testS_X);    
   
    [Best_pos_S,~,curve_S]=SSA(pop,Max_iteration,lb,ub,dim,fobj_S);   
    B_S = Best_pos_S>0.5; 
    train_New = trainS_X(B_S,:);
    test_New = testS_X(B_S,:);
    b=bayeslda(1);
    b=train(b,train_New,trainS_Y);
    S_predict_labelTrain_S=classify(b,train_New);S_predict_labelTrain_S(S_predict_labelTrain_S>0)=1;S_predict_labelTrain_S(S_predict_labelTrain_S<0)=-1;
    S_accuracyTrain_S=sum(S_predict_labelTrain_S==trainS_Y)/length(trainS_Y);
    S_predict_labelTest_S=classify(b,test_New);S_predict_labelTest_S(S_predict_labelTest_S>0)=1;S_predict_labelTest_S(S_predict_labelTest_S<0)=-1;
    S_accuracyTest_S=sum(S_predict_labelTest_S==testS_Y)/length(testS_Y);
    %% 随机游走麻雀
    fobj_all = @(x) fun_BLDA_f1(x,train_Y,train_X,test_Y,test_X);    
   
    [R_Best_pos_all,~,R_curve_all]=RandomWalkSSA(pop,Max_iteration,lb,ub,dim,fobj_all);   
    R_B_all = R_Best_pos_all>0.5; 
    train_New = train_X(R_B_all,:);
    test_New = test_X(R_B_all,:);
    b=bayeslda(1);
    b=train(b,train_New,train_Y);
    R_S_predict_labelTrain_all=classify(b,train_New);R_S_predict_labelTrain_all(R_S_predict_labelTrain_all>0)=1;R_S_predict_labelTrain_all(R_S_predict_labelTrain_all<0)=-1;
    R_S_accuracyTrain_all=sum(R_S_predict_labelTrain_all==train_Y)/length(train_Y);
    R_S_predict_labelTest_all=classify(b,test_New);R_S_predict_labelTest_all(R_S_predict_labelTest_all>0)=1;R_S_predict_labelTest_all(R_S_predict_labelTest_all<0)=-1;
    R_S_accuracyTest_all=sum(R_S_predict_labelTest_all==test_Y)/length(test_Y);
    
    fobj_H = @(x) fun_BLDA_f1(x,trainH_Y,trainH_X,testH_Y,testH_X);    
   
    [R_Best_pos_H,~,R_curve_H]=RandomWalkSSA(pop,Max_iteration,lb,ub,dim,fobj_H);   
    R_B_H = R_Best_pos_H>0.5; 
    train_New = trainH_X(R_B_H,:);
    test_New = testH_X(R_B_H,:);
    b=bayeslda(1);
    b=train(b,train_New,trainH_Y);
    R_S_predict_labelTrain_H=classify(b,train_New);R_S_predict_labelTrain_H(R_S_predict_labelTrain_H>0)=1;R_S_predict_labelTrain_H(R_S_predict_labelTrain_H<0)=-1;
    R_S_accuracyTrain_H=sum(R_S_predict_labelTrain_H==trainH_Y)/length(trainH_Y);
    R_S_predict_labelTest_H=classify(b,test_New);R_S_predict_labelTest_H(R_S_predict_labelTest_H>0)=1;R_S_predict_labelTest_H(R_S_predict_labelTest_H<0)=-1;
    R_S_accuracyTest_H=sum(R_S_predict_labelTest_H==testH_Y)/length(testH_Y);
    
    fobj_F = @(x) fun_BLDA_f1(x,trainF_Y,trainF_X,testF_Y,testF_X);    
   
    [R_Best_pos_F,~,R_curve_F]=RandomWalkSSA(pop,Max_iteration,lb,ub,dim,fobj_F);   
    R_B_F = R_Best_pos_F>0.5; 
    train_New = trainF_X(R_B_F,:);
    test_New = testF_X(R_B_F,:);
    b=bayeslda(1);
    b=train(b,train_New,trainF_Y);
    R_S_predict_labelTrain_F=classify(b,train_New);R_S_predict_labelTrain_F(R_S_predict_labelTrain_F>0)=1;R_S_predict_labelTrain_F(R_S_predict_labelTrain_F<0)=-1;
    R_S_accuracyTrain_F=sum(R_S_predict_labelTrain_H==trainF_Y)/length(trainF_Y);
    R_S_predict_labelTest_F=classify(b,test_New);R_S_predict_labelTest_F(R_S_predict_labelTest_F>0)=1;R_S_predict_labelTest_F(R_S_predict_labelTest_F<0)=-1;
    R_S_accuracyTest_F=sum(R_S_predict_labelTest_F==testF_Y)/length(testF_Y);
    
    fobj_S = @(x) fun_BLDA_f1(x,trainS_Y,trainS_X,testS_Y,testS_X);    
   
    [R_Best_pos_S,Best_score,R_curve_S]=RandomWalkSSA(pop,Max_iteration,lb,ub,dim,fobj_S);   
    R_B_S = R_Best_pos_S>0.5; 
    train_New = trainS_X(R_B_S,:);
    test_New = testS_X(R_B_S,:);
    b=bayeslda(1);
    b=train(b,train_New,trainS_Y);
    R_S_predict_labelTrain_S=classify(b,train_New);R_S_predict_labelTrain_S(R_S_predict_labelTrain_S>0)=1;R_S_predict_labelTrain_S(R_S_predict_labelTrain_S<0)=-1;
    R_S_accuracyTrain_S=sum(R_S_predict_labelTrain_S==trainS_Y)/length(trainS_Y);
    R_S_predict_labelTest_S=classify(b,test_New);R_S_predict_labelTest_S(R_S_predict_labelTest_S>0)=1;R_S_predict_labelTest_S(R_S_predict_labelTest_S<0)=-1;
    R_S_accuracyTest_S=sum(R_S_predict_labelTest_S==testS_Y)/length(testS_Y);
    %%
    acc=[S_accuracyTrain_all S_accuracyTest_all S_accuracyTrain_H S_accuracyTest_H S_accuracyTrain_F S_accuracyTest_F S_accuracyTrain_S S_accuracyTest_S ...
        R_S_accuracyTrain_all R_S_accuracyTest_all R_S_accuracyTrain_H R_S_accuracyTest_H R_S_accuracyTrain_F R_S_accuracyTest_F R_S_accuracyTrain_S R_S_accuracyTest_S]; 
    B=[B_all; B_H; B_F; B_S; R_B_all; R_B_H ;R_B_F; R_B_S]; 
    curve=[curve_all; curve_H ;curve_F; curve_S ;R_curve_all; R_curve_H ;R_curve_F; R_curve_S];
    TEST_Y.test_Y = test_Y; TEST_Y.testH_Y = testH_Y; TEST_Y.testF_Y = testF_Y; TEST_Y.testS_Y = testS_Y;
   % PRE_TEST_Y.predict_labelTest_all=predict_labelTest_all;PRE_TEST_Y.predict_labelTest_h=predict_labelTest_h;PRE_TEST_Y.predict_labelTest_f=predict_labelTest_f;PRE_TEST_Y.predict_labelTest_s=predict_labelTest_s;
    PRE_TEST_Y.S_predict_labelTest_all=S_predict_labelTest_all;PRE_TEST_Y.S_predict_labelTest_H=S_predict_labelTest_H;PRE_TEST_Y.S_predict_labelTest_F=S_predict_labelTest_F;PRE_TEST_Y.S_predict_labelTest_S=S_predict_labelTest_S;
    PRE_TEST_Y.R_S_predict_labelTest_all=R_S_predict_labelTest_all;PRE_TEST_Y.R_S_predict_labelTest_H=R_S_predict_labelTest_H;PRE_TEST_Y.R_S_predict_labelTest_F=R_S_predict_labelTest_F;PRE_TEST_Y.R_S_predict_labelTest_S=R_S_predict_labelTest_S;
    %PRE_TEST_Y=[  predict_labelTest_f predict_labelTest_s S_predict_labelTest_all S_predict_labelTest_H S_predict_labelTest_F S_predict_labelTest_S R_S_predict_labelTest_all R_S_predict_labelTest_H R_S_predict_labelTest_F R_S_predict_labelTest_S];
   savepath=['I:\基于麻雀算法的特征选择用于EEG抑郁症识别\Results_f1\',num2str(i),'.mat'];
   save(savepath,'acc','B','curve','TEST_Y','PRE_TEST_Y');
   Acc=cat(1,Acc,acc);
   BestPos(i).B=B;
   Curve(i).curve=curve;
   TestLabel(i).TEST_Y =TEST_Y;
   Pre_TestLabel(i).PRE_TEST_Y =PRE_TEST_Y;
end
save(['I:\基于麻雀算法的特征选择用于EEG抑郁症识别\Results_f1\','Sum.mat'],'Acc','BestPos','TestLabel','Pre_TestLabel','Curve');