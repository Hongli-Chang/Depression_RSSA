%% 基于麻雀搜索算法同步优化特征选择-抑郁症识别。
%% by Jack旭 ： https://mianbaoduo.com/o/bread/mbd-YZaTlppv
% 意大利葡萄酒种类识别
%% 清空环境变量
close all;
clear;
clc;
format compact;
%% 数据提取
addpath('/redhdd/changhongli/SSA_Feature_selected/');
% 载入测试数据wine,其中包含的数据为classnumber = 3,wine:178*13的矩阵,wine_labes:178*1的列向量
  %load chapter_wineClass.mat;
  load('/redhdd/changhongli/SSA_Feature_selected/cross5/LSTM_1.mat');
 %load('LSTM_1.mat'); 
% load wine.mat
% load wine_test.mat
%画出测试数据的box可视化图
% figure;
% boxplot(wine,'orientation','horizontal','labels',categories);
% title('wine数据的box可视化图','FontSize',12);
% xlabel('属性值','FontSize',12);
% grid on;

% % 画出测试数据的分维可视化图
% figure
% subplot(3,5,1);
% hold on
% for run = 1:178
%     plot(run,wine_labels(run),'*');
% end
% xlabel('样本','FontSize',10);
% ylabel('类别标签','FontSize',10);
% title('class','FontSize',10);
% for run = 2:14
%     subplot(3,5,run);
%     hold on;
%     str = ['attrib ',num2str(run-1)];
%     for i = 1:178
%         plot(i,wine(i,run-1),'*');
%     end
%     xlabel('样本','FontSize',10);
%     ylabel('属性值','FontSize',10);
%     title(str,'FontSize',10);
% end

% 选定训练集和测试集

% % 将第一类的1-30,第二类的60-95,第三类的131-153做为训练集
% % train_wine = [wine(1:30,:);wine(60:95,:);wine(131:153,:)];
%  train_wine = [wine(1:30,:);wine(60:95,:);wine(131:153,:)];
% % 相应的训练集的标签也要分离出来
% % train_wine_labels = [wine_labels(1:30);wine_labels(60:95);wine_labels(131:153)];
% train_wine_labels = [wine_labels(1:30);wine_labels(60:95);wine_labels(131:153)];
% % 将第一类的31-59,第二类的96-130,第三类的154-178做为测试集
% % test_wine = [wine(31:59,:);wine(96:130,:);wine(154:178,:)];
% test_wine = [wine(31:59,:);wine(96:130,:);wine(154:178,:)];
% % 相应的测试集的标签也要分离出来
% % test_wine_labels = [wine_labels(31:59);wine_labels(96:130);wine_labels(154:178)];
% test_wine_labels = [wine_labels(31:59);wine_labels(96:130);wine_labels(154:178)];

%% 数据预处理
% 数据预处理,将训练集和测试集归一化到[0,1]区间
test_Y=test_y_a(1,:);
train_Y=train_y_a(1,:);
test_X=reshape(test_X,[5*128  size(test_X,3)]);
test_X=test_X;
train_X=reshape(train_X,[5*128  size(train_X,3)]);
train_X=train_X;
[mtrain,ntrain] = size(train_X);
[mtest,ntest] = size(test_X);

dataset = [train_X,test_X];
% mapminmax为MATLAB自带的归一化函数
[dataset_scale,ps] = mapminmax(dataset,0,1);

train_X = dataset_scale(:,1:ntrain);
test_X = dataset_scale(:, (ntrain+1):(ntrain+ntest));
%% 智能算法优化SVM训练，若使用网络训练将这部分与下面优化部分删除即可。
tic
%%  麻雀参数设置
% 定义优化参数的个数，在该场景中，优化参数的个数为数据集特征总数 。
%目标函数
fobj = @(x) fun_BLDA(x,train_Y,train_X,test_Y,test_X); 
% 优化参数的个数 特征维度
dim = size(train_X,1); %特征维度
% 优化参数的取值下限，[0,1],大于0.5为选择该特征，小于0.5为不选择该特征
lb = 0;
ub = 1;

%%  参数设置
pop =100; %麻雀数量
Max_iteration=50;%最大迭代次数             
%% 优化(这里主要调用函数)
[Best_pos,Best_score,curve]=RandomWalkSSA(pop,Max_iteration,lb,ub,dim,fobj); 
h=figure;
plot(curve,'linewidth',1.5);
xlabel('迭代次数');
ylabel('适应度值');
title('麻雀收敛曲线');
grid on;
saveas(h,'curve.fig')

B = Best_pos>0.5; %大于0.5为1，小于0.5为0
train_wineNew = train_X(B,:);
test_wineNew = test_X(B,:);
%% SVM网络预测
b=bayeslda(1);
b=train(b,train_wineNew,train_Y);
predict_labelTrain=classify(b,train_wineNew);predict_labelTrain(predict_labelTrain>0)=1;predict_labelTrain(predict_labelTrain<0)=-1;
accuracyTrain=sum(predict_labelTrain==train_Y)/length(train_Y);
predict_labelTest=classify(b,test_wineNew);predict_labelTest(predict_labelTest>0)=1;predict_labelTest(predict_labelTest<0)=-1;
accuracyTest=sum(predict_labelTest==test_Y)/length(test_Y);
%% 结果分析
l=figure
hold on;
plot(train_Y,'o');
plot(predict_labelTrain,'r*');
xlabel('训练集样本','FontSize',12);
ylabel('类别标签','FontSize',12);
legend('实际训练集分类','预测训练集分类');
title('SSA特征选择后训练集的实际分类和预测分类图','FontSize',12);
grid on;
saveas(l,'ssatrain.fig')
m=figure;
hold on;
plot(test_Y,'o');
plot(predict_labelTest,'r*');
xlabel('测试集样本','FontSize',12);
ylabel('类别标签','FontSize',12);
legend('实际测试集分类','预测测试集分类');
title('SSA特征选择后测试集的实际分类和预测分类图','FontSize',12);
grid on;
hold off;
saveas(m,'ssatest.fig')

b=bayeslda(1);
b=train(b,train_X,train_Y);
predict_labelTrain1=classify(b,train_X);predict_labelTrain1(predict_labelTrain1>0)=1;predict_labelTrain1(predict_labelTrain1<0)=-1;
accuracyTrain1=sum(predict_labelTrain1==train_Y)/length(train_Y);
predict_labelTest1=classify(b,test_X);predict_labelTest1(predict_labelTest1>0)=1;predict_labelTest1(predict_labelTest1<0)=-1;
accuracyTest1=sum(predict_labelTest1==test_Y)/length(test_Y);

%% 结果分析
n=figure
hold on;
plot(train_Y,'o');
plot(predict_labelTrain,'r*');
xlabel('训练集样本','FontSize',12);
ylabel('类别标签','FontSize',12);
legend('实际训练集分类','预测训练集分类');
title('基础SVM训练集的实际分类和预测分类图','FontSize',12);
grid on;
saveas(n,'rawsvmtrain.fig');
o=figure;
hold on;
plot(test_Y,'o');
plot(predict_labelTest,'r*');
xlabel('测试集样本','FontSize',12);
ylabel('类别标签','FontSize',12);
legend('实际测试集分类','预测测试集分类');
title('基础SVM测试集的实际分类和预测分类图','FontSize',12);
grid on;
hold off;
saveas(o,'rawsvmtest.fig');

disp(['基础SVM训练集最终预测准确率：',num2str(accuracyTrain1(1))])
disp(['基础SVM测试集最终预测准确率：',num2str(accuracyTest1(1))])

disp(['SSA特征选择后SVM训练集最终预测准确率：',num2str(accuracyTrain(1))])
disp(['SSA特征选择后SVM测试集最终预测准确率：',num2str(accuracyTest(1))])
disp(['总特征数：',num2str(size(train_X,1))])
disp(['麻雀算法选择的特征总数：',num2str(size(train_wineNew,1))])
disp(['麻雀算法选择的特征(0为不选择，1为选择):',num2str(B)]);

