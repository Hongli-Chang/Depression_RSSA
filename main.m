%% 基于麻雀搜索算法同步优化特征选择。
%% by Jack旭 ： https://mianbaoduo.com/o/bread/mbd-YZaTlppv
% 意大利葡萄酒种类识别
%% 清空环境变量
close all;
clear;
clc;
format compact;
%% 数据提取

% 载入测试数据wine,其中包含的数据为classnumber = 3,wine:178*13的矩阵,wine_labes:178*1的列向量
  load chapter_wineClass.mat;
% load wine.mat
% load wine_test.mat
%画出测试数据的box可视化图
figure;
boxplot(wine,'orientation','horizontal','labels',categories);
title('wine数据的box可视化图','FontSize',12);
xlabel('属性值','FontSize',12);
grid on;

% 画出测试数据的分维可视化图
figure
subplot(3,5,1);
hold on
for run = 1:178
    plot(run,wine_labels(run),'*');
end
xlabel('样本','FontSize',10);
ylabel('类别标签','FontSize',10);
title('class','FontSize',10);
for run = 2:14
    subplot(3,5,run);
    hold on;
    str = ['attrib ',num2str(run-1)];
    for i = 1:178
        plot(i,wine(i,run-1),'*');
    end
    xlabel('样本','FontSize',10);
    ylabel('属性值','FontSize',10);
    title(str,'FontSize',10);
end

% 选定训练集和测试集

% 将第一类的1-30,第二类的60-95,第三类的131-153做为训练集
% train_wine = [wine(1:30,:);wine(60:95,:);wine(131:153,:)];
 train_wine = [wine(1:30,:);wine(60:95,:);wine(131:153,:)];
% 相应的训练集的标签也要分离出来
% train_wine_labels = [wine_labels(1:30);wine_labels(60:95);wine_labels(131:153)];
train_wine_labels = [wine_labels(1:30);wine_labels(60:95);wine_labels(131:153)];
% 将第一类的31-59,第二类的96-130,第三类的154-178做为测试集
% test_wine = [wine(31:59,:);wine(96:130,:);wine(154:178,:)];
test_wine = [wine(31:59,:);wine(96:130,:);wine(154:178,:)];
% 相应的测试集的标签也要分离出来
% test_wine_labels = [wine_labels(31:59);wine_labels(96:130);wine_labels(154:178)];
test_wine_labels = [wine_labels(31:59);wine_labels(96:130);wine_labels(154:178)];

%% 数据预处理
% 数据预处理,将训练集和测试集归一化到[0,1]区间

[mtrain,ntrain] = size(train_wine);
[mtest,ntest] = size(test_wine);

dataset = [train_wine;test_wine];
% mapminmax为MATLAB自带的归一化函数
[dataset_scale,ps] = mapminmax(dataset',0,1);
dataset_scale = dataset_scale';

train_wine = dataset_scale(1:mtrain,:);
test_wine = dataset_scale( (mtrain+1):(mtrain+mtest),: );
%% 智能算法优化SVM训练，若使用网络训练将这部分与下面优化部分删除即可。
tic
%%  麻雀参数设置
% 定义优化参数的个数，在该场景中，优化参数的个数为数据集特征总数 。
%目标函数
fobj = @(x) fun(x,train_wine_labels,train_wine,test_wine_labels,test_wine); 
% 优化参数的个数 特征维度
dim = size(train_wine,2); %特征维度
% 优化参数的取值下限，[0,1],大于0.5为选择该特征，小于0.5为不选择该特征
lb = 0;
ub = 1;

%%  参数设置
pop =10; %麻雀数量
Max_iteration=50;%最大迭代次数             
%% 优化(这里主要调用函数)
[Best_pos,Best_score,curve]=SSA(pop,Max_iteration,lb,ub,dim,fobj); 
figure
plot(curve,'linewidth',1.5);
xlabel('迭代次数');
ylabel('适应度值');
title('麻雀收敛曲线');
grid on;

c = 2;  
g = 2; 
toc
% 用优化得到的特征进行训练和测试
cmd = ['-s 0 -t 2 ', '-c ', num2str(c), ' -g ', num2str(g), ' -q'];
B = Best_pos>0.5; %大于0.5为1，小于0.5为0
train_wineNew = train_wine(:,B);
model = libsvmtrain(train_wine_labels, train_wineNew, cmd);
test_wineNew = test_wine(:,B);
%% SVM网络预测
[predict_labelTrain, accuracyTrain,~] = libsvmpredict(train_wine_labels, train_wineNew, model);
[predict_labelTest, accuracyTest,~] = libsvmpredict(test_wine_labels, test_wineNew, model);
%% 结果分析
figure
hold on;
plot(train_wine_labels,'o');
plot(predict_labelTrain,'r*');
xlabel('训练集样本','FontSize',12);
ylabel('类别标签','FontSize',12);
legend('实际训练集分类','预测训练集分类');
title('SSA特征选择后训练集的实际分类和预测分类图','FontSize',12);
grid on;

figure;
hold on;
plot(test_wine_labels,'o');
plot(predict_labelTest,'r*');
xlabel('测试集样本','FontSize',12);
ylabel('类别标签','FontSize',12);
legend('实际测试集分类','预测测试集分类');
title('SSA特征选择后测试集的实际分类和预测分类图','FontSize',12);
grid on;
hold off;


%% 基础SVM预测结果
% 用优化得到的特征进行训练和测试
cmd = ['-s 0 -t 2 ', '-c ', num2str(c), ' -g ', num2str(g), ' -q'];
model = libsvmtrain(train_wine_labels, train_wine, cmd);
%% SVM网络预测
[predict_labelTrain1, accuracyTrain1,~] = libsvmpredict(train_wine_labels, train_wine, model);
[predict_labelTest1, accuracyTest1,~] = libsvmpredict(test_wine_labels, test_wine, model);%% 结果分析
%% 结果分析
figure
hold on;
plot(train_wine_labels,'o');
plot(predict_labelTrain,'r*');
xlabel('训练集样本','FontSize',12);
ylabel('类别标签','FontSize',12);
legend('实际训练集分类','预测训练集分类');
title('基础SVM训练集的实际分类和预测分类图','FontSize',12);
grid on;

figure;
hold on;
plot(test_wine_labels,'o');
plot(predict_labelTest,'r*');
xlabel('测试集样本','FontSize',12);
ylabel('类别标签','FontSize',12);
legend('实际测试集分类','预测测试集分类');
title('基础SVM测试集的实际分类和预测分类图','FontSize',12);
grid on;
hold off;


disp(['基础SVM训练集最终预测准确率：',num2str(accuracyTrain1(1))])
disp(['基础SVM测试集最终预测准确率：',num2str(accuracyTest1(1))])

disp(['SSA特征选择后SVM训练集最终预测准确率：',num2str(accuracyTrain(1))])
disp(['SSA特征选择后SVM测试集最终预测准确率：',num2str(accuracyTest(1))])
disp(['总特征数：',num2str(size(train_wine,2))])
disp(['麻雀算法选择的特征总数：',num2str(size(train_wineNew,2))])
disp(['麻雀算法选择的特征(0为不选择，1为选择):',num2str(B)]);

