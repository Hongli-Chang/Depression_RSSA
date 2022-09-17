%% ������ȸ�����㷨ͬ���Ż�����ѡ��
%% by Jack�� �� https://mianbaoduo.com/o/bread/mbd-YZaTlppv
% ��������Ѿ�����ʶ��
%% ��ջ�������
close all;
clear;
clc;
format compact;
%% ������ȡ

% �����������wine,���а���������Ϊclassnumber = 3,wine:178*13�ľ���,wine_labes:178*1��������
  load chapter_wineClass.mat;
% load wine.mat
% load wine_test.mat
%�����������ݵ�box���ӻ�ͼ
figure;
boxplot(wine,'orientation','horizontal','labels',categories);
title('wine���ݵ�box���ӻ�ͼ','FontSize',12);
xlabel('����ֵ','FontSize',12);
grid on;

% �����������ݵķ�ά���ӻ�ͼ
figure
subplot(3,5,1);
hold on
for run = 1:178
    plot(run,wine_labels(run),'*');
end
xlabel('����','FontSize',10);
ylabel('����ǩ','FontSize',10);
title('class','FontSize',10);
for run = 2:14
    subplot(3,5,run);
    hold on;
    str = ['attrib ',num2str(run-1)];
    for i = 1:178
        plot(i,wine(i,run-1),'*');
    end
    xlabel('����','FontSize',10);
    ylabel('����ֵ','FontSize',10);
    title(str,'FontSize',10);
end

% ѡ��ѵ�����Ͳ��Լ�

% ����һ���1-30,�ڶ����60-95,�������131-153��Ϊѵ����
% train_wine = [wine(1:30,:);wine(60:95,:);wine(131:153,:)];
 train_wine = [wine(1:30,:);wine(60:95,:);wine(131:153,:)];
% ��Ӧ��ѵ�����ı�ǩҲҪ�������
% train_wine_labels = [wine_labels(1:30);wine_labels(60:95);wine_labels(131:153)];
train_wine_labels = [wine_labels(1:30);wine_labels(60:95);wine_labels(131:153)];
% ����һ���31-59,�ڶ����96-130,�������154-178��Ϊ���Լ�
% test_wine = [wine(31:59,:);wine(96:130,:);wine(154:178,:)];
test_wine = [wine(31:59,:);wine(96:130,:);wine(154:178,:)];
% ��Ӧ�Ĳ��Լ��ı�ǩҲҪ�������
% test_wine_labels = [wine_labels(31:59);wine_labels(96:130);wine_labels(154:178)];
test_wine_labels = [wine_labels(31:59);wine_labels(96:130);wine_labels(154:178)];

%% ����Ԥ����
% ����Ԥ����,��ѵ�����Ͳ��Լ���һ����[0,1]����

[mtrain,ntrain] = size(train_wine);
[mtest,ntest] = size(test_wine);

dataset = [train_wine;test_wine];
% mapminmaxΪMATLAB�Դ��Ĺ�һ������
[dataset_scale,ps] = mapminmax(dataset',0,1);
dataset_scale = dataset_scale';

train_wine = dataset_scale(1:mtrain,:);
test_wine = dataset_scale( (mtrain+1):(mtrain+mtest),: );
%% �����㷨�Ż�SVMѵ������ʹ������ѵ�����ⲿ���������Ż�����ɾ�����ɡ�
tic
%%  ��ȸ��������
% �����Ż������ĸ������ڸó����У��Ż������ĸ���Ϊ���ݼ��������� ��
%Ŀ�꺯��
fobj = @(x) fun(x,train_wine_labels,train_wine,test_wine_labels,test_wine); 
% �Ż������ĸ��� ����ά��
dim = size(train_wine,2); %����ά��
% �Ż�������ȡֵ���ޣ�[0,1],����0.5Ϊѡ���������С��0.5Ϊ��ѡ�������
lb = 0;
ub = 1;

%%  ��������
pop =10; %��ȸ����
Max_iteration=50;%����������             
%% �Ż�(������Ҫ���ú���)
[Best_pos,Best_score,curve]=SSA(pop,Max_iteration,lb,ub,dim,fobj); 
figure
plot(curve,'linewidth',1.5);
xlabel('��������');
ylabel('��Ӧ��ֵ');
title('��ȸ��������');
grid on;

c = 2;  
g = 2; 
toc
% ���Ż��õ�����������ѵ���Ͳ���
cmd = ['-s 0 -t 2 ', '-c ', num2str(c), ' -g ', num2str(g), ' -q'];
B = Best_pos>0.5; %����0.5Ϊ1��С��0.5Ϊ0
train_wineNew = train_wine(:,B);
model = libsvmtrain(train_wine_labels, train_wineNew, cmd);
test_wineNew = test_wine(:,B);
%% SVM����Ԥ��
[predict_labelTrain, accuracyTrain,~] = libsvmpredict(train_wine_labels, train_wineNew, model);
[predict_labelTest, accuracyTest,~] = libsvmpredict(test_wine_labels, test_wineNew, model);
%% �������
figure
hold on;
plot(train_wine_labels,'o');
plot(predict_labelTrain,'r*');
xlabel('ѵ��������','FontSize',12);
ylabel('����ǩ','FontSize',12);
legend('ʵ��ѵ��������','Ԥ��ѵ��������');
title('SSA����ѡ���ѵ������ʵ�ʷ����Ԥ�����ͼ','FontSize',12);
grid on;

figure;
hold on;
plot(test_wine_labels,'o');
plot(predict_labelTest,'r*');
xlabel('���Լ�����','FontSize',12);
ylabel('����ǩ','FontSize',12);
legend('ʵ�ʲ��Լ�����','Ԥ����Լ�����');
title('SSA����ѡ�����Լ���ʵ�ʷ����Ԥ�����ͼ','FontSize',12);
grid on;
hold off;


%% ����SVMԤ����
% ���Ż��õ�����������ѵ���Ͳ���
cmd = ['-s 0 -t 2 ', '-c ', num2str(c), ' -g ', num2str(g), ' -q'];
model = libsvmtrain(train_wine_labels, train_wine, cmd);
%% SVM����Ԥ��
[predict_labelTrain1, accuracyTrain1,~] = libsvmpredict(train_wine_labels, train_wine, model);
[predict_labelTest1, accuracyTest1,~] = libsvmpredict(test_wine_labels, test_wine, model);%% �������
%% �������
figure
hold on;
plot(train_wine_labels,'o');
plot(predict_labelTrain,'r*');
xlabel('ѵ��������','FontSize',12);
ylabel('����ǩ','FontSize',12);
legend('ʵ��ѵ��������','Ԥ��ѵ��������');
title('����SVMѵ������ʵ�ʷ����Ԥ�����ͼ','FontSize',12);
grid on;

figure;
hold on;
plot(test_wine_labels,'o');
plot(predict_labelTest,'r*');
xlabel('���Լ�����','FontSize',12);
ylabel('����ǩ','FontSize',12);
legend('ʵ�ʲ��Լ�����','Ԥ����Լ�����');
title('����SVM���Լ���ʵ�ʷ����Ԥ�����ͼ','FontSize',12);
grid on;
hold off;


disp(['����SVMѵ��������Ԥ��׼ȷ�ʣ�',num2str(accuracyTrain1(1))])
disp(['����SVM���Լ�����Ԥ��׼ȷ�ʣ�',num2str(accuracyTest1(1))])

disp(['SSA����ѡ���SVMѵ��������Ԥ��׼ȷ�ʣ�',num2str(accuracyTrain(1))])
disp(['SSA����ѡ���SVM���Լ�����Ԥ��׼ȷ�ʣ�',num2str(accuracyTest(1))])
disp(['����������',num2str(size(train_wine,2))])
disp(['��ȸ�㷨ѡ�������������',num2str(size(train_wineNew,2))])
disp(['��ȸ�㷨ѡ�������(0Ϊ��ѡ��1Ϊѡ��):',num2str(B)]);

