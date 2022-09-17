%% ������ȸ�����㷨ͬ���Ż�����ѡ��-����֢ʶ��
%% by Jack�� �� https://mianbaoduo.com/o/bread/mbd-YZaTlppv
% ��������Ѿ�����ʶ��
%% ��ջ�������
close all;
clear;
clc;
format compact;
%% ������ȡ
addpath('/redhdd/changhongli/SSA_Feature_selected/');
% �����������wine,���а���������Ϊclassnumber = 3,wine:178*13�ľ���,wine_labes:178*1��������
  %load chapter_wineClass.mat;
  load('/redhdd/changhongli/SSA_Feature_selected/cross5/LSTM_1.mat');
 %load('LSTM_1.mat'); 
% load wine.mat
% load wine_test.mat
%�����������ݵ�box���ӻ�ͼ
% figure;
% boxplot(wine,'orientation','horizontal','labels',categories);
% title('wine���ݵ�box���ӻ�ͼ','FontSize',12);
% xlabel('����ֵ','FontSize',12);
% grid on;

% % �����������ݵķ�ά���ӻ�ͼ
% figure
% subplot(3,5,1);
% hold on
% for run = 1:178
%     plot(run,wine_labels(run),'*');
% end
% xlabel('����','FontSize',10);
% ylabel('����ǩ','FontSize',10);
% title('class','FontSize',10);
% for run = 2:14
%     subplot(3,5,run);
%     hold on;
%     str = ['attrib ',num2str(run-1)];
%     for i = 1:178
%         plot(i,wine(i,run-1),'*');
%     end
%     xlabel('����','FontSize',10);
%     ylabel('����ֵ','FontSize',10);
%     title(str,'FontSize',10);
% end

% ѡ��ѵ�����Ͳ��Լ�

% % ����һ���1-30,�ڶ����60-95,�������131-153��Ϊѵ����
% % train_wine = [wine(1:30,:);wine(60:95,:);wine(131:153,:)];
%  train_wine = [wine(1:30,:);wine(60:95,:);wine(131:153,:)];
% % ��Ӧ��ѵ�����ı�ǩҲҪ�������
% % train_wine_labels = [wine_labels(1:30);wine_labels(60:95);wine_labels(131:153)];
% train_wine_labels = [wine_labels(1:30);wine_labels(60:95);wine_labels(131:153)];
% % ����һ���31-59,�ڶ����96-130,�������154-178��Ϊ���Լ�
% % test_wine = [wine(31:59,:);wine(96:130,:);wine(154:178,:)];
% test_wine = [wine(31:59,:);wine(96:130,:);wine(154:178,:)];
% % ��Ӧ�Ĳ��Լ��ı�ǩҲҪ�������
% % test_wine_labels = [wine_labels(31:59);wine_labels(96:130);wine_labels(154:178)];
% test_wine_labels = [wine_labels(31:59);wine_labels(96:130);wine_labels(154:178)];

%% ����Ԥ����
% ����Ԥ����,��ѵ�����Ͳ��Լ���һ����[0,1]����
test_Y=test_y_a(1,:);
train_Y=train_y_a(1,:);
test_X=reshape(test_X,[5*128  size(test_X,3)]);
test_X=test_X;
train_X=reshape(train_X,[5*128  size(train_X,3)]);
train_X=train_X;
[mtrain,ntrain] = size(train_X);
[mtest,ntest] = size(test_X);

dataset = [train_X,test_X];
% mapminmaxΪMATLAB�Դ��Ĺ�һ������
[dataset_scale,ps] = mapminmax(dataset,0,1);

train_X = dataset_scale(:,1:ntrain);
test_X = dataset_scale(:, (ntrain+1):(ntrain+ntest));
%% �����㷨�Ż�SVMѵ������ʹ������ѵ�����ⲿ���������Ż�����ɾ�����ɡ�
tic
%%  ��ȸ��������
% �����Ż������ĸ������ڸó����У��Ż������ĸ���Ϊ���ݼ��������� ��
%Ŀ�꺯��
fobj = @(x) fun_BLDA(x,train_Y,train_X,test_Y,test_X); 
% �Ż������ĸ��� ����ά��
dim = size(train_X,1); %����ά��
% �Ż�������ȡֵ���ޣ�[0,1],����0.5Ϊѡ���������С��0.5Ϊ��ѡ�������
lb = 0;
ub = 1;

%%  ��������
pop =100; %��ȸ����
Max_iteration=50;%����������             
%% �Ż�(������Ҫ���ú���)
[Best_pos,Best_score,curve]=RandomWalkSSA(pop,Max_iteration,lb,ub,dim,fobj); 
h=figure;
plot(curve,'linewidth',1.5);
xlabel('��������');
ylabel('��Ӧ��ֵ');
title('��ȸ��������');
grid on;
saveas(h,'curve.fig')

B = Best_pos>0.5; %����0.5Ϊ1��С��0.5Ϊ0
train_wineNew = train_X(B,:);
test_wineNew = test_X(B,:);
%% SVM����Ԥ��
b=bayeslda(1);
b=train(b,train_wineNew,train_Y);
predict_labelTrain=classify(b,train_wineNew);predict_labelTrain(predict_labelTrain>0)=1;predict_labelTrain(predict_labelTrain<0)=-1;
accuracyTrain=sum(predict_labelTrain==train_Y)/length(train_Y);
predict_labelTest=classify(b,test_wineNew);predict_labelTest(predict_labelTest>0)=1;predict_labelTest(predict_labelTest<0)=-1;
accuracyTest=sum(predict_labelTest==test_Y)/length(test_Y);
%% �������
l=figure
hold on;
plot(train_Y,'o');
plot(predict_labelTrain,'r*');
xlabel('ѵ��������','FontSize',12);
ylabel('����ǩ','FontSize',12);
legend('ʵ��ѵ��������','Ԥ��ѵ��������');
title('SSA����ѡ���ѵ������ʵ�ʷ����Ԥ�����ͼ','FontSize',12);
grid on;
saveas(l,'ssatrain.fig')
m=figure;
hold on;
plot(test_Y,'o');
plot(predict_labelTest,'r*');
xlabel('���Լ�����','FontSize',12);
ylabel('����ǩ','FontSize',12);
legend('ʵ�ʲ��Լ�����','Ԥ����Լ�����');
title('SSA����ѡ�����Լ���ʵ�ʷ����Ԥ�����ͼ','FontSize',12);
grid on;
hold off;
saveas(m,'ssatest.fig')

b=bayeslda(1);
b=train(b,train_X,train_Y);
predict_labelTrain1=classify(b,train_X);predict_labelTrain1(predict_labelTrain1>0)=1;predict_labelTrain1(predict_labelTrain1<0)=-1;
accuracyTrain1=sum(predict_labelTrain1==train_Y)/length(train_Y);
predict_labelTest1=classify(b,test_X);predict_labelTest1(predict_labelTest1>0)=1;predict_labelTest1(predict_labelTest1<0)=-1;
accuracyTest1=sum(predict_labelTest1==test_Y)/length(test_Y);

%% �������
n=figure
hold on;
plot(train_Y,'o');
plot(predict_labelTrain,'r*');
xlabel('ѵ��������','FontSize',12);
ylabel('����ǩ','FontSize',12);
legend('ʵ��ѵ��������','Ԥ��ѵ��������');
title('����SVMѵ������ʵ�ʷ����Ԥ�����ͼ','FontSize',12);
grid on;
saveas(n,'rawsvmtrain.fig');
o=figure;
hold on;
plot(test_Y,'o');
plot(predict_labelTest,'r*');
xlabel('���Լ�����','FontSize',12);
ylabel('����ǩ','FontSize',12);
legend('ʵ�ʲ��Լ�����','Ԥ����Լ�����');
title('����SVM���Լ���ʵ�ʷ����Ԥ�����ͼ','FontSize',12);
grid on;
hold off;
saveas(o,'rawsvmtest.fig');

disp(['����SVMѵ��������Ԥ��׼ȷ�ʣ�',num2str(accuracyTrain1(1))])
disp(['����SVM���Լ�����Ԥ��׼ȷ�ʣ�',num2str(accuracyTest1(1))])

disp(['SSA����ѡ���SVMѵ��������Ԥ��׼ȷ�ʣ�',num2str(accuracyTrain(1))])
disp(['SSA����ѡ���SVM���Լ�����Ԥ��׼ȷ�ʣ�',num2str(accuracyTest(1))])
disp(['����������',num2str(size(train_X,1))])
disp(['��ȸ�㷨ѡ�������������',num2str(size(train_wineNew,1))])
disp(['��ȸ�㷨ѡ�������(0Ϊ��ѡ��1Ϊѡ��):',num2str(B)]);

