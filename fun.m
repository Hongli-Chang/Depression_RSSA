%% ������ȸ�����㷨ͬ���Ż�����ѡ��
function  fitness = fun(x,train_wine_labels,train_wine,test_wine_labels,test_wine)
c = 2;  
g = 2; 
featureNum = size(test_wine,2);
B = x>0.5; %����0.5Ϊ1��С��0.5Ϊ0
%��֤������һ��������ѡ��
if sum(B) == 0
   B(randi(featureNum))  = 1;
end
%���Ϊ1����ѡ����������ѵ��
train_wineNew = train_wine(:,B);
cmd = [' -c ',num2str(2),' -g ',num2str(2)];
model=libsvmtrain(train_wine_labels,train_wineNew,cmd); % SVMģ��ѵ��
[~, accuracy,~]=libsvmpredict(train_wine_labels,train_wineNew,model); % SVMģ��Ԥ�⼰�侫��

test_wineNew = test_wine(:,B);
[~, accuracy1,~]=libsvmpredict(test_wine_labels,test_wineNew,model); % SVMģ��Ԥ�⼰�侫��


fitness = accuracy(1) + accuracy1(1) + 1 - sum(B)/featureNum;
fitness = -fitness;%�Ӹ���ת��Ϊ����Сֵ
end