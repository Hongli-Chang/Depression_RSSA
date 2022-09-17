%% 基于麻雀搜索算法同步优化特征选择。
function  fitness = fun(x,train_wine_labels,train_wine,test_wine_labels,test_wine)
c = 2;  
g = 2; 
featureNum = size(test_wine,2);
B = x>0.5; %大于0.5为1，小于0.5为0
%保证至少有一个特征被选中
if sum(B) == 0
   B(randi(featureNum))  = 1;
end
%如果为1，则选择特征用于训练
train_wineNew = train_wine(:,B);
cmd = [' -c ',num2str(2),' -g ',num2str(2)];
model=libsvmtrain(train_wine_labels,train_wineNew,cmd); % SVM模型训练
[~, accuracy,~]=libsvmpredict(train_wine_labels,train_wineNew,model); % SVM模型预测及其精度

test_wineNew = test_wine(:,B);
[~, accuracy1,~]=libsvmpredict(test_wine_labels,test_wineNew,model); % SVM模型预测及其精度


fitness = accuracy(1) + accuracy1(1) + 1 - sum(B)/featureNum;
fitness = -fitness;%加负号转换为求最小值
end