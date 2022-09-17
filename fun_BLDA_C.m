
function  fitness = fun_BLDA_C(x,train_labels,train_X,test_labels,test_X)

featureNum = 128;
B = x>0.5; %����0.5Ϊ1��С��0.5Ϊ0
%��֤������һ��������ѡ��
if sum(B) == 0
   B(randi(featureNum))  = 1;
end
%���Ϊ1����ѡ����������ѵ��
    train_X = reshape(train_X,[5,128,size(train_X,2)]);
    train_X=train_X(:,B,:);
    test_X = reshape(test_X,[5,128,size(test_X,2)]);
    test_X=test_X(:,B,:);
    train_New = reshape(train_X,[size(train_X,2)*5,size(train_X,3)]);
    test_New = reshape(test_X,[size(test_X,2)*5,size(test_X,3)]);

b=bayeslda(1);
b=train(b,train_New,train_labels);
predict_labelTrain=classify(b,train_New);predict_labelTrain(predict_labelTrain>0)=1;predict_labelTrain(predict_labelTrain<0)=-1;
accuracy=sum(predict_labelTrain==train_labels)/length(train_labels);
predict_labelTest=classify(b,test_New);predict_labelTest(predict_labelTest>0)=1;predict_labelTest(predict_labelTest<0)=-1;
accuracy1=sum(predict_labelTest==test_labels)/length(test_labels);


fitness = accuracy(1) + accuracy1(1) + 1 - sum(B)/featureNum;
fitness = -fitness;%�Ӹ���ת��Ϊ����Сֵ
end