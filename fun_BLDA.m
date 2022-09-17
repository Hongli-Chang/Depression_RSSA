function  fitness = fun_BLDA(x,train_labels,train_X,test_labels,test_X)

    featureNum = size(test_X,1);
    B = x>0.5; %����0.5Ϊ1��С��0.5Ϊ0
    %��֤������һ��������ѡ��
    if sum(B) == 0
       B(randi(featureNum))  = 1;
    end
    %���Ϊ1����ѡ����������ѵ��
    train_wineNew = train_X(B,:);
    test_wineNew = test_X(B,:);
    b=bayeslda(1);
    b=train(b,train_wineNew,train_labels);
    predict_labelTrain=classify(b,train_wineNew);predict_labelTrain(predict_labelTrain>0)=1;predict_labelTrain(predict_labelTrain<0)=-1;
    accuracy=sum(predict_labelTrain==train_labels)/length(train_labels);
    predict_labelTest=classify(b,test_wineNew);predict_labelTest(predict_labelTest>0)=1;predict_labelTest(predict_labelTest<0)=-1;
    accuracy1=sum(predict_labelTest==test_labels)/length(test_labels);

    fitness = accuracy(1) + accuracy1(1) + 1 - sum(B)/featureNum;
    fitness = -fitness;%�Ӹ���ת��Ϊ����Сֵ
end