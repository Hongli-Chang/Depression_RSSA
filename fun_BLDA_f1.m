function  fitness1 = fun_BLDA_f1(x,train_labels,train_X,test_labels,test_X)
    featureNum = size(test_X,1);
    B = x>0.5; %大于0.5为1，小于0.5为0
    %保证至少有一个特征被选中
    if sum(B) == 0
       B(randi(featureNum))  = 1;
    end
    %如果为1，则选择特征用于训练
    train_wineNew = train_X(B,:);
    test_wineNew = test_X(B,:);
    b=bayeslda(1);
    b=train(b,train_wineNew,train_labels);
    
    predict_labelTrain=classify(b,train_wineNew);
    predict_labelTrain(predict_labelTrain>0)=1;
    predict_labelTrain(predict_labelTrain<0)=0;
    train_labels(train_labels<0)=0;
    [score1,recall1,precision1]=f1_score(train_labels, predict_labelTrain);
    %accuracy=sum(predict_labelTrain==train_labels)/length(train_labels);
    
    predict_labelTest=classify(b,test_wineNew);
    predict_labelTest(predict_labelTest>0)=1;
    predict_labelTest(predict_labelTest<0)=0;
    test_labels(test_labels<0)=0;
    [score2,recall2,precision2]=f1_score(test_labels, predict_labelTest);
    %accuracy1=sum(predict_labelTest==test_labels)/length(test_labels);
    fitness1 = score1(1) + score2(1) + 1 - sum(B)/featureNum; fitness1 = -fitness1;
    %fitness2 = recall1(1) + recall2(1) + 1 - sum(B)/featureNum; fitness2 = -fitness2;
    %fitness3 = precision1(1) + precision2(1) + 1 - sum(B)/featureNum; fitness3 = -fitness3;
    %加负号转换为求最小值
end