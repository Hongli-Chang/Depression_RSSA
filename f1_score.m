function [score,recall,precision]=f1_score(label, predict) 
    %M=zeros(2,2);
   % M(1,1)=size(find(find(label==0)==find(predict==0)),2);
    M=confusionmat(label,predict,'order',[0 1]);%以下两行为二分类时用
    %M=TP FN;
   %   FP TN;
   
    recall=M(1,1)/(M(1,1) +M(1,2)); %SE: TP/(TP+FN) 
    precision=M(1,1)/(M(1,1) +M(2,1));%:TP/(TP+FP) 
    
    score=2*precision*recall/(precision+recall); 
end