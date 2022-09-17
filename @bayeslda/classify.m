function varargout = classify(b, x)
% prediction procedure for bayeslda   bayeslda的预测程序
% INPUT:
%    b          - object of type bayeslda  bayeslda类型的对象
%    x          - m*n matrix containing n feature vectors of size m*1 
%                  m * n矩阵，其包含大小为m * 1的n个特征向量
%
% OUTPUT:
%    varargout  - if classify is called with one output argument an array
%                 containing the mean value of the predictive distribution 
%                 for each example in x is returned 
%        如果使用一个输出参数调用分类，则返回包含x中每个示例的预测分布的平均值的数组
%               - if classify is called with two output arguments the 
%                 mean value and the variance of the predictive
%                 distribution are returned
%             如果使用两个输出参数调用分类，则返回预测分布的平均值和方差
% Author: Ulrich Hoffmann - EPFL, 2006
% Copyright: Ulrich Hoffmann - EPFL
%
% The algorithm implemented here was originally described by 
% MacKay, D. J. C., 1992. Bayesian interpolation.
% Neural Computation 4 (3), pp. 415-447.


%% add feature that is constantly one (bias term)添加的功能，是一个不断（偏项）
x = [x; ones(1,size(x,2))];    


%% compute mean of predictive distributions计算预测分布的平均值
m = b.w'*x;


%% if one output argument return mean only如果一个输出参数只返回平均值
if nargout == 1
    varargout(1) = {m};    
end


%% if two output arguments compute and return variance also如果两个输出参数也计算并返回方差
% if nargout == 2
%     s = zeros(1,size(x,2));
%     for i = 1:size(x,2);
%         s(i) = x(:,i)'*b.p*x(:,i) + (1/b.beta);
%     end
%     varargout(1) = {m};
%     varargout(2) = {s};
% end
% 
% if nargout > 2
%     fprintf('Too many output arguments!\n');
end