function b = bayeslda(verbose)
% constructor for class bayeslda
% Bayesian Linear Discriminant Analysis
%
% METHODS:
%       train    - learns a linear discriminant from training examples
%       classify - classfies new examples
%    getevidence - returns the log evidence for the learned discriminant
%
% Author: Ulrich Hoffmann - EPFL, 2006
% Copyright: Ulrich Hoffmann - EPFL
%注释：
%Bayeslda类的构造函数
%贝叶斯线性判别分析

%? 方法：
%??????? train - 从训练实例中学习线性判别
%??????? classify - 分类新例子
%???? getevidence - 返回学习判别式的日志证据
%% set verbose flag 设置详细标志
if nargin == 1
    b.verbose = verbose;    % if set train gives verbose output如果训练给出详细输出
else
    b.verbose = 0;
end


%% define attributes of object定义对象的属性
b.evidence = 0;             % log evidence 日志证据
b.beta = 0;                 % inverse variance of noise噪声的逆方差 
b.alpha = 0;                % inverse variance of prior 先前的逆方差
b.w  = [];                  % weight vector (mean of posterior)权衡向量（后方的平均值）
b.p  = [];                  % precision matrix of posterior后验精度矩阵


%% initialize class
b = class(b,'bayeslda');