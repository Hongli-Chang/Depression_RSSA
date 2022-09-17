function b = train(b, x, y)
% training procedure for Bayesian LDA  贝叶斯LDA训练程序
% INPUT:
%    b       - object of type bayeslda  bayeslda类型的对象
%    x       - m*n matrix containing n feature vectors of size m*1 
%                m * n矩阵，其包含大小为m * 1的n个特征向量
%    y       - 1*n matrix containing class labels (-1,1) 
%                包含类标签的1 * n矩阵（-1,1）
% OUTPUT:
%    b       - updated object of type bayeslda 更新的对象为bayeslda
%
% Author: Ulrich Hoffmann - EPFL, 2006
% Copyright: Ulrich Hoffmann - EPFL
%
% The algorithm implemented here was originally described by
% MacKay, D. J. C., 1992. Bayesian interpolation.
% Neural Computation 4 (3), pp. 415-447.


%% compute regression targets from class labels (to do lda via regression)
%从类标签计算回归目标（通过回归做lda）
n_posexamples = sum(y==1);
n_negexamples = sum(y==-1);
n_examples    = n_posexamples + n_negexamples;
y(y==1) = n_examples/n_posexamples;
y(y==-1) = -n_examples/n_negexamples;


%% add feature that is constantly one (bias term)添加的功能，是一个不断（偏项）
x = [x; ones(1,size(x,2))];  


%% initialize variables for fast iterative estimation of alpha and beta
%初始化变量以快速迭代估计α和β
n_features = size(x,1);            % dimension of feature vectors 特征向量维数
d_beta = inf;                      % (initial) diff. between new and old beta  （初始）差异 新旧beta之间
d_alpha = inf;                     % (initial) diff. between new and old alpha （初始）差异 新旧阿尔法之间
alpha    = 25;                     % (initial) inverse variance of prior distribution（初始）先前分布的逆方差
%alpha    = 50; 
biasalpha = 0.00000001;            % (initial) inverse variance of prior for bias term（初始）偏差项之前的逆方差
 %biasalpha = 0.000001;  
 beta     = 1;                      % (initial) inverse variance around targets（初始）围绕目标的逆方差
%beta     = 100; 
stopeps  = 0.0001;                 % desired precision for alpha and beta所需的alpha和beta精度
%stopeps  = 0.01;  
i        = 1;                      % keeps track of number of iterations跟踪迭代次数
maxit    = 500;                    % maximal number of iterations 最大迭代次数
[v,d] = eig(x*x');                 % needed for fast estimation of alpha and beta 需要快速估计α和β 
vxy    = v'*x*y';                  % dito
d = diag(d);                       % dito
e = ones(n_features-1,1);          % dito


%% estimate alpha and beta iteratively迭代地估计α和β
while ((d_alpha > stopeps) || (d_beta > stopeps)) && (i < maxit);
    alphaold = alpha;
    betaold  = beta;
    m = beta*v*((beta*d+[alpha*e; biasalpha]).^(-1).*vxy);
    err = sum((y-m'*x).^2);
    gamma = sum(beta*d./(beta*d+[alpha*e; biasalpha]));
    alpha = gamma/(m'*m);
    beta  = (n_examples - gamma)/err;
    if b.verbose
        fprintf('Iteration %i: alpha = %f, beta = %f\n',i,alpha,beta);
    end
    d_alpha = abs(alpha-alphaold);
    d_beta  = abs(beta-betaold);
    i = i + 1;
end


%% process results of estimation 过程估算结果
if (i < maxit)
    
    % compute the log evidence计算日志证据
    % this can be used for simple model selection tasks这可以用于简单的模型选择任务
    % (see MacKays paper)（见MacKays论文）
    b.evidence = (n_features/2)*log(alpha) + (n_examples/2)*log(beta) - ...
    (beta/2)*err - (alpha/2)*m'*m - ...
    0.5*sum(log((beta*d+[alpha*e; biasalpha]))) - (n_examples/2)*log(2*pi);

    % store alpha, beta, the posterior mean and the posterrior precision-
    % matrix in class attributes存储α，β，类属性中的后验均值和后验精度矩阵
    b.alpha = alpha;
    b.beta  = beta;
    b.w     = m;
    b.p     = v*diag((beta*d+[alpha*e; biasalpha]).^-1)*v';
 
    if b.verbose
        fprintf('Optimization of alpha and beta successfull.\n');
        fprintf('The logevidence is %f.\n',b.evidence);
    end

else

    fprintf('Optimization of alpha and beta did not converge after %i iterations.\n',maxit);
    fprintf('Giving up.');

end