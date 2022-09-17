function b = train(b, x, y)
% training procedure for Bayesian LDA  ��Ҷ˹LDAѵ������
% INPUT:
%    b       - object of type bayeslda  bayeslda���͵Ķ���
%    x       - m*n matrix containing n feature vectors of size m*1 
%                m * n�����������СΪm * 1��n����������
%    y       - 1*n matrix containing class labels (-1,1) 
%                �������ǩ��1 * n����-1,1��
% OUTPUT:
%    b       - updated object of type bayeslda ���µĶ���Ϊbayeslda
%
% Author: Ulrich Hoffmann - EPFL, 2006
% Copyright: Ulrich Hoffmann - EPFL
%
% The algorithm implemented here was originally described by
% MacKay, D. J. C., 1992. Bayesian interpolation.
% Neural Computation 4 (3), pp. 415-447.


%% compute regression targets from class labels (to do lda via regression)
%�����ǩ����ع�Ŀ�꣨ͨ���ع���lda��
n_posexamples = sum(y==1);
n_negexamples = sum(y==-1);
n_examples    = n_posexamples + n_negexamples;
y(y==1) = n_examples/n_posexamples;
y(y==-1) = -n_examples/n_negexamples;


%% add feature that is constantly one (bias term)��ӵĹ��ܣ���һ�����ϣ�ƫ�
x = [x; ones(1,size(x,2))];  


%% initialize variables for fast iterative estimation of alpha and beta
%��ʼ�������Կ��ٵ������Ʀ��ͦ�
n_features = size(x,1);            % dimension of feature vectors ��������ά��
d_beta = inf;                      % (initial) diff. between new and old beta  ����ʼ������ �¾�beta֮��
d_alpha = inf;                     % (initial) diff. between new and old alpha ����ʼ������ �¾ɰ�����֮��
alpha    = 25;                     % (initial) inverse variance of prior distribution����ʼ����ǰ�ֲ����淽��
%alpha    = 50; 
biasalpha = 0.00000001;            % (initial) inverse variance of prior for bias term����ʼ��ƫ����֮ǰ���淽��
 %biasalpha = 0.000001;  
 beta     = 1;                      % (initial) inverse variance around targets����ʼ��Χ��Ŀ����淽��
%beta     = 100; 
stopeps  = 0.0001;                 % desired precision for alpha and beta�����alpha��beta����
%stopeps  = 0.01;  
i        = 1;                      % keeps track of number of iterations���ٵ�������
maxit    = 500;                    % maximal number of iterations ����������
[v,d] = eig(x*x');                 % needed for fast estimation of alpha and beta ��Ҫ���ٹ��Ʀ��ͦ� 
vxy    = v'*x*y';                  % dito
d = diag(d);                       % dito
e = ones(n_features-1,1);          % dito


%% estimate alpha and beta iteratively�����ع��Ʀ��ͦ�
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


%% process results of estimation ���̹�����
if (i < maxit)
    
    % compute the log evidence������־֤��
    % this can be used for simple model selection tasks��������ڼ򵥵�ģ��ѡ������
    % (see MacKays paper)����MacKays���ģ�
    b.evidence = (n_features/2)*log(alpha) + (n_examples/2)*log(beta) - ...
    (beta/2)*err - (alpha/2)*m'*m - ...
    0.5*sum(log((beta*d+[alpha*e; biasalpha]))) - (n_examples/2)*log(2*pi);

    % store alpha, beta, the posterior mean and the posterrior precision-
    % matrix in class attributes�洢�����£��������еĺ����ֵ�ͺ��龫�Ⱦ���
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