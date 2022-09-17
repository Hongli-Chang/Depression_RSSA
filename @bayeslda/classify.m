function varargout = classify(b, x)
% prediction procedure for bayeslda   bayeslda��Ԥ�����
% INPUT:
%    b          - object of type bayeslda  bayeslda���͵Ķ���
%    x          - m*n matrix containing n feature vectors of size m*1 
%                  m * n�����������СΪm * 1��n����������
%
% OUTPUT:
%    varargout  - if classify is called with one output argument an array
%                 containing the mean value of the predictive distribution 
%                 for each example in x is returned 
%        ���ʹ��һ������������÷��࣬�򷵻ذ���x��ÿ��ʾ����Ԥ��ֲ���ƽ��ֵ������
%               - if classify is called with two output arguments the 
%                 mean value and the variance of the predictive
%                 distribution are returned
%             ���ʹ����������������÷��࣬�򷵻�Ԥ��ֲ���ƽ��ֵ�ͷ���
% Author: Ulrich Hoffmann - EPFL, 2006
% Copyright: Ulrich Hoffmann - EPFL
%
% The algorithm implemented here was originally described by 
% MacKay, D. J. C., 1992. Bayesian interpolation.
% Neural Computation 4 (3), pp. 415-447.


%% add feature that is constantly one (bias term)��ӵĹ��ܣ���һ�����ϣ�ƫ�
x = [x; ones(1,size(x,2))];    


%% compute mean of predictive distributions����Ԥ��ֲ���ƽ��ֵ
m = b.w'*x;


%% if one output argument return mean only���һ���������ֻ����ƽ��ֵ
if nargout == 1
    varargout(1) = {m};    
end


%% if two output arguments compute and return variance also��������������Ҳ���㲢���ط���
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