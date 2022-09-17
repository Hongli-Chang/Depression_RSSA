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
%ע�ͣ�
%Bayeslda��Ĺ��캯��
%��Ҷ˹�����б����

%? ������
%??????? train - ��ѵ��ʵ����ѧϰ�����б�
%??????? classify - ����������
%???? getevidence - ����ѧϰ�б�ʽ����־֤��
%% set verbose flag ������ϸ��־
if nargin == 1
    b.verbose = verbose;    % if set train gives verbose output���ѵ��������ϸ���
else
    b.verbose = 0;
end


%% define attributes of object������������
b.evidence = 0;             % log evidence ��־֤��
b.beta = 0;                 % inverse variance of noise�������淽�� 
b.alpha = 0;                % inverse variance of prior ��ǰ���淽��
b.w  = [];                  % weight vector (mean of posterior)Ȩ���������󷽵�ƽ��ֵ��
b.p  = [];                  % precision matrix of posterior���龫�Ⱦ���


%% initialize class
b = class(b,'bayeslda');