function e = getevidence(b)
% returns log evidence from object b从对象b返回日志证据
%
% INPUT:
%    b       - object of type bayeslda  bayeslda类型的对象
%
% OUTPUT:
%    e       - log evidence日志证据
%
% Author: Ulrich Hoffmann - EPFL, 2006
% Copyright: Ulrich Hoffmann - EPFL

e = b.evidence;