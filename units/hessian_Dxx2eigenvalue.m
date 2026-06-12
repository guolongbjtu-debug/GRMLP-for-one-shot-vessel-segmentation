function [Lambda1,Lambda2,v1x,v1y,v2x,v2y,angle1,angle2] = hessian_Dxx2eigenvalue(Dxx,Dxy,Dyy)
% 20240709 于李兆基科技大厦，用于求Hessian矩阵的特征值Lambda和特征向量v
% |Lambda1| < |Lambda2| 
% 已经验证了和eig2image是一样的了，lambda1是特征值绝对值小的那个，对应曲率小，沿着血管方向；lambda2则相反。

%...求lambda，v
    % 求lambda 
    tmp = sqrt((Dxx - Dyy).^2 + 4*Dxy.^2);
    Lambda1 = 0.5*(Dxx + Dyy - tmp);
    Lambda2 = 0.5*(Dxx + Dyy + tmp);
    % 求v2 = v1 
    v2x = 2*Dxy;
    v2y = Dyy-Dxx+tmp;
    mag = sqrt(v2x.^2 + v2y.^2); 
    check = (mag ~= 0);   % 不等于0的check
    v2x(check) = v2x(check)./mag(check);
    v2y(check) = v2y(check)./mag(check); % 归一化
    v1x = -v2y;
    v1y = v2x;
    
%...确保和Lambda1<2,v和Lambdal对应 
    % 交换lambda 
    check = abs(Lambda1)>abs(Lambda2);  %1>2要交换的check
    mid = Lambda1;
    Lambda1(check) = Lambda2(check);  % 1 > 2 则交换
    Lambda2(check) = mid(check);
    % 交换vx
    mid = v1x;
    v1x(check) = v2x(check);  % 1 > 2 则交换
    v2x(check) = mid(check);
    % 交换vy 
    mid = v1y;
    v1y(check) = v2y(check);  % 1 > 2 则交换
    v2y(check) = mid(check);
    angle1 = atan2(v1x,v1y);
    angle2 = atan2(v2x,v2y);
end