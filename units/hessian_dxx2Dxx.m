function  [Dxx,Dxy,Dyy] = hessian_dxx2Dxx(dxx,dxy,dyy,sigmas)
% 在计算图像响应后乘以 $\sigma^2$ 来进行尺度归一化：为了抵消高斯微分核本身随尺度变化而发生的幅度衰减，
% 从而使得响应值的大小能够反映图像结构的本质强度，而不是所使用的滤波器的尺度。
% 郭龙 2025/08/30 于李兆基
%...遍历sigmas
    for i = 1:length(sigmas)
        sigma = sigmas(i);
%...求hessian矩阵 Dxx,Dxy,Dyy
        dxx = (sigma^2)*dxx; % 分别代表图像的二阶空间导数
        dxy = (sigma^2)*dxy;    
        dyy = (sigma^2)*dyy; % 当你使用高斯核进行卷积时，由于高斯核的平滑效果，计算得到的二阶导数实际上会比原始图像的二阶导数要小。
                             % 乘以 𝜎2，实际上是在尝试恢复原始图像中二阶导数的“真实”值。
        Dxx(:,:,i) = dxx;
        Dxy(:,:,i) = dxy;
        Dyy(:,:,i) = dyy;
    end
end