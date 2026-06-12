% clear,clc,close all
% img = imread('img_enhenced.png');             % 原图
% img = preprocessing_gray_img(img);
% X_neighborhood = extract_neighborhood(img,9);   % 329960*49

function neighborhood_data = extract_neighborhood(img,k)
%EXTRACT7X7 高效提取 7×7 邻域
%   neighborhood_data = EXTRACT7X7(IMG)
%   输入：
%       img - m×n 的灰度图像（double/uint8/uint16 均可）
%   输出：
%       neighborhood_data - (m*n)×49 的矩阵，每一行对应一个像素 7×7 邻域
%                         按列优先顺序展开（与 conv2 一致）

    %---- 参数 ----
    padW = (k-1)/2;     % 3

    %---- 边界填充 ----
    %  replicate 方式与 conv2 默认一致，也可改成 'symmetric'/'circular'
    img_pad = padarray(img, [padW padW], 'replicate', 'both');

    %---- 用 im2col 滑窗展开 ----
    %  得到 49×(m*n) 的矩阵，每列是一个 7×7 邻域按列优先拉成的向量
    cols = im2col(img_pad, [k k], 'sliding');

    %---- 转置即可 ----
    neighborhood_data = cols.';   % 现在大小是 (m*n)×49
end