% 
% %...mask
%     [maskimg_binary, map] = imread('22_training_mask.gif', 'Frames', 'all');
%     maskimg_binary = preprocessing_gray_img(maskimg_binary);
%     maskimg_binary = preprocessing_binarize_image(maskimg_binary,0.5);
%     figure();imshow(maskimg_binary)
% 
% n = 5;  % 侵入5个像素
% result = invade_black_to_white(maskimg_binary, n);
%  figure();imshow(result)


function result_mask = invade_black_to_white(maskimg_binary, n, shape_type)
% 将黑色背景向白色前景侵入n个像素
% 输入：
%   maskimg_binary - 二值图像（黑背景=0，白前景=1）
%   n - 侵入的像素数
%   shape_type - 结构元素形状，可选：'disk'(默认), 'square', 'diamond'
% 输出：
%   result_mask - 处理后的图像

    % 参数验证
    if n <= 0
        error('侵入像素数n必须大于0');
    end
    
    % 设置默认结构元素形状
    if nargin < 3
        shape_type = 'disk';
    end
    
    % 确保输入是二值图像
    if ~islogical(maskimg_binary)
        maskimg_binary = logical(maskimg_binary);
    end
    
    % 根据形状类型创建结构元素
    switch lower(shape_type)
        case 'disk'
            se = strel('disk', n);
        case 'square'
            se = strel('square', 2*n+1);
        case 'diamond'
            se = strel('diamond', n);
        otherwise
            warning('不支持的形状类型，使用默认disk');
            se = strel('disk', n);
    end
    
    % 方法1：直接对背景进行膨胀（更直观的方法）
    % 这里我们实际上是对背景（黑色区域）进行膨胀
    % 而imdilate是扩张白色区域，所以需要先取反
    background = ~maskimg_binary;  % 背景变为白色
    dilated_background = imdilate(background, se);
    result_mask = ~dilated_background;  % 恢复格式
    
    % 方法2：对前景进行腐蚀（等价操作）
    % 这种方法更直接：侵入黑色 = 腐蚀白色前景
    % result_mask = imerode(maskimg_binary, se);
    
    % 注意：方法1和方法2是等价的，选择一种即可
    % 方法2更直接，但方法1更容易理解"侵入黑色"的概念
end