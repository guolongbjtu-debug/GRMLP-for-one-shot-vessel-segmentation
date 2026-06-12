function eroded_image = postprocessing_erode(binary_image, erosion_size)
% 使用形态学腐蚀缩小血管宽度
%
% 输入:
%   binary_image - 二值血管图像
%   erosion_size - 腐蚀大小（像素）
%
% 输出:
%   eroded_image - 腐蚀后的二值图像

    % 创建圆形结构元素
    se = strel('disk', erosion_size);
    
    % 执行腐蚀操作
    eroded_image = imerode(binary_image, se);
end