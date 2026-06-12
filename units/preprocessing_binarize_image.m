function binary_img = preprocessing_binarize_image(gray_img, threshold)
    % binarize_image: 将灰度图像二值化
    % 输入:
    %   gray_img - 输入的灰度图像 (二维矩阵)
    %   threshold - 二值化阈值 (0-1 范围)
    %               如果未提供，使用 Otsu 方法自动选择
    % 输出:
    %   binary_img - 二值化后的图像 (逻辑矩阵, 0 或 1)

    % 检查输入是否为灰度图像
    if size(gray_img, 3) == 3
        error('输入的图像不是灰度图像，请先将其转换为灰度图像');
    end
    
    % 如果未提供阈值，则使用 Otsu 方法计算阈值
    if nargin < 2
        threshold = graythresh(gray_img); % graythresh 返回 [0, 1] 范围，需要乘以 255
    end
   
    binary_img = gray_img > threshold; % 进行二值化
    binary_img = logical(binary_img); % 将二值化结果转换为逻辑类型
end
