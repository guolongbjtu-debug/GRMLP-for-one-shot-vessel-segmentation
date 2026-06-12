
function output_img = postprocessing_expansion(BW_img, expansion_size)
    % 输入:
    %   BW_img - 二值血管图像
    %   expansion_size - 膨胀大小（像素）
    %
    % 输出:
    %   output_img - 膨胀后的二值图像
    
    se = strel('disk', expansion_size);
    output_img = imdilate(BW_img, se); % 膨胀
end