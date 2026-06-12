function processed_img = preprocessing_remove_small_regions(binary_img, min_size)
    % remove_small_regions: 移除二值图像中小于指定像素数的连通域
    % 输入:
    %   binary_img - 二值图像，逻辑类型或0/1矩阵
    %   min_size - 连通域的最小像素数，低于此值的连通域将被移除
    % 输出:
    %   processed_img - 处理后的二值图像

    % 确保输入是逻辑类型
    binary_img = logical(binary_img);

    % 连通域检测
    conn_comp = bwconncomp(binary_img);  % 检测连通域

    % 统计每个连通域的像素数量
    num_pixels = cellfun(@numel, conn_comp.PixelIdxList);

    % 将小于指定大小的连通域置为黑色
    for i = 1:length(num_pixels)
        if num_pixels(i) < min_size
            binary_img(conn_comp.PixelIdxList{i}) = 0;  % 设为黑色
        end
    end
    % 返回处理后的二值图像
    processed_img = binary_img;
end