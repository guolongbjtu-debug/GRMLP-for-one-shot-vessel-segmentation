% BW_img = imread('16-target-bw.png');
% BW_img = [];
% dis = 20;
% img_Connected_domain_filtering = connected_domain_filtering(BW_img, dis);
% figure()
% imshow(img_Connected_domain_filtering)
% 
% 

function img_Connected_domain_filtering = connected_domain_filtering(BW_img, dis)
    % 删除较远距离的连通域
    % 输入:
    %   BW_img - 二值图像，逻辑类型或0/1矩阵
    %   dis - 连通域距离主连通域的最近距离超过dis就删除
    % 输出:
    %   img_Connected_domain_filtering - 处理后的二值图像
      
    % --- 1. 连通域分析
    CC = bwconncomp(BW_img);
    stats = regionprops(CC, 'Area', 'PixelIdxList', 'PixelList');
    
    % --- 2. 按大小排序
    [~, sortIdx] = sort([stats.Area], 'descend');
    stats = stats(sortIdx);
    
    % --- 3. 最大连通域
    largest = stats(1).PixelList;

    % --- 4. 计算每个连通域到最大连通域的最近距离
    distances = zeros(length(stats),1);
    for i = 2:length(stats)  % 从第2个开始，因为第1个是最大
        pts = stats(i).PixelList;          % 当前连通域的像素点
        D = pdist2(double(pts), double(largest)); % 点对点欧氏距离
        distances(i) = min(D(:));          % 最近的距离
    end
     
    % --- 5. 删除所有距离 > dis 的连通域
    img_Connected_domain_filtering = BW_img;
    for i = 2:length(stats) % 从第2个开始
        if distances(i) > dis
            img_Connected_domain_filtering(stats(i).PixelIdxList) = 0;
        end
    end

   
end