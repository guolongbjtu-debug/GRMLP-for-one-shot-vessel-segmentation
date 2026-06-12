function vessel_binary2 = postprocessing_fill_holes(vessel_binary,minsize)
    % 有的血管内部会存在一些小的孔洞，用4取反+连通域检测的方法填充这些空洞
    % 计算连通域的时候，用了4邻域而不用8邻域
    bw = vessel_binary;
    bw_inv = ~bw; % 先取反
    cc = bwconncomp(bw_inv,4);   % 找黑色连通的4邻域，8邻域效果不是很好
    stats = regionprops(cc, 'Area'); % 得到每个连通域的像素索引
    idxRemove = find([stats.Area] < minsize);  % 找到小于阈值的连通域
    for i = 1:numel(idxRemove)
        bw_inv(cc.PixelIdxList{idxRemove(i)}) = 0;   % 把这些小连通域像素点置为白色
    end
    vessel_binary2 = ~bw_inv; % 再取反，回到原图定义：黑色=0，白色=1
end