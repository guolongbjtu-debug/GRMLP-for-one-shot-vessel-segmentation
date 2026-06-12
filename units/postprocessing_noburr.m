function  vessels_binary3 = postprocessing_noburr(vessels_binary,n)
    % 去除二值图像的毛刺，n是去除次数
    % 输入：vessels_binary二值图
    % n是循环次数
    for i =  1:n
        % 步骤1: 提取骨架skeleton，并计算骨架盲端endpoints ==1
        skeleton = bwmorph(vessels_binary, 'skel', Inf);
        endpoint_kernel = [1 1 1; 1 0 1; 1 1 1];
        skeleton_neighbor = conv2(single(skeleton), endpoint_kernel, 'same');   % 骨架邻居
        skeleton_endpoints = skeleton & (skeleton_neighbor == 1);               % 骨架盲端：邻居数量为1

        % 步骤2：血管-骨架盲端
        vessels_binary2 = vessels_binary - skeleton_endpoints;
        
        % 步骤3：血管+血管洞洞
        vessel_neighbor = conv2(single(vessels_binary2), endpoint_kernel, 'same');  
        holes = (vessel_neighbor == 8);                                         % 血管洞洞：邻居数量为8
        vessels_binary3 = vessels_binary2 + holes;  % 补洞洞
        vessels_binary3 = vessels_binary3>=1;
        
        % 循环n次
        vessels_binary = vessels_binary3;
        i = i + 1;
    end
end
