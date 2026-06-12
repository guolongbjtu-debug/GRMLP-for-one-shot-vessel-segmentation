% clear,clc,close all
% addpath('D:\OFF\清华大学云盘_挂载缓存\郭龙\我的资料库\代码_血管\AAAAA血管图像处理\20250912-重新梳理所有方法\20250826-血管优化\20251017-合并')
% pathtest()
% 
% img_target = imread('2-target.png');
% img_MLP_binary = imread('2png_img_MLP_binary_threshold=0.03.png');
% figure()
% imshow(img_target);title('img traget')
% figure()
% imshow(img_MLP_binary);title('img mlp binary')
% 
% 
% %          % remove_small   remove_distinct  no_burr_times  fill_holes     exp参数        这个值最小是exp参数
% X_opt2 =     [   100,             1,             10,              8,       0.2,   0.45,   0.35    ];    %  mor参数
% [recall, dice, img_fore_gray,img_fore_binary] = obj1(img_MLP_binary,img_target,X_opt2);
% recall
% dice
% figure()
% subplot(1,2,1);imshow(img_fore_gray);title('img fore gray')
% subplot(1,2,2);imshow(img_fore_binary);title('img fore binary')
% 

function   [recall, dice, img_fore_gray, img_fore_binary] = obj1(img_MLP_binary,img_target,X_opt2)
    x_remove_small = X_opt2(1);
    x_remove_dis = X_opt2(2);
    x_burr_times = X_opt2(3);
    x_holes = X_opt2(4);
    delta = X_opt2(5);    % exp用的参数
    delta2 = X_opt2(6);
    threshold_binary = X_opt2(7); %exp之后用于评价的的参数

    %...对MLP进行形态学处理，以此是remove，dis，burr，holes
        img_mor = preprocessing_remove_small_regions(img_MLP_binary,x_remove_small);         % 去除小颗粒
        img_mor = postprocessing_connected_domain_filtering(img_mor, x_remove_dis);    % remove除远端的连通域  
        img_mor = postprocessing_noburr(img_mor,x_burr_times);     % reduce burrs
        img_mor = postprocessing_fill_holes(img_mor,x_holes); % fill holes
        img_mor_binary = img_mor;
    
    %...膨胀
        dis = bwdist(img_mor_binary);
        % delta = 0.1;
        % delta2 = 0.45;
        weight_img = delta + (1-delta) .* exp(-delta2 .* dis);  % 这个就是图像
        img_fore_gray = weight_img;
        img_fore_binary = preprocessing_binarize_image(img_fore_gray,threshold_binary);
        cal_asd = 0;
        [recall, dice, asd, others] = calculate_evaluation_metric(img_target, img_fore_binary, cal_asd);

    % %...膨胀图/img_fore
    %     dis = bwdist(img_mor_binary);
    %     weight_img = delta + (1-delta) .* exp(-delta2 .* dis);  % 这个就是图像
    %     img_fore_gray = weight_img;
    % 
    % %...计算目标函数
    %     [dice_score, dice_loss] = weighted_dice_loss(img_fore_gray, img_target);
    %     bce_loss = weighted_bce_loss(img_fore_gray, img_target);
    %     para = [dice_loss,bce_loss];

    %...计算paras.
    %...这里需要注意，因为膨胀图本身是灰度图，因为和概率相关，但是计算recall和dice需要转化为二值图(binary相关)，因此膨胀不是很好计算recall和dice
    %...所以是不是需要找一个新的目标函数呢？想一想
    %...输入是前景膨胀的概率图，对比的是targetimg。所以可以是targetimg上面膨胀部分的点乘平均值，越接近1越好。
    %...dss说diceloss和bceloss不错
    %...20251110
    %...试了一下加上expand部分去优化，效果一般，所以先考虑分开优化试一下
end

