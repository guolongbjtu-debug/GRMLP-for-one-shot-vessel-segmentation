% clear,clc
% target_img = imread('vessel-target-binary.png');
% load("D:\OFF\清华大学云盘_挂载缓存\郭龙\我的资料库\代码_血管\AAAAA血管图像处理\20250912-重新梳理所有方法\20250826-血管优化\20250916-beyond+frangi（not done）\zusersdata3_combination.mat")
% predicted_img = combination;
% [iou, dice, recall, others] = calculate_iou_dice(target_img, predicted_img);
% recall
% others
% subplot(1,2,1);imshow(target_img);
% subplot(1,2,2);imshow(predicted_img)

function [iou, dice, recall, others] = calculate_iou_dice(target_img, predicted_img)
    % CALCUATE_IOU_DICE 计算两个二值图像的 IoU 和 Dice 系数
    %   输入:
    %       target_img: 真实的标签图像 (logical, 512x512)
    %       predicted_img:    模型的预测图像 (logical, 512x512)
    %       这个顺序不能乱
    %   输出:
    %       iou:  交并比 (Intersection over Union)
    %       dice: Dice 系数
    %=========================================================

    %...确保输入图像是 logical 类型且大小相同
        if ~islogical(target_img) || ~islogical(predicted_img)
            error('输入图像必须是 logical 类型。');
        end
        
        if ~isequal(size(target_img), size(predicted_img))
            error('输入图像的大小必须相同。');
        end
    
    %...calculate recall, dice, iou  
        TP = sum((predicted_img == 1) & (target_img == 1), 'all'); % 预测为血管，真值也是血管 (True Positive)
        TN = sum((predicted_img == 0) & (target_img == 0), 'all'); % 预测为背景，真值也是背景 (True Negative)
        FP = sum((predicted_img == 1) & (target_img == 0), 'all'); % 预测为血管，但真值是背景 (False Positive)
        FN = sum((predicted_img == 0) & (target_img == 1), 'all'); % 预测为背景，但真值是血管 (False Negative)
        recall = TP/(TP+FN);    % Recall 真实血管像素中被成功分类的比例。衡量了避免漏诊的能力。
        dice = (2 * TP) / (2 * TP + FP + FN);  % F1-Score (Dice) 精确率和召回率的调和平均数 
        iou = TP / (TP + FP + FN);           % IoU  预测区域和真实区域的重合程度。
    
    others.recall = recall;    % 避免漏诊
    others.dice = dice;        % 综合性能
    others.iou = iou;          % 和dice重叠，但是dice常用于医学图像，iou常用于计算机视觉
    % others.accuracy =  (TP + TN) / (TP + TN + FP + FN);  % 所有像素中分对的比例:不重要，因为背景多极度的类别不平衡。不要单独使用Accuracy
    % others.specificity = TN / (TN + FP);   % 真实背景像素中被成功分类的比例: 不重要，和recall重叠
    others.MCC = (TP*TN - FP*FN) / sqrt((TP+FP) * (TP+FN) * (TN+FP) * (TN+FN)); % MCC
    


    % 提取边界点 (bwperim: 二值边界提取)
    target_perim    = bwperim(target_img);
    predicted_perim = bwperim(predicted_img);
    % 获取坐标 (find返回行列索引，转成[x,y])
    [ty, tx] = find(target_perim);    % Ground truth 边界点
    [py, px] = find(predicted_perim); % Predicted 边界点

    target_points    = [tx, ty];
    predicted_points = [px, py];

    % ----------- 计算点到点的距离 -------------
    % 从 target -> predicted
    D1 = pdist2(target_points, predicted_points);
    d_target_to_pred = min(D1, [], 2);

    % 从 predicted -> target
    D2 = pdist2(predicted_points, target_points);
    d_pred_to_target = min(D2, [], 2);

    % ----------- 计算指标 -------------
    % 平均表面距离 (ASD)
    asd = (mean(d_target_to_pred) + mean(d_pred_to_target)) / 2

    % Hausdorff Distance (HD)
    all_distances = [d_target_to_pred; d_pred_to_target];
    hd95 = prctile(all_distances, 100) ; %  95%HD
end