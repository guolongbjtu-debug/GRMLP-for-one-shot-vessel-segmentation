%=======================================================================
% GUO LONG 20250919, lee shaukee building
% 【input】
%   target_img
%   predicted_img
%   cal_asd      默认等于0：不计算asd，因为他很占用时间
% 【output】
%   recall
%   dice
%   asd
%   others = recall dice asd HD95 HD90
% 【evaluation metric】
%   *recall：    important       避免漏检
%   *dice：      important       最常用，衡量总体重叠程度
%   *asd：       important       最全面
%   MCC：        可选            背景很多的时候，虽然比accuracy好，但也好不了太多，可解释性不强
%   HD95：       可选            效果不好
%   iou：        not important   和dice重叠
%   HD：         not important   受到个别特别差点的影响大
%   accuracy：   not important   类别不平衡：背景太多了，不要单独使用Accuracy
%   specificity：not important   和recall重叠
%=======================================================================

% clear,clc
% target_img = imread('vessel-target-binary.png');
% load("D:\OFF\清华大学云盘_挂载缓存\郭龙\我的资料库\代码_血管\AAAAA血管图像处理\20250912-重新梳理所有方法\20250826-血管优化\20250916-beyond+frangi（not done）\zusersdata3_combination.mat")
% predicted_img = combination_binary;
% [recall, dice, asd, others] = calculate_evaluation_metric(target_img, predicted_img);   % 
% others
% subplot(1,2,1);imshow(target_img);
% subplot(1,2,2);imshow(predicted_img)

function [recall, dice, asd, others] = calculate_evaluation_metric(target_img, predicted_img, cal_asd)
    % CALCUATE_IOU_DICE 计算两个二值图像的 IoU 和 Dice 系数
    %   输入:
    %       target_img: 真实的标签图像 (logical, 512x512)
    %       predicted_img:    模型的预测图像 (logical, 512x512)
    %       这个顺序不能乱
    %   输出:
    %       iou:  交并比 (Intersection over Union)
    %       dice: Dice 系数
    %=========================================================

    %...判断是否进行asd运算
        if nargin < 3 || isempty(cal_asd)
            cal_asd = 0;  % 默认不计算
        end

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
        others.recall = TP/(TP+FN);    % Recall 真实血管像素中被成功分类的比例。衡量了避免漏诊的能力。
        others.dice = (2 * TP) / (2 * TP + FP + FN);  % F1-Score (Dice) 精确率和召回率的调和平均数 
        others.iou = TP / (TP + FP + FN);           % IoU  预测区域和真实区域的重合程度                                          % IoU和dice功能重叠，但是dice常用于医学图像，iou常用于计算机视觉                
        % others.accuracy =  (TP + TN) / (TP + TN + FP + FN);  % 所有像素中分对的比例:不重要，因为背景多极度的类别不平衡。不要单独使用Accuracy
        % others.specificity = TN / (TN + FP);   % 真实背景像素中被成功分类的比例: 不重要，和recall重叠
        % others.MCC = (TP*TN - FP*FN) / sqrt((TP+FP) * (TP+FN) * (TN+FP) * (TN+FN)); % MCC
    
    %...calculate asd and hd95
        if cal_asd == 1
            target_perim    = bwperim(target_img); % 提取边界点 (bwperim: 二值边界提取)
            predicted_perim = bwperim(predicted_img);
            [ty, tx] = find(target_perim);    % 获取坐标 (find返回行列索引，转成[x,y])Ground truth 边界点
            [py, px] = find(predicted_perim); % Predicted 边界点
            target_points    = [tx, ty];
            predicted_points = [px, py];
            D1 = pdist2(target_points, predicted_points);  % 从 target -> predicted的点距离
            d_target_to_pred = min(D1, [], 2);
            D2 = pdist2(predicted_points, target_points); % 从 predicted -> target的点距离
            d_pred_to_target = min(D2, [], 2);
            % ASD
            others.asd = (mean(d_target_to_pred) + mean(d_pred_to_target)) / 2;
            % Hausdorff Distance (HD)
            all_distances = [d_target_to_pred; d_pred_to_target];
            others.hd95 = prctile(all_distances, 95) ; %  95%HD
            others.hd90 = prctile(all_distances, 90) ; %  90%HD
        else
            others.asd = 512;
        end
        
    %...output evaluation metric
        recall = others.recall;
        dice = others.dice;
        asd = others.asd;
end