% clear,clc,close all
% img = imread("9png_highrecall_gray_index297.png");
% targetimg = imread("9png-target.png");
% delta = 0.001;
% [recall,dice] = get_recall_dice_curve(img,targetimg,delta);
% plot(-recall,-dice)


function [recall,dice] = get_recall_dice_curve(img,targetimg,delta)
    img = preprocessing_gray_img(img);
    i = 0;
    for threshold = [0:delta:1]
        i = i +1;
        binary = preprocessing_binarize_image(img,threshold);      
        [recall(i), dice(i), asd, others] = calculate_evaluation_metric_iou(targetimg, binary, 0);           
    end
end