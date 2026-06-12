% clear,clc,close all
% addpath('D:\OFF\清华大学云盘_挂载缓存\郭龙\我的资料库\代码_血管\AAAAA血管图像处理\20250912-重新梳理所有方法\20250826-血管优化\20251017-合并');
% pathtest
% x = [7	6.65541524273151	342.339656533528	0.837783606613558	5.90268809231928	78	5	2.90808205982884	1	6.46690646394536	55	6	2.47984657102415	0.679113886895763	3.26109399914454	60	0.0379497893689643	0.641158001789251	0.679925158360654	3.78643051516470	73	50	10000];
% x(20) = x(20)*2;
% x(23) = eps;
% % x = [8	1.34340373330805	188.414747673618	0.885761420582638	4.72222741328360	31	5	1.42529678646716	0.612073763476053	8.10450171568259	61	5	2.04784365375116	0.865150354756531	7.73447121660273	80	0.376230721430155	0.101297406634707	0.812877579394453	2.94653187845752	29	18	0.0537065413032240];
% % x = [8	1.66666701295083	202.179723226131	0.866429998949726	4.66039552940049	33	5	1.41492264681859	0.628736146785373	7.82319971328648	60	5	1.96178192118188	0.873810053691031	8.05223960584816	75	0.373120213251007	0.130413711743693	0.796260046828190	3.10376774744467	30	18	0.0733153048800270];
% %...初始灰度图像准备
%     img = imread('2.png');             % 原图
%     img = preprocessing_gray_img(img);
% %...手工标注的二值图像准备
%     targetimg_binary = imread('2-target.png');   % 手工标注
%     targetimg_binary = preprocessing_gray_img(targetimg_binary) ;
%     targetimg_binary = preprocessing_binarize_image(targetimg_binary, 0.5);
% 
% %...debug    
%     [paras,img1_gray,img2_gray,img3_gray,img4_gray,img1_binary,img2_binary,img3_binary,img4_binary,img5_gray,img5_binary] = obj(x,img,targetimg_binary,0);


% ==================================目标函数================================
function [paras,img1_gray,img2_gray,img3_gray,img4_gray,img1_binary,img2_binary,img3_binary,img4_binary,img5_gray,img5_binary]= obj(X,img,targetimg_binary,cal_asd)
    % =====================================================================
    % GUO LONG 20251022 lee shaukee building 1043-3 room
    % 输入：设计变量X, 以及计算参数的图像img，net，targetimg，cal_asd
    % 输出：目标函数y
    % =====================================================================
        if nargin < 3 % targetimg_binary的默认值
            targetimg_binary = ones(512,512);  
            targetimg_binary = preprocessing_binarize_image(targetimg_binary,0.5);
            fprintf('\n no targetimg input, default value is ones(512,512)')
        end
        if nargin < 4 % targetimg_binary的默认值
            cal_asd = 0;
            fprintf('\n no targetimg input, default values is cal_asd = 0 ')
        end
        if ~islogical(targetimg_binary)        % targetimg_binary的默认值是否是logical的检测
            error('targetimg_binary必须是logical类型，但输入的是 %s 类型', class(targetimg_binary));
        end
        
        frangi.N = X(1);
        frangi.A = X(2);
        frangi.B_fenzhi1 = X(3);    % 这个参数好像不太对
        frangi.adjust = X(4);
        frangi.binary =  X(5)/1000;
        frangi.remove = X(6);

        beyond_N = X(7);    % sigma范围
        beyond_tau = X(8);  
        beyond_adjust = X(9);
        beyond_binary = X(10)/100;  
        beyond_remove = X(11);

        zhang_N = X(12);    % sigma范围
        zhang_tau = X(13);  
        zhang_adjust = X(14);
        zhang_binary = X(15)/100;  
        zhang_remove = X(16);  

        w1 = X(17);
        w2 = X(18);
        combination_adjust = X(19);
        combination_binary = X(20)/100;
        combination_remove = X(21);
        dis = X(22);
        gama = X(23);

    %...img1：frangi（sigma=1）
        [img1_gray,img1_binary] = Filter_frangi(X(1:6),img);

    %...img2： Jerman
        [img2_gray,img2_binary]= Filter_Jerman(X(7:11),img);

    %...img3:zhang
        [img3_gray,img3_binary]= Filter_Zhang(X(12:16),img);
    
    %...img4
        img4_gray = (w1*img1_gray + w2*img2_gray + (1-w1-w2)*img3_gray);
        img4_gray = imadjust(img4_gray, [0 combination_adjust]);   
        img4_binary = preprocessing_binarize_image(img4_gray,combination_binary);
        img4_binary = preprocessing_remove_small_regions(img4_binary, combination_remove);
        img4_binary = postprocessing_connected_domain_filtering(img4_binary, dis);

    %...img5 
        Guo_img_gray = Filter_Guo(img,[frangi.N,gama]);
        img5_gray = (w1*img1_gray + w2*img2_gray + (1-w1-w2)*img3_gray).*Guo_img_gray;
        img5_gray = imadjust(img5_gray, [0 combination_adjust]);   
        img5_binary = preprocessing_binarize_image(img5_gray,combination_binary);
        img5_binary = preprocessing_remove_small_regions(img5_binary, combination_remove);
        img5_binary = postprocessing_connected_domain_filtering(img5_binary, dis);

    %...计算图片的性能参数
        [recall, dice, asd, paras.img1] = calculate_evaluation_metric(targetimg_binary, img1_binary, cal_asd);
        [recall, dice, asd, paras.img2] = calculate_evaluation_metric(targetimg_binary, img2_binary, cal_asd);
        [recall, dice, asd, paras.img3] = calculate_evaluation_metric(targetimg_binary, img3_binary, cal_asd);
        [recall, dice, asd, paras.img4] = calculate_evaluation_metric(targetimg_binary, img4_binary, cal_asd);
        [recall, dice, asd, paras.img5] = calculate_evaluation_metric(targetimg_binary, img5_binary, cal_asd);
end
