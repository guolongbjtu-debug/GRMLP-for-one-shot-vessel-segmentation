% GRMLP_singlescale_inference Run the singlescale GRMLP + small-scale filter on one test sample.
%
% Loads the cfg.testindex-th test sample (image + mask + target), loads a trained
% GRMLP checkpoint (selected by cfg.testtime), predicts a vessel probability map,
% fuses it with a small-scale filter into the VPE output, scores
% recall-dice against the target, and saves results to result/<checkpoint_folder_name>/.
%
% Dataset, model_variant, testindex, testtime come from get_config.m.

clear; clc; close all;
fprintf('===============inference start========================\n')

... get setting dir from get_config.m
    script_dir = fileparts(mfilename('fullpath'));
    addpath(script_dir);
    cfg = get_config();
    project_root = cfg.project_root;
    test_dir    = fullfile(cfg.processed_dir, 'test');  % dataset_normalization/<dataset>/test
    image_dir   = fullfile(test_dir, 'images');         % .../test/images
    mask_dir    = fullfile(test_dir, 'masks');          % .../test/masks
    target_dir  = fullfile(test_dir, 'targets');        % .../test/targets
    paras = cfg.paras.(cfg.model_variant);


... pick the testindex-th test sample and load its image + mask
    testindex = cfg.testindex;
    image_files  = dir(fullfile(image_dir,  '*.png'));
    mask_files   = dir(fullfile(mask_dir,   '*.png'));
    target_files = dir(fullfile(target_dir, '*.png'));
    if ~(numel(image_files) == numel(mask_files) && numel(image_files) == numel(target_files))
        error('Test sample counts differ across folders (images=%d, masks=%d, targets=%d).\nCannot pair by position.', ...
              numel(image_files), numel(mask_files), numel(target_files));
    end
    n_samples = numel(image_files);
    fprintf('Total test samples: %d  (cfg.testindex can be 1..%d; current=%d)\n', n_samples, n_samples, testindex);
    if testindex < 1 || testindex > n_samples
        error('testindex=%d is out of range [1, %d] (number of test samples).', testindex, n_samples);
    end
    image_path  = fullfile(image_files(testindex).folder,  image_files(testindex).name);
    mask_path   = fullfile(mask_files(testindex).folder,   mask_files(testindex).name);
    target_path = fullfile(target_files(testindex).folder, target_files(testindex).name);
    fprintf('Testing sample %d/%d:  img=%s  mask=%s  target=%s\n', ...
            testindex, n_samples, image_files(testindex).name, mask_files(testindex).name, target_files(testindex).name);

    % mask: grayscale -> binary -> expand boundary by 3px (reduces edge artifacts)
    [maskimg_binary, ~] = imread(mask_path);
    maskimg_binary = preprocessing_gray_img(maskimg_binary);
    maskimg_binary = preprocessing_binarize_image(maskimg_binary, 0.5);
    invaded_maskimg_binary = invade_black_to_white(maskimg_binary, 5);

    % image: grayscale, then restrict to the RoI
    img = imread(image_path);
    img = preprocessing_gray_img(img);
    img = img .* maskimg_binary;

    figure();
    subplot(2,3,1); imshow(invaded_maskimg_binary); title('mask img');
    subplot(2,3,2); imshow(img); title('initial img');
    fprintf('======================================================\n')
    fprintf('imread test image successfully.\n');


... RoI model: DRIVE already provides RoI mask, so we directly use it to generate high-resolution feature binary map for GRMLP training
    img_hrf_binary = ones(512, 512); % initialize high-resolution feature binary map
    img_hrf_binary(maskimg_binary == 0) = 0; % set non-RoI areas to 0
    img_hrf_binary = img_hrf_binary > 0; % convert to binary
    subplot(2,3,3);imshow(img_hrf_binary);title('ROI binary map')
    fprintf('======================================================\n')
    fprintf('RoI module finish successfully.\n');
    fprintf('======================================================\n')


... VPE model: get trained GRMLP result
    testtime = cfg.testtime;
    checkpoint_dir = cfg.checkpoint_dir;
    
    % find the checkpoint folder based on testtime (e.g. 'latest' or '2024-05-01T12-00-00'), fallback to latest if not found
    if isempty(testtime) || strcmpi(testtime, 'none')
        checkpoint_folders = dir(fullfile(checkpoint_dir, sprintf('*_%s_*', cfg.model_variant))); % list all checkpoint folders when testtime is none
        checkpoint_folders = checkpoint_folders([checkpoint_folders.isdir]); % keep only folders (skip legacy flat .mat)
        if ~isempty(checkpoint_folders)
            [~, latest_idx] = max([checkpoint_folders.datenum]); % find the latest by modification date (dir sorts by name, not time)
            checkpoint_folders = checkpoint_folders(latest_idx);
        end
    else
        checkpoint_folders = dir(fullfile(checkpoint_dir, sprintf('%s_%s_*', testtime, cfg.model_variant))); % find checkpoint folder by testtime prefix
        checkpoint_folders = checkpoint_folders([checkpoint_folders.isdir]); % keep only folders (skip legacy flat .mat)
    end
    if isempty(checkpoint_folders) % fallback to the latest checkpoint folder if no match
        checkpoint_folders = dir(fullfile(checkpoint_dir, sprintf('*_%s_*', cfg.model_variant))); % list all checkpoint folders
        checkpoint_folders = checkpoint_folders([checkpoint_folders.isdir]); % keep only folders (skip legacy flat .mat)
        [~, latest_idx] = max([checkpoint_folders.datenum]); % find the latest by modification date
        checkpoint_folders = checkpoint_folders(latest_idx);
        fprintf('No checkpoint found for testtime=%s, using latest: %s\n', testtime, checkpoint_folders(1).name);
    end
    checkpoint_folder = fullfile(checkpoint_dir, checkpoint_folders(1).name); % path to trained GRMLP checkpoint folder


    % get the result file's name (.m result)
    for i = 1:length(paras.maxsigma)
        GRMLP_files = dir(fullfile(checkpoint_folder, sprintf('*_%s_*.mat', cfg.model_variant))); % find .mat file inside the folder
        resultfile_path(i).name = fullfile(GRMLP_files(i).folder, GRMLP_files(i).name); % path to .mat checkpoint
    end


... VPE model: multi-scale GRMLP inference
    img_MLP_list = cell(1, length(paras.maxsigma));
    for i = 1:length(paras.maxsigma)
        fprintf('inferencing the %d-th GRMLP model, sigma = %d..................\n', i,paras.maxsigma(i));
        load(resultfile_path(i).name,'result', 'paras'); % load trained GRMLP result and parameters
        maxsigma = paras.maxsigma(i);
        [X_Lambda,X_neighborhood,useDS,Lambda_xsize,Lambda_ysize] = GRMLP_inputdata(img,maxsigma,paras.neighborhood_size); % build per-pixel neighborhood features
        Y_pred = predict(result.net, useDS);                   % run the trained GRMLP network
        img_MLP = reshape(Y_pred, Lambda_xsize, Lambda_ysize); % reshape flat predictions to the image grid
        img_MLP_list{i} = img_MLP.*invaded_maskimg_binary;
    end

    % multi-scale combination
    img_MLP_multiscale = zeros(512,512);
    for i = 1:length(paras.maxsigma)
        img_MLP_multiscale = img_MLP_multiscale + img_MLP_list{i};
    end
    img_MLP_multiscale = img_MLP_multiscale/max(max(img_MLP_multiscale));
    subplot(2,3,4);imshow(img_MLP_multiscale)

%%
... VPE model: small-scale filter
    N = 3;
    tau = 2;
    remove = 200;  % default num
    adjust = 1;    % no use
    binary = 0;    % no use
    x = [N,tau,adjust,binary,remove];
    [smallscale_gray,~] = Filter_Jerman(x,img);
    invaded_maskimg_binary = invade_black_to_white(maskimg_binary, 10);
    smallscale_gray = smallscale_gray.*invaded_maskimg_binary;
    smallscale_gray = imadjust(smallscale_gray,[0,adjust]);

    binary = quantile(smallscale_gray(:), 0.88); % small-scale filter threshold (before binarization)  ============
    smallscale_binary = smallscale_gray>binary;
    smallscale_binary = preprocessing_remove_small_regions(smallscale_binary,remove);
    smallscale_filter = smallscale_binary.*smallscale_gray;
    subplot(2,3,5);imshow(smallscale_filter);title('small-scale filter')

    imgVPE = (img_MLP_multiscale + smallscale_filter)/max(img_MLP_multiscale(:) + smallscale_filter(:));   % fuse small-scale filter into the GRMLP result
    fprintf('VPE module finish successfully.\n');
    fprintf('======================================================\n')




... cal metrics
    targetimg_binary = imread(target_path);
    targetimg_binary = preprocessing_gray_img(targetimg_binary);
    targetimg_binary = preprocessing_binarize_image(targetimg_binary,0.5);

    [para_smallscale.recall,para_smallscale.dice] = get_recall_dice_curve(smallscale_gray,targetimg_binary,0.001);
    best_smallscale_dice = max(para_smallscale.dice);

    [para_mlp.recall,para_mlp.dice] = get_recall_dice_curve(img_MLP_multiscale,targetimg_binary,0.001);
    best_mlp_dice = max(para_mlp.dice);

    [para_VPE.recall,para_VPE.dice] = get_recall_dice_curve(imgVPE,targetimg_binary,0.001);
    best_VPE_dice = max(para_VPE.dice);
    
    best_VPE_idx = find(para_VPE.dice == best_VPE_dice, 1);
    best_VPE_threshold = (best_VPE_idx - 1) * 0.001;
    imgVPE_bestbinary = preprocessing_binarize_image(imgVPE, best_VPE_threshold);
    imgVPE_bestbinary = preprocessing_remove_small_regions(imgVPE_bestbinary,200);
    imgVPE_bestbinary = postprocessing_noburr(imgVPE_bestbinary,10);
    imgVPE_bestbinary = postprocessing_fill_holes(imgVPE_bestbinary,5);

    imgVPE2 = imgVPE_bestbinary.*imgVPE;
    [para_VPE2.recall,para_VPE2.dice] = get_recall_dice_curve(imgVPE2,targetimg_binary,0.001);
    subplot(2,3,6);imshow(imgVPE2);title('combined')


    figure()
    hold on
    scatter(-para_smallscale.recall, -para_smallscale.dice, 'MarkerEdgeAlpha',0.1,'MarkerFaceAlpha',0.1,'MarkerFaceColor','b','MarkerEdgeColor','r');
    scatter(-para_mlp.recall, -para_mlp.dice, 'MarkerEdgeAlpha',0.1,'MarkerFaceAlpha',0.1,'MarkerFaceColor','g','MarkerEdgeColor','r');
    scatter(-para_VPE.recall, -para_VPE.dice, 'MarkerEdgeAlpha',0.1,'MarkerFaceAlpha',0.1,'MarkerFaceColor','r','MarkerEdgeColor','r');
    best_dice = max([best_smallscale_dice, best_mlp_dice, best_VPE_dice]);
    title(sprintf('VPE''s recall-dice (best dice = %.3f)', best_dice))
    legend('small-scale filter', 'GRMLP', 'VPE')


    
... save results
    save_path = fullfile(project_root, 'result', checkpoint_folders(1).name);
    if ~isfolder(save_path)
        mkdir(save_path);
    end
    fprintf('Results will be saved to: %s\n', save_path);

    imgVPE_uint8 = uint8(imgVPE * 255);
    imwrite(imgVPE_uint8, fullfile(save_path, sprintf('%d_imgVPE.png', testindex)));

    imgVPE_bestbinary_uint8 = uint8(imgVPE_bestbinary * 255);
    imwrite(imgVPE_bestbinary_uint8, fullfile(save_path, sprintf('%d_imgVPE_bestbinary_dice%.3f.png', testindex, best_VPE_dice)));
    fprintf('imgVPE best-dice binary saved (dice=%.3f).\n', best_VPE_dice);
    fprintf('======================================================\n')
    fprintf('===================inference finish===================\n')
    fprintf('======================================================\n')
