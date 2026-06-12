% GRMLP_multiscale_training Train the multiscale GRMLP on a single sample.
%
% Loads ONE normalized training sample (image + mask + target) from
% dataset_normalization/<DATASET>/training/, trains the GRMLP network, and
% saves the trained result + paras to checkpoints/<tag>_<variant>_<dataset>_.../
%
% Dataset, model_variant and paras all come from get_config.m.

clear; clc; close all;

... get setting parameters from get_config.m
    script_dir = fileparts(mfilename('fullpath'));
    addpath(script_dir);
    cfg = get_config();
    paras = cfg.paras.(cfg.model_variant);                   % training parameters for the active variant (from get_config.m)
    training_dir = fullfile(cfg.processed_dir, 'training');  % dataset_normalization/<dataset>/training
    image_dir   = fullfile(training_dir, 'images');          % .../training/images
    mask_dir    = fullfile(training_dir, 'masks');           % .../training/masks
    target_dir  = fullfile(training_dir, 'targets');         % .../training/targets

... check sample count: GRMLP trains on exactly one image + mask + target
    image_files  = dir(fullfile(image_dir,  '*.png'));
    mask_files   = dir(fullfile(mask_dir,   '*.png'));
    target_files = dir(fullfile(target_dir, '*.png'));
    if numel(image_files) ~= 1
        error('Expected exactly 1 image in %s, found %d.', image_dir, numel(image_files));
    end
    if numel(mask_files) ~= 1
        error('Expected exactly 1 mask in %s, found %d.', mask_dir, numel(mask_files));
    end
    if numel(target_files) ~= 1
        error('Expected exactly 1 target in %s, found %d.', target_dir, numel(target_files));
    end
    
... read the training image
    image_path  = fullfile(image_files(1).folder,  image_files(1).name);
    img = preprocessing_gray_img(imread(image_path));
    mask_path   = fullfile(mask_files(1).folder,   mask_files(1).name);
    maskimg_binary = preprocessing_binarize_image(preprocessing_gray_img(imread(mask_path)), 0.5);    % binarize mask at 0.5
    target_path = fullfile(target_files(1).folder,  target_files(1).name);
    targetimg_binary = preprocessing_binarize_image(preprocessing_gray_img(imread(target_path)), 0.5); % binarize target at 0.5
  
    if ~isequal(size(img), size(maskimg_binary), size(targetimg_binary))
        error('Image, mask, and target sizes must match.');
    end

    img = img .* maskimg_binary;   % restrict the image to the RoI (zero out background)
    fprintf('Loaded training image:  %s\n', image_path);
    fprintf('Loaded training mask:   %s\n', mask_path);
    fprintf('Loaded training target: %s\n', target_path);


... training and save results
    
    ... create checkpoint folder
        checkpoint_dir = cfg.checkpoint_dir;    % e.g., 'checkpoints' (defined in get_config.m)
        if ~isfolder(checkpoint_dir)
            mkdir(checkpoint_dir);
        end


    ... checkpoint_name and folder
        time_tag = datestr(now, 'mmddHHMM');
        checkpoint_name = sprintf("%s_%s_%s_maxsigma%d-%d_InitLR%de-3_valpatience%d", ...
            time_tag, cfg.model_variant, cfg.dataset_name, min(paras.maxsigma),max(paras.maxsigma), paras.InitialLearnRate*1000, paras.ValidationPatience);
        checkpoint_folder = fullfile(checkpoint_dir, checkpoint_name); 
        if ~isfolder(checkpoint_folder)         % create folder for this checkpoint if not exist
            mkdir(checkpoint_folder);
        end

        
    for index = 1:length(paras.maxsigma)
        ... training
            para_tmp = paras;
            para_tmp.maxsigma = paras.maxsigma(index);
            result = GRMLP(img,targetimg_binary,maskimg_binary,para_tmp);  % train; result.net holds the trained network
        ... save one by one
            resultfile_name = sprintf("%s_%s_%s_maxsigma%d_InitLR%de-3_valpatience%d", ...
                time_tag, cfg.model_variant, cfg.dataset_name, para_tmp.maxsigma, paras.InitialLearnRate*1000, paras.ValidationPatience);
            checkpoint_path = fullfile(checkpoint_folder, resultfile_name + ".mat");
            save(checkpoint_path, 'result', 'paras', 'image_path', 'mask_path', 'target_path');
            fprintf('Saved GRMLP result maxsigma =%d : %s\n', para_tmp.maxsigma, checkpoint_path);
    end 







