


function cfg = get_config()
% GET_CONFIG Shared configuration for the pipeline scripts.
%   
%   prepare_dataset / GRMLP_multiscale_training / GRMLP_multiscale_inference all call this, so the dataset
%   selection and training parameters stay consistent across the whole pipeline.
%
%   Edit cfg.dataset_name below to choose the dataset.
%
%   Raw input lives at  dataset/<dataset_name>/{training,test}/...
%   Normalized output at dataset_normalization/<dataset_name>/{training,test}/...
%
%   adds units/ and GRMLP_Architecture/ to the MATLAB path, so callers do not need to addpath them separately.

... ===== dataset selection (edit this) =====
    cfg.dataset_name = 'DRIVE_enhenced';
    

... ===== training parameters (edit this) =====
    cfg.paras.multiscale.maxsigma            = [3,4,5,6,7];     % [5], [3,5], [3,5,7] also work reasonably well
    cfg.paras.multiscale.InitialLearnRate    = 7e-3;            % needs to find a balance between convergence and stability
    cfg.paras.multiscale.ValidationPatience  = 15;              % 15
    cfg.paras.multiscale.ValidationFrequency = 10;              % 10
    cfg.paras.multiscale.MaxEpochs           = 30;
    cfg.paras.multiscale.MiniBatchSize       = 128;
    cfg.paras.multiscale.neighborhood_size   = 7;               % do not change neighborhood side



... ===== inference parameters (edit this) =====
    cfg.testindex = 2;              % which test sample to run (1-based position in the sorted test list)
    ... For easier per-image inspection, only one image is processed at a time during inference.
    cfg.testtime  = '06121122';     % use the time tag in the checkpoint name, e.g., '06120952' for June 12, 9:52am
    cfg.testtime  = 'none';         % 'none' to use latest checkpoint



... ===== derived paths (do not edit) =====
    cfg.model_variant = 'multiscale';   
    script_dir = fileparts(mfilename('fullpath'));        % .../scripts
    cfg.project_root   = fileparts(script_dir);
    cfg.raw_dir        = fullfile(cfg.project_root, 'dataset',              cfg.dataset_name);
    cfg.processed_dir  = fullfile(cfg.project_root, 'dataset_normalization', cfg.dataset_name);
    cfg.units_dir      = fullfile(cfg.project_root, 'units');
    cfg.grmlp_dir      = fullfile(cfg.project_root, 'GRMLP_Architecture');
    cfg.checkpoint_dir = fullfile(cfg.project_root, 'checkpoints');
    
    restoredefaultpath;  % clear any previously added paths to avoid conflicts
    addpath(cfg.units_dir);
    addpath(cfg.grmlp_dir);
    addpath(script_dir);
    pathtest
end
