% PREPARE_DATASET Normalize a raw dataset into PNGs for GRMLP training/inference.
%
% Raw input:   dataset/<DATASET>/{training,test}/{image,mask,target}/...
% Normalized:  dataset_normalization/<DATASET>/{training,test}/{images,masks,targets}/...
%   - images:  grayscale, resized to 512x512, values in [0, 1]
%   - masks:   binary {0, 1}
%   - targets: binary {0, 1}
%
% <DATASET> = cfg.dataset_name in get_config.m (edit there to switch dataset).
%
% A file's category is decided from its folder/file name:
%   contains "mask"                                     -> masks/
%   contains "target"/"manual"/"label"/"annotation"     -> targets/
%   otherwise                                            -> images/
%
% Steps (see ... sections below):
%   load settings -> select dataset -> discover files -> process each -> summarize

... get setting parameters from get_config.m
    clear; clc; close all
    restoredefaultpath;
    
    script_dir = fileparts(mfilename('fullpath'));
    addpath(script_dir);
    cfg = get_config();
    raw_dir       = cfg.raw_dir;       % dataset/<dataset_name>
    processed_dir = cfg.processed_dir; % dataset_normalization/<dataset_name>
    image_exts = {'.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp', '.gif'};

... select and validate dataset
    raw_root = fullfile(cfg.project_root, 'dataset');
    ds = dir(raw_root); ds = ds([ds.isdir]);
    ds_names = setdiff({ds.name}, {'.', '..'});
    fprintf('Available datasets: %s\n', strjoin(ds_names, ', '));
    if isempty(ds_names)
        error('No dataset folders found under %s.\nExpected dataset/<name>/{training,test}/...', raw_root);
    end
    if ~ismember(cfg.dataset_name, ds_names)
        error('Selected dataset "%s" not found under %s.\nAvailable: %s', ...
            cfg.dataset_name, raw_root, strjoin(ds_names, ', '));
    end
    fprintf('Selected dataset: %s\n', cfg.dataset_name);

... discover raw files and reset counters
    files = dir(fullfile(raw_dir, '**', '*'));
    counts = struct();                 % counts.<split>.<category>, plus counts.failed
    counts.training.images  = 0;
    counts.training.masks   = 0;
    counts.training.targets = 0;
    counts.test.images      = 0;
    counts.test.masks       = 0;
    counts.test.targets     = 0;
    counts.failed           = 0;
    fprintf('Preparing dataset\n');
    fprintf('  Raw:       %s\n', raw_dir);
    fprintf('  Processed: %s\n\n', processed_dir);

... process each file: classify -> normalize -> write PNG
    for i = 1:numel(files)
        if files(i).isdir
            continue;                  % skip subfolders
        end

        source_path = fullfile(files(i).folder, files(i).name);
        [~, base_name, ext] = fileparts(source_path);

        if ~any(strcmpi(ext, image_exts))
            continue;                  % skip non-image files
        end

        % resolve where this file belongs: split (training/test) + category (images/masks/targets)
        rel_folder = get_relative_folder(files(i).folder, raw_dir);
        [split_name, split_rel_folder] = get_dataset_split(rel_folder);
        category = classify_dataset_item(split_rel_folder, base_name);
        output_subdir = strip_category_folder(split_rel_folder);

        % build output path: processed/<split>/<category>/<subdir>/<name>.png
        output_dir = fullfile(processed_dir, split_name, category, output_subdir);
        if ~isfolder(output_dir)
            mkdir(output_dir);
        end
        output_path = fullfile(output_dir, [base_name, '.png']);

        try
            img = read_image_as_gray01(source_path);
            img = imresize(img, [512, 512]);

            % normalize: keep grayscale for images, binarize masks/targets at 0.5
            switch category
                case 'images'
                    out_img = img;
                case {'masks', 'targets'}
                    out_img = img >= 0.5;
                otherwise
                    error('Unknown category: %s', category);
            end

            % validate split, then tally this file under counts.<split>.<category>
            if ~ismember(split_name, {'training', 'test'})
                error('Unknown dataset split: %s', split_name);
            end
            counts.(split_name).(category) = counts.(split_name).(category) + 1;

            imwrite(out_img, output_path);
            fprintf('[%s/%s] %s -> %s\n', split_name, category, source_path, output_path);
        catch ME
            counts.failed = counts.failed + 1;
            warning('Failed to process %s: %s', source_path, ME.message);
        end
    end

... print summary and check for failures
    fprintf('\nDone.\n');
    fprintf('  training image number:      %d\n', counts.training.images);
    fprintf('  training mask number:       %d\n', counts.training.masks);
    fprintf('  training target number:     %d\n', counts.training.targets);
    fprintf('  test image number:          %d\n', counts.test.images);
    fprintf('  test mask number:           %d\n', counts.test.masks);
    fprintf('  test target number:         %d\n', counts.test.targets);
    fprintf('  failed number:              %d\n', counts.failed);

    if counts.failed > 0
        error('Dataset preparation finished with %d failed file(s).', counts.failed);
    end



% =========================================================================
% local helper functions
% =========================================================================

function img = read_image_as_gray01(path)
% Read an image (indexed or RGB) and convert to grayscale double in [0, 1].
    [raw_img, cmap] = imread(path);

    if ~isempty(cmap)
        img = ind2gray(raw_img, cmap);
    else
        img = raw_img;
    end

    if ndims(img) == 3
        img = rgb2gray(img);
    end

    img = im2double(img);
    img = min(max(img, 0), 1);
end

function category = classify_dataset_item(rel_folder, base_name)
% Classify a file as 'images' / 'masks' / 'targets' from its folder + name.
    text = lower([rel_folder, filesep, base_name]);

    if contains(text, 'mask')
        category = 'masks';
    elseif contains(text, 'target') || contains(text, 'manual') || ...
            contains(text, 'label') || contains(text, 'annotation')
        category = 'targets';
    else
        category = 'images';
    end
end

function [split_name, remaining_folder] = get_dataset_split(rel_folder)
% Pick the split ('training'/'test') from the first folder segment; return the rest.
    parts = split_path(rel_folder);

    if isempty(parts)
        error('Raw images must be under dataset/<name>/training or dataset/<name>/test.');
    end

    first_part = lower(parts{1});

    if any(strcmp(first_part, {'training', 'train'}))
        split_name = 'training';
    elseif any(strcmp(first_part, {'test', 'testing'}))
        split_name = 'test';
    else
        error('Raw images must be under dataset/<name>/training or dataset/<name>/test. Found: %s', rel_folder);
    end

    if isscalar(parts)
        remaining_folder = '';
    else
        remaining_folder = fullfile(parts{2:end});
    end
end

function rel_folder = get_relative_folder(folder_path, root_path)
% Strip the root prefix (and any leading separator) to get a path relative to raw_dir.
    rel_folder = folder_path;

    if startsWith(rel_folder, root_path)
        rel_folder = rel_folder(numel(root_path) + 1:end);
    end

    if startsWith(rel_folder, filesep)
        rel_folder = rel_folder(2:end);
    end
end

function output_subdir = strip_category_folder(rel_folder)
% Drop folder segments up to & including the first category-named folder
% (image/mask/target/etc.); keep whatever follows as the output subdir.
    if isempty(rel_folder)
        output_subdir = '';
        return;
    end

    parts = split_path(rel_folder);
    lower_parts = lower(parts);

    category_words = {'image', 'img', 'enhanced', 'enhenced', ...
                      'mask', 'target', 'manual', 'label', 'annotation'};
    is_category_seg = false(size(lower_parts));
    for k = 1:numel(category_words)
        is_category_seg = is_category_seg | contains(lower_parts, category_words{k});
    end
    category_idx = find(is_category_seg, 1, 'first');

    if isempty(category_idx)
        kept_parts = parts;                       % no category folder -> keep all segments
    else
        kept_parts = parts(category_idx + 1:end); % drop up to first category folder
    end

    if isempty(kept_parts)
        output_subdir = '';
    else
        output_subdir = fullfile(kept_parts{:});
    end
end

function parts = split_path(path_text)
% Split a path into its non-empty folder components (cell array of char vectors).
    if isempty(path_text)
        parts = {};
        return;
    end

    parts = strsplit(path_text, filesep);
    parts = parts(~cellfun('isempty', parts));
end
