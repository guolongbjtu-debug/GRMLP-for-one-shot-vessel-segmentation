
function result = GRMLP(img, targetimg_binary, maskimg_binary, paras)
% GRMLP Train the GRMLP network on a single vessel image.
%   result = GRMLP(img, targetimg_binary, maskimg_binary, paras)
%
%   Inputs:
%       img              - grayscale vessel image (masked by RoI)
%       targetimg_binary - binary ground-truth annotation
%       maskimg_binary   - binary RoI mask
%       paras            - struct with fields:
%             .maxsigma, .InitialLearnRate, .MaxEpochs, .MiniBatchSize,
%             .ValidationPatience, .ValidationFrequency, .neighborhood_size
%
%   Output:
%       result.net           - trained network
%       result.info          - training info struct
%       result.mu_XLambda    - mean of Lambda features (for inference normalization)
%       result.sigma_XLambda - std  of Lambda features
%       result.mu_XNeigh     - mean of neighborhood features
%       result.sigma_XNeigh  - std  of neighborhood features

    %...======== Extract features ========
        %... Hessian eigenvalue features
            sigmas = 1:1:paras.maxsigma;
            for i = 1:length(sigmas)
                sigma = sigmas(i);
                [dxx, dxy, dyy, ~, ~] = hessian_img2dxx(img, sigma);
                [Dxx, Dxy, Dyy] = hessian_dxx2Dxx(dxx, dxy, dyy, sigma);
                [Lambda1, Lambda2, ~, ~, ~, ~, ~, ~] = hessian_Dxx2eigenvalue(Dxx, Dxy, Dyy);
                Dxx_all(:,:,i) = dxx;
                Dxy_all(:,:,i) = dxy;
                Dyy_all(:,:,i) = dyy;
                Lambda1_all(:,:,i) = Lambda1;
                Lambda2_all(:,:,i) = Lambda2;
            end

            Lambda = cat(3, Lambda1_all, Lambda2_all);              % e.g. 584x565x20
            Lambda_2d  = reshape(Lambda, [], 2*paras.maxsigma);     % Nx(2*maxsigma), Lambda flattened
            mask_1d = reshape(maskimg_binary, [], 1);                % Nx1, mask flattened
            X_Lambda = Lambda_2d(mask_1d, :);                       % only keep pixels inside mask

        %... Neighborhood features
            X_neighborhood = extract_neighborhood(img, paras.neighborhood_size);   % Nx49
            X_neighborhood = X_neighborhood(mask_1d, :);     % only keep pixels inside mask

        %... Labels
            Y = reshape(targetimg_binary, [], 1);            % Nx1
            Y = double(Y);                                   % sigmoid requires double
            Y = Y(mask_1d, :);

        %... Z-score standardization
            mu_XLambda = mean(X_Lambda, 1);
            sigma_XLambda = std(X_Lambda, 0, 1);

            mu_XNeigh = mean(X_neighborhood, 1);
            sigma_XNeigh = std(X_neighborhood, 0, 1);

            X_Lambda = (X_Lambda - mu_XLambda) ./ (sigma_XLambda + eps);
            X_neighborhood = (X_neighborhood - mu_XNeigh) ./ (sigma_XNeigh + eps);

        %... Train / validation split
            numSamples = size(X_Lambda, 1);
            idx = randperm(numSamples);
            trainRatio = 0.8;
            numTrain = round(trainRatio * numSamples);
            trainIdx = idx(1:numTrain);
            testIdx  = idx(numTrain+1:end);

            X_train_input1 = X_Lambda(trainIdx, :);
            X_train_input2 = X_neighborhood(trainIdx, :);
            Y_train = Y(trainIdx, :);

            X_test_input1 = X_Lambda(testIdx, :);
            X_test_input2 = X_neighborhood(testIdx, :);
            Y_test = Y(testIdx, :);

        %... Convert data to trainNetwork format
            dsX1 = arrayDatastore(X_train_input1, 'IterationDimension', 1);
            dsX2 = arrayDatastore(X_train_input2, 'IterationDimension', 1);
            dsY  = arrayDatastore(Y_train, 'IterationDimension', 1);
            dsCombined = combine(dsX1, dsX2, dsY);
            trainDS = transform(dsCombined, @(data) formatMultiInput(data));

            dsX1 = arrayDatastore(X_test_input1, 'IterationDimension', 1);
            dsX2 = arrayDatastore(X_test_input2, 'IterationDimension', 1);
            dsY  = arrayDatastore(Y_test, 'IterationDimension', 1);
            dsCombined = combine(dsX1, dsX2, dsY);
            testDS = transform(dsCombined, @(data) formatMultiInput(data));

        lambdaDim = size(X_Lambda, 2);
        neighDim  = size(X_neighborhood, 2);

    %... Lambda branch (coarse Hessian features)
        lambdaBranch = [
            featureInputLayer(lambdaDim, Name="lambda_input")

            fullyConnectedLayer(64, Name="lambda_fc1")
            batchNormalizationLayer(Name="lambda_bn1")
            reluLayer(Name="lambda_relu1")
            dropoutLayer(0.2, Name="lambda_drop1")      % 3 FC layers with Dropout for regularization

            fullyConnectedLayer(64, Name="lambda_fc2")
            batchNormalizationLayer(Name="lambda_bn2")
            reluLayer(Name="lambda_relu2")
            dropoutLayer(0.2, Name="lambda_drop2")

            fullyConnectedLayer(64, Name="lambda_fc3")
            batchNormalizationLayer(Name="lambda_bn3")
            reluLayer(Name="lambda_relu3")
            ];

    %... Neighborhood branch (fine-grained "finger" and "tail" vessel details)
        neighBranch = [
            featureInputLayer(neighDim, Name="neigh_input")

            fullyConnectedLayer(128, Name="neigh_fc1")
            batchNormalizationLayer(Name="neigh_bn1")
            reluLayer(Name="neigh_relu1")
            dropoutLayer(0.1, Name="neigh_drop1")       % lighter Dropout to preserve fine-grained features

            fullyConnectedLayer(64, Name="neigh_fc2")
            batchNormalizationLayer(Name="neigh_bn2")
            reluLayer(Name="neigh_relu2")
            ];

    %... Main branch (fusion + classification)
        mainBranch = [
            fullyConnectedLayer(128, Name="main_fc1")
            batchNormalizationLayer(Name="bn1")
            reluLayer(Name="relu1")
            dropoutLayer(0.3, Name="main_drop1")        % heavy dropout for regularization

            fullyConnectedLayer(64, Name="main_fc2")
            batchNormalizationLayer(Name="bn2")
            reluLayer(Name="relu2")
            dropoutLayer(0.2, Name="main_drop2")        % moderate dropout

            fullyConnectedLayer(32, Name="main_fc3")
            batchNormalizationLayer(Name="bn3")
            reluLayer(Name="relu3")
            dropoutLayer(0.1, Name="main_drop3")        % light dropout

            fullyConnectedLayer(1, Name="main_out")
            sigmoidLayer(Name="sigmoid")
            regressionLayer('Name', 'regOutput')
            ];

    %... Attention fusion block
        attentionFusionBlock = [
            % ---- Gate-weight generation: neighborhood guides attention ----
            % input: neigh_relu2
            % 49-dim input through bottleneck to 32-dim for compact representation
            fullyConnectedLayer(32, Name="attn_gate_fc1")   % 32-dim bottleneck to discard noise, keeping key attention features
            batchNormalizationLayer(Name="attn_gate_bn1")   % BN to stabilize training
            reluLayer(Name="attn_gate_relu")
            dropoutLayer(0.1, Name="attn_gate_drop")

            fullyConnectedLayer(64, Name="attn_gate_fc2")   % project back to 64-dim to match lambda branch (multiplicationLayer requires same dims)
            sigmoidLayer(Name="attn_gate_sigmoid")          % gate weight range [0,1]

            % ---- Apply gate weight ----
            % inputs: attn_gate_sigmoid (weight) and lambda_relu3 (features)
            multiplicationLayer(2, Name="attn_weighted_lambda")

            % ---- Residual addition: weighted + original ----
            additionLayer(2, Name="attn_residual")

            % ---- Concatenate with neighborhood branch ----
            concatenationLayer(1, 2, Name="attn_concat")    % output 128-dim (64+64)

            % ---- Layer normalization to stabilize training ----
            layerNormalizationLayer(Name="attn_layernorm")
        ];

    %... Build network graph
        lgraph = layerGraph();
        lgraph = addLayers(lgraph, lambdaBranch);
        lgraph = addLayers(lgraph, neighBranch);
        lgraph = addLayers(lgraph, attentionFusionBlock);
        lgraph = addLayers(lgraph, mainBranch);

    %... Cross-branch / skip connections
        lgraph = connectLayers(lgraph, "lambda_relu3", "attn_weighted_lambda/in2");
        lgraph = connectLayers(lgraph, "neigh_relu2", "attn_gate_fc1");              % path 1: generate gate
        lgraph = connectLayers(lgraph, "neigh_relu2", "attn_concat/in2");            % path 2: direct concatenation
        lgraph = connectLayers(lgraph, "lambda_relu3", "attn_residual/in2");         % residual connection to preserve original lambda info
        lgraph = connectLayers(lgraph, "attn_layernorm", "main_fc1");                % connect fusion output to main branch

    %... Training options
        options = trainingOptions("adam", ...
            'MaxEpochs', paras.MaxEpochs, ...
            'MiniBatchSize', paras.MiniBatchSize, ...
            'InitialLearnRate', paras.InitialLearnRate, ...
            'LearnRateSchedule', 'piecewise', ...              % piecewise learning rate decay
            'LearnRateDropPeriod', 15, ...                     % drop every 15 epochs
            'LearnRateDropFactor', 0.7, ...                    % learning rate drop factor
            'L2Regularization', 3e-5, ...                      % weight decay to prevent overfitting
            'Shuffle', 'every-epoch', ...
            'ValidationData', testDS, ...
            'ValidationFrequency', paras.ValidationFrequency, ...
            'ValidationPatience', paras.ValidationPatience, ... % early stopping patience
            'Verbose', true, ...
            'Plots', 'training-progress');

        [net, info] = trainNetwork(trainDS, lgraph, options);

        result.net = net;
        result.info = info;
        result.mu_XLambda = mu_XLambda;
        result.sigma_XLambda = sigma_XLambda;
        result.mu_XNeigh = mu_XNeigh;
        result.sigma_XNeigh = sigma_XNeigh;

end




function dataOut = formatMultiInput(data)
%FORMATMULTIINPUT Reshape combined datastore output for multi-input network.
%   data    - 1x3 cell array {X1row, X2row, Yrow}
%   dataOut - 1x3 cell array {X1col, X2col, Ycol} (column vectors)

    X1 = data{1};
    X2 = data{2};
    Y  = data{3};
    dataOut = {X1', X2', Y'};
end
