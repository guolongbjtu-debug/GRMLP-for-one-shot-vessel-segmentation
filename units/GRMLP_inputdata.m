function [X_Lambda, X_neighborhood, useDS, x, y] = GRMLP_inputdata(img, maxsigma, X_neighborhood_size)
%GRMLP_INPUTDATA Extract Hessian eigenvalue and neighborhood features for GRMLP inference.
%   img       - input grayscale image
%   maxsigma  - maximum sigma for multi-scale Hessian computation
%   X_neighborhood_size - window size for neighborhood feature extraction
%   X_Lambda  - normalized eigenvalue feature matrix (N-by-2*maxsigma)
%   X_neighborhood - normalized neighborhood feature matrix (N-by-K)
%   useDS     - combined datastore ready for trainNetwork / predict
%   x, y      - spatial dimensions of the original image

    % preallocate arrays
    [h, w] = size(img); % h = image height, w = image width
    Dxx_all = zeros(h, w, maxsigma);    % second-order partial derivatives dxx
    Dxy_all = zeros(h, w, maxsigma);    % second-order partial derivatives dxy
    Dyy_all = zeros(h, w, maxsigma);    % second-order partial derivatives dyy
    Lambda1_all = zeros(h, w, maxsigma); % first eigenvalue maps
    Lambda2_all = zeros(h, w, maxsigma); % second eigenvalue maps

    for i = 1:maxsigma % loop over sigma scales
        sigma = i;
        [dxx, dxy, dyy, ~, ~] = hessian_img2dxx(img, sigma);
        [Dxx, Dxy, Dyy] = hessian_dxx2Dxx(dxx, dxy, dyy, sigma);
        [Lambda1, Lambda2, ~, ~, ~, ~, ~, ~] = hessian_Dxx2eigenvalue(Dxx, Dxy, Dyy);
        Dxx_all(:,:,i) = dxx;
        Dxy_all(:,:,i) = dxy;
        Dyy_all(:,:,i) = dyy;
        Lambda1_all(:,:,i) = Lambda1;
        Lambda2_all(:,:,i) = Lambda2;
    end
    Lambda = cat(3, Lambda1_all, Lambda2_all); % concat eigenvalue maps
    [x, y, z] = size(Lambda);
    X_Lambda = reshape(Lambda, [x*y, z]); % flatten to N-by-(2*maxsigma)

    % neighborhood features
    X_neighborhood = extract_neighborhood(img, X_neighborhood_size);

    % z-score normalization
    mu_XLambda = mean(X_Lambda, 1);
    sigma_XLambda = std(X_Lambda, 0, 1);
    mu_XNeigh = mean(X_neighborhood, 1);
    sigma_XNeigh = std(X_neighborhood, 0, 1);
    X_Lambda = (X_Lambda - mu_XLambda) ./ (sigma_XLambda + eps);
    X_neighborhood = (X_neighborhood - mu_XNeigh) ./ (sigma_XNeigh + eps);

    % build multi-input datastore for trainNetwork / predict
    dsX1 = arrayDatastore(X_Lambda, 'IterationDimension', 1);
    dsX2 = arrayDatastore(X_neighborhood, 'IterationDimension', 1);
    dsCombined = combine(dsX1, dsX2);
    useDS = transform(dsCombined, @(data) formatMultiInput(data));
end


function dataOut = formatMultiInput(data)
%FORMATMULTIINPUT Reshape combined datastore output into column vectors for multi-input network.
%   data    - 1x2 cell array {X1row, X2row} from combined datastore
%   dataOut - 1x2 cell array {X1col, X2col} with transposed column vectors

    X1 = data{1};
    X2 = data{2};
    dataOut = {X1', X2'};
end
