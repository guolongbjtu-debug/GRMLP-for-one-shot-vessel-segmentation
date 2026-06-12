function [dxx,dxy,dyy,X,Y] = hessian_img2dxx(img,Sigma)
    %  This function Hessian2 Filters the image with 2nd derivatives of a 
    %  Gaussian with parameter Sigma.
    % 
    % [Dxx,Dxy,Dyy] = Hessian2(I,Sigma);
    % 
    % inputs,
    %   I : The image, class preferable double or single
    %   Sigma : The sigma of the gaussian kernel used
    %
    % outputs,
    %   dxx, dxy, dyy: The 2nd derivatives
    %
    % example,
    %   I = im2double(imread('moon.tif'));
    %   [Dxx,Dxy,Dyy] = Hessian2(I,2);
    %   figure, imshow(Dxx,[]);

    if nargin < 2, Sigma = 1; end
    % Make kernel coordinates
    [Y,X] = ndgrid(-round(3*Sigma):round(3*Sigma));   % 3sigma原则，已经改了x和y的顺序

    % Build the gaussian 2nd derivatives filters
    DGaussxx = 1/(2*pi*Sigma^4) * (X.^2/Sigma^2 - 1) .* exp(-(X.^2 + Y.^2)/(2*Sigma^2));
    DGaussxy = 1/(2*pi*Sigma^6) * (X .* Y)           .* exp(-(X.^2 + Y.^2)/(2*Sigma^2));
    DGaussyy = DGaussxx';

    dxx = imfilter(img,DGaussxx,'conv');
    dxy = imfilter(img,DGaussxy,'conv');
    dyy = imfilter(img,DGaussyy,'conv');
end