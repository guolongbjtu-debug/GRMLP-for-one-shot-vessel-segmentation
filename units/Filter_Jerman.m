% % debug
% I = imread('2.png');
% I = preprocessing_gray_img(I);
% sigmas = [1:1:6];
% spacing = [1;1];
% tau = 1;
% brightondark = false;
% vesselness = vesselness2D(I, sigmas, spacing, tau, brightondark);
% figure()
% imshow(vesselness)
% fprintf('vesselness函数没问题')
% 
% X = [6,1,0.5,5,10];
% [Jerman_gray,Jerman_binary] = Filter_jerman(X,I);
% figure()
% imshow(Jerman_gray)
% fprintf('vesselness函数没问题')

function  [Jerman_gray,Jerman_binary] = Filter_jerman(X,img)
    beyond.N = X(1);    % sigma范围
    beyond.tau = X(2);  
    beyond.adjust = X(3);
    beyond.binary = X(4)/100;  
    beyond.remove = X(5);  
    
    threshold.adjust = beyond.adjust;   % 在beyond_frangi中用到了
    threshold.binary = beyond.binary;
    threshold.remove = beyond.remove;
    
    N = beyond.N;
    tau = beyond.tau;

    % 加载要处理的图像并且preprocess the input a little bit
    % img = imread('vessel-ori.png');
    Ip = single(img);
    thr = prctile(Ip(Ip(:)>0),1) * 0.9;
    Ip(Ip<=thr) = thr;
    Ip = Ip - min(Ip(:));
    Ip = Ip ./ max(Ip(:));    
        
    % 确定要输入的参数
    % tau = 2;   % 优化参数之一
    % threshold.binary = 0.05;  % 输出图像的二值化参数
    % threshold.remove = 50;    % 输出图像的二值化参数
    
    % 计算Vf输出图和性能
    output_gray = vesselness2D(Ip, [0.5:0.5:N], [1;1], tau, false);
    % 拉伸：将这个狭窄的输入区间 [0, 0.25] 线性拉伸到整个输出范围 [0, 1]。
    output_gray = mat2gray(output_gray);  
    output_gray = imadjust(output_gray, [0 threshold.adjust]);   
    % 阈值法二值化  
    outputimg_binary = preprocessing_binarize_image(output_gray, threshold.binary); 
    outputimg_binary = preprocessing_remove_small_regions(outputimg_binary, threshold.remove);   
    Jerman_gray = output_gray;
    Jerman_binary = outputimg_binary;
    % [img_high_recall_gray,img_high_recall_binary] = Jerman_filter(img,img,beyond.tau,threshold,beyond.N);
end






function vesselness = vesselness2D(I, sigmas, spacing, tau, brightondark)
    % calculates the vesselness probability map (local tubularity) of a 2D
    % input image
    % 
    % vesselness = vesselness2D(I, sigmas, spacing, tau, brightondark)
    % 
    % inputs,
    %   I : 2D image
    %   sigmas : vector of scales on which the vesselness is computed
    %   spacing : input image spacing resolution - during hessian matrix 
    %       computation, the gaussian filter kernel size in each dimension can 
    %       be adjusted to account for different image spacing for different
    %       dimensions 
    %   tau : (between 0.5 and 1) : parameter that controls response uniformity
    %       - lower tau -> more intense output response            
    %   brightondark: (true/false) : are vessels (tubular structures) bright on 
    %       dark background or dark on bright (default for 2D is false)
    %
    % outputs,
    %   vesselness: maximum vesselness response over scales sigmas
    %
    % example:
    %   V = vesselness2D(I, 1:5, [1;1], 1, false);
    %
    % Function written by T. Jerman, University of Ljubljana (October 2014)
    % Based on code by D. Kroon, University of Twente (May 2009)
    
    verbose = 1;
    
    if nargin<5
        brightondark = false; % default mode for 2D is dark vessels compared to the background
    end
    
    I = single(I);
    
    for j = 1:length(sigmas)
        
        if verbose
            %disp(['Current filter scale (sigma): ' num2str(sigmas(j)) ]);
        end
        
        [~, Lambda2] = imageEigenvalues(I,sigmas(j),spacing,brightondark); 
        if brightondark == true
            Lambda2 = -Lambda2;
        end  
        
        % proposed filter at current scale
        Lambda3 = Lambda2;
        
        Lambda_rho = Lambda3;
        Lambda_rho(Lambda3 > 0 & Lambda3 <= tau .* max(Lambda3(:))) = tau .* max(Lambda3(:));
        Lambda_rho(Lambda3 <= 0) = 0;
        response = Lambda2.*Lambda2.*(Lambda_rho-Lambda2).* 27 ./ (Lambda2 + Lambda_rho).^3;    
        
        response(Lambda2 >= Lambda_rho./2 & Lambda_rho > 0) = 1;    
        response(Lambda2 <= 0 | Lambda_rho <= 0) = 0;
        response(~isfinite(response)) = 0;   
        
        %max response over multiple scales
        if(j==1)
            vesselness = response;
        else        
            vesselness = max(vesselness,response);
        end
            
        clear response Lambda2 Lambda3
    end
    
    vesselness = vesselness ./ max(vesselness(:)); % should not be really needed   
    vesselness(vesselness < 1e-2) = 0;
end


function [Lambda1, Lambda2] = imageEigenvalues(I,sigma,spacing,brightondark)
    % calculates the two eigenvalues for each voxel in a volume
    
    % Calculate the 2D hessian
    [Hxx, Hyy, Hxy] = Hessian2D(I,sigma,spacing);
    
    % Correct for scaling
    c=sigma.^2;
    Hxx = c*Hxx; 
    Hxy = c*Hxy;
    Hyy = c*Hyy;
    
    % reduce computation by computing vesselness only where needed
    % S.-F. Yang and C.-H. Cheng, 揊ast computation of Hessian-based
    % enhancement filters for medical images,? Comput. Meth. Prog. Bio., vol.
    % 116, no. 3, pp. 215?225, 2014.
    B1 = - (Hxx+Hyy);
    B2 = Hxx .* Hyy - Hxy.^2;
    
    T = ones(size(B1));
    
    if brightondark == true
        T(B1<0) = 0;
        T(B2==0 & B1 == 0) = 0;
    else
        T(B1>0) = 0;
        T(B2==0 & B1 == 0) = 0;
    end
    
    clear B1 B2;
    
    indeces = find(T==1);
    
    Hxx = Hxx(indeces);
    Hyy = Hyy(indeces);
    Hxy = Hxy(indeces);
    
    % Calculate eigen values
    [Lambda1i,Lambda2i]=eigvalOfHessian2D(Hxx,Hxy,Hyy);
    
    clear Hxx Hyy Hxy;
    
    Lambda1 = zeros(size(T));
    Lambda2 = zeros(size(T));
    
    Lambda1(indeces) = Lambda1i;
    Lambda2(indeces) = Lambda2i;
    
    % some noise removal
    Lambda1(~isfinite(Lambda1)) = 0;
    Lambda2(~isfinite(Lambda2)) = 0;
    
    Lambda1(abs(Lambda1) < 1e-4) = 0;
    Lambda2(abs(Lambda2) < 1e-4) = 0;
end

function [Dxx, Dyy, Dxy] = Hessian2D(I,Sigma,spacing)
    %  filters the image with an Gaussian kernel
    %  followed by calculation of 2nd order gradients, which aprroximates the
    %  2nd order derivatives of the image.
    % 
    % [Dxx, Dyy, Dxy] = Hessian2D(I,Sigma,spacing)
    % 
    % inputs,
    %   I : The image, class preferable double or single
    %   Sigma : The sigma of the gaussian kernel used. If sigma is zero
    %           no gaussian filtering.
    %   spacing : input image spacing
    %
    % outputs,
    %   Dxx, Dyy, Dxy: The 2nd derivatives
    
    if nargin < 3, Sigma = 1; end
    
    if(Sigma>0)
        F=imgaussian(I,Sigma,spacing);
    else
        F=I;
    end
    
    % Create first and second order diferentiations
    Dy=gradient2(F,'y');
    Dyy=(gradient2(Dy,'y'));
    clear Dy;
    
    Dx=gradient2(F,'x');
    Dxx=(gradient2(Dx,'x'));
    Dxy=(gradient2(Dx,'y'));
    clear Dx;
end

function D = gradient2(F,option)
    % Example:
    %
    % Fx = gradient2(F,'x');
    
    [k,l] = size(F);
    D  = zeros(size(F),class(F)); 
    
    switch lower(option)
    case 'x'
        % Take forward differences on left and right edges
        D(1,:) = (F(2,:) - F(1,:));
        D(k,:) = (F(k,:) - F(k-1,:));
        % Take centered differences on interior points
        D(2:k-1,:) = (F(3:k,:)-F(1:k-2,:))/2;
    case 'y'
        D(:,1) = (F(:,2) - F(:,1));
        D(:,l) = (F(:,l) - F(:,l-1));
        D(:,2:l-1) = (F(:,3:l)-F(:,1:l-2))/2;
    otherwise
        disp('Unknown option')
    end
end
function I=imgaussian(I,sigma,spacing,siz)
    % IMGAUSSIAN filters an 1D, 2D color/greyscale or 3D image with an 
    % Gaussian filter. This function uses for filtering IMFILTER or if 
    % compiled the fast  mex code imgaussian.c . Instead of using a 
    % multidimensional gaussian kernel, it uses the fact that a Gaussian 
    % filter can be separated in 1D gaussian kernels.
    %
    % J=IMGAUSSIAN(I,SIGMA,SIZE)
    %
    % inputs,
    %   I: 2D input image
    %   SIGMA: The sigma used for the Gaussian kernel
    %   SPACING: input image spacing
    %   SIZ: Kernel size (single value) (default: sigma*6)
    % 
    % outputs,
    %   I: The gaussian filtered image
    %
    
    if(~exist('siz','var')), siz=sigma*6; end
    
    if(sigma>0)
    
        % Filter each dimension with the 1D Gaussian kernels\
        x=-ceil(siz/spacing(1)/2):ceil(siz/spacing(1)/2);
        H = exp(-(x.^2/(2*(sigma/spacing(1))^2)));
        H = H/sum(H(:));    
        Hx=reshape(H,[length(H) 1]);
        
        x=-ceil(siz/spacing(2)/2):ceil(siz/spacing(2)/2);
        H = exp(-(x.^2/(2*(sigma/spacing(2))^2)));
        H = H/sum(H(:));    
        Hy=reshape(H,[1 length(H)]);
        
        I=imfilter(imfilter(I,Hx, 'same' ,'replicate'),Hy, 'same' ,'replicate');
    end
end

function [Lambda1,Lambda2]=eigvalOfHessian2D(Dxx,Dxy,Dyy)
    % This function calculates the eigen values from the
    % hessian matrix, sorted by abs value
    
    % Compute the eigenvectors of J, v1 and v2
    tmp = sqrt((Dxx - Dyy).^2 + 4*Dxy.^2);
    
    % Compute the eigenvalues
    mu1 = 0.5*(Dxx + Dyy + tmp);
    mu2 = 0.5*(Dxx + Dyy - tmp);
    
    % Sort eigen values by absolute value abs(Lambda1)<abs(Lambda2)
    check=abs(mu1)>abs(mu2);
    
    Lambda1=mu1; Lambda1(check)=mu2(check);
    Lambda2=mu2; Lambda2(check)=mu1(check);
end