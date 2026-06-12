% % debug area
% %...初始灰度图像准备
% clear,clc
%     img = imread('2.png');             % 原图
%     img = preprocessing_gray_img(img);
% 
% x = [8,eps];
% outimg_gray = Filter_Guo(img,x);
% figure()
% imshow(outimg_gray)
% fprintf('Filter_Guo没问题');



function outimg_gray = Filter_Guo(img,X)
    N = X(1);
    gamaa = X(2);
    
    %...second-class parameters
    sigmas = [1:1:N];
    %...run frangi
    img_gray = preprocessing_gray_img(img);  
    for i = 1:length(sigmas)  
        sigma = sigmas(i);
        [dxx,dxy,dyy,X,Y] = hessian_img2dxx(img_gray,sigma); %dxx
        [Dxx,Dxy,Dyy] = hessian_dxx2Dxx(dxx,dxy,dyy,sigma);  %Dxx
        [Lambda1,Lambda2,v1x,v1y,v2x,v2y,angle1,angle2] = hessian_Dxx2eigenvalue(Dxx,Dxy,Dyy); % lambda
        Dxx_all(:,:,i) = dxx;
        Dxy_all(:,:,i) = dxy;
        Dyy_all(:,:,i) = dyy;
        Lambda1_all(:,:,i) = Lambda1;
        Lambda2_all(:,:,i) = Lambda2;
    end
    % fliter
    for i = 1:size(Lambda1_all,3)
        lambda1 = Lambda1_all(:,:,i);
        lambda2 = Lambda2_all(:,:,i);
        lambda2(lambda2==0) = eps; % 计算血管函数Ifiltered
        % Rb = abs(lambda1./lambda2);
        % S = sqrt(lambda1.^2  + lambda2.^2);
        C = (abs(lambda2) - abs(lambda1))/(abs(lambda2) + abs(lambda1));
        Ifiltered = ones(size(lambda1))-exp(-C.^2/(2*gamaa^2));
        Ifiltered(lambda2<0)=0;   % 黑背景还是白背景
        ALLfiltered(:,:,i) = Ifiltered;
    end
    outimg_gray = max(ALLfiltered,[],3);
    outimg_gray = reshape(outimg_gray,size(lambda1));
    outimg_gray = mat2gray(outimg_gray);
end
