function [frangi_gray,frangi_binary] = Filter_frangi(X,img)
    frangi.N = X(1);
    frangi.A = X(2);
    frangi.B_fenzhi1 = X(3);    % 这个参数好像不太对
    frangi.adjust = X(4);
    frangi.binary =  X(5)/1000;
    frangi.remove = X(6);
    
    N = frangi.N;    
    A = frangi.A;
    B_fenzhi1 = frangi.B_fenzhi1;
    threshold2.adjust = frangi.adjust;
    threshold2.binary = frangi.binary;  
    threshold2.remove = frangi.remove;  

    %...second-class parameters
    sigmas = [1:1:N];
    B = 1/B_fenzhi1;
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
    % frangi
    frangi_gray = hessian_eig2vf_frangi(Lambda1_all,Lambda2_all,A,B);       % vf
    frangi_gray = mat2gray(frangi_gray);                             % imadjust
    frangi_gray = imadjust(frangi_gray, [0 threshold2.adjust]);   
    frangi_binary = preprocessing_binarize_image(frangi_gray, threshold2.binary); 
    frangi_binary = preprocessing_remove_small_regions(frangi_binary, threshold2.remove);
end