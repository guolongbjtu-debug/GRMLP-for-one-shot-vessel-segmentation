function outimg_gray = hessian_eig2vf(Lambda1_all,Lambda2_all,A,B)
    % beta = 0.5，对应A=2;
    % c = 15，对应B=1/450
    for i = 1:size(Lambda1_all,3)
        lambda1 = Lambda1_all(:,:,i);
        lambda2 = Lambda2_all(:,:,i);
        lambda2(lambda2==0) = eps; % 计算血管函数Ifiltered
        Rb = abs(lambda1./lambda2);
        S = sqrt(lambda1.^2  + lambda2.^2);

        Ifiltered = exp(-A*Rb.^2) .*(ones(size(lambda1))-exp(-B*S.^2));
        Ifiltered(lambda2<0)=0;   % 黑背景还是白背景
        ALLfiltered(:,:,i) = Ifiltered;
    end
    outimg_gray = max(ALLfiltered,[],3);
    outimg_gray = reshape(outimg_gray,size(lambda1));
    outimg_gray = mat2gray(outimg_gray);
end