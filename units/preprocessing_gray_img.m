function double_img = preprocessing_gray_img(img) 
    % 图像灰度化为01区间
    % 20250827
    if max(max(img)) > 1
        double_img = double(img)/255;
        if size(img, 3) == 3
            img = rgb2gray(img);  % 将彩色图像转换为灰度图像
            double_img = double(img) / 255;
        end
        if islogical(img)
            double_img = double(img);  % 将 logical 图像转换为 double 格式
        end
    else
        double_img= double(img);
    end
end