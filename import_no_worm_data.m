function [no_worm_data] = import_no_worm_data()

%Create image value matrix
no_worm_data = zeros(10201, 4);

for i = 1:5480
    
    %Import file names iterativley and store image pixel values in cval 
    %This file path is exclusive to my computer!
    filename = sprintf('%s_%d.%s','C:\Users\Paul\Desktop\MLProj4\worm_images\0 (no worm)\image', i ,'png');
    [cval] = imread(filename);
    
    %Transpose cval and turn it into a column vector
    cval_t = cval';
    re_cval = cval_t(:); 
    
    %Store images column vector into a matrix
    no_worm_data(:, i) = re_cval;
end

end

