%Function to take 101x101 pixel worm pictures and convert them to
%usable data.
function [worm_data] = import_worm_data(num_of_pics)

%Create image value matrix
worm_data = zeros(10201, num_of_pics);

for i = 1:num_of_pics
    
    %Import file names iterativley and store image pixel values in cval 
    %This file path is exclusive to my computer!
    filename = sprintf('%s_%d.%s','C:\Users\Paul\Desktop\MLProj4\worm_images\1 (worm)\image', i ,'png');
    [cval] = imread(filename);
    
    %Transpose cval and turn it into a column vector
    cval_t = cval';
    re_cval = cval_t(:); 
    
    %Store images column vector into a matrix
    worm_data(:, i) = re_cval;
end

end
