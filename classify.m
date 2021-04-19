%Simple function to return a row vector of 1s and 0s classifying the 
%test data

function [y] = classify(y_hat)
m = mean(y_hat);
y = zeros(1, size(y_hat, 2));

    for i = 1:size(y_hat, 2)
        if y_hat(1, i) >= m
            y(1, i) = 1;
        end
        if y_hat(1, i) < m
            y(1, i) = 0;
        end
    end
    
end
