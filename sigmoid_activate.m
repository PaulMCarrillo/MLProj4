%Function for sigmoid activation function. Using the following formula: 
%1/(1 + exp(-x))

function [y] = sigmoid_activate(x_entries)

for i = 1:size(x_entries, 2)
    ans = 1/(1 + exp(-x_entries(1, i)));
    y(1, i) = ans;
end

end
