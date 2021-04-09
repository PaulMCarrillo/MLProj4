clear;close;clc;
N = 5500;
n1 = 4000;
n2 = N-n1;
[x1, x2] = import_worm_data(N);
data = [x1';x2'];

Aw = data(1:n1,:);
Awt = data(n1+1:N,:);
Anw = data(N+1:N+n1,:);
Anwt = data(N+n1+1:2*N,:);
X = [Aw;Anw];
Xt = [Awt;Anwt];
% [coeff,score,latent] = pca(X');
[coeff, score, ~, ~, explained, ~] = pca(X,'Rows','all', 'Algorithm', 'eig');
[coefft, scoret, ~, ~, explainedt, ~] = pca(Xt,'Rows','all', 'Algorithm', 'eig');

nof = 200;

A = score(:,1:nof);
At = scoret(:,1:nof);

A = [A A.^2];
At = [At At.^2];

A1 = [A ones(2*n1,1)];
At1 = [At ones(2*n2,1)];

t = [ones(n1,1); zeros(n1,1)];
tt = [ones(n2,1); zeros(n2,1)];
[weight, grad,cout] = logistic(A,t);

y1 = A1*weight;
y2 = sigmoid_activate(y1');
y3 = classify(y2);
[TP,TN,FP,FN,a,p,r,f1,s] = confusion_matrix(t,y3)
max(grad)
mean(grad)

disp('testing accuracy')

y1 = At1*weight;
y2 = sigmoid_activate(y1');
y3 = classify(y2);
[TP1,TN1,FP1,FN1,a1,p1,r1,f11,s1] = confusion_matrix(tt,y3')




function [worm_data, no_worm_data] = import_worm_data(num_of_pics)
    
    val = 0.36;
    sz = ceil(101*val);
    g = sz*sz;
    worm_data = zeros(g, num_of_pics);
    no_worm_data = zeros(g, num_of_pics);
    for i = 1:num_of_pics

        filename = sprintf('%s_%d.%s','C:\Users\Student\Desktop\Celegans_Train\1\image', i ,'png');
        [cval] = imread(filename);
        cval_t = (cval);
        cval_t = imresize(cval_t,val);
        cval_t = imbinarize(cval_t);
        re_cval = cval_t(:)'; 

        worm_data(:, i) = re_cval;
    end

    for i = 1:num_of_pics

        filename = sprintf('%s_%d.%s','C:\Users\Student\Desktop\Celegans_Train\0\image', i ,'png');
        [cval] = imread(filename);
        cval_t = (cval);
        cval_t = imresize(cval_t,val);
        cval_t = imbinarize(cval_t);
        re_cval = cval_t(:)'; 

        no_worm_data(:, i) = re_cval;
    end
end


function [w,J,cout] = logistic(A,t)
    % A = data(known)
    % t = target(known)
    iter = 1500;
    [N,d] = size(A);
    o = ones(N,1);
    x = [A o];
    rng(100);
    w = rand(d+1,1);
    R =1e-2;
    cout = 0;
%     Beta = 0.9;
%     Vm = 0;
    for i =1:iter
        cout = cout +1;
        J = 0;
        for j = 1:N
            J = J + (sigmoid_activate(x(j,:)*w)-t(j))*x(j,:);
        end
        
%         Vn = Vm*Beta + (1-Beta)*J ;
%         w = w-R*Vn';
        w = w-R*J';
        
        if(J <= 0)
            break;
        end  
%         Vm = Vn;
    end

end

function [y] = sigmoid_activate(x_entries)
y = zeros(1,size(x_entries, 2));
for i = 1:size(x_entries, 2)
    answ = 1/(1 + exp(-x_entries(1, i)));
    y(1, i) = answ;
end

end

function [y] = classify(y_hat)

y = zeros(1, size(y_hat, 2));

    for i = 1:size(y_hat, 2)
        if y_hat(1, i) >= 0.5
            y(1, i) = 1;
        end
        if y_hat(1, i) < 0.5
            y(1, i) = 0;
        end
    end
    
end

function [TP,TN,FP,FN, Accuracy,Precision,Recall,F1_Score,Specificity] = confusion_matrix(a,b)

    TP = 0;
    TN = 0;
    FP = 0;
    FN = 0;
    for i=1:length(a)
        
        if(a(i) == 1)
            if(b(i) == 1)
                TP = TP + 1;
            else
                FN = FN + 1;
            end
        else
            if(b(i) == 0)
                TN = TN + 1;
            else
                FP = FP + 1;
            end
        end
        
    end

    Accuracy = (TP + TN) / (TP + FP + TN + FN);
    Precision = TP / (TP + FP);
    Recall = TP / (TP + FN); % Sensitivity
    F1_Score =  2 * (Recall*Precision) / (Recall + Precision);
    Specificity = TN / (TN + FP);

end





