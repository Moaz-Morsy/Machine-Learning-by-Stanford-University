function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]; % C = 1;
sigma = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]; % sigma = 0.3;
%values = zeros(length(C)*length(sigma),3);
%values(:,2) = repelem(C(:),8);
%values(:,3) = repmat(sigma(:),[8,1]);
% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

%k = 0;
%for i=1:length(C)
%    for j=1:length(sigma)
%        k = k+1;
%        model= svmTrain(X, y, C(i),...
%            @(x1, x2) gaussianKernel(x1, x2, sigma(j)));
%        predictions = svmPredict(model, Xval);
%        values(k,1) = mean(double(predictions ~= yval));
%        values(k,2) = C(i);
%        values(k,3) = sigma(j);
%    end    
%end
%for m=1:size(values,1)
%    model= svmTrain(X, y, values(m,2),...
%            @(x1, x2) gaussianKernel(x1, x2, values(m,3)));
%    predictions = svmPredict(model, Xval);
%    values(m,1) = mean(double(predictions ~= yval));
%end

%[~,index] = min(values(:,1));
%C = values(index,2);
%sigma = values(index,3);

V=[];
for i=1:length(C)
    for j=1:length(sigma)
        model= svmTrain(X, y, C(i),...
            @(x1, x2) gaussianKernel(x1, x2, sigma(j)));
        predictions = svmPredict(model, Xval);
        error = mean(double(predictions ~= yval));
        V = [V;[error,C(i),sigma(j)]];
    end    
end

[~,index] = min(V(:,1));
C = V(index,2);
sigma = V(index,3);


% =========================================================================

end
