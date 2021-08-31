
clear ; close all; clc

%% ============= Part 1: Data preparation =============

load('dataset.mat'); % the extracted feature
load('sam_labels'); % the experiment version
load('ID.mat'); % the corresponding subjects of the trials   


data.X{1,1} = dataset(1:216,:);
data.X{2,1} = dataset(217:end,:);
data.Y{1,1} = sample_labels(1:216,:);
data.Y{2,1} = sample_labels(217:end,:);

sam_id1 = sample_ids(1:216,:);
sam_id2 = sample_ids(1:216,:);
id1=1:12;
id2=1:12;

% group information of features
group_info = [];
for info=1:20
    group = info*ones(1,12);
    group_info = [group_info, group];
end

%indices for 10-fold
kfold = 10;
indices=crossvalind('Kfold',12,10); 
indices2=crossvalind('Kfold',12,10);

%Parameter Setting
ori_pararange =[1e-5 1e-4 1e-3 1e-2 1e-1];
lambda(1) = ori_pararange(4);
lambda(2) = ori_pararange(3);
lambda(3) = ori_pararange(2); 

 for k=1:10  

    % Hierarchical Sampling 
    %testing set
    test = find(indices == k,1);   
    testid = find(sam_id1 == test);
    test_data1=data.X{1,1}(testid,:); 
    test2 = find(indices2 == k,1); 
    testid2 = find(sam_id2 == test2);
    test_data2=data.X{2,1}(testid2,:);
    test_label1=data.Y{1,1}(testid,:);
    test_label2=data.Y{2,1}(testid2,:); 
    
    data.test_data = [test_data1;test_data2];
    X2 = zscore(data.test_data,0,1);
    y2 = [test_label1;test_label2]; 
    
    %training set    
    trainid = find(sam_id1 ~= test );     
    train_data1=data.X{1,1}(trainid,:);
    trainid2 = find(sam_id2 ~= test2 );  
    train_data2=data.X{2,1}(trainid2,:);
    train_label1=data.Y{1,1}(trainid,:);
    train_label2=data.Y{2,1}(trainid2,:);
    
    data.train_data = [train_data1;train_data2];
    X1 = zscore(data.train_data, 0, 1);
    y1 = [train_label1;train_label2];  
              
 

%% ============= Part 2: Regularization and Accuracies =============

    % Initialize fitting parameters
    initial_theta = zeros(size(X1, 2), 1);
    PX = get_connectivity(X1,2);
    % Set Options
    options = optimset('GradObj', 'on', 'MaxIter',400,'TolFun',1e-5,'TolX',1e-5 );%1000
    
    % Optimize
    [theta, J, exit_flag] = ...
        fminunc(@(t)(jccb_costFunction(t, X1, y1, lambda,PX,group_info)), initial_theta, options);
    
    % Predict
    p = predict1(theta, X2);
    acc =  (sum(p()==y2()))/(size(y2,1));
    ACC(1,k)=acc;


 end


