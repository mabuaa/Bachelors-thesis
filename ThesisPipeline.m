% Loading data
% Here I load all the datasets that I might be interested in

%Loading MNIST

% load('C:\Users\matej\Desktop\Bachelor''s thesis\MNIST2.mat');

% load 'mnist.mat';
load('C:\Users\matej\OneDrive\Desktop\benchmark\benchmark_MNIST_tSNE.mat');
% img_size = 28;
% images = transpose(reshape(transpose(cell2mat(transpose(img))), img_size^2, 60000));
% labelsMnist = labels.*(labels~=10);
% testimages = transpose(reshape(transpose(cell2mat(transpose(img_test))), img_size^2, 10000));
% testlabels = labels_test.*(labels_test~=10);


%Loading Wine dataset

load('C:\Users\matej\OneDrive\Desktop\Bachelor''s thesis\data\wine_X.mat');
load('C:\Users\matej\OneDrive\Desktop\Bachelor''s thesis\data\wine_y.mat');


%Loading bank

load('C:\Users\matej\OneDrive\Desktop\Bachelor''s thesis\data\bank_X.csv');
load('C:\Users\matej\OneDrive\Desktop\Bachelor''s thesis\data\bank_y.csv');


%Loading cifar10

load('C:\Users\matej\OneDrive\Desktop\Bachelor''s thesis\data\cifar10_X.csv');
load('C:\Users\matej\OneDrive\Desktop\Bachelor''s thesis\data\cifar10_y.csv');

% 
%Loading cnae9

load('C:\Users\matej\OneDrive\Desktop\Bachelor''s thesis\data\cnae9_X.csv');
load('C:\Users\matej\OneDrive\Desktop\Bachelor''s thesis\data\cnae9_y.csv');

% 
%Loading coil20

load('C:\Users\matej\OneDrive\Desktop\Bachelor''s thesis\data\coil20_X.csv');
load('C:\Users\matej\OneDrive\Desktop\Bachelor''s thesis\data\coil20_y.csv');


%Loading epileptic

load('C:\Users\matej\OneDrive\Desktop\Bachelor''s thesis\data\epileptic_X.csv');
load('C:\Users\matej\OneDrive\Desktop\Bachelor''s thesis\data\epileptic_y.csv');


%Loading fashion_mnist

load('C:\Users\matej\OneDrive\Desktop\Bachelor''s thesis\data\fashion_mnist_X.csv');
load('C:\Users\matej\OneDrive\Desktop\Bachelor''s thesis\data\fashion_mnist_y.csv');


%Loading har

load('C:\Users\matej\OneDrive\Desktop\Bachelor''s thesis\data\har_X.csv');
load('C:\Users\matej\OneDrive\Desktop\Bachelor''s thesis\data\har_y.csv');


%Loading hatespeech

load('C:\Users\matej\OneDrive\Desktop\Bachelor''s thesis\data\hatespeech_X.csv');
load('C:\Users\matej\OneDrive\Desktop\Bachelor''s thesis\data\hatespeech_y.csv');


%Loading hiva

load('C:\Users\matej\OneDrive\Desktop\Bachelor''s thesis\data\hiva_X.csv');
load('C:\Users\matej\OneDrive\Desktop\Bachelor''s thesis\data\hiva_y.csv');


%Loading imdb

load('C:\Users\matej\OneDrive\Desktop\Bachelor''s thesis\data\imdb_X.csv');
load('C:\Users\matej\OneDrive\Desktop\Bachelor''s thesis\data\imdb_y.csv');


%Loading orl

load('C:\Users\matej\OneDrive\Desktop\Bachelor''s thesis\data\orl_X.csv');
load('C:\Users\matej\OneDrive\Desktop\Bachelor''s thesis\data\orl_y.csv');


%Loading secom

load('C:\Users\matej\OneDrive\Desktop\Bachelor''s thesis\data\secom_X.csv');
load('C:\Users\matej\OneDrive\Desktop\Bachelor''s thesis\data\secom_y.csv');


%Loading seismic

load('C:\Users\matej\OneDrive\Desktop\Bachelor''s thesis\data\seismic_X.csv');
load('C:\Users\matej\OneDrive\Desktop\Bachelor''s thesis\data\seismic_y.csv');


%Loading sentiment

load('C:\Users\matej\OneDrive\Desktop\Bachelor''s thesis\data\sentiment_X.csv');
load('C:\Users\matej\OneDrive\Desktop\Bachelor''s thesis\data\sentiment_y.csv');


%Loading sms

load('C:\Users\matej\OneDrive\Desktop\Bachelor''s thesis\data\sms_X.csv');
load('C:\Users\matej\OneDrive\Desktop\Bachelor''s thesis\data\sms_y.csv');


%Loading spambase

load('C:\Users\matej\OneDrive\Desktop\Bachelor''s thesis\data\spambase_X.csv');
load('C:\Users\matej\OneDrive\Desktop\Bachelor''s thesis\data\spambase_y.csv');


%Loading svhn

load('C:\Users\matej\OneDrive\Desktop\Bachelor''s thesis\data\svhn_X.csv');
load('C:\Users\matej\OneDrive\Desktop\Bachelor''s thesis\data\svhn_y.csv');



%% Loading file with embeddings and errors cell arrays

% load('C:\Users\matej\OneDrive\Desktop\Bachelor''s thesis\updated_pipeline\experiments_05.mat');
% load('C:\Users\matej\OneDrive\Desktop\Bachelor''s thesis\updated_pipeline\experiments_10.mat');

%% Puting the datasets in a cell array

nod = 19; % number of datasets that I am curently interested in / had prepared previously
data = cell(nod,1);
labels = cell(nod,1);
titels = cell(nod,1);


fullTSNEtimes = 10;
experiments = 30;

DandPfullTSNE = cell(nod, 2);
timesDandPfullTSNE = cell(nod,2);
full_tSNE_X = cell(nod,fullTSNEtimes);
losses_fullTSNE = cell(nod, fullTSNEtimes+1);
times_fullTSNE = cell(nod, fullTSNEtimes+1);

embeddings = cell(nod,4*experiments);
errors = cell(nod,4*(experiments+1));
times = cell(nod, 4*(experiments+1) );

sampled_indices = cell(nod, 3*experiments +1);


embeddings_afterKNNreg = cell(nod, 4*experiments);
errors_afterKNNreg = cell(nod, 4*(experiments+1));
times_afterKNNreg = cell(nod, 4*(experiments+1));



% data{1} = images;
% labels{1} = labelsMnist;
% titels{1} = 'MNIST data';
%
% data{2} = rescale(wine_X,'InputMin',min(wine_X),'InputMax',max(wine_X));
% labels{2} = wine_y;
% titels{2} = 'data of 3 types of wine';
%
% data{3} = rescale(bank_X,'InputMin',min(bank_X),'InputMax',max(bank_X));
% labels{3} = bank_y;
% titels{3} = 'bank data';

% data{4} = rescale(cifar10_X,'InputMin',min(cifar10_X),'InputMax',max(cifar10_X));
% labels{4} = cifar10_y;
% titels{4} = 'cifar10 data';
%
% data{5} = rescale(cnae9_X,'InputMin',min(cnae9_X),'InputMax',max(cnae9_X));
% labels{5} = cnae9_y;
% titels{5} = 'cnae9 data';
% 
% data{6} = rescale(coil20_X,'InputMin',min(coil20_X),'InputMax',max(coil20_X));
% labels{6} = coil20_y;
% titels{6} = 'coil20 data';
%
% data{7} = rescale(epileptic_X,'InputMin',min(epileptic_X),'InputMax',max(epileptic_X));
% labels{7} = epileptic_y;
% titels{7} = 'epileptic data';

% data{8} = rescale(fashion_mnist_X,'InputMin',min(fashion_mnist_X),'InputMax',max(fashion_mnist_X));
% labels{8} = fashion_mnist_y;
% titels{8} = 'fashion mnist data';
%
% data{9} = rescale(har_X,'InputMin',min(har_X),'InputMax',max(har_X));
% labels{9} = har_y;
% titels{9} = 'har data';

% data{10} = rescale(hatespeech_X,'InputMin',min(hatespeech_X),'InputMax',max(hatespeech_X));
% labels{10} = hatespeech_y;
% titels{10} = 'hatespeech data';

% data{11} = rescale(hiva_X,'InputMin',min(hiva_X),'InputMax',max(hiva_X));
% labels{11} = hiva_y;
% titels{11} = 'hiva data';

% data{12} = rescale(imdb_X,'InputMin',min(imdb_X),'InputMax',max(imdb_X));
% labels{12} = imdb_y;
% titels{12} = 'imdb data';
%
% data{13} = rescale(orl_X,'InputMin',min(orl_X),'InputMax',max(orl_X));
% labels{13} = orl_y;
% titels{13} = 'orl data';

% data{14} = rescale(secom_X,'InputMin',min(secom_X),'InputMax',max(secom_X));
% labels{14} = secom_y;
% titels{14} = 'secom data';
%
% data{15} = rescale(seismic_X,'InputMin',min(seismic_X),'InputMax',max(seismic_X));
% labels{15} = seismic_y;
% titels{15} = 'seismic data';
%
% data{16} = rescale(sentiment_X,'InputMin',min(sentiment_X),'InputMax',max(sentiment_X));
% labels{16} = sentiment_y;
% titels{16} = 'sentiment data';
%
% data{17} = rescale(sms_X,'InputMin',min(sms_X),'InputMax',max(sms_X));
% labels{17} = sms_y;
% titels{17} = 'sms data';
%
% data{18} = rescale(spambase_X,'InputMin',min(spambase_X),'InputMax',max(spambase_X));
% labels{18} = spambase_y;
% titels{18} = 'spambase data';

%data{19} = rescale(svhn_X,'InputMin',min(svhn_X),'InputMax',max(svhn_X));
% labels{19} = svhn_y;
% titels{19} = 'svhn data';



%% Running multiple full-t-SNE

% Choosing the dataset to work with now

datai = 10   % index of the dataset from the cell array that I want to apply t-SNE to now

X = data{datai};
Y = labels{datai};
n = size(X,1);

rng default
perplexity = 30;
tic
D = XtoD(X);
t = toc;
timesDandPfullTSNE{datai, 1} = t;
P = d2p(D, perplexity, 1e-5);
t = toc;
timesDandPfullTSNE{datai, 2} = t;
DandPfullTSNE{datai, 1} = D;
DandPfullTSNE{datai, 2} = P;

time_fullTSNE = zeros(fullTSNEtimes,1);
loss_fullTSNE = zeros(fullTSNEtimes,2);
for i = 1:fullTSNEtimes
    i
    tic
    ydata = tsne_p(P);
    t = toc;
    full_tSNE_X{datai, i} = ydata;
    loss_fullTSNE(i,1) = trustworthiness(X,ydata,12);
    loss_fullTSNE(i,2) = continuity(X,ydata,12);
    time_fullTSNE(i) = t;
    times_fullTSNE{datai, i} = t;
    losses_fullTSNE{datai, i} = loss_fullTSNE(i, :);
end
avgtime_fullTSNE = mean(time_fullTSNE);
times_fullTSNE{datai, fullTSNEtimes+1} = avgtime_fullTSNE;
avgloss_fullTSNE = mean(loss_fullTSNE);
losses_fullTSNE{datai, fullTSNEtimes+1} = avgloss_fullTSNE;

%% Hubs
datai = 18
load(strcat('datai', int2str(datai), 'fullTSNEandPrerequisits.mat'));
tic
[sortedD,nearest] = sort(D,2);

k = 100;
k_nearest = nearest(:,2:k+1);

%save(strcat('datai', int2str(datai), 'fullTSNEandPrerequisits.mat'));

%for ftsnei = 1:fullTSNEtimes
    ftsnei = 1;
    
    X2D = full_tSNE_X{datai, ftsnei};
    
    %% Testing out multiple datasets
    
    %for datai = 3:19
    
    %% Determining sample size - multiple sizes
    
    for p = 5:5:50  % percentage of total dataset to sample
        
        nss = floor(n*p/100); % smaller subset size
        
        
        %% Random sampling
        
        %function available in section Functions
        
        
        %% Random walk sampling
        
        L = 10; % Number of steps for each random walk
        M = nss;
        
        %function available in section Functions
        
        
        %% New random walk sampling
        
        % [w, idx] = weights_dist(X);
        % weights_and_indices{datai,1} = w;
        % weights_and_indices{datai,2} = idx;
        
        %% Hubs
        
        hub_scores = zeros(nss,1);
        for i = 1:nss
            hub_scores(i) = sum(sum(k_nearest == i));
        end
        [hub_scores_sorted, I] = sort(hub_scores);
        hubs = I(1:nss);
        sampled_indices{datai,3*experiments + 1} = hubs;
        
        Xh = X(hubs,:);
        Yh = Y(hubs);
        full_Xh2D = X2D(hubs, :);
        th1 = toc;
        
        %% Applying t-SNE to the chosen dataset (presampled)
        
        
        %% Random sampling
        
        st = 1;
        K = 5;  % used for KNN regression
        
        rng default;
        losses_r = zeros(experiments,3);
        losses_rKNN = zeros(experiments,3);
        time_r = zeros(experiments,1);
        time_rKNN = zeros(experiments,1);
        for i = 1:experiments
            
            [datai, ftsnei, p, st, i ]
            tic
            [Xr, Yr, sampled_idx_r] = random_sampling(X,Y,nss);
            Dtrunc = D(sampled_idx_r, sampled_idx_r);
            Xr2D = tsne_d(Dtrunc);
            t = toc;
            sampled_indices{datai,i} = sampled_idx_r;
            regressionX2D_r = knn_regression(X, Xr2D, sampled_idx_r, K);
            Xr2Dknn = [Xr2D; regressionX2D_r];
            tar = toc;
            time_r(i) = t;
            time_rKNN(i) = tar;
            times{datai, i} = t;
            times_afterKNN{datai, i} = tar;
            embeddings{datai, i} = Xr2D;
            embeddings_afterKNNreg{datai, i} = Xr2Dknn;
            
            losses_r(i,1) = trustworthiness(Xr, Xr2D, 12);
            losses_r(i,2) = continuity(Xr, Xr2D, 12);
            full_Xr2D = X2D(sampled_idx_r, :);
            losses_r(i,3) = procrustes(full_Xr2D, Xr2D);
            errors{datai, i} = losses_r(i,:);
            
            losses_rKNN(i,1) = trustworthiness(X, Xr2Dknn, 12);
            losses_rKNN(i,2) = continuity(X, Xr2Dknn, 12);
            losses_rKNN(i,3) = procrustes(X2D, Xr2Dknn);
            errors_afterKNNreg{datai, i} = losses_rKNN(i,:);
        end
        avg_loss_r = mean(losses_r,1)
        errors{datai, experiments + 1} = avg_loss_r;
        avg_loss_rKNN = mean(losses_rKNN,1)
        errors_afterKNNreg{datai, experiments + 1} = avg_loss_rKNN;
        avg_time_r = mean(time_r)
        times{datai, experiments+1} = avg_time_r;
        avg_time_rKNN = mean(time_rKNN)
        times_afterKNN{datai, experiments+1} = avg_time_rKNN;
        
        savethis = strcat('datai', int2str(datai),'ftsnei', int2str(ftsnei), 'percentage',int2str(p), 'up to st', int2str(st), '.mat');
        save(savethis, 'embeddings', 'errors', 'times', 'embeddings_afterKNNreg', 'errors_afterKNNreg', 'times_afterKNNreg', 'sampled_indices');
        
        %% (Old) random walk sampling with inverted distances weights
        st = 2;
        
        losses_rw = zeros(experiments,3);
        losses_rwKNN = zeros(experiments,3);
        time_rw = zeros(experiments, 1);
        time_rwKNN = zeros(experiments, 1);
        for i = 1:experiments
            [datai, ftsnei, p, st, i ]
            tic
            [w, idx] = weights_dist2(X, 1);
            [Xrw, Yrw, sampled_idx_rw] = new_random_walk_sampling(X,Y, w, idx, M,L);
            Dtrunc = D(sampled_idx_rw, sampled_idx_rw);
            Xrw2D = tsne_d(Dtrunc);
            t = toc;
            sampled_indices{datai, experiments + i} = sampled_idx_rw;
            regressionX2D_rw = knn_regression(X, Xrw2D, sampled_idx_rw, K);
            Xrw2Dknn = [Xrw2D; regressionX2D_rw];
            tar = toc;
            time_rw(i) = t;
            time_rwKNN(i) = tar;
            times{datai, experiments+1 + i} = t;
            times_afterKNN{datai, experiments+1 + i} = tar;
            embeddings{datai, experiments + i} = Xrw2D;
            embeddings_afterKNNreg{datai, experiments + i} = Xrw2Dknn;
     
            losses_rw(i,1) = trustworthiness(Xrw, Xrw2D, 12);
            losses_rw(i,2) = continuity(Xrw, Xrw2D, 12);
            full_Xrw2D = X2D(sampled_idx_rw, :);
            losses_rw(i,3) = procrustes(full_Xrw2D, Xrw2D);
            errors{datai, experiments+1 + i} = losses_rw(i,:);
            
            losses_rwKNN(i,1) = trustworthiness(X, Xrw2Dknn, 12);
            losses_rwKNN(i,2) = continuity(X, Xrw2Dknn, 12);
            losses_rwKNN(i,3) = procrustes(X2D, Xrw2Dknn);
            errors_afterKNNreg{datai, experiments+1 + i} = losses_rwKNN(i,:);
        end
        avg_loss_rw = mean(losses_rw,1)
        errors{datai, 2*(experiments + 1)} = avg_loss_rw;
        avg_loss_rwKNN = mean(losses_rwKNN,1)
        errors_afterKNNreg{datai, 2*(experiments + 1)} = avg_loss_rwKNN;
        avg_time_rw = mean(time_rw)
        times{datai, 2*(experiments+1)} = avg_time_rw;
        avg_time_rwKNN = mean(time_rwKNN)
        times_afterKNN{datai, 2*(experiments+1)} = avg_time_rwKNN;
        
        
        savethis = strcat('datai', int2str(datai),'ftsnei', int2str(ftsnei), 'percentage',int2str(p), 'up to st', int2str(st), '.mat');
        save(savethis, 'embeddings', 'errors', 'times', 'embeddings_afterKNNreg', 'errors_afterKNNreg', 'times_afterKNNreg', 'sampled_indices');
        
        %% New random walk sampling with t-SNE native probability matrix P
        st = 3;
        
        rng default;
        losses_nrw = zeros(experiments,3);
        losses_nrwKNN = zeros(experiments,3);
        time_nrw = zeros(experiments, 1);
        time_nrwKNN = zeros(experiments, 1);
        for i = 1:experiments
            [datai, ftsnei, p, st, i ]
            tic
            [w, idx] = weights_dist21(X);
            [Xnrw, Ynrw, sampled_idx_nrw] = new_random_walk_sampling(X,Y, w, idx, M,L);
            Dtrunc = D(sampled_idx_nrw, sampled_idx_nrw);
            Xnrw2D = tsne_d(Dtrunc);
            t = toc;
            sampled_indices{datai, 2*experiments + i} = sampled_idx_nrw;
            regressionX2D_nrw = knn_regression(X, Xnrw2D, sampled_idx_nrw, K);
            Xnrw2Dknn = [Xnrw2D; regressionX2D_nrw];
            tar = toc;
            time_nrw(i) = t;
            time_nrwKNN(i) = tar;
            times{datai, 2*(experiments+1) + i} = t;
            times_afterKNN{datai, 2*(experiments+1) + i} = tar;
            embeddings{datai, 2*experiments + i} = Xnrw2D;
            embeddings_afterKNNreg{datai, 2*experiments + i} = Xnrw2Dknn;
            
            losses_nrw(i,1) = trustworthiness(Xnrw, Xnrw2D, 12);
            losses_nrw(i,2) = continuity(Xnrw, Xnrw2D, 12);
            full_Xnrw2D = X2D(sampled_idx_nrw, :);
            losses_nrw(i,3) = procrustes(full_Xnrw2D, Xnrw2D);
            errors{datai, 2*(experiments+1) + i} = losses_nrw(i,:);
            
            losses_nrwKNN(i,1) = trustworthiness(X, Xnrw2Dknn, 12);
            losses_nrwKNN(i,2) = continuity(X, Xnrw2Dknn, 12);
            losses_nrwKNN(i,3) = procrustes(X2D, Xnrw2Dknn);
            errors_afterKNNreg{datai, 2*(experiments+1) + i} = losses_nrwKNN(i,:);
        end
        avg_loss_nrw = mean(losses_nrw,1)
        errors{datai, 3*(experiments + 1)} = avg_loss_nrw;
        avg_loss_nrwKNN = mean(losses_nrwKNN,1)
        errors_afterKNNreg{datai, 3*(experiments + 1)} = avg_loss_nrwKNN;
        avg_time_nrw = mean(time_nrw)
        times{datai, 3*(experiments+1)} = avg_time_nrw;
        avg_time_nrwKNN = mean(time_nrwKNN)
        times_afterKNN{datai, 3*(experiments+1)} = avg_time_nrwKNN;
        
        
        savethis = strcat('datai', int2str(datai),'ftsnei', int2str(ftsnei), 'percentage',int2str(p), 'up to st', int2str(st), '.mat');
        save(savethis, 'embeddings', 'errors', 'times', 'embeddings_afterKNNreg', 'errors_afterKNNreg', 'times_afterKNNreg', 'sampled_indices');
        
        
        %% Hubs sampling
        st = 4;
        
        rng default;
        losses_h = zeros(experiments,3);
        losses_hKNN = zeros(experiments,3);
        time_h = zeros(experiments,1);
        time_hKNN = zeros(experiments,1);
        for i = 1:experiments
            
            [datai, ftsnei, p, st, i ]
            tic
            Dtrunc = D(hubs, hubs);
            Xh2D = tsne_d(Dtrunc);
            th2 = toc;
            regressionX2D_h = knn_regression(X, Xh2D, hubs, K);
            Xh2Dknn = [Xh2D; regressionX2D_h];
            tar = toc;
            time_h(i) = th1+th2;
            time_hKNN(i) = th1+tar;
            times{datai, 3*(experiments+1)+i} = th1+th2;
            times_afterKNN{datai, 3*(experiments+1)+i} = th1+tar;
            embeddings{datai, 3*(experiments)+i} = Xh2D;
            embeddings_afterKNNreg{datai, 3*(experiments)+i} = Xh2Dknn;
            
            losses_h(i,1) = trustworthiness(Xh, Xh2D, 12);
            losses_h(i,2) = continuity(Xh, Xh2D, 12);
            full_Xh2D = X2D(hubs, :);
            losses_h(i,3) = procrustes(full_Xh2D, Xh2D);
            errors{datai, 3*(experiments+1)+i} = losses_h(i,:);
            
            losses_hKNN(i,1) = trustworthiness(X, Xh2Dknn, 12);
            losses_hKNN(i,2) = continuity(X, Xh2Dknn, 12);
            losses_hKNN(i,3) = procrustes(X2D, Xh2Dknn);
            errors_afterKNNreg{datai, 3*(experiments+1)+ i} = losses_hKNN(i,:);
        end
        avg_loss_h = mean(losses_h,1)
        errors{datai, 4*(experiments + 1)} = avg_loss_r;
        avg_loss_hKNN = mean(losses_hKNN,1)
        errors_afterKNNreg{datai, 4*(experiments + 1)} = avg_loss_hKNN;
        avg_time_h = mean(time_h)
        times{datai, 4*(experiments+1)} = avg_time_h;
        avg_time_hKNN = mean(time_hKNN)
        times_afterKNN{datai, 4*(experiments+1)} = avg_time_hKNN;
        
        
        savethis = strcat('datai', int2str(datai),'ftsnei', int2str(ftsnei), 'percentage',int2str(p), 'up to st', int2str(st), '.mat');
        save(savethis, 'embeddings', 'errors', 'times', 'embeddings_afterKNNreg', 'errors_afterKNNreg', 'times_afterKNNreg', 'sampled_indices');
        
        
    end        % from the for-loop for running experiments on multiple sampling percentages
%end        % from the for-loop for comparing with different full tsne embeddings


% %% Actully visualizing the data
%
% figure(1)
% gscatter(X2D(:,1), X2D(:,2), Y);
% title(strcat('t-SNE applied to: ',titels{datai}, ' without samling'), 'FontSize', 8);
%
% figure(2)
% a = 1:size(X,1);
% a = a';
% unsampled_idx_r = setdiff(a, sampled_idx_r);
% idx_afterKNN = [sampled_idx_r; unsampled_idx_r];
% gscatter(Xr2Dknn(:,1), Xr2Dknn(:,2), Y(idx_afterKNN));
% title(strcat('t-SNE applied to: ',titels{datai}, ' using random samling and KNN reg'), 'FontSize', 8);
%
% figure(3)
% a = 1:size(X,1);
% a = a';
% unsampled_idx_nrw = setdiff(a, sampled_idx_nrw);
% idx_afterKNN = [sampled_idx_nrw; unsampled_idx_nrw];
% gscatter(Xnrw2Dknn(:,1), Xnrw2Dknn(:,2), Y(idx_afterKNN));
% title(strcat('t-SNE applied to: ',titels{datai}, ' using new random walk samling and KNN reg'), 'FontSize', 8);
%
% figure(4)
% a = 1:size(X,1);
% a = a';
% unsampled_idx_h = setdiff(a, hubs);
% idx_afterKNN = [hubs; unsampled_idx_h];
% gscatter(Xh2Dknn(:,1), Xh2Dknn(:,2), Y(idx_afterKNN));
% title(strcat('t-SNE applied to: ',titels{datai}, ' using hubness samling and KNN reg'), 'FontSize', 8);

%% Saving data if needed

%save('C:\Users\matej\Desktop\Bachelor''s thesis\MNIST4.mat');
%save('C:\Users\matej\Desktop\Bachelor''s thesis\Wine1.mat');

% save('C:\Users\matej\Desktop\Bachelor''s thesis\bank.mat');
% save('C:\Users\matej\Desktop\Bachelor''s thesis\cifar10.mat');
% save('C:\Users\matej\Desktop\Bachelor''s thesis\cnae9.mat');
% save('C:\Users\matej\Desktop\Bachelor''s thesis\coil20.mat');
% save('C:\Users\matej\Desktop\Bachelor''s thesis\epileptic.mat');
% save('C:\Users\matej\Desktop\Bachelor''s thesis\fashion_mnist.mat');
% save('C:\Users\matej\Desktop\Bachelor''s thesis\har.mat');
% save('C:\Users\matej\Desktop\Bachelor''s thesis\hatespeech.mat');
% save('C:\Users\matej\Desktop\Bachelor''s thesis\hiva.mat');
% save('C:\Users\matej\Desktop\Bachelor''s thesis\imdb.mat');
% save('C:\Users\matej\Desktop\Bachelor''s thesis\orl.mat');
% save('C:\Users\matej\Desktop\Bachelor''s thesis\secom.mat');
% save('C:\Users\matej\Desktop\Bachelor''s thesis\seismic.mat');
% save('C:\Users\matej\Desktop\Bachelor''s thesis\sentiment.mat');
% save('C:\Users\matej\Desktop\Bachelor''s thesis\sms.mat');
% save('C:\Users\matej\Desktop\Bachelor''s thesis\spambase.mat');
% save('C:\Users\matej\Desktop\Bachelor''s thesis\svhn.mat');

% save('C:\Users\matej\Desktop\Bachelor''s thesis\Multiple6e5p.mat');
% save('C:\Users\matej\Desktop\Bachelor''s thesis\Multiple6e10p.mat');
% save('C:\Users\matej\Desktop\Bachelor''s thesis\Multiple6e15p.mat');

% save('C:\Users\matej\OneDrive\Desktop\Bachelor''s thesis\NewRandomWalkWine10p.mat');
% save('C:\Users\matej\OneDrive\Desktop\Bachelor''s thesis\NewRandomWalkMultiple10p.mat', 'embeddings', 'errors');

% save('C:\Users\matej\OneDrive\Desktop\Bachelor''s thesis\updated_pipeline\experiments_10.mat', 'embeddings', 'errors');
% save('C:\Users\matej\OneDrive\Desktop\Bachelor''s thesis\updated_pipeline\experiments_05.mat');
%% Functions

function [Xr, Yr, usedIdx] = random_sampling(X, Y, nss)
r = randperm(size(X,1)); % Using simple random sampling for now

Xr = X(r(1:nss),:);
Yr = Y(r(1:nss));
usedIdx = r(1:nss);
usedIdx = usedIdx';
end

function [Xrw, Yrw, usedIdx] = random_walk_sampling(X,Y, nearestHD, M,L)

n = size(X,1);

Xrw = zeros(M,size(X,2));
Yrw = zeros(M,1);
usedIdx = zeros(M,1);

for i = 1:M
    j = randi([1,n]);
    % while sum(usedIdx==j)==1
    %     j = randi([1,n]);
    % end
    for a = 1:L
        u = rand();
        stepTo = floor(log(u)/log(0.5))+2;
        if stepTo > 100
            stepTo= 100;
        end
        %    while sum(usedIdx==nearestHD(j,stepTo))==1
        %        u = rand();
        %        stepTo = floor(log(u)/log(0.5))+2;
        %        if stepTo > 100
        %            stepTo= 100;
        %        end
        %    end
        j = nearestHD(j,stepTo);
        
    end
    
    Xrw(i,:) = X(j,:);
    sizewithduplicates = size(Xrw)
    Xrw = unique(Xrw,'rows');
    sizewithoutduplicates = size(Xrw)
    Yrw(i) = Y(j);
    usedIdx(i) = j;
    
end

end



function [Xrw, Yrw, usedIdx] = new_random_walk_sampling(X,Y, w, idx, M,L)

n = size(X,1);

Xrw = zeros(M,size(X,2));
Yrw = zeros(M,1);
usedIdx = zeros(M,1);

for i = 1:M
    j = randi([1,n]);
    while sum(usedIdx==j)==1
        j = randi([1,n]);
    end
    for a = 1:L
        len = sum(w(j,:));
        u = len*rand();
        c = 1;
        sum_weights = w(j,c);
        while u > sum_weights
            c = c+1;
            sum_weights = sum_weights + w(j,c);
        end
        stepTo = idx(j,c);
        j = stepTo;
        
    end
    
    Xrw(i,:) = X(j,:);
    Yrw(i) = Y(j);
    usedIdx(i) = j;
    w = w.*(idx ~= j);
    
end

end

function [w, i] = weights_dist1(X)

n = size(X,1);
d = pdist(X);
ds = squareform(d);
rds = ds./sum(ds,2);
ords = ones(n)-rds;
nords = ords./((n-2)*ones(n));
[w, i] = sort(nords,2, 'descend');
w = w(:,2:end);
i = i(:,2:end);
end

function [w, i] = weights_dist21(X)

P = tsneV(X);
[w,i] = sort(P, 2, 'descend');

end


function trust = trustworthiness(XHD, XLD, k)
n = size(XHD,1);
nearestHD = knnsearch(XHD, XHD, 'K', n);
%nearestHD = nearestHD(:,2:end);
nearestLD = knnsearch(XLD, XLD, 'K', k+1);

sum_i = 0;

for i = 1:n
    Uk = setdiff(nearestLD(i,:),nearestHD(i,1:k));
    Uk_size  = size(Uk,2);
    sum_j = 0;
    for j = 1:Uk_size
        sum_j = sum_j + find(nearestHD(i,:)==Uk(j));
    end
    sum_i = sum_i + sum_j;
end

trust = 1 - (2*sum_i)/(n*k*(2*n - 3*k - 1));

end


function cont = continuity(XHD, XLD, k)
n = size(XHD,1);
nearestHD = knnsearch(XHD, XHD, 'K', k+1);
nearestLD = knnsearch(XLD, XLD, 'K', n);
%nearestLD = nearestLD(:,2:end);

sum_i = 0;

for i = 1:n
    Vk = setdiff(nearestHD(i,:),nearestLD(i,1:k));
    Vk_size  = size(Vk,2);
    sum_j = 0;
    for j = 1:Vk_size
        sum_j = sum_j + find(nearestLD(i,:)==Vk(j));
    end
    
    
    sum_i = sum_i + sum_j;
end

cont = 1 - (2*sum_i)/(n*k*(2*n - 3*k - 1));

end


function regressionX2D = knn_regression(X, sampledX2D, usedIdx, k)  % X is the full HD data matrix,
% sampledX2D is the 2D embedding of the sample,
% usedIdx are the indices of the sampled datapoints, i.e. sampledX2D = tsne(X(usedIdx, :)),
% k is the parameter for KNN regression
a = 1:size(X,1);
a = a';
unsampled_idx = setdiff(a, usedIdx);
regressionX2D = zeros(size(unsampled_idx,1),2);     % for saving the 2D coordinates of the previously unsampled datapoints
sampledX = X(usedIdx, :);
nss = size(sampledX2D,1);       % sample size
for i = 1:size(unsampled_idx,1)
    idx = unsampled_idx(i);
    b = ones(nss,1)*X(idx, :);
    c = b - sampledX;
    d = sqrt(sum(c.^2,2));      % d is a column vector containing the distances from one unsampled point to all the sampled points in HD
    nss1 = [1:nss]';
    d_idx = [d, nss1];
    [e, ind] = sort(d);         % ind is for the indices of nearest sampled neighbors in HD, sorted
    kNdist = d_idx(ind(1:k), :);    % truncating sorted distances and indices to the top k
    f = kNdist(:,1);
    
    %w = distancesToWeights1(f);
    w = distancesToWeights2(f, 0.1);    % w is a 1xk row vector containing the weights
    s = sampledX2D(kNdist(:,2),:);      % taking the k nearest neighbors' 2D coordinates, s is a kx2 vector
    
    regressionX2D(i,:) = w*s;
end
end

function w = distancesToWeights1(f) % f is a column vector, w is a row vector in the end

fs = f./sum(f);
ofs = ones(k,1) - fs;
w = ofs./(k-1);
w = w';
end

function [w] = distancesToWeights2(d, a) % d is a column vector, w is a row vector in the end

% functionality explained in the picture
d = d+eps;
k = d./min(d);
invka = ones(size(d,1),1)./(k+a);
x = 1/sum(invka) - a;
w = (x+a)*invka;
w = w';
end

function [w, i] = weights_dist2(X, a)
n = size(X,1);
d = pdist(X);
ds = squareform(d);
[sds, i] = sort(ds, 2, 'ascend');
d = sds(:, 2:end);
d = d+eps;
i = i(:, 2:end);
k = d./min(d, [], 2);
invka = ones(n,n-1)./(k+a);
x = ones(n,1)./sum(invka,2) - a;
w = ((x+a)*ones(1,n-1)).*invka;

end

function [P, beta] = d2p(D, u, tol)
%D2P Identifies appropriate sigma's to get kk NNs up to some tolerance
%
%   [P, beta] = d2p(D, kk, tol)
%
% Identifies the required precision (= 1 / variance^2) to obtain a Gaussian
% kernel with a certain uncertainty for every datapoint. The desired
% uncertainty can be specified through the perplexity u (default = 15). The
% desired perplexity is obtained up to some tolerance that can be specified
% by tol (default = 1e-4).
% The function returns the final Gaussian kernel in P, as well as the
% employed precisions per instance in beta.
%
%
% (C) Laurens van der Maaten, 2008
% Maastricht University


if ~exist('u', 'var') || isempty(u)
    u = 15;
end
if ~exist('tol', 'var') || isempty(tol)
    tol = 1e-4;
end

% Initialize some variables
n = size(D, 1);                     % number of instances
P = zeros(n, n);                    % empty probability matrix
beta = ones(n, 1);                  % empty precision vector
logU = log(u);                      % log of perplexity (= entropy)

% Run over all datapoints
for i=1:n
    
    if ~rem(i, 500)
        %disp(['Computed P-values ' num2str(i) ' of ' num2str(n) ' datapoints...']);
    end
    
    % Set minimum and maximum values for precision
    betamin = -Inf;
    betamax = Inf;
    
    % Compute the Gaussian kernel and entropy for the current precision
    [H, thisP] = Hbeta(D(i, [1:i - 1, i + 1:end]), beta(i));
    
    % Evaluate whether the perplexity is within tolerance
    Hdiff = H - logU;
    tries = 0;
    while abs(Hdiff) > tol && tries < 50
        
        % If not, increase or decrease precision
        if Hdiff > 0
            betamin = beta(i);
            if isinf(betamax)
                beta(i) = beta(i) * 2;
            else
                beta(i) = (beta(i) + betamax) / 2;
            end
        else
            betamax = beta(i);
            if isinf(betamin)
                beta(i) = beta(i) / 2;
            else
                beta(i) = (beta(i) + betamin) / 2;
            end
        end
        
        % Recompute the values
        [H, thisP] = Hbeta(D(i, [1:i - 1, i + 1:end]), beta(i));
        Hdiff = H - logU;
        tries = tries + 1;
    end
    
    % Set the final row of P
    P(i, [1:i - 1, i + 1:end]) = thisP;
end
% disp(['Mean value of sigma: ' num2str(mean(sqrt(1 ./ beta)))]);
% disp(['Minimum value of sigma: ' num2str(min(sqrt(1 ./ beta)))]);
% disp(['Maximum value of sigma: ' num2str(max(sqrt(1 ./ beta)))]);
end



% Function that computes the Gaussian kernel values given a vector of
% squared Euclidean distances, and the precision of the Gaussian kernel.
% The function also computes the perplexity of the distribution.
function [H, P] = Hbeta(D, beta)
P = exp(-D * beta);
sumP = sum(P);
H = log(sumP) + beta * sum(D .* P) / sumP;
% why not: H = exp(-sum(P(P > 1e-5) .* log(P(P > 1e-5)))); ???
P = P / sumP;
end

function P = tsneV(X, labels, no_dims, initial_dims, perplexity)
%TSNE Performs symmetric t-SNE on dataset X
%
%   mappedX = tsne(X, labels, no_dims, initial_dims, perplexity)
%   mappedX = tsne(X, labels, initial_solution, perplexity)
%
% The function performs symmetric t-SNE on the NxD dataset X to reduce its
% dimensionality to no_dims dimensions (default = 2). The data is
% preprocessed using PCA, reducing the dimensionality to initial_dims
% dimensions (default = 30). Alternatively, an initial solution obtained
% from an other dimensionality reduction technique may be specified in
% initial_solution. The perplexity of the Gaussian kernel that is employed
% can be specified through perplexity (default = 30). The labels of the
% data are not used by t-SNE itself, however, they are used to color
% intermediate plots. Please provide an empty labels matrix [] if you
% don't want to plot results during the optimization.
% The low-dimensional data representation is returned in mappedX.
%
%
% (C) Laurens van der Maaten, 2010
% University of California, San Diego


if ~exist('labels', 'var')
    labels = [];
end
if ~exist('no_dims', 'var') || isempty(no_dims)
    no_dims = 2;
end
if ~exist('initial_dims', 'var') || isempty(initial_dims)
    initial_dims = min(50, size(X, 2));
end
if ~exist('perplexity', 'var') || isempty(perplexity)
    perplexity = 30;
end

% First check whether we already have an initial solution
if numel(no_dims) > 1
    initial_solution = true;
    ydata = no_dims;
    no_dims = size(ydata, 2);
    perplexity = initial_dims;
else
    initial_solution = false;
end

% Normalize input data
X = X - min(X(:));
X = X / max(X(:));
X = bsxfun(@minus, X, mean(X, 1));

%     % Perform preprocessing using PCA
%     if ~initial_solution
%         disp('Preprocessing data using PCA...');
%         if size(X, 2) < size(X, 1)
%             C = X' * X;
%         else
%             C = (1 / size(X, 1)) * (X * X');
%         end
%         [M, lambda] = eig(C);
%         [lambda, ind] = sort(diag(lambda), 'descend');
%         M = M(:,ind(1:initial_dims));
%         lambda = lambda(1:initial_dims);
%         if ~(size(X, 2) < size(X, 1))
%             M = bsxfun(@times, X' * M, (1 ./ sqrt(size(X, 1) .* lambda))');
%         end
%         X = bsxfun(@minus, X, mean(X, 1)) * M;
%         clear M lambda ind
%     end

% Compute pairwise distance matrix
sum_X = sum(X .^ 2, 2);
D = bsxfun(@plus, sum_X, bsxfun(@plus, sum_X', -2 * (X * X')));

% Compute joint probabilities
P = d2p(D, perplexity, 1e-5);                                           % compute affinities using fixed perplexity
clear D

%     % Run t-SNE
%     if initial_solution
%         ydata = tsne_p(P, labels, ydata);
%     else
%         ydata = tsne_p(P, labels, no_dims);
%     end
end


function D = XtoD(X)
sum_X = sum(X .^ 2, 2);
D = bsxfun(@plus, sum_X, bsxfun(@plus, sum_X', -2 * (X * X')));
end


function ydata = tsne_p(P, labels, no_dims)
%TSNE_P Performs symmetric t-SNE on affinity matrix P
%
%   mappedX = tsne_p(P, labels, no_dims)
%
% The function performs symmetric t-SNE on pairwise similarity matrix P
% to create a low-dimensional map of no_dims dimensions (default = 2).
% The matrix P is assumed to be symmetric, sum up to 1, and have zeros
% on the diagonal.
% The labels of the data are not used by t-SNE itself, however, they
% are used to color intermediate plots. Please provide an empty labels
% matrix [] if you don't want to plot results during the optimization.
% The low-dimensional data representation is returned in mappedX.
%
%
% (C) Laurens van der Maaten, 2010
% University of California, San Diego


if ~exist('labels', 'var')
    labels = [];
end
if ~exist('no_dims', 'var') || isempty(no_dims)
    no_dims = 2;
end

% First check whether we already have an initial solution
if numel(no_dims) > 1
    initial_solution = true;
    ydata = no_dims;
    no_dims = size(ydata, 2);
else
    initial_solution = false;
end

% Initialize some variables
n = size(P, 1);                                     % number of instances
momentum = 0.5;                                     % initial momentum
final_momentum = 0.8;                               % value to which momentum is changed
mom_switch_iter = 250;                              % iteration at which momentum is changed
stop_lying_iter = 100;                              % iteration at which lying about P-values is stopped
max_iter = 1000;                                    % maximum number of iterations
epsilon = 500;                                      % initial learning rate
min_gain = .01;                                     % minimum gain for delta-bar-delta

% Make sure P-vals are set properly
P(1:n + 1:end) = 0;                                 % set diagonal to zero
P = 0.5 * (P + P');                                 % symmetrize P-values
P = max(P ./ sum(P(:)), realmin);                   % make sure P-values sum to one
const = sum(P(:) .* log(P(:)));                     % constant in KL divergence
if ~initial_solution
    P = P * 4;                                      % lie about the P-vals to find better local minima
end

% Initialize the solution
if ~initial_solution
    ydata = .0001 * randn(n, no_dims);
end
y_incs  = zeros(size(ydata));
gains = ones(size(ydata));

% Run the iterations
for iter=1:max_iter
    
    % Compute joint probability that point i and j are neighbors
    sum_ydata = sum(ydata .^ 2, 2);
    num = 1 ./ (1 + bsxfun(@plus, sum_ydata, bsxfun(@plus, sum_ydata', -2 * (ydata * ydata')))); % Student-t distribution
    num(1:n+1:end) = 0;                                                 % set diagonal to zero
    Q = max(num ./ sum(num(:)), realmin);                               % normalize to get probabilities
    
    % Compute the gradients (faster implementation)
    L = (P - Q) .* num;
    y_grads = 4 * (diag(sum(L, 1)) - L) * ydata;
    
    % Update the solution
    gains = (gains + .2) .* (sign(y_grads) ~= sign(y_incs)) ...         % note that the y_grads are actually -y_grads
        + (gains * .8) .* (sign(y_grads) == sign(y_incs));
    gains(gains < min_gain) = min_gain;
    y_incs = momentum * y_incs - epsilon * (gains .* y_grads);
    ydata = ydata + y_incs;
    ydata = bsxfun(@minus, ydata, mean(ydata, 1));
    
    % Update the momentum if necessary
    if iter == mom_switch_iter
        momentum = final_momentum;
    end
    if iter == stop_lying_iter && ~initial_solution
        P = P ./ 4;
    end
    
    %         % Print out progress
    %         if ~rem(iter, 10)
    %             cost = const - sum(P(:) .* log(Q(:)));
    %             disp(['Iteration ' num2str(iter) ': error is ' num2str(cost)]);
    %         end
    
    %         % Display scatter plot (maximally first three dimensions)
    %         if ~rem(iter, 10) && ~isempty(labels)
    %             if no_dims == 1
    %                 scatter(ydata, ydata, 9, labels, 'filled');
    %             elseif no_dims == 2
    %                 scatter(ydata(:,1), ydata(:,2), 9, labels, 'filled');
    %             else
    %                 scatter3(ydata(:,1), ydata(:,2), ydata(:,3), 40, labels, 'filled');
    %             end
    %             axis tight
    %             axis off
    %             drawnow
    %         end
end
end

function ydata = tsne_d(D, labels, no_dims, perplexity)
%TSNE_D Performs symmetric t-SNE on the pairwise Euclidean distance matrix D
%
%   mappedX = tsne_d(D, labels, no_dims, perplexity)
%   mappedX = tsne_d(D, labels, initial_solution, perplexity)
%
% The function performs symmetric t-SNE on the NxN pairwise Euclidean
% distance matrix D to construct an embedding with no_dims dimensions
% (default = 2). An initial solution obtained from an other dimensionality
% reduction technique may be specified in initial_solution.
% The perplexity of the Gaussian kernel that is employed can be specified
% through perplexity (default = 30). The labels of the data are not used
% by t-SNE itself, however, they are used to color intermediate plots.
% Please provide an empty labels matrix [] if you don't want to plot
% results during the optimization.
% The data embedding is returned in mappedX.
%
%
% (C) Laurens van der Maaten, 2010
% University of California, San Diego


if ~exist('labels', 'var')
    labels = [];
end
if ~exist('no_dims', 'var') || isempty(no_dims)
    no_dims = 2;
end
if ~exist('perplexity', 'var') || isempty(perplexity)
    perplexity = 30;
end

% First check whether we already have an initial solution
if numel(no_dims) > 1
    initial_solution = true;
    ydata = no_dims;
    no_dims = size(ydata, 2);
else
    initial_solution = false;
end

% Compute joint probabilities
D = D / max(D(:));                                                      % normalize distances
P = d2p(D .^ 2, perplexity, 1e-5);                                      % compute affinities using fixed perplexity

% Run t-SNE
if initial_solution
    ydata = tsne_p(P, labels, ydata);
else
    ydata = tsne_p(P, labels, no_dims);
end
end
