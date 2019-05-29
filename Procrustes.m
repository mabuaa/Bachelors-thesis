datai = 13
load(strcat('C:\Users\matej\OneDrive\Desktop\Bachelor''s thesis\AAA\datai', int2str(datai), '\datai', int2str(datai), 'fullTSNEandPrerequisits.mat'));

for p = 5:5:50
    load(strcat('C:\Users\matej\OneDrive\Desktop\Bachelor''s thesis\AAA\datai', int2str(datai), '\percentage',int2str(p), 'up to st', int2str(4), '.mat'));
    procrustes_errors = zeros(experiments+1, 4, 10);
    procrustes_errors_afterKNN = zeros(experiments+1, 4,10);
    for ftsnei = 1:10
        X2D = full_tSNE_X{datai, ftsnei};
        %% Random sampling
        for i = 1:experiments
            sampled_idx_r = sampled_indices{datai,i};
            Xr2D = embeddings{datai, i};
            Xr2Dknn = embeddings_afterKNNreg{datai, i};
            full_Xr2D = X2D(sampled_idx_r, :);
            procrustes_errors(i, 1, ftsnei) = procrustes(full_Xr2D, Xr2D);
            procrustes_errors_afterKNN(i,1, ftsnei) = procrustes(X2D, Xr2Dknn);
        end
        avg_loss = mean(procrustes_errors(1:(end-1),1, ftsnei));
        procrustes_errors(experiments+1,1, ftsnei) = avg_loss;
        avg_loss_afterKNN = mean(procrustes_errors_afterKNN(1:(end-1),1, ftsnei));
        procrustes_errors_afterKNN(experiments+1,1, ftsnei) = avg_loss;
        %% (Old) random walk sampling
        for i = 1:experiments
            sampled_idx_r = sampled_indices{datai,experiments+i};
            Xr2D = embeddings{datai, experiments+i};
            Xr2Dknn = embeddings_afterKNNreg{datai,experiments+ i};
            full_Xr2D = X2D(sampled_idx_r, :);
            procrustes_errors(i, 2, ftsnei) = procrustes(full_Xr2D, Xr2D);
            procrustes_errors_afterKNN(i,2, ftsnei) = procrustes(X2D, Xr2Dknn);
        end
        avg_loss = mean(procrustes_errors(1:(end-1),2, ftsnei));
        procrustes_errors(experiments+1,2, ftsnei) = avg_loss;
        avg_loss_afterKNN = mean(procrustes_errors_afterKNN(1:(end-1),2, ftsnei));
        procrustes_errors_afterKNN(experiments+1,2, ftsnei) = avg_loss;
        %% (New) random walk sampling
        for i = 1:experiments
            sampled_idx_r = sampled_indices{datai,2*experiments+i};
            Xr2D = embeddings{datai, 2*experiments+i};
            Xr2Dknn = embeddings_afterKNNreg{datai,2*experiments+ i};
            full_Xr2D = X2D(sampled_idx_r, :);
            procrustes_errors(i, 3, ftsnei) = procrustes(full_Xr2D, Xr2D);
            procrustes_errors_afterKNN(i,3, ftsnei) = procrustes(X2D, Xr2Dknn);
        end
        avg_loss = mean(procrustes_errors(1:(end-1),3));
        procrustes_errors(experiments+1,3) = avg_loss;
        avg_loss_afterKNN = mean(procrustes_errors_afterKNN(1:(end-1),3, ftsnei));
        procrustes_errors_afterKNN(experiments+1,3, ftsnei) = avg_loss;
        %% Hubness sampling
        for i = 1:experiments
            sampled_idx_r = sampled_indices{datai,3*experiments+1};
            Xr2D = embeddings{datai, 3*experiments+i};
            Xr2Dknn = embeddings_afterKNNreg{datai,3*experiments+ i};
            full_Xr2D = X2D(sampled_idx_r, :);
            procrustes_errors(i, 4, ftsnei) = procrustes(full_Xr2D, Xr2D);
            procrustes_errors_afterKNN(i,4, ftsnei) = procrustes(X2D, Xr2Dknn);
        end
        avg_loss = mean(procrustes_errors(1:(end-1),4, ftsnei));
        procrustes_errors(experiments+1,4, ftsnei) = avg_loss;
        avg_loss_afterKNN = mean(procrustes_errors_afterKNN(1:(end-1),4, ftsnei));
        procrustes_errors_afterKNN(experiments+1,4, ftsnei) = avg_loss;
    end
    savethis = strcat('C:\Users\matej\OneDrive\Desktop\Bachelor''s thesis\AAA\datai', int2str(datai), '\percentage',int2str(p), 'up to st', int2str(4), 'with procrustes.mat');
    save(savethis, 'embeddings', 'errors', 'times', 'embeddings_afterKNNreg', 'errors_afterKNNreg', 'times_afterKNNreg', 'sampled_indices', 'procrustes_errors', 'procrustes_errors_afterKNN');
end