datai = 6;
load(strcat('C:\Users\matej\OneDrive\Desktop\Bachelor''s thesis\AAA\datai', int2str(datai), '\datai', int2str(datai), 'fullTSNEandPrerequisits.mat'));

 figure(1)
X2D = full_tSNE_X{datai, 8};
 gscatter(X2D(:,1), X2D(:,2), Y)
 title('t-SNE applied to coil20 data')

figure(2)
p = 25;
load(strcat('C:\Users\matej\OneDrive\Desktop\Bachelor''s thesis\AAA\datai', int2str(datai), '\percentage',int2str(p), 'up to st', int2str(4), 'with procrustes.mat'));
plotlabels = {'Random sampling', 'First random walk sampling', 'Second random walk sampling', 'Hubness sampling'};
z=7
for w = 1:4
    subplot(2,2,w)
    X = embeddings_afterKNNreg{datai, (w-1)*30+z};
    if w ~= 4
    Y = sampled_indices{datai, (w-1)*3+z};
    else
        Y = sampled_indices{datai, (w-1)*3+1};
    end
    a = 1:size(X,1);
    a = a';
    b = setdiff(a, Y);
    idx_afterKNN = [Y; b];
    gscatter(X(:,1), X(:,2), coil20_y(idx_afterKNN))
    l = strcat('t-SNE applied to :',plotlabels{w}, ' of data');
    title(l)
    
end