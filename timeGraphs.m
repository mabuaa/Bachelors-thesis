l = {'orl', 'seismic', 'har', 'svhn', 'cnae9', 'coil20', 'secom', 'bank', 'cifar10'};
figure(1)
w = 1;
for datai = [13, 15, 9, 19, 5, 6, 14, 3, 4]
    subplot(3,3,w)
    measures = zeros(6,9);
    load(strcat('C:\Users\matej\OneDrive\Desktop\Bachelor''s thesis\AAA\datai', int2str(datai), '\datai', int2str(datai), 'fullTSNEandPrerequisits.mat'));
    fulltime = avgtime_fullTSNE + timesDandPfullTSNE{datai, 2};
    measures(1,:) = 10:5:50;
    measures(6,:) = fulltime*ones(1,9);
    j = 1;
    for p = 10:5:50
        load(strcat('C:\Users\matej\OneDrive\Desktop\Bachelor''s thesis\AAA\datai', int2str(datai), '\percentage',int2str(p), 'up to st', int2str(4), 'with procrustes.mat'));
        for st = 1:4
            measures(st+1,j) = times{datai, st*(experiments+1)} + timesDandPfullTSNE{datai, 1};
        end
        j= j+1;
    end
    
    measures = measures';
    plot(measures(:,1),measures(:,2), 'b-o', measures(:,1), measures(:,3), 'm-s', measures(:,1), measures(:,4), 'g-o', measures(:,1), measures(:,5), 'k-h',measures(:,1),measures(:,6), 'b-*')
    legend('rs', 'rw_1', 'rw_2', 'hs', 'full');
    title(strcat('Data  "', l{w}, '"'));
    w = w+1;
end