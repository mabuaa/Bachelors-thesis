% m = [0,0,0,0,0,0; 5,4,6,3,5,6; -1.5,-1,-1.9,-1,-3,-1; 2,-5,5,1.9,1,1];
m = [randperm(10,6);randperm(10,6);randperm(10,6);randperm(10,6);randperm(10,6);randperm(10,6);randperm(10,6)];

sigma = normalize(rand(6));
sigma = sigma*sigma';

r1 = mvnrnd(m(1,:),sigma,300);
r2 = mvnrnd(m(2,:),sigma,300);
r3 = mvnrnd(m(3,:),sigma,300);
r4 = mvnrnd(m(4,:),sigma,300);
r5 = mvnrnd(m(5,:),sigma,300);
r6 = mvnrnd(m(6,:),sigma,300);
r7 = mvnrnd(m(7,:),sigma,300);

data = [r1;r2;r3;r4;r5;r6;r7];

pcaX = pca(data);
pcaX = data*pcaX(:,1:2);

tsneX = tsne(data);

l = [ones(300,1);2*ones(300,1);3*ones(300,1);4*ones(300,1);5*ones(300,1);6*ones(300,1);7*ones(300,1)];

figure(1)
gscatter(pcaX(:,1),pcaX(:,2),l)
title('PCA applied to a mixture of multivariate Gaussians')

figure(2)
gscatter(tsneX(:,1),tsneX(:,2),l)
title('t-SNE applied to a mixture of multivariate Gaussians')