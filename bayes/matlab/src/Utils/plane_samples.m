% config
n = 60;
sigma = 0.1;

% choose a grid of points
[x y] = meshgrid((1:n)/n);

% gaussian kernel
L = exp(- (bsxfun(@minus,x(:),x(:)').^2 + ...
           bsxfun(@minus,y(:),y(:)').^2) / sigma^2);

% sample
samplea = sample_dpp(decompose_kernel(L));
l = length(samplea);
sampleb = randsample(n*n,l);
  
% plot
figure(209);

subplot(1,2,1);
plot(x(samplea),y(samplea),'b.');
axis([0 1.02 0 1.02]);
axis square;
set(gca,'YTick',[]);
set(gca,'XTick',[]);
xlabel('DPP');

subplot(1,2,2);
plot(x(sampleb),y(sampleb),'r.');
axis([0 1.02 0 1.02]);
axis square;
set(gca,'YTick',[]);
set(gca,'XTick',[]);
xlabel('Independent');

set(gcf,'Position',[100 100 400 200]);
