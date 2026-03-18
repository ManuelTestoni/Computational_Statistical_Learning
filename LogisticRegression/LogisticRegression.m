clear 
close all

load output_wine
n_classes = 3;

tl = templateLinear('Learner','logistic','Regularization','lasso');
Mdl = fitcecoc(t,y,'Learners',tl,'coding','onevsall');
w = zeros(2,n_classes);
b = zeros(1,n_classes);
for i = 1 : n_classes
    w(:,i) = Mdl.BinaryLearners{i,1}.Beta;
    b(i) = Mdl.BinaryLearners{i,1}.Bias;
end
prob_tot = @(w,b,x) sum(exp(x*w+b));
sigma = @(w,b,x,p) exp(x*w+b)/p;
colori = 'ykrbgc';
x1v = linspace(min(t(:,1)),max(t(:,1)),100);
x2v = linspace(min(t(:,2)),max(t(:,2)),100);
figure
for i = 1:n_classes
    plot(t(y == i,1),t(y == i,2),['o' colori(mod(i,6)+1)])
    hold on
end
[X1,X2] = meshgrid(x1v,x2v);
X1 = X1(:); X2 = X2(:);
for i = 1 : length(X1)
    p_tot = prob_tot(w,b,[X1(i),X2(i)]);
    s(1) = sigma(w(:,1),b(1),[X1(i),X2(i)],p_tot);
    s(2) = sigma(w(:,2),b(2),[X1(i),X2(i)],p_tot);
    s(3) = sigma(w(:,3),b(3),[X1(i),X2(i)],p_tot);
    [~,imax] = max(s);
    plot(X1(i),X2(i),['.' colori(imax+1)])
end

[X1,X2] = meshgrid(x1v,x2v);
S1 = zeros(100);
S2 = zeros(100);
S3 = zeros(100);
for i = 1 : 100
    for j = 1 : 100
        p_tot = prob_tot(w,b,[X1(i,j),X2(i,j)]);
        S1(i,j) = sigma(w(:,1),b(1),[X1(i,j),X2(i,j)],p_tot);
        S2(i,j) = sigma(w(:,2),b(2),[X1(i,j),X2(i,j)],p_tot);
        S3(i,j) = sigma(w(:,3),b(3),[X1(i,j),X2(i,j)],p_tot);
    end
end
figure, surf(X1,X2,S1), hold on, surf(X1,X2,S2), surf(X1,X2,S3)