clear 
close all

%dataset = 'wine_scale.txt';
%[x,y] = leggi_light(dataset);

[x,y] = wine_dataset;
x = x';
[y,~] = find(y == 1);

[l,d] = size(x);
n_classes = max(y);
[y,is] = sort(y);
x = x(is,:);
li = zeros(1,n_classes);
ind = cell(n_classes);
smi = zeros(n_classes,d);
for i = 1:n_classes
    ind{i} = find(y == i);
    li(i) = length(ind{i});
    smi(i,:) = mean(x(ind{i},:));
end
sm = mean(x);

SB = zeros(d);
SW = zeros(d);
for i = 1:n_classes
    SB = SB + li(i)*(smi(i,:)'-sm')*(smi(i,:)-sm);
    SW = SW + (x(ind{i},:)-repmat(smi(i,:),li(i),1))'*(x(ind{i},:)-repmat(smi(i,:),li(i),1));
end

[V,D] = eig(SB,SW);
lambda = diag(D);
[lambda,inds] = sort(lambda,'descend');
if n_classes == 3
    w1 = V(:,inds(1))/norm(V(:,inds(1)));
    w2 = V(:,inds(2))/norm(V(:,inds(2)));
    t = zeros(l,2);
    colori = 'krb';
    figure
    for i = 1:3
        t(ind{i},:) = x(ind{i},:)*[w1,w2];
        plot(t(ind{i},1),t(ind{i},2),['o' colori(i)])
        hold on
    end
else
    w1 = V(:,inds(1))/norm(V(:,inds(1)));
    w2 = V(:,inds(2))/norm(V(:,inds(2)));
    w3 = V(:,inds(3))/norm(V(:,inds(3)));
    t = zeros(l,3);
    colori = 'ykrbgcm';
    figure
    for i = 1:n_classes
        t(ind{i},:) = x(ind{i},:)*[w1,w2,w3];
        plot3(t(ind{i},1),t(ind{i},2),t(ind{i},3),['o' colori(mod(i,7)+1)])
        hold on
    end
end

%save output_wine_new t y