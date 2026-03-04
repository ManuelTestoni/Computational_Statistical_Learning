clear 
close all

load fisheriris
n = size(meas,1);
n_neib = 5;
rng('default')
ind = randperm(n);
n_test = ceil(n/4);
n = n - n_test;
x = meas(ind(1:n),:);
y = species(ind(1:n));
test = meas(ind(n+1:end),:);
y_test = species(ind(n+1:end));

figure
ind = find(contains(y,'setosa'));
plot3(x(ind,1),x(ind,2),x(ind,3),'or')
hold on
ind = find(contains(y,'versicolor'));
plot3(x(ind,1),x(ind,2),x(ind,3),'ok')
ind = find(contains(y,'virginica'));
plot3(x(ind,1),x(ind,2),x(ind,3),'ob')
grid on

pred = cell(n_test,1);
for k = 1 : n_test
    dist = zeros(n,1);
    for i = 1 : n
        dist(i) = norm(x(i,:)-test(k,:));
    end
    [ds,is] = sort(dist);
    [s,~,j] = unique(y(is(1:n_neib)));
    pred{k} = s{mode(j)};
    switch pred{k}
        case 'setosa'
            plot3(test(k,1),test(k,2),test(k,3),'og','markerfacecolor','r','markersize',8)
        case 'versicolor'
            plot3(test(k,1),test(k,2),test(k,3),'og','markerfacecolor','k','markersize',8)
        case 'virginica'
            plot3(test(k,1),test(k,2),test(k,3),'og','markerfacecolor','b','markersize',8)
    end
end

Mdl = fitcknn(x,y,'NumNeighbors',n_neib);
label = predict(Mdl,test);
for k = 1 : n_test
    switch label{k}
        case 'setosa'
            plot3(test(k,1),test(k,2),test(k,3),'xr','markersize',16)
        case 'versicolor'
            plot3(test(k,1),test(k,2),test(k,3),'xk','markersize',16)
        case 'virginica'
            plot3(test(k,1),test(k,2),test(k,3),'xb','markersize',16)
    end
end