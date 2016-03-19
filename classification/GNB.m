function GNB()
	[X, Y, K] = read();
	[mu, sigma, theta] = train(X(:,:), Y, K);
	[X_test, Y_test]=read_test();
	Y_t=zeros(length(Y_test), 1);
	for i=1:length(Y_test)
		Y_t(i)=predict(mu, sigma, theta, X_test(i,:), K);
	end
	sum(Y_t==Y_test)/length(Y_test)
end

function [X_test, Y_test] = read_test()
	X_test = load('data/iris.test');
	Y_test = X_test(:, 1);	
	X_test = X_test(:, 2:(size(X_test)(2)));
end

function [X, Y, K]=read()
	X = load('data/iris.train');
	Y = X(:, 1);
	X = X(:, 2:(size(X)(2)));
	K = unique(Y);
end

function [mu, sigma, theta] = train(X, Y, K)
	mu=zeros(length(K), size(X)(2));
	sigma=zeros(length(K), size(X)(2));
	theta=zeros(length(K), 1);
	for i = 1:length(K)
		mu(i,:) = mean(X(Y==K(i), :));		
		sigma(i,:) = var(X(Y==K(i), :));
		theta(i)=sum(Y==K(i))/length(Y);
	end	
end

function y=predict(mu, sigma, theta, x, K)
	t=zeros(length(K), 1);
	for i=1:length(K)
		t(i)=log(theta(i))+sum(log(1./(sqrt(2*pi)*sigma(i,:)))-(x-mu(i,:)).*(x-mu(i,:))./(2*sigma(i,:)));
	end
	[value, index]=max(t);
	y=K(index);
end
