clc;
clear;
close all;

%% Вхідні та цільові дані
P = [5 4 3 2];          % Вхідна послідовність
T = [10 20 30 40];      % Бажана вихідна послідовність

%% ---------------------------------------------------------
% ЕКСПЕРИМЕНТ 1
% Адаптивна лінійна мережа з 2 блоками затримки
%% ---------------------------------------------------------

disp('=== Мережа з 2 блоками затримки ===');

delays = 1:2;   % Два блоки затримки

% Створення лінійної нейронної мережі
net = newlin(minmax(P),1,delays,0.01);

% Підготовка часових рядів
[Xs,Xi,Ai,Ts] = preparets(net,num2cell(P),num2cell(T));

% Навчання мережі
net.trainParam.epochs = 200;
net = train(net,Xs,Ts,Xi,Ai);

% Робота мережі
Y = net(Xs,Xi,Ai);

% Перетворення результатів
Ynum = cell2mat(Y);

disp('Вихід мережі:');
disp(Ynum);

% Помилка
error = gsubtract(Ts,Y);
perf = perform(net,Ts,Y);

disp(['Середньоквадратична помилка: ', num2str(perf)]);

%% Графік
figure;
plot(cell2mat(Ts),'b-o','LineWidth',2);
hold on;
plot(Ynum,'r-*','LineWidth',2);

grid on;
xlabel('Номер елемента');
ylabel('Значення');
title('Порівняння бажаного та фактичного виходу');

legend('Бажаний вихід','Вихід мережі');

%% ---------------------------------------------------------
% ЕКСПЕРИМЕНТ 2
% Дослідження впливу кількості блоків затримки
%% ---------------------------------------------------------

disp(' ');
disp('=== Дослідження кількості блоків затримки ===');

maxDelays = 10;
errors = zeros(1,maxDelays);

for d = 1:maxDelays
    
    % Створення мережі
    netTest = newlin(minmax(P),1,1:d,0.01);
    
    % Підготовка даних
    [Xs,Xi,Ai,Ts] = preparets(netTest,num2cell(P),num2cell(T));
    
    % Навчання
    netTest.trainParam.epochs = 200;
    netTest = train(netTest,Xs,Ts,Xi,Ai);
    
    % Робота мережі
    Y = netTest(Xs,Xi,Ai);
    
    % Обчислення помилки
    errors(d) = perform(netTest,Ts,Y);
    
    fprintf('Кількість блоків затримки: %d --> Помилка: %f\n', ...
        d, errors(d));
end

%% Пошук оптимальної кількості блоків
[minError,optDelay] = min(errors);

disp(' ');
disp(['Оптимальна кількість блоків затримки: ', num2str(optDelay)]);
disp(['Мінімальна помилка: ', num2str(minError)]);

%% Графік залежності помилки
figure;

plot(1:maxDelays,errors,'m-o','LineWidth',2);

grid on;
xlabel('Кількість блоків затримки');
ylabel('Помилка');
title('Вплив кількості блоків затримки на помилку мережі');