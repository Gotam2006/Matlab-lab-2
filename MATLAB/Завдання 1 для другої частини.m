clc;
clear;
close all;

%% =========================================
% ЧАСТИНА 1 — XOR
%% =========================================

% Вхідні дані XOR
P = [0 0 1 1;
     0 1 0 1];

% Цільові значення
T = [0 1 1 0];

% Створення мережі
net_xor = feedforwardnet(2);

% Функції активації
net_xor.layers{1}.transferFcn = 'logsig';
net_xor.layers{2}.transferFcn = 'logsig';

% Метод навчання
net_xor.trainFcn = 'trainlm';

% Навчання
net_xor = train(net_xor, P, T);

% Тестування
Yxor = net_xor(P);

disp('Результати XOR:');
disp(round(Yxor));

%% =========================================
% ЧАСТИНА 2 — РОЗПІЗНАВАННЯ БУКВ
%% =========================================

% Кількість букв
N = 26;

% Розмір символу 5x7 = 35
alphabet = zeros(35, N);

%% Генерація простих випадкових букв
% (щоб код гарантовано працював)

rng(1);

for i = 1:N
    alphabet(:,i) = round(rand(35,1));
end

%% Цільові вектори
targets = eye(N);

%% Створення мережі

net = feedforwardnet(20);

net.layers{1}.transferFcn = 'logsig';
net.layers{2}.transferFcn = 'logsig';

net.trainFcn = 'trainlm';

%% Навчання мережі

net = train(net, alphabet, targets);

%% Тестування без шуму

Y = net(alphabet);

[~, predicted] = max(Y);
[~, actual] = max(targets);

accuracy = sum(predicted == actual) / N * 100;

fprintf('\nТочність без шуму: %.2f %%\n', accuracy);

%% =========================================
% ДОСЛІДЖЕННЯ ШУМУ
%% =========================================

noise_levels = 0:0.05:0.5;

errors = zeros(length(noise_levels),1);

for n = 1:length(noise_levels)

    sigma = noise_levels(n);

    total_error = 0;

    for letter = 1:N

        for k = 1:10

            % Генерація шуму
            noise = sigma * randn(35,1);

            % Обмеження шуму
            noise(noise > 1) = 1;
            noise(noise < -1) = -1;

            % Зашумлений вектор
            noisy_input = alphabet(:,letter) + noise;

            % Вихід мережі
            output = net(noisy_input);

            % Помилка
            err = norm(output - targets(:,letter));

            total_error = total_error + err;

        end
    end

    % Середня помилка
    errors(n) = total_error / (N * 10);

end

%% =========================================
% ПОБУДОВА ГРАФІКА
%% =========================================

figure;

plot(noise_levels, errors, '-ob', ...
    'LineWidth', 2, ...
    'MarkerSize', 8);

grid on;

xlabel('Рівень шуму');
ylabel('Середня помилка');

title('Залежність помилки мережі від рівня шуму');

xlim([0 0.5]);

disp('Графік побудовано успішно.');