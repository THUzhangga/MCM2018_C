%[num letter] = xlsread('C:\Users\dell\Desktop\美赛\ProblemCData.xlsx',1);
%letter(1,:) = [];
%letter = cell2mat(letter);




dc23 = xlsread('C:\Users\dell\Desktop\美赛\weight.xlsx',1);
n = size(dc23,1);
[X,lambda] = eig(dc23);
max_lambda = max(max(lambda));
CI = (max_lambda - n) / (n - 1);
RI = [0 0 0.58 0.90 1.12 1.24 1.32 1.41 1.45 1.49 1.51];
CR = CI / RI(n);
fprintf('CR = %d\n',CR);
if CR > 0.1
    fprintf('第一步一致性分析不通过\n');
    return;
end
pos = find(lambda == max_lambda, 1);
vector23 = X(:,pos);

dc12 = xlsread('C:\Users\dell\Desktop\美赛\weight.xlsx',2,'A1:X4');
lambda_set = [];
vector_set = [];
for i = 1:6
    temp = dc12(1:4, (4i-3):4i);
    [X,lambda] = eig(temp);
    max_lambda = max(max(lambda));
    pos = find(lambda == max_lambda, 1);
    vector_temp = X(:,pos);
    lambda_set = [lambda_set max_lambda];
    vector_set = [vector_set vector_temp];
end

vector = vector_set * vector23;

temp = (lambda_set - 4) ./ 2;
CI2 = temp * vector23;
CR = CI + CI2 / RI(4);
fprintf('CR2 = %d\n',CR);
if CR > 0.1
    fprintf('第二步一致性分析不通过\n');
    return;
end
vector
return;