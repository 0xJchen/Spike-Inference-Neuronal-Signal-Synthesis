clc;clear;close all;
% load('D:\neuron\NeuralNetwork\DeepLearning\project\data\data_20200709_LPV1_pre.mat');
% calsignal = data_20200709_LPV1_pre;

load('D:\neuron\NeuralNetwork\DeepLearning\project\data\data_20201212_pva1_pre_adjust2.mat')
calsignal = data_20201212_pva1_pre_adjust2;
sn =size(calsignal,3);
cn =size(calsignal,2);
%%%  biao
biao_on = zeros(sn,15);
biao_off = zeros(sn,15);
for s = 2 : sn-1
    tag = diff( squeeze( tag_sum_20201212(:,1,s) ) );
%     figure(s)
%     plot(tag_sum_20201212(:,1,s) )
    for i = 1:3600-5
        if tag(i) < 10 && tag(i+1) > 10       % find onset
            [row,col,latest]  = find(biao_on(s,:),1,'last');
            if isempty(latest)
                biao_on(s,1) = i+2;
            else
                if i+2 - latest  < 20
                    i;
                else
                    biao_on(s, col+1) = i+2;
                end
            end
        end
        if tag(i) < -10 && tag(i+1) > -10
            [row,col,latest]  = find(biao_off(s,:),1,'last');
            [tmp,tmp1,latest_on]  = find(biao_on(s,:),1,'last');
            if isempty(latest) && ~isempty(latest_on) 
                biao_off(s,1) = i;
            else
                if i+1 - latest  < 20
                    biao_off(s, col) = i;
                else
                    biao_off(s, col+1) = i;
                end
            end
        end
    end
end

tmp1 = biao_on(biao_on>0);
tmp2 = biao_off(biao_off>0);
numel(tmp1)
numel(tmp2)
tmp = biao_off - biao_on;
duration = max(abs(tmp),[],2)
biao_on(1,1:13) = [19:280:3600] ;
biao_on(sn,1:13) = [19:280:3600] ;
biao_off(1,1:13) = [19:280:3600] ;
biao_off(sn,1:13) = [19:280:3600] ;

%%%
inN=[1:38];
exN = [8,2,5,4,6,10,11,12,1,15,20,100,101,102,103,201,202,203,204,104,205,206,105,106,107,108,207,208,209,21,22,23,24,25,26,27,29,30,28,40,50,60,70,80,75];
exN = exN+40;
exN = unique(exN);
res = [];
in_ex = [];
class = [];
nb = 0;
th = 0.15;

for c =  inN
    c;
    data = squeeze(calsignal(:,c,:));
    for s = 1 :sn
        ocos = squeeze(data(:,s));
        biao = biao_on(s,:);
        biao = biao(biao>0);
        for t = 1:numel(biao)
            if biao(t)+59 <= 3600 && biao(t)-15 >0
                ocot = squeeze( ocos(biao(t)-15:3 : biao(t)+59 ,:) );
                f0 =  mean(squeeze( ocot(1:5) ) );
                ocot = (ocot -   f0) / f0;
                if max(ocot) >= th
                    nb = nb+1;
                    in_ex = [in_ex;ocot'] ;
                    class(nb,1) = -1;
                end
            end
        end
    end
end
nin = nb
for c =  exN
    c;
    data = squeeze(calsignal(:,c,:));
    for s = 1: sn
        ocos = squeeze(data(:,s));
        biao = biao_on(s,:);
        biao = biao(biao>0);

        for t = 1:numel(biao) %randperm( numel(biao),1 )
            if biao(t)+59 <= 3600 && biao(t)-15 >0
                ocot = squeeze( ocos(biao(t)-15:3 : biao(t)+59 ,:) );
                f0 =  mean(squeeze( ocot(1:5) ) );
                ocot = (ocot -   f0) / f0;
                if max(ocot) >= th
                    if nb == nin*2
                        continue 
                    end
                    nb = nb+1;
                    in_ex = [in_ex;ocot'] ;
                    class(nb,1) = 1;
                end
            end
        end
    end
end
nb
%%%

load('D:\neuron\NeuralNetwork\DeepLearning\project\data\data_20200709_LPV1_pre.mat')
calsignal = data_20200709_LPV1_pre;
sn =size(calsignal,3);
cn =size(calsignal,2);
%%%  biao
biao_on = zeros(sn,15);
biao_off = zeros(sn,15);
for s = 2 : sn-1
    tag = diff( squeeze( tag_sum_20200709(1,:,s) ) );
%     figure(s)
%     plot(tag_sum_20200709(1,:,s) )
    for i = 1:3600-5
        if tag(i) < 10 && tag(i+1) > 10       % find onset
            [row,col,latest]  = find(biao_on(s,:),1,'last');
            if isempty(latest)
                biao_on(s,1) = i+2;
            else
                if i+2 - latest  < 20
                    i;
                else
                    biao_on(s, col+1) = i+2;
                end
            end
        end
        if tag(i) < -10 && tag(i+1) > -10
            [row,col,latest]  = find(biao_off(s,:),1,'last');
            [tmp,tmp1,latest_on]  = find(biao_on(s,:),1,'last');
            if isempty(latest) && ~isempty(latest_on) 
                biao_off(s,1) = i;
            else
                if i+1 - latest  < 20
                    biao_off(s, col) = i;
                else
                    biao_off(s, col+1) = i;
                end
            end
        end
    end
end

tmp1 = biao_on(biao_on>0);
tmp2 = biao_off(biao_off>0);
numel(tmp1)
numel(tmp2)
tmp = biao_off - biao_on;
duration = max(abs(tmp),[],2)
biao_on(1,1:15) = [19:251:3600] ;
biao_on(sn,1:15) = [19:251:3600] ;
biao_off(1,1:15) = [19:251:3600] ;
biao_off(sn,1:15) = [19:251:3600] ;



inN=[90 92 56 99 3 68 106 14];
exN = [8,2,5,4,6,10,11,12,1,15,20,100,75];
exN = unique(exN);
res = [];
th = 0.15;
newin = nb
for c =  inN
    c;
    data = squeeze(calsignal(:,c,:));
    for s = 1 :sn
        ocos = squeeze(data(:,s));
        biao = biao_on(s,:);
        biao = biao(biao>0);
        for t = 1:numel(biao)
            if biao(t)+59 <= 3600 && biao(t)-15 >0
                ocot = squeeze( ocos(biao(t)-15:3 : biao(t)+59 ,:) );
                f0 =  mean(squeeze( ocot(1:5) ) );
                ocot = (ocot -   f0) / f0;
                if max(ocot) >= th
                    nb = nb+1;
                    in_ex = [in_ex;ocot'] ;
                    class(nb,1) = -1;
                end
            end
        end
    end
end
newin = nb-newin
newex = 0
for c =  exN
    c;
    data = squeeze(calsignal(:,c,:));
    for s = 1: sn
        ocos = squeeze(data(:,s));
        biao = biao_on(s,:);
        biao = biao(biao>0);

        for t = 1:numel(biao) %randperm( numel(biao),1 )
            if biao(t)+59 <= 3600 && biao(t)-15 >0
                ocot = squeeze( ocos(biao(t)-15:3 : biao(t)+59 ,:) );
                f0 =  mean(squeeze( ocot(1:5) ) );
                ocot = (ocot -   f0) / f0;
                if max(ocot) >= th
                    if newex == newin
                        continue 
                    end
                    nb = nb+1;
                    newex =newex+1;
                    in_ex = [in_ex;ocot'] ;
                    class(nb,1) = 1;
                end
            end
        end
    end
end
nb







%%

testid = randperm(nb,240);
trainid = [1:nb];
trainid(testid) = [];
x_train = in_ex(trainid,:);
mean(x_train,'all')   %0.048
std(x_train,[],'all') %0.146
y_train = class(trainid);
x_test = in_ex(testid,:);
y_test = class(testid);


acc = 0.0;
rng(i);
Mdl = fitcsvm(x_train,y_train,  'Standardize',false ,'KernelFunction','rbf',  'BoxConstraint',Inf,'ClassNames',[-1 1]); 
% CVMdl = crossval(Mdl);                       % cross- validate Mdl
% cvlabel = kfoldPredict(CVMdl);
% newLabel = predict(Mdl,features_test);          % if want to predict
pred = predict(Mdl,x_test) ;
acc = sum(pred== y_test)/numel(y_test)  %0.67083
% test_acc = sum((cvlabel == class))/numel(class)


%%
load('D:\neuron\Support\data_20180827_Mouse1_linear.mat');
calsignal =cat(2, data_Z1_20180827,data_Z2_20180827,data_Z3_20180827);
clearvars  data_Z1_20180827 data_Z2_20180827 data_Z3_20180827
mdate = '0827';
cn = size(calsignal,2);
sn = size(calsignal,3);

stimulitype =name_20180827;
load('D:\neuron\NeuralNetwork\DeepLearning\project\data\biaos_0827_startAndEndFrameOfStimulus.mat')

%%% reshape
nb = 0;
in_ex = [];
th = 0.15;
cellidx = [];
for c =  1:cn
    data = squeeze(calsignal(:,c,:));
    for s = 1 :sn
        ocos = squeeze(data(:,s));
        biao = biao_on(s,:);
        biao = biao(biao>0);
        for t = 1:numel(biao)
            if biao(t)+19 <= 1200 && biao(t)-5 >0
                ocot = squeeze( ocos(biao(t)-5 : biao(t)+19 ,:) );
                f0 =  mean(squeeze( ocot(1:5) ) );
                ocot = (ocot -   f0) / f0;
                if max(ocot) >= th
                    nb = nb+1;
                    in_ex  =[in_ex; ocot'] ;
                    cellidx = [cellidx;c];
                end
            end
        end
    end
end
nb

mean(in_ex,'all')   %0.031
std(in_ex,[],'all') %0.144
newLabel = predict(Mdl,in_ex); 
res= zeros(cn,1);
for c =1:cn
    onecell = find(cellidx == c);
    celllabel = newLabel(onecell);
    res(c,1) = mean(celllabel);
end
figure()
histogram(res,15);
title('Distribution of the average score for all cells','FontSize',15);
xlabel('Score')
ylabel('#cell')
ax = gca;
ax.XAxis.FontSize = 15;
ax.YAxis.FontSize = 15;
saveas(gcf, strcat('D:\neuron\NeuralNetwork\DeepLearning\project\final\','SVMscore.png' ))
    
% set(gca,'YScale','log')
% find(res<0)
find(res<-0.165)
