 for myi = 1:10
     clc;close all;
     myi
    format shortg
    c = clock;
    ctime = join(string(fix(c)),'_')
    %%Import data
    load('D:\neuron\Support\data_20180827_Mouse1_linear.mat');
    calsignal =cat(2, data_Z1_20180827,data_Z2_20180827,data_Z3_20180827);
    clearvars  data_Z1_20180827 data_Z2_20180827 data_Z3_20180827
    cn = size(calsignal,2);
    sn = size(calsignal,3);

    deconv_signal = zeros(size(calsignal));
    for c = 1:cn
        for s =  1:sn
            Data = calsignal(:,c,s);
            fs=10; % sample frequency is 10Hz
            N=length(Data);
            t=[0:1/fs:(N-1)/fs]'; 
            OutputData(1)=Data(1);
            CutoffFreq=0.2;%Cutoff frequency
            FilterFactor=2*3.14*1/fs*CutoffFreq;
            for i=2:length(Data)
              OutputData(i)=FilterFactor*Data(i)+(1-FilterFactor)*OutputData(i-1);%one order low pass filter
            end

            t1 = [0:1:(N-1)]'; 
            diff_B = diff(OutputData);
            diff_B = max(0,diff_B);
            diff_B = diff_B+min(Data);       
            diff_B = [diff_B,diff_B(end)];
            deconv_signal(:,c,s) = diff_B;
        end
    end

    % figure
    % histogram(deconv_signal)

    %% check population res for 1200*259*14 dataset
    load('D:\neuron\NeuralNetwork\DeepLearning\project\data\biaos_0827_startAndEndFrameOfStimulus.mat')
    load('D:\neuron\NeuralNetwork\DeepLearning\project\final\duration_20180827.mat');
    calsignal = deconv_signal;
    cn = 259;
    sn = 14;

    calsignal_minus_median_by_cell = zeros(size(calsignal));
    for c = 1:cn
        cell  = squeeze(calsignal(:,c,:));
        mean(cell,'all');
        cell = (cell-median(cell,'all'))/median(cell,'all');
        calsignal_minus_median_by_cell(:,c,:) = cell;
    end
    test = squeeze(calsignal(:,2,:));
    test = (squeeze(calsignal(:,2,2))-median( test ,'all')) /median( test ,'all');
    ref = calsignal_minus_median_by_cell(:,2,2);
    res = sum(test == ref);
    if res == 1200
        disp('minus_median_pass')
    end
    % calsignal_minus_median_by_cell = calsignal;

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% reshape for SVM
    nb = 0; % number of total trials
    for s = 1:sn
        biao = biao_on(s,:);
        for b = 1:40
            if biao(b)+28 <= 1200 && biao(b)>0 % site of slice point, ADJUSTABLE
                nb = nb+ 1;
            end
        end
    end
    nb  
    
    class = zeros(1, nb);
    reshape_signal = zeros(nb, cn*29);
    nb = 0;
    for s = 1:sn
        biao = biao_on(s,:);
        for b = 1:40
            if biao(b)+28 <= 1200 && biao(b)>0%site of slice point, ADJUSTABLE
                nb = nb + 1;
                slice = squeeze(calsignal_minus_median_by_cell(biao(b):biao(b)+28 ,:,s));  
                slice = reshape(slice,[1,cn*29]);
                reshape_signal(nb, : ) = slice;
                class(nb) = s;
            end
        end
    end
    nb 

    ref = calsignal_minus_median_by_cell(biao_on(1,2):biao_on(1,2)+28 ,2,1)';
    test = reshape_signal(2,30:29*2);
    if sum(ref == test)==29
        disp('reshape_signal pass')
    end

    rescale_reshape_signal = reshape_signal*5/std(reshape_signal,[],'all');
%     figure
%     histogram(rescale_reshape_signal)
%     std(rescale_reshape_signal,[],'all')

    testid = randperm(nb,50);
    trainid = [1:nb];
    trainid(testid) = [];
    x_train = rescale_reshape_signal(trainid,:);
    y_train = class(trainid);
    x_test = rescale_reshape_signal(testid,:);
    y_test = class(testid);
    
    figure
    histogram(x_train)
    Mdl = fitcecoc(x_train,y_train);
    pred = predict(Mdl,x_test) ;
    acc = sum(pred' == y_test)/numel(y_test)
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  rescale and Interpolant for SNN
    rescale_calsignal = calsignal_minus_median_by_cell;
    rescale_calsignal = rescale_calsignal*5/std(rescale_calsignal,[],'all');
    figure(1)
    histogram(rescale_calsignal)
    std(rescale_calsignal,[],'all')
    saveas(gcf, strcat('D:\neuron\NeuralNetwork\DeepLearning\project\final\',ctime,'HistInput.png' ))

    time = 12000;
    spike = zeros(sn,cn,time);
    for s = 1:sn
        for c = 1:cn
            fr = squeeze(rescale_calsignal(:,c,s));
            x = [1:1200];
            v =fr;
            F = griddedInterpolant(x,v,'pchip','nearest');
            xq = [0.1 :0.1:1200];
            vq = F(xq);
            spike(s,c,:) = vq;
        end
    end
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% SNN
    cn = 259;
    % Ii = [11;12;30;34;35;48;70;73;80;83;102;169;182;201;242];
    Ii=[1;11;12;30;32;34;35;36;66;72;73;101;102;103;104;168;169;257;258];
    %[1;11;12;28;30;32;33;34;35;48;66;72;73;79;101;102;103;104;168;169;182;187;257]
    %[1;11;12;28;30;32;33;34;35;66;72;73;101;102;103;104;168;169;182;257]
    %[1;11;12;28;30;32;33;34;35;66;72;73;101;102;103;104;168;169;182;257]
    Ie = [1:cn];
    Ie(Ii) = 0;
    Ie = Ie(Ie>0);
    Ni=numel(Ii);  Ne=cn-Ni; 
    re=rand(Ne,1); ri=rand(Ni,1);
    a= 0.02*ones(cn,1);
    a(Ii) = 0.02+0.08*ri;
    b = 0.2*ones(cn,1);
    b(Ii) = 0.25-0.05*ri;
    c =  -65*ones(cn,1);
    c(Ie) = -65+15*re.^2;
    d = 2*ones(cn,1);
    d(Ie) = 8-6*re.^2;
    S = [0.5*rand(Ne+Ni,cn)];
    S(:,Ii) =  -rand(Ne+Ni,Ni);
    S(1:1+size(S,1):end) = 0;
    v=-65*ones(cn,1); % Initial values of v
    u=b.*v; % Initial values of u
    initS = S;
    %%% train
    output = zeros(sn, cn,time);

    firings=[]; % spike timings
    v_recorder = zeros(1,time);
    u_recorder = zeros(1,time);
    at=0;
    for s=1  :sn
        for t=1 :time 
            at = at+1;
            I = squeeze(spike(s,:,t) )' ;
            fired=find(v>=30); % indices of spikes
            if ~isempty(fired) && ~isempty(firings)
                nearest_neighbor = find( firings(:,1)== firings(end,1));
                S = nearest_neighbor_STDP(S,firings(nearest_neighbor ,:),fired,at);
            end
            firings=[firings;at+0*fired,fired];
            v(fired)=c(fired);
            u(fired)=u(fired)+d(fired);
            I=I+sum(S(:,fired),2);
            v=v+0.5*(0.04*v.^2+5*v+140-u+I); % step 0.5 ms
            v=v+0.5*(0.04*v.^2+5*v+140-u+I); % for numeric stability
            u=u+a.*(b.*v-u); 
            v_recorder(at) = v(58);
            u_recorder(at) = u(58);
            output(s, fired,t) = 1;
        end
    end
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% end of SNN training
    figure(2)
    set(gcf,'Units','centimeters','Position',[0 5 35 10]);
    plot(firings(:,1),firings(:,2),'.');
    title('population spike')
    saveas(gcf, strcat('D:\neuron\NeuralNetwork\DeepLearning\project\final\',ctime,'SNNPopuSpk.png' ))

    figure(3)
    set(gcf,'Units','centimeters','Position',[0 5 20 10]);
    subplot(1,2,1)
    plot(v_recorder);
    title('membrane potential of neuron 58')
    subplot(1,2,2)
    plot(u_recorder);
    title('relative [Ca^{2+}] in neuron 58')
    saveas(gcf, strcat('D:\neuron\NeuralNetwork\DeepLearning\project\final\',ctime,'SNNNeu58Act.png' ))

    figure(4)
    set(gcf,'Units','centimeters','Position',[0 5 35 10]);
    subplot(1,3,1)
    imagesc(initS)
    colormap default
    caxis([min(S,[],'all') max(S,[],'all')])
    title('init w')
    subplot(1,3,2)
    imagesc(S)
    colormap default
    title('learned w')
    colorbar;
    subplot(1,3,3)
    imagesc(S-initS)
    colormap default
    title('change of w')
    colorbar;
    saveas(gcf, strcat('D:\neuron\NeuralNetwork\DeepLearning\project\final\',ctime,'SNNFuncConnec.png' ))

    SNNresults = zeros(1200,cn,sn);
    for f = 1  :1200
        tmp = sum(output(:, :,(f-1)*10+1:f*10 ),3);
        SNNresults(f,:,:) = tmp';
    end
    ref = sum(output(4,50,1:10));
    test = (SNNresults(1,50 ,4));
    if sum(ref == test) == numel(test)
        disp('SNN results pass')
    end

    sumSNNresults= zeros(sn, cn);
    for s = 1:sn
        biao = biao_on(s,:);
        nb=0;
        for b = 1:40
            if biao(b)+28 <= 1200 && biao(b)>0%site of slice point, ADJUSTABLE
                nb = nb + 1;
                slice = squeeze(SNNresults(biao(b):biao(b)+28 ,:,s));  
                sumSNNresults(s, : ) = sumSNNresults(s, : )+sum(slice);
            end
        end
        sumSNNresults(s, : ) = sumSNNresults(s, : )/nb;
    end
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%plot SNN spikes
    Y = sumSNNresults;
    mymap = [1 0 0
        1 .125 .125
        1 0.25 0.25
        1 .5 .5
        1 .625 .625
        1 .75 .75
        1 .825 .825
        1 1 1
        ];
    mymap = flipud(mymap);
    load('D:\neuron\NeuralNetwork\DeepLearning\project\final\name_20180827.mat');
    figure(5)
    set(gcf,'Units','centimeters','Position',[0 0 35 20]);
    c3 = [min(Y,[],'all'), max(Y,[],'all')];
    for s = 1:14
        subplot(2,7,s)
        imagesc(Y(s,:)')
        colormap(mymap)
        caxis(c3)
        tmp = string(name_20180827(s));
        tmp = strrep(tmp,'_','-');
        title(tmp)
    end
    colorbar
    sgtitle({'SNN response to each stimulus'})
    saveas(gcf, strcat('D:\neuron\NeuralNetwork\DeepLearning\project\final\',ctime,'SNNSpkRes.png' ))

    
    % % % %%%%%%%%%%%55%% SVM test SNN   NOT compelete !
% % % 
% % %     SNNresults_minus_mean_by_cell = zeros(size(SNNresults));
% % %     for c = 1:cn
% % %         cell  = squeeze(SNNresults(:,c,:));
% % %         if mean(cell,'all') >0
% % %             cell = (cell-mean(cell,'all'))/mean(cell,'all');
% % %         end
% % %         SNNresults_minus_mean_by_cell(:,c,:) = cell;
% % %     end
% % %     test = squeeze(SNNresults(:,2,:));
% % %     test = (squeeze(SNNresults(:,2,2))-mean( test ,'all')) /mean( test ,'all');
% % %     ref = SNNresults_minus_mean_by_cell(:,2,2);
% % %     res = sum(test == ref);
% % %     if res == 1200
% % %         disp('SNN_minus_mean_pass')
% % %     end
% % % 
% % %     nb = 0; % number of total trials
% % %     for s = 1:sn
% % %         biao = biao_on(s,:);
% % %         for b = 1:40
% % %             if biao(b)+28 <= 1200 && biao(b)>0 % site of slice point, ADJUSTABLE
% % %                 nb = nb+ 1;
% % %             end
% % %         end
% % %     end
% % %     nb  
% % %     class = zeros(1, nb);
% % %     reshape_signal = zeros(nb, cn*29);
% % %     nb = 0;
% % %     for s = 1:sn
% % %         biao = biao_on(s,:);
% % %         for b = 1:40
% % %             if biao(b)+28 <= 1200 && biao(b)>0%site of slice point, ADJUSTABLE
% % %                 nb = nb + 1;
% % %                 slice = squeeze(SNNresults_minus_mean_by_cell(biao(b):biao(b)+28 ,:,s));  
% % %                 slice = reshape(slice,[1,cn*29]);
% % %                 reshape_signal(nb, : ) = slice;
% % %                 class(nb) = s;
% % %             end
% % %         end
% % %     end
% % %     nb 
% % % 
% % %     ref = SNNresults_minus_mean_by_cell(biao_on(1,2):biao_on(1,2)+28 ,2,1)';
% % %     test = reshape_signal(2,30:29*2);
% % %     if sum(ref == test)==29
% % %         disp('SNN_reshape_signal pass')
% % %     end
% % %     
% % % %     rescale_reshape_signal = 
% % %     rescale_reshape_signal = reshape_signal*5/std(reshape_signal,[],'all');
% % %     figure
% % %     histogram(rescale_reshape_signal)
% % % %     std(rescale_reshape_signal,[],'all')
% % % 
% % % %     testid = randperm(nb,50);
% % % %     trainid = [1:nb];
% % % %     trainid(testid) = [];
% % % %     x_train = rescale_reshape_signal(trainid,:);



    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% functional connectivity
    audairN =[110,84,58];
    audN = [199,159,29];
    % airN = [8,13,24,27,36,53,58,70,76,84,98,99,101,110,164,184,240];
    nonairN = [8,13,24,27,53,70,76,98,99,164,184,240];    % air responsive but not aud-air responsive
    intersect(Ii,audairN)
    intersect(Ii,audN)
    intersect(Ii,nonairN)


    c1i = initS(audairN, audN);
    c1i = reshape(c1i,[numel(c1i),1]);
    c1 = S(audairN, audN);
    c1 = reshape(c1,[numel(c1),1]);
    c1imean = mean(c1i);
    c1isem =  std(c1i)/sqrt(numel(c1i));
    c1mean = mean(c1);
    c1sem =  std(c1)/sqrt(numel(c1));

    c2i = initS(audairN,audairN);
    c2i = c2i(c2i~= 0);
    c2 = S(audairN,audairN);
    c2 = c2(c2~= 0);
    c2imean = mean(c2i);
    c2isem =  std(c2i)/sqrt(numel(c2i));
    c2mean = mean(c2);
    c2sem =  std(c2)/sqrt(numel(c2));

    c3i = initS(audN,audN);
    c3i = c3i(c3i~= 0);
    c3 = S(audN,audN);
    c3 = c3(c3~= 0);
    c3imean = mean(c3i);
    c3isem =  std(c3i)/sqrt(numel(c3i));
    c3mean = mean(c3);
    c3sem =  std(c3)/sqrt(numel(c3));

    c4i = initS(nonairN,nonairN);
    c4i = c4i(c4i~= 0);
    c4 = S(nonairN,nonairN);
    c4 = c4(c4~= 0);
    c4imean = mean(c4i);
    c4isem =  std(c4i)/sqrt(numel(c4i));
    c4mean = mean(c4);
    c4sem =  std(c4)/sqrt(numel(c4));

    c5i = initS(Ie,Ie);
    c5i = c5i(c5i ~= 0);
    c5 = S(Ie,Ie);
    c5 = c5(c5~= 0);
    c5imean = mean(c5i);
    c5isem =  std(c5i)/sqrt(numel(c5i));
    c5mean = mean(c5);
    c5sem =  std(c5)/sqrt(numel(c5));

    data = [c1imean,c1mean,c2imean,c2mean,c3imean,c3mean,c4imean,c4mean,c5imean,c5mean];
    sem = [c1isem,c1sem,c2isem,c2sem,c3isem,c3sem,c4isem,c4sem,c5isem,c5sem];

    figure(6)
    b = bar(data,0.5,'w','LineWidth',2);
    b.FaceColor = 'flat';
    hold on
    er = errorbar(data,sem);   
    er.Color = [0 0 0];                            
    er.LineStyle = 'none';  
    hold off
    ylabel("Functional connectivity / A.U."); 
    xticklabels({'aud to audair-pre','aud to audair-post','intra-audair-pre','intra-audair-post','intra-aud-pre','intra-aud-post','intra-non-air-pre','intra-non-air-post','Intra-Ex-pre','intra-Ex-post'});
    xtickangle(45)
    ax = gca;
    ax.XAxis.FontSize = 15;
    ax.YAxis.FontSize = 15;
    title('Inter-ensemble connection','FontSize',15)
    saveas(gcf, strcat('D:\neuron\NeuralNetwork\DeepLearning\project\final\',ctime,'FuncConnec.png' ))
    
    c1i =  initS(audairN, audN);
    c1i = reshape(c1i,[numel(c1i),1]);
    c1 = S(audairN, audN);
    c1 = reshape(c1,[numel(c1),1]);
    deltac1 = c1-c1i;
    ddeltac1 = deltac1./c1i;
    d1mean = mean(ddeltac1);
    d1sem =  std(ddeltac1)/sqrt(numel(ddeltac1));
    
    c2i = initS(audairN,audairN);
    c2i = c2i(c2i~= 0);
    c2 = S(audairN,audairN);
    c2 = c2(c2~= 0);
    deltac2 = c2-c2i;
    ddeltac2 = deltac2./c2i;
    d2mean = mean(ddeltac2);
    d2sem =  std(ddeltac2)/sqrt(numel(ddeltac2));
    
    c3i = initS(audN,audN);
    c3i = c3i(c3i~= 0);
    c3 = S(audN,audN);
    c3 = c3(c3~= 0);
    deltac3 = c3-c3i;
    ddeltac3 = deltac3./c3i;
    d3mean = mean(ddeltac3);
    d3sem =  std(ddeltac3)/sqrt(numel(ddeltac3));
    
    c4i = initS(nonairN,nonairN);
    c4i = c4i(c4i~= 0);
    c4 = S(nonairN,nonairN);
    c4 = c4(c4~= 0);
    deltac4 = c4-c4i;
    ddeltac4 = deltac4./c4i;
    d4mean = mean(ddeltac4);
    d4sem =  std(ddeltac4)/sqrt(numel(ddeltac4));
    
    c5i = initS(Ie,Ie);
    c5i = c5i(c5i~= 0);
    c5 = S(Ie,Ie);
    c5 = c5(c5~= 0);
    deltac5 = c5-c5i;
    ddeltac5 = deltac5./c5i;
    d5mean = mean(ddeltac5);
    d5sem =  std(ddeltac5)/sqrt(numel(ddeltac5));
    
    figure(7)
    data = [d1mean,d2mean,d3mean,d4mean,d5mean];
    sem = [d1sem,d2sem,d3sem,d4sem,d5sem];
    b = bar(data,0.5,'w','LineWidth',2);
    b.FaceColor = 'flat';
    hold on
    er = errorbar(data,sem);   
    er.Color = [0 0 0];                            
    er.LineStyle = 'none';  
    hold off
    ylabel("delta connectivity / init connectivity"); 
    xticklabels({'aud to audair','intra-audair','intra-aud','intra-non-air','intra-Ex'})
    xtickangle(45)
    ax = gca;
    ax.XAxis.FontSize = 15;
    ax.YAxis.FontSize = 15;
    title('Fraction change of inter-ensemble connection','FontSize',15)
    saveas(gcf, strcat('D:\neuron\NeuralNetwork\DeepLearning\project\final\',ctime,'FracChangeFuncConnec.png' ))
    
    
    
    %%%%%%%%%%%%%%%%%%%%%%%
    c1i = initS(audairN, Ii);
    c1i = reshape(c1i,[numel(c1i),1]);
    c1 = S(audairN, Ii);
    c1 = reshape(c1,[numel(c1),1]);
    c1imean = mean(c1i);
    c1isem =  std(c1i)/sqrt(numel(c1i));
    c1mean = mean(c1);
    c1sem =  std(c1)/sqrt(numel(c1));

    c2i = initS(audN, Ii);
    c2i = reshape(c2i,[numel(c2i),1]);
    c2 = S(audN, Ii);
    c2 = reshape(c2,[numel(c2),1]);
    c2imean = mean(c2i);
    c2isem =  std(c2i)/sqrt(numel(c2i));
    c2mean = mean(c2);
    c2sem =  std(c2)/sqrt(numel(c2));

    c3i = initS(nonairN, Ii);
    c3i = reshape(c3i,[numel(c3i),1]);
    c3 = S(nonairN, Ii);
    c3 = reshape(c3,[numel(c3),1]);
    c3imean = mean(c3i);
    c3isem =  std(c3i)/sqrt(numel(c3i));
    c3mean = mean(c3);
    c3sem =  std(c3)/sqrt(numel(c3));

    c4i = initS(Ie, Ii);
    c4i = reshape(c4i,[numel(c4i),1]);
    c4 = S(Ie, Ii);
    c4 = reshape(c4,[numel(c4),1]);
    c4imean = mean(c4i);
    c4isem =  std(c4i)/sqrt(numel(c4i));
    c4mean = mean(c4);
    c4sem =  std(c4)/sqrt(numel(c4));

    data = [c1imean,c1mean,c2imean,c2mean,c3imean,c3mean, c4imean,c4mean];
    sem = [c1isem,c1sem,c2isem,c2sem,c3isem,c3sem,c4isem,c4sem ];

    figure(8)
    b = bar(data,0.5,'w','LineWidth',2);
    b.FaceColor = 'flat';
    hold on
    er = errorbar(data,sem);   
    er.Color = [0 0 0];                            
    er.LineStyle = 'none';  
    hold off
    ylabel("Functional connectivity / A.U."); 
    xticklabels({'Inh to audair-pre','Inh to audair-post','Inh to aud-pre','Inh to aud-post','Inh to non air-pre','Inh to non air-post',  'Inh to all Ex-pre',  'Inh to all Ex-post'})
    xtickangle(45)
    ax = gca;
    ax.XAxis.FontSize = 15;
    ax.YAxis.FontSize = 15;
    title('Afferent inhibitory connection to each ensemble','FontSize',15)
    saveas(gcf, strcat('D:\neuron\NeuralNetwork\DeepLearning\project\final\',ctime,'InhiConnec.png' ))

    
    
    
    c1i = initS(audairN, Ii);
    c1i = reshape(c1i,[numel(c1i),1]);
    c1 = S(audairN, Ii);
    c1 = reshape(c1,[numel(c1),1]);
    deltac1 = c1-c1i;
    ddeltac1 = deltac1./c1i;
    d1mean = mean(ddeltac1);
    d1sem =  std(ddeltac1)/sqrt(numel(ddeltac1));

    c2i = initS(audN, Ii);
    c2i = reshape(c2i,[numel(c2i),1]);
    c2 = S(audN, Ii);
    c2 = reshape(c2,[numel(c2),1]);
    deltac2 = c2-c2i;
    ddeltac2 = deltac2./c2i;
    d2mean = mean(ddeltac2);
    d2sem =  std(ddeltac2)/sqrt(numel(ddeltac2));

    c3i = initS(nonairN, Ii);
    c3i = reshape(c3i,[numel(c3i),1]);
    c3 = S(nonairN, Ii);
    c3 = reshape(c3,[numel(c3),1]);
    deltac3 = c3-c3i;
    ddeltac3 = deltac3./c3i   ;
    % c3i(find(isnan(ddeltac3)))
    d3mean = mean(ddeltac3)   ;
    d3sem =  std(ddeltac3)/sqrt(numel(ddeltac3));

    c4i = initS(Ie, Ii);
    c4i = reshape(c4i,[numel(c4i),1]);
    c4 = S(Ie, Ii);
    c4 = reshape(c4,[numel(c4),1]);
    deltac4= c4-c4i;
    ddeltac4 = deltac4./c4i;
    ddeltac4(isnan(ddeltac4)) = [];
    d4mean = mean(ddeltac4);
    d4sem =  std(ddeltac4)/sqrt(numel(ddeltac4));

    figure(9)
    data = [d1mean,d2mean,d3mean,d4mean];
    sem = [d1sem,d2sem,d3sem,d4sem ];
    b = bar(data,0.5,'w','LineWidth',2);
    b.FaceColor = 'flat';
    hold on
    er = errorbar(data,sem);   
    er.Color = [0 0 0];                            
    er.LineStyle = 'none';  
    hold off
    ylabel("delta connectivity / init connectivity"); 
    xticklabels({'audair','aud','non-air','total Ex'})
    xtickangle(45)
    ax = gca;
    ax.XAxis.FontSize = 15;
    ax.YAxis.FontSize = 10;
    title('Fraction change of afferent inhibitory connection','FontSize',15)
    saveas(gcf, strcat('D:\neuron\NeuralNetwork\DeepLearning\project\final\',ctime,'FracChangeInhiConnec.png' ));
end


%%
function con = nearest_neighbor_STDP(connection, allSpike, activeN,t)
Ii = [11;12;30;34;35;48;70;73;80;83;102;169;182;201;242];    
Ap =0.003 ;
Am = -Ap*0.5;
taup = 20;   % 20 ms
taum = 60;
previousN = allSpike(:,2);
for i=1:length(activeN)
    neut = activeN(i);
    for j=1:length(previousN)
        neutm = previousN(j);
        dt = t - allSpike(j,1);
        fdt_m = Am*exp(-dt/taum);
        fdt_p = Ap*exp(-dt/taup);
        if ~any(Ii == neut) && ~any(Ii == neutm)
            connection(neutm,neut) = connection(neutm,neut).*(fdt_m + 1);
            connection(neut,neutm) = connection(neut,neutm).*((fdt_p + 1)');
        elseif any(Ii == neut) && ~any(Ii == neutm)
            connection(neutm,neut) = connection(neutm,neut).*(fdt_m + 1);  % In to Ex
            connection(neut,neutm) = connection(neut,neutm).*((-fdt_p + 1)'); % Ex to In
        elseif ~any(Ii == neut) && any(Ii == neutm)
            connection(neutm,neut) = connection(neutm,neut).*(-fdt_m + 1);  % Ex to In
            connection(neut,neutm) = connection(neut,neutm).*((fdt_p + 1)'); % In to Ex
        else
%             disp(t);
%             disp('inh to inh');
        end
    end
end
con = connection;
end
