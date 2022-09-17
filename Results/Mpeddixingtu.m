clear all;
clc;
load('channelSelect.mat');


subplot(2,4,1);%negtive
fig1=mean(:,1);
topoplotEEG(fig1,'128loc.txt','electrodes','labels','maplimits',[0,max(mean)]);

subplot(2,4,2);%negtive
fig2=mean(:,2);
topoplotEEG(fig2,'128loc.txt','electrodes','labels','maplimits',[0,max(mean)]);

subplot(2,4,3);%negtive
fig3=mean(:,3);
topoplotEEG(fig3,'128loc.txt','electrodes','labels','maplimits',[0,max(mean)]);

subplot(2,4,4);%negtive
fig4=mean(:,4);
topoplotEEG(fig4,'128loc.txt','electrodes','labels','maplimits',[0,max(mean)]);

subplot(2,4,5);%negtive
fig5=mean(:,5);
topoplotEEG(fig5,'128loc.txt','electrodes','labels','maplimits',[0,max(mean)]);

subplot(2,4,6);%negtive
fig6=mean(:,6);
topoplotEEG(fig6,'128loc.txt','electrodes','labels','maplimits',[0,max(mean)]);

subplot(2,4,7);%negtive
fig7=mean(:,7);
topoplotEEG(fig7,'128loc.txt','electrodes','labels','maplimits',[0,max(mean)]);

%¥Ãº§ Ù–‘
subplot(2,4,8)
fig8=mean(:,8);
topoplotEEG(fig8,'128loc.txt','electrodes','labels','maplimits',[0,max(mean)]);
