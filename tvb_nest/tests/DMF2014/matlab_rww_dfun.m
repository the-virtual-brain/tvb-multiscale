function out = matlab_rww_dfun(sn, sg, coupling)
    % corresponding TVB parameters:
    taon = 100.0;   % tau_e
    taog = 10.0;    % tau_i
    gamma = 0.641;  % gamma_e * 1000
    JN = 0.15;      % J_N
    J = 1.0;        % J_i
    I0 = 0.382;     % I_o
    Jexte = 1.;     % W_e
    Jexti = 0.7;    % W_i
    w = 1.4;        % w_p
    C = 2.0;        % G

    % corresponding TVB variables:
    % Constraints in TVB are now external to the model and performed by the integrator
%    sn(sn>1) = 1;   % S_e
%    sn(sn<0) = 0;
%
%    sg(sg>1) = 1;   % S_i
%    sg(sg<0) = 0;

    xn = I0*Jexte + w*JN*sn + JN*C*coupling - J*sg; % x_e
    xg = I0*Jexti + JN*sn - sg;  % x_i

    rn = phie(xn);  % H_e
    rg = phii(xg);  % H_i

    dsn = -sn/taon + (1-sn)*gamma.*rn./1000.;  % dS_e
    dsg = -sg/taog + rg./1000.;  % dS_i

    out = {dsn, dsg};  % , rn, rg, xn, xg
end


% Code from DMF_Deco2014.m
%clear
%close all
%clc
%
%%%%%%%%%%%% Load SC
%sc_data  = load('test_SC.mat');
%C        = sc_data.sc_cap;
%
%%%%%%%%%%%% Simulation and model parameters
%Nnew     = size(C,1);
%dtt      = 1e-3;   % Sampling rate of simulated neuronal activity (seconds)
%ds       = 100;    % BOLD downsampling rate
%dt=0.1;
%tmax=10000; % Simulation length in ms
%tspan=0:dt:tmax;
%
%taon=100;
%taog=10;
%gamma=0.641;
%sigma=0.001;
%JN=0.15;
%J=ones(Nnew,1);
%I0=0.382;
%Jexte=1.;
%Jexti=0.7;
%w=1.4;
%wee=1.0;
%neuro_act=zeros(tmax,Nnew);
%
%
% sn=0.001*ones(Nnew,1);
% sg=0.001*ones(Nnew,1);
% nn=1;
% j=0;
% for i=1:1:length(tspan)
%  xn=I0*Jexte+w*JN*sn+wee*JN*C*sn-J.*sg;
%  xg=I0*Jexti+JN*sn-sg;
%  rn=phie(xn);
%  rg=phii(xg);
%  sn=sn+dt*(-sn/taon+(1-sn)*gamma.*rn./1000.)+(dt)*sigma*randn(Nnew,1);
%  sn(sn>1) = 1;
%  sn(sn<0) = 0;
%  sg=sg+dt*(-sg/taog+rg./1000.)+(dt)*sigma*randn(Nnew,1);
%  sg(sg>1) = 1;
%  sg(sg<0) = 0;
%  j=j+1;
%  if j==10
%   neuro_act(nn,:)=sn';
%   nn=nn+1
%   j=0;
%  end
% end
