function result=phii(x)
% corresponding TVB parameters:
    g=0.087; % d_i
    I=177.;  % b_i
    c=615.;  % a_i
    y=c*x-I;
    result = y./(1-exp(-g*y));
end
