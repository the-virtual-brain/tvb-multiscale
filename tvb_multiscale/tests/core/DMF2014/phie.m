function result=phie(x)
% corresponding TVB parameters:
    g=0.16;  % d_e
    I=125.;  % b_e
    c=310.;  % a_e
    y=c*x-I;
    %y1=x-I;
    result = y./(1-exp(-g*y));
end
