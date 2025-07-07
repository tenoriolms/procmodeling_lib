## LISTA DE VARIAVEIS ##
# y = [X, enz_F, enz_B, enz_Xy]
# y[0] = dydt_0 = X, Biomassa
# y[1] = dydt_1 = enz_F, Enzima celulase
# y[2] = dydt_2 = enz_B, Enzima Beta-glicosidase
# y[3] = dydt_3 = enz_Xy, Enzima xilanase
S0 = 2

#Equação para correção da fase inicial (f) até 16h
Tl = 14.4
f = ( 1 - exp(-t/Tl) )

#Concentração de substrato (S) e sua variação (dSdt)
qs = -0.027
S = S0*exp( qs*( t + Tl*exp(-t/Tl) - Tl) )
dSdt = f*qs*S

#Variação do microorganismo - X
u_m = 0.2
Ks = 0.83
Xm = 40
r = 0.6
ux = u_m*( S/(Ks+S) )*(( 1 - (y[0]/Xm) )**r)
u_md = 0.18
Ksd = 0.47
uxd = f*u_md*( S/(Ksd+S) )

dydt[0] = y[0]*(ux-uxd)

# Variação de cellulases (FPase) - enz_F
mf = 6.
Fm = 4000
n = 0.46
Kfs = 43.86
Kif = 0.98
Kfd = 0.023
dydt[1] = f*mf*y[0]*( (1-(y[1]/Fm))**n )*( -Kfs*dSdt )*( 1/(1+(S/Kif)) ) - y[1]*Kfd

# Variação de b-glucosidase
mb = 20.94
Bm = 19940.28
o = 1.60
Kbs = 12.22
Kib = 6.58
Kbd = 0.0019
dydt[2] = f*mb*y[0]*( (1-(y[2]/Bm))**o )*( -Kbs*dSdt )*( 1/(1+(S/Kib)) ) - y[2]*Kbd

# Variação de xylanase
mXy = 565.37
Xym = 131240.70
p = 0.44
KXys = 28.59
KiXy = 2.16
KXyd = 0.014
dydt[3] = f*mXy*y[0]*( (1-(y[3]/Xym))**p )*( -KXys*dSdt )*( 1/(1+(S/KiXy)) ) - y[3]*KXyd




