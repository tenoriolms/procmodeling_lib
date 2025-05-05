## LISTA DE VARIAEIS ##
# y = [C, G2, G, H, X, Ef]
# y[0] = C, Celulose        => dydt_0
# y[1] = G2, Celobiose      => dydt_1
# y[2] = G, Glicose         => dydt_2
# y[3] = H, Hemicelulose    => dydt_3
# y[4] =  X, Xilose         => dydt_4
# y[5] =  Ef, Enzima livre

solids0 = 10
enzyme0 = 15
#g/L
celulose0 = solids0*0.66*10 
#g/L
enzima_total = (36/(203*1000))*( enzyme0*celulose0) 



Emax = 8.32
Kad = 7.16
div_Eb_S = ( Emax*Kad*y[5] )/( 1 + Kad*y[5])


Ebc = div_Eb_S*y[0]

Ebh = div_Eb_S*y[3]

S = y[0] + y[3] + solids0*0.252*10

Rs = S/(solids0*10)


# r1 - Cellulose (C) to cellobiose (G2)
K1r = 0.177
K1IG2 = 0.402
K1IG = 2.71
K1IX = 2.15
r1 = ( K1r*Ebc*Rs*S )/( 1 + y[1]/K1IG2 + y[2]/K1IG + y[4]/K1IX)

# r2 - Cellulose (C) to glucose (G)
K2r = 8.81
K2IG2 = 119.6
K2IG = 4.69
K2IX = 0.095
r2 = ( K2r*Ebc*Rs*S )/( 1 + y[1]/K2IG2 + y[2]/K2IG + y[4]/K2IX)

# r3 - Cellobiose (G2) to glucose (G)
K3r = 201.0
K3M = 26.6
K3IG = 11.06
K3IX = 1.023
r3 = ( K3r*y[5]*y[1] )/( K3M*(1 + y[2]/K3IG + y[4]/K3IX) + y[1] )

# r4 - Hemicellulose (H) to xylose (X)
K4r = 1.634
K4IG2 = 16.25
K4IG = 4.0
K4IX = 154.0
r4 = ( K4r*Ebh*Rs*S )/( 1 + y[1]/K4IG2 + y[2]/K4IG + y[4]/K4IX)

# Variação de celulose
dydt[0] = -r1 - r2
# Variação de celobiose
dydt[1] = 1.056*r1 - r3
# Variação de glicose

dydt[2] = 1.111*r2 + 1.053*r3
# Variação de hemicelulose
dydt[3] = -r4
# Variação de xilose
dydt[4] = 1.136*r4
# Variação de enzima livre
# noise = np.random.normal(-0.001,0.0015) #  μ = 0, σ = 2, size = length of x or y. Choose μ and σ wisely.
dydt[5] = 0