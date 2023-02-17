#!/usr/bin/env python3

"""Defines a dictionary with the scattering functions for all elements
   Key is a string with the element
   Value is the scattering function as described in Brown P J, Fox A G, Maslen E N,
   O'Keefe M A and Willis B T M 2004 Intensity of diffraction intensities
   International Tables for Crystallography Volume C: Mathematical, Physical, and Chemical Tables
   ed E Prince (Norwell, MA: Kluwer Academic Publishers) pp 55495
   x in the anonymous function is the sin(theta) / lambda
"""
import numpy as np

scattering_factor = {
    'H': lambda x: 0.489918*np.exp(-20.6593*np.square(x))+0.262003*np.exp(-7.74039*np.square(x))+
         0.196767*np.exp(-49.5519*np.square(x))+0.049879*np.exp(-2.20159*np.square(x))+0.001305 ,
    'He1-': lambda x: 0.897661*np.exp(-53.1368*np.square(x))+0.565616*np.exp(-15.187*np.square(x))+
            0.415815*np.exp(-186.576*np.square(x))+0.116973*np.exp(-3.56709*np.square(x))+0.002389 ,
    'He': lambda x: 0.8734*np.exp(-9.1037*np.square(x))+0.6309*np.exp(-3.3568*np.square(x))+
          0.3112*np.exp(-22.9276*np.square(x))+0.178*np.exp(-0.9821*np.square(x))+0.0064  ,
    'Li': lambda x: 1.1282*np.exp(-3.9546*np.square(x))+0.7508*np.exp(-1.0524*np.square(x))+
          0.6175*np.exp(-85.3905*np.square(x))+0.4653*np.exp(-68.261*np.square(x))+0.0377  ,
    'Li1+': lambda x: 0.6968*np.exp(-4.6237*np.square(x))+0.7888*np.exp(-1.9557*np.square(x))+
            0.3414*np.exp(-0.6316*np.square(x))+0.7029*np.exp(-0.542*np.square(x))+0.0167  ,
    'Be': lambda x: 1.5919*np.exp(-43.6427*np.square(x))+1.1278*np.exp(-1.8623*np.square(x))+
          0.5391*np.exp(-103.483*np.square(x))+0.7029*np.exp(-0.542*np.square(x))+0.0385  ,
    'Be2+': lambda x: 6.2603*np.exp(-0.0027*np.square(x))+0.8849*np.exp(-0.8313*np.square(x))+
            0.7993*np.exp(-2.2758*np.square(x))+0.1647*np.exp(-5.1146*np.square(x))-6.1092  ,
    'B': lambda x: 2.0545*np.exp(-23.2185*np.square(x))+1.3326*np.exp(-1.021*np.square(x))+
         1.0979*np.exp(-60.3498*np.square(x))+0.7068*np.exp(-0.1403*np.square(x))-0.1932  ,
    'C': lambda x: 2.31*np.exp(-20.8439*np.square(x))+1.02*np.exp(-10.2075*np.square(x))+
         1.5886*np.exp(-0.5687*np.square(x))+0.865*np.exp(-51.6512*np.square(x))+0.2156 ,
    'Cval': lambda x: 2.26069*np.exp(-22.6907*np.square(x))+1.56165*np.exp(-0.656665*np.square(x))+
            1.05075*np.exp(-9.75618*np.square(x))+0.839259*np.exp(-55.5949*np.square(x))+0.286977 ,
    'N': lambda x: 12.2126*np.exp(-0.0057*np.square(x))+3.1322*np.exp(-9.8933*np.square(x))+
         2.0125*np.exp(-28.9975*np.square(x))+1.1663*np.exp(-0.5826*np.square(x))-11.529 ,
    'O': lambda x: 3.0485*np.exp(-13.2771*np.square(x))+2.2868*np.exp(-5.7011*np.square(x))+
         1.5463*np.exp(-0.3239*np.square(x))+0.867*np.exp(-32.9089*np.square(x))+0.2508 ,
    'O1-': lambda x: 4.1916*np.exp(-12.8573*np.square(x))+1.63969*np.exp(-4.17236*np.square(x))+
           1.52673*np.exp(47.0179*np.square(x))+20.307*np.exp(0.01404*np.square(x))+21.9412 ,
    'F': lambda x: 3.5392*np.exp(-10.2825*np.square(x))+2.6412*np.exp(-4.2944*np.square(x))+
         1.517*np.exp(-0.2615*np.square(x))+1.0243*np.exp(-26.1476*np.square(x))+0.2776 ,
    'F1-': lambda x: 3.6322*np.exp(-5.27756*np.square(x))+3.51057*np.exp(-14.7353*np.square(x))+
           1.26064*np.exp(-0.442258*np.square(x))+0.940706*np.exp(-47.3437*np.square(x))+0.653396 ,
    'Ne': lambda x: 3.9553*np.exp(-8.4042*np.square(x))+3.1125*np.exp(-3.4262*np.square(x))+
          1.4546*np.exp(-0.2306*np.square(x))+1.1251*np.exp(-21.7184*np.square(x))+0.3515 ,
    'Na': lambda x: 4.7626*np.exp(-3.285*np.square(x))+3.1736*np.exp(-8.8422*np.square(x))+
          1.2674*np.exp(-0.3136*np.square(x))+1.1128*np.exp(-129.424*np.square(x))+0.676 ,
    'Na1+': lambda x: 3.2565*np.exp(-2.6671*np.square(x))+3.9362*np.exp(-6.1153*np.square(x))+
            1.3998*np.exp(-0.2001*np.square(x))+1.0032*np.exp(-14.039*np.square(x))+0.404 ,
    'Mg': lambda x: 5.4204*np.exp(-2.8275*np.square(x))+2.1735*np.exp(-79.2611*np.square(x))+
          1.2269*np.exp(-0.3808*np.square(x))+2.3073*np.exp(-7.1937*np.square(x))+0.8584 ,
    'Mg2+': lambda x: 3.4988*np.exp(-2.1676*np.square(x))+3.8378*np.exp(-4.7542*np.square(x))+
            1.3284*np.exp(-0.185*np.square(x))+0.8497*np.exp(-10.1411*np.square(x))+0.4853 ,
    'Al': lambda x: 6.4202*np.exp(-3.0387*np.square(x))+1.9002*np.exp(-0.7426*np.square(x))+
          1.5936*np.exp(-31.5472*np.square(x))+1.9646*np.exp(-85.0886*np.square(x))+1.1151 ,
    'Al3+': lambda x: 4.17448*np.exp(-1.93816*np.square(x))+3.3876*np.exp(-4.14553*np.square(x))+
            1.20296*np.exp(-0.228753*np.square(x))+0.528137*np.exp(-8.28524*np.square(x))+0.706786 ,
    'Si': lambda x: 6.2915*np.exp(-2.4386*np.square(x))+3.0353*np.exp(-32.3337*np.square(x))+
          1.9891*np.exp(-0.6785*np.square(x))+1.541*np.exp(-81.6937*np.square(x))+1.1407 ,
    'Sival': lambda x: 5.66269*np.exp(-2.6652*np.square(x))+3.07164*np.exp(-38.6634*np.square(x))+
             2.62446*np.exp(-0.916946*np.square(x))+1.3932*np.exp(-93.5458*np.square(x))+1.24707 ,
    'Si4+': lambda x: 4.43918*np.exp(-1.64167*np.square(x))+3.20345*np.exp(-3.43757*np.square(x))+
            1.19453*np.exp(-0.2149*np.square(x))+0.41653*np.exp(-6.65365*np.square(x))+0.746297 ,
    'P': lambda x: 6.4345*np.exp(-1.9067*np.square(x))+4.1791*np.exp(-27.157*np.square(x))+
         1.78*np.exp(-0.526*np.square(x))+1.4908*np.exp(-68.1645*np.square(x))+1.1149 ,
    'S': lambda x: 6.9053*np.exp(-1.4679*np.square(x))+5.2034*np.exp(-22.2151*np.square(x))+
         1.4379*np.exp(-0.2536*np.square(x))+1.5863*np.exp(-56.172*np.square(x))+0.8669 ,
    'Cl': lambda x: 11.4604*np.exp(-0.0104*np.square(x))+7.1964*np.exp(-1.1662*np.square(x))+
          6.2556*np.exp(-18.5194*np.square(x))+1.6455*np.exp(-47.7784*np.square(x))-9.5574 ,
    'Cl1-': lambda x: 18.2915*np.exp(-0.0066*np.square(x))+7.2084*np.exp(-1.1717*np.square(x))+
            6.5337*np.exp(-19.5424*np.square(x))+2.3386*np.exp(-60.4486*np.square(x))-16.378 ,
    'Ar': lambda x: 7.4845*np.exp(-0.9072*np.square(x))+6.7723*np.exp(-14.8407*np.square(x))+
          0.6539*np.exp(-43.8983*np.square(x))+1.6442*np.exp(-33.3929*np.square(x))+1.4445 ,
    'K': lambda x: 8.2186*np.exp(-12.7949*np.square(x))+7.4398*np.exp(-0.7748*np.square(x))+
         1.0519*np.exp(-213.187*np.square(x))+0.8659*np.exp(-41.6841*np.square(x))+1.4228  ,
    'Ca': lambda x: 8.6266*np.exp(-10.4421*np.square(x))+7.3873*np.exp(-0.6599*np.square(x))+
          1.5899*np.exp(-85.7484*np.square(x))+1.0211*np.exp(-178.437*np.square(x))+1.3751 ,
    'Ca2+': lambda x: 15.6348*np.exp(0.0074*np.square(x))+7.9518*np.exp(-0.6089*np.square(x))+
            8.4372*np.exp(-10.3116*np.square(x))+0.8537*np.exp(-25.9905*np.square(x))-14.875 ,
    'Sc': lambda x: 9.189*np.exp(-9.0213*np.square(x))+7.3679*np.exp(-0.5729*np.square(x))+
          1.6409*np.exp(-136.108*np.square(x))+1.468*np.exp(-51.3531*np.square(x))+1.3329 ,
    'Sc3+': lambda x: 13.4008*np.exp(-0.29854*np.square(x))+8.0273*np.exp(-7.9629*np.square(x))+
            1.65943*np.exp(0.28604*np.square(x))+1.57936*np.exp(-16.0662*np.square(x))-6.6667 ,
    'Ti': lambda x: 9.7595*np.exp(-7.8508*np.square(x))+7.3558*np.exp(-0.5*np.square(x))+
          1.6991*np.exp(-35.6338*np.square(x))+1.9021*np.exp(-116.105*np.square(x))+1.2807 ,
    'Ti2+': lambda x: 9.11423*np.exp(-7.5243*np.square(x))+7.62174*np.exp(-0.457585*np.square(x))+
            2.2793*np.exp(-19.5361*np.square(x))+0.087899*np.exp(-61.6558*np.square(x))+0.897155 ,
    'Ti3+': lambda x: 17.7344*np.exp(-0.22061*np.square(x))+8.73816*np.exp(-7.04716*np.square(x))+
            5.25691*np.exp(0.15762*np.square(x))+1.92134*np.exp(-15.9768*np.square(x))-14.652 ,
    'Ti4+': lambda x: 19.5114*np.exp(-0.178847*np.square(x))+8.23473*np.exp(-6.67018*np.square(x))+
            2.01341*np.exp(0.29263*np.square(x))+1.5208*np.exp(-12.9464*np.square(x))-13.28 ,
    'V': lambda x: 10.2971*np.exp(-6.8657*np.square(x))+7.3511*np.exp(-0.4385*np.square(x))+
         2.0703*np.exp(-26.8938*np.square(x))+2.0571*np.exp(-102.478*np.square(x))+1.2199 ,
    'V2+': lambda x: 10.106*np.exp(-6.8818*np.square(x))+7.3541*np.exp(-0.4409*np.square(x))+
           2.2884*np.exp(-20.3004*np.square(x))+0.0223*np.exp(-115.122*np.square(x))+1.2298 ,
    'V3+': lambda x: 9.43141*np.exp(-6.39535*np.square(x))+7.7419*np.exp(-0.383349*np.square(x))+
           2.15343*np.exp(-15.1908*np.square(x))+0.016865*np.exp(-63.969*np.square(x))+0.656565 ,
    'V5+': lambda x: 15.6887*np.exp(-0.679003*np.square(x))+8.14208*np.exp(-5.40135*np.square(x))+
           2.03081*np.exp(-9.97278*np.square(x))-9.576*np.exp(-0.940464*np.square(x))+1.7143 ,
    'Cr': lambda x: 10.6406*np.exp(-6.1038*np.square(x))+7.3537*np.exp(-0.392*np.square(x))+
          3.324*np.exp(-20.2626*np.square(x))+1.4922*np.exp(-98.7399*np.square(x))+1.1832 ,
    'Cr2+': lambda x: 9.54034*np.exp(-5.66078*np.square(x))+7.7509*np.exp(-0.344261*np.square(x))+
            3.58274*np.exp(-13.3075*np.square(x))+0.509107*np.exp(-32.4224*np.square(x))+0.616898 ,
    'Cr3+': lambda x: 9.6809*np.exp(-5.59463*np.square(x))+7.81136*np.exp(-0.334393*np.square(x))+
            2.87603*np.exp(-12.8288*np.square(x))+0.113575*np.exp(-32.8761*np.square(x))+0.518275 ,
    'Mn': lambda x: 11.2819*np.exp(-5.3409*np.square(x))+7.3573*np.exp(-0.3432*np.square(x))+
          3.0193*np.exp(-17.8674*np.square(x))+2.2441*np.exp(-83.7543*np.square(x))+1.0896 ,
    'Mn2+': lambda x: 10.8061*np.exp(-5.2796*np.square(x))+7.362*np.exp(-0.3435*np.square(x))+
            3.5268*np.exp(-14.343*np.square(x))+0.2184*np.exp(-41.3235*np.square(x))+1.0874 ,
    'Mn3+': lambda x: 9.84521*np.exp(-4.91797*np.square(x))+7.87194*np.exp(-0.294393*np.square(x))+
            3.56531*np.exp(-10.8171*np.square(x))+0.323613*np.exp(-24.1281*np.square(x))+0.393974 ,
    'Mn4+': lambda x: 9.96253*np.exp(-4.8485*np.square(x))+7.97057*np.exp(-0.283303*np.square(x))+
            2.76067*np.exp(-10.4852*np.square(x))+0.054447*np.exp(-27.573*np.square(x))+0.251877 ,
    'Fe': lambda x: 11.7695*np.exp(-4.7611*np.square(x))+7.3573*np.exp(-0.3072*np.square(x))+
          3.5222*np.exp(-15.3535*np.square(x))+2.3045*np.exp(-76.8805*np.square(x))+1.0369 ,
    'Fe2+': lambda x: 11.0424*np.exp(-4.6538*np.square(x))+7.374*np.exp(-0.3053*np.square(x))+
            4.1346*np.exp(-12.0546*np.square(x))+0.4399*np.exp(-31.2809*np.square(x))+1.0097 ,
    'Fe3+': lambda x: 11.1764*np.exp(-4.6147*np.square(x))+7.3863*np.exp(-0.3005*np.square(x))+
            3.3948*np.exp(-11.6729*np.square(x))+0.0724*np.exp(-38.5566*np.square(x))+0.9707 ,
    'Co': lambda x: 12.2841*np.exp(-4.2791*np.square(x))+7.3409*np.exp(-0.2784*np.square(x))+
          4.0034*np.exp(-13.5359*np.square(x))+2.3488*np.exp(-71.1692*np.square(x))+1.0118 ,
    'Co2+': lambda x: 11.2296*np.exp(-4.1231*np.square(x))+7.3883*np.exp(-0.2726*np.square(x))+
            4.7393*np.exp(-10.2443*np.square(x))+0.7108*np.exp(-25.6466*np.square(x))+0.9324 ,
    'Ni': lambda x: 12.8376*np.exp(-3.8785*np.square(x))+7.292*np.exp(-0.2565*np.square(x))+
          4.4438*np.exp(-12.1763*np.square(x))+2.38*np.exp(-66.3421*np.square(x))+1.0341 ,
    'Ni2+': lambda x: 11.4166*np.exp(-3.6766*np.square(x))+7.4005*np.exp(-0.2449*np.square(x))+
            5.3442*np.exp(-8.873*np.square(x))+0.9773*np.exp(-22.1626*np.square(x))+0.8614 ,
    'Ni3+': lambda x: 10.7806*np.exp(-3.5477*np.square(x))+7.75868*np.exp(-0.22314*np.square(x))+
            5.22746*np.exp(-7.64468*np.square(x))+0.847114*np.exp(-16.9673*np.square(x))+0.386044 ,
    'Cu': lambda x: 13.338*np.exp(-3.5828*np.square(x))+7.1676*np.exp(-0.247*np.square(x))+
          5.6158*np.exp(-11.3966*np.square(x))+1.6735*np.exp(-64.8126*np.square(x))+1.191 ,
    'Cu1+': lambda x: 11.9475*np.exp(-3.3669*np.square(x))+7.3573*np.exp(-0.2274*np.square(x))+
            6.2455*np.exp(-8.6625*np.square(x))+1.5578*np.exp(-25.8487*np.square(x))+0.89 ,
    'Cu2+': lambda x: 11.8168*np.exp(-3.37484*np.square(x))+7.11181*np.exp(-0.244078*np.square(x))+
            5.78135*np.exp(-7.9876*np.square(x))+1.14523*np.exp(-19.897*np.square(x))+1.14431 ,
    'Zn': lambda x: 14.0743*np.exp(-3.2655*np.square(x))+7.0318*np.exp(-0.2333*np.square(x))+
          5.1652*np.exp(-10.3163*np.square(x))+2.41*np.exp(-58.7097*np.square(x))+1.3041 ,
    'Zn2+': lambda x: 11.9719*np.exp(-2.9946*np.square(x))+7.3862*np.exp(-0.2031*np.square(x))+
            6.4668*np.exp(-7.0826*np.square(x))+1.394*np.exp(-18.0995*np.square(x))+0.7807 ,
    'Ga': lambda x: 15.2354*np.exp(-3.0669*np.square(x))+6.7006*np.exp(-0.2412*np.square(x))+
          4.3591*np.exp(-10.7805*np.square(x))+2.9623*np.exp(-61.4135*np.square(x))+1.7189 ,
    'Ga3+': lambda x: 12.692*np.exp(-2.81262*np.square(x))+6.69883*np.exp(-0.22789*np.square(x))+
            6.06692*np.exp(-6.36441*np.square(x))+1.0066*np.exp(-14.4122*np.square(x))+1.53545 ,
    'Ge': lambda x: 16.0816*np.exp(-2.8509*np.square(x))+6.3747*np.exp(-0.2516*np.square(x))+
          3.7068*np.exp(-11.4468*np.square(x))+3.683*np.exp(-54.7625*np.square(x))+2.1313 ,
    'Ge4+': lambda x: 12.9172*np.exp(-2.53718*np.square(x))+6.70003*np.exp(-0.205855*np.square(x))+
            6.06791*np.exp(-5.47913*np.square(x))+0.859041*np.exp(-11.603*np.square(x))+1.45572 ,
    'As': lambda x: 16.6723*np.exp(-2.6345*np.square(x))+6.0701*np.exp(-0.2647*np.square(x))+
          3.4313*np.exp(-12.9479*np.square(x))+4.2779*np.exp(-47.7972*np.square(x))+2.531 ,
    'Se': lambda x: 17.0006*np.exp(-2.4098*np.square(x))+5.8196*np.exp(-0.2726*np.square(x))+
          3.9731*np.exp(-15.2372*np.square(x))+4.3543*np.exp(-43.8163*np.square(x))+2.8409 ,
    'Br': lambda x: 17.1789*np.exp(-2.1723*np.square(x))+5.2358*np.exp(-16.5796*np.square(x))+
          5.6377*np.exp(-0.2609*np.square(x))+3.9851*np.exp(-41.4328*np.square(x))+2.9557 ,
    'Br1-': lambda x: 17.1718*np.exp(-2.2059*np.square(x))+6.3338*np.exp(-19.3345*np.square(x))+
            5.5754*np.exp(-0.2871*np.square(x))+3.7272*np.exp(-58.1535*np.square(x))+3.1776 ,
    'Kr': lambda x: 17.3555*np.exp(-1.9384*np.square(x))+6.7286*np.exp(-16.5623*np.square(x))+
          5.5493*np.exp(-0.2261*np.square(x))+3.5375*np.exp(-39.3972*np.square(x))+2.825 ,
    'Rb': lambda x: 17.1784*np.exp(-1.7888*np.square(x))+9.6435*np.exp(-17.3151*np.square(x))+
          5.1399*np.exp(-0.2748*np.square(x))+1.5292*np.exp(-164.934*np.square(x))+3.4873 ,
    'Rb1+': lambda x: 17.5816*np.exp(-1.7139*np.square(x))+7.6598*np.exp(-14.7957*np.square(x))+
            5.8981*np.exp(-0.1603*np.square(x))+2.7817*np.exp(-31.2087*np.square(x))+2.0782 ,
    'Sr': lambda x: 17.5663*np.exp(-1.5564*np.square(x))+9.8184*np.exp(-14.0988*np.square(x))+
          5.422*np.exp(-0.1664*np.square(x))+2.6694*np.exp(-132.376*np.square(x))+2.5064 ,
    'Sr2+': lambda x: 18.0874*np.exp(-1.4907*np.square(x))+8.1373*np.exp(-12.6963*np.square(x))+
            2.5654*np.exp(24.5651*np.square(x))+34.193*np.exp(0.0138*np.square(x))+41.4025 ,
    'Y': lambda x: 17.776*np.exp(-1.4029*np.square(x))+10.2946*np.exp(-12.8006*np.square(x))+
         5.72629*np.exp(-0.125599*np.square(x))+3.26588*np.exp(-104.354*np.square(x))+1.91213 ,
    'Y3+': lambda x: 17.9268*np.exp(-1.35417*np.square(x))+9.1531*np.exp(-11.2145*np.square(x))+
           1.76795*np.exp(22.6599*np.square(x))+33.108*np.exp(0.01319*np.square(x))+40.2602 ,
    'Zr': lambda x: 17.8765*np.exp(-1.27618*np.square(x))+10.948*np.exp(-11.916*np.square(x))+
          5.41732*np.exp(-0.117622*np.square(x))+3.65721*np.exp(-87.6627*np.square(x))+2.06929 ,
    'Zr4+': lambda x: 18.1668*np.exp(-1.2148*np.square(x))+10.0562*np.exp(-10.1483*np.square(x))+
            1.01118*np.exp(-21.6054*np.square(x))-2.6479*np.exp(0.10276*np.square(x))+9.41454 ,
    'Nb': lambda x: 17.6142*np.exp(-1.18865*np.square(x))+12.0144*np.exp(-11.766*np.square(x))+
          4.04183*np.exp(-0.204785*np.square(x))+3.53346*np.exp(-69.7957*np.square(x))+3.75591 ,
    'Nb3+': lambda x: 19.8812*np.exp(-0.019175*np.square(x))+18.0653*np.exp(-1.13305*np.square(x))+
            11.0177*np.exp(-10.1621*np.square(x))+1.94715*np.exp(-28.3389*np.square(x))-12.912 ,
    'Nb5+': lambda x: 17.9163*np.exp(-1.12446*np.square(x))+13.3417*np.exp(-0.028781*np.square(x))+
            10.799*np.exp(-9.28206*np.square(x))+0.337905*np.exp(-25.7228*np.square(x))-6.3934 ,
    'Mo': lambda x: 3.7025*np.exp(-0.2772*np.square(x))+17.2356*np.exp(-1.0958*np.square(x))+
          12.8876*np.exp(-11.004*np.square(x))+3.7429*np.exp(-61.6584*np.square(x))+4.3875 ,
    'Mo3+': lambda x: 21.1664*np.exp(-0.014734*np.square(x))+18.2017*np.exp(-1.03031*np.square(x))+
            11.7423*np.exp(-9.53659*np.square(x))+2.30951*np.exp(-26.6307*np.square(x))-14.421 ,
    'Mo5+': lambda x: 21.0149*np.exp(-0.014345*np.square(x))+18.0992*np.exp(-1.02238*np.square(x))+
            11.4632*np.exp(-8.78809*np.square(x))+0.740625*np.exp(-23.3452*np.square(x))-14.316 ,
    'Mo6+': lambda x: 17.8871*np.exp(-1.03649*np.square(x))+11.175*np.exp(-8.48061*np.square(x))+
            6.57891*np.exp(-0.058881*np.square(x))+0*np.exp(-0*np.square(x))+0.344941 ,
    'Tc': lambda x: 19.1301*np.exp(-0.864132*np.square(x))+11.0948*np.exp(-8.14487*np.square(x))+
          4.64901*np.exp(-21.5707*np.square(x))+2.71263*np.exp(-86.8472*np.square(x))+5.40428 ,
    'Ru': lambda x: 19.2674*np.exp(-0.80852*np.square(x))+12.9182*np.exp(-8.43467*np.square(x))+
          4.86337*np.exp(-24.7997*np.square(x))+1.56756*np.exp(-94.2928*np.square(x))+5.37814 ,
    'Ru3+': lambda x: 18.5638*np.exp(-0.847329*np.square(x))+13.2885*np.exp(-8.37164*np.square(x))+
            9.32602*np.exp(-0.017662*np.square(x))+3.00964*np.exp(-22.887*np.square(x))-3.1892 ,
    'Ru4+': lambda x: 18.5003*np.exp(-0.844582*np.square(x))+13.1787*np.exp(-8.12534*np.square(x))+
            4.71304*np.exp(-0.36495*np.square(x))+2.18535*np.exp(-20.8504*np.square(x))+1.42357 ,
    'Rh': lambda x: 19.2957*np.exp(-0.751536*np.square(x))+14.3501*np.exp(-8.21758*np.square(x))+
          4.73425*np.exp(-25.8749*np.square(x))+1.28918*np.exp(-98.6062*np.square(x))+5.328 ,
    'Rh3+': lambda x: 18.8785*np.exp(-0.764252*np.square(x))+14.1259*np.exp(-7.84438*np.square(x))+
            3.32515*np.exp(-21.2487*np.square(x))-6.1989*np.exp(0.01036*np.square(x))+11.8678 ,
    'Rh4+': lambda x: 18.8545*np.exp(-0.760825*np.square(x))+13.9806*np.exp(-7.62436*np.square(x))+
            2.53464*np.exp(-19.3317*np.square(x))-5.6526*np.exp(0.0102*np.square(x))+11.2835 ,
    'Pd': lambda x: 19.3319*np.exp(-0.698655*np.square(x))+15.5017*np.exp(-7.98929*np.square(x))+
          5.29537*np.exp(-25.2052*np.square(x))+0.605844*np.exp(-76.8986*np.square(x))+5.26593 ,
    'Pd2+': lambda x: 19.1701*np.exp(-0.696219*np.square(x))+15.2096*np.exp(-7.55573*np.square(x))+
            4.32234*np.exp(-22.5057*np.square(x))+0*np.exp(-0*np.square(x))+5.2916 ,
    'Pd4+': lambda x: 19.2493*np.exp(-0.683839*np.square(x))+14.79*np.exp(-7.14833*np.square(x))+
            2.89289*np.exp(-17.9144*np.square(x))-7.9492*np.exp(-0.005127*np.square(x))+13.0174 ,
    'Ag': lambda x: 19.2808*np.exp(-0.6446*np.square(x))+16.6885*np.exp(-7.4726*np.square(x))+
          4.8045*np.exp(-24.6605*np.square(x))+1.0463*np.exp(-99.8156*np.square(x))+5.179 ,
    'Ag1+': lambda x: 19.1812*np.exp(-0.646179*np.square(x))+15.9719*np.exp(-7.19123*np.square(x))+
            5.27475*np.exp(-21.7326*np.square(x))+0.357534*np.exp(-66.1147*np.square(x))+5.21572 ,
    'Ag2+': lambda x: 19.1643*np.exp(-0.645643*np.square(x))+16.2456*np.exp(-7.18544*np.square(x))+
            4.3709*np.exp(-21.4072*np.square(x))+0*np.exp(-0*np.square(x))+5.21404 ,
    'Cd': lambda x: 19.2214*np.exp(-0.5946*np.square(x))+17.6444*np.exp(-6.9089*np.square(x))+
          4.461*np.exp(-24.7008*np.square(x))+1.6029*np.exp(-87.4825*np.square(x))+5.0694 ,
    'Cd2+': lambda x: 19.1514*np.exp(-0.597922*np.square(x))+17.2535*np.exp(-6.80639*np.square(x))+
            4.47128*np.exp(-20.2521*np.square(x))+0*np.exp(-0*np.square(x))+5.11937 ,
    'In': lambda x: 19.1624*np.exp(-0.5476*np.square(x))+18.5596*np.exp(-6.3776*np.square(x))+
          4.2948*np.exp(-25.8499*np.square(x))+2.0396*np.exp(-92.8029*np.square(x))+4.9391 ,
    'In3+': lambda x: 19.1045*np.exp(-0.551522*np.square(x))+18.1108*np.exp(-6.3247*np.square(x))+
            3.78897*np.exp(-17.3595*np.square(x))+0*np.exp(-0*np.square(x))+4.99635 ,
    'Sn': lambda x: 19.1889*np.exp(-5.8303*np.square(x))+19.1005*np.exp(-0.5031*np.square(x))+
          4.4585*np.exp(-26.8909*np.square(x))+2.4663*np.exp(-83.9571*np.square(x))+4.7821 ,
    'Sn2+': lambda x: 19.1094*np.exp(-0.5036*np.square(x))+19.0548*np.exp(-5.8378*np.square(x))+
            4.5648*np.exp(-23.3752*np.square(x))+0.487*np.exp(-62.2061*np.square(x))+4.7861 ,
    'Sn4+': lambda x: 18.9333*np.exp(-5.764*np.square(x))+19.7131*np.exp(-0.4655*np.square(x))+
            3.4182*np.exp(-14.0049*np.square(x))+0.0193*np.exp(0.7583*np.square(x))+3.9182 ,
    'Sb': lambda x: 19.6418*np.exp(-5.3034*np.square(x))+19.0455*np.exp(-0.4607*np.square(x))+
          5.0371*np.exp(-27.9074*np.square(x))+2.6827*np.exp(-75.2825*np.square(x))+4.5909 ,
    'Sb3+': lambda x: 18.9755*np.exp(-0.467196*np.square(x))+18.933*np.exp(-5.22126*np.square(x))+
            5.10789*np.exp(-19.5902*np.square(x))+0.288753*np.exp(-55.5113*np.square(x))+4.69626 ,
    'Sb5+': lambda x: 19.8685*np.exp(-5.44853*np.square(x))+19.0302*np.exp(-0.467973*np.square(x))+
            2.41253*np.exp(-14.1259*np.square(x))+0*np.exp(-0*np.square(x))+4.69263 ,
    'Te': lambda x: 19.9644*np.exp(-4.81742*np.square(x))+19.0138*np.exp(-0.420885*np.square(x))+
          6.14487*np.exp(-28.5284*np.square(x))+2.5239*np.exp(-70.8403*np.square(x))+4.352 ,
    'I': lambda x: 20.1472*np.exp(-4.347*np.square(x))+18.9949*np.exp(-0.3814*np.square(x))+
         7.5138*np.exp(-27.766*np.square(x))+2.2735*np.exp(-66.8776*np.square(x))+4.0712 ,
    'I1-': lambda x: 20.2332*np.exp(-4.3579*np.square(x))+18.997*np.exp(-0.3815*np.square(x))+
           7.8069*np.exp(-29.5259*np.square(x))+2.8868*np.exp(-84.9304*np.square(x))+4.0714 ,
    'Xe': lambda x: 20.2933*np.exp(-3.9282*np.square(x))+19.0298*np.exp(-0.344*np.square(x))+
          8.9767*np.exp(-26.4659*np.square(x))+1.99*np.exp(-64.2658*np.square(x))+3.7118 ,
    'Cs': lambda x: 20.3892*np.exp(-3.569*np.square(x))+19.1062*np.exp(-0.3107*np.square(x))+
          10.662*np.exp(-24.3879*np.square(x))+1.4953*np.exp(-213.904*np.square(x))+3.3352 ,
    'Cs1+': lambda x: 20.3524*np.exp(-3.552*np.square(x))+19.1278*np.exp(-0.3086*np.square(x))+
            10.2821*np.exp(-23.7128*np.square(x))+0.9615*np.exp(-59.4565*np.square(x))+3.2791 ,
    'Ba': lambda x: 20.3361*np.exp(-3.216*np.square(x))+19.297*np.exp(-0.2756*np.square(x))+
          10.888*np.exp(-20.2073*np.square(x))+2.6959*np.exp(-167.202*np.square(x))+2.7731 ,
    'Ba2+': lambda x: 20.1807*np.exp(-3.21367*np.square(x))+19.1136*np.exp(-0.28331*np.square(x))+
            10.9054*np.exp(-20.0558*np.square(x))+0.77634*np.exp(-51.746*np.square(x))+3.02902 ,
    'La': lambda x: 20.578*np.exp(-2.94817*np.square(x))+19.599*np.exp(-0.244475*np.square(x))+
          11.3727*np.exp(-18.7726*np.square(x))+3.28719*np.exp(-133.124*np.square(x))+2.14678 ,
    'La3+': lambda x: 20.2489*np.exp(-2.9207*np.square(x))+19.3763*np.exp(-0.250698*np.square(x))+
            11.6323*np.exp(-17.8211*np.square(x))+0.336048*np.exp(-54.9453*np.square(x))+2.4086 ,
    'Ce': lambda x: 21.1671*np.exp(-2.81219*np.square(x))+19.7695*np.exp(-0.226836*np.square(x))+
          11.8513*np.exp(-17.6083*np.square(x))+3.33049*np.exp(-127.113*np.square(x))+1.86264 ,
    'Ce3+': lambda x: 20.8036*np.exp(-2.77691*np.square(x))+19.559*np.exp(-0.23154*np.square(x))+
            11.9369*np.exp(-16.5408*np.square(x))+0.612376*np.exp(-43.1692*np.square(x))+2.09013 ,
    'Ce4+': lambda x: 20.3235*np.exp(-2.65941*np.square(x))+19.8186*np.exp(-0.21885*np.square(x))+
            12.1233*np.exp(-15.7992*np.square(x))+0.144583*np.exp(-62.2355*np.square(x))+1.5918 ,
    'Pr': lambda x: 22.044*np.exp(-2.77393*np.square(x))+19.6697*np.exp(-0.222087*np.square(x))+
          12.3856*np.exp(-16.7669*np.square(x))+2.82428*np.exp(-143.644*np.square(x))+2.0583 ,
    'Pr3+': lambda x: 21.3727*np.exp(-2.6452*np.square(x))+19.7491*np.exp(-0.214299*np.square(x))+
            12.1329*np.exp(-15.323*np.square(x))+0.97518*np.exp(-36.4065*np.square(x))+1.77132 ,
    'Pr4+': lambda x: 20.9413*np.exp(-2.54467*np.square(x))+20.0539*np.exp(-0.202481*np.square(x))+
            12.4668*np.exp(-14.8137*np.square(x))+0.296689*np.exp(-45.4643*np.square(x))+1.24285 ,
    'Nd': lambda x: 22.6845*np.exp(-2.66248*np.square(x))+19.6847*np.exp(-0.210628*np.square(x))+
          12.774*np.exp(-15.885*np.square(x))+2.85137*np.exp(-137.903*np.square(x))+1.98486 ,
    'Nd3+': lambda x: 21.961*np.exp(-2.52722*np.square(x))+19.9339*np.exp(-0.199237*np.square(x))+
            12.12*np.exp(-14.1783*np.square(x))+1.51031*np.exp(-30.8717*np.square(x))+1.47588 ,
    'Pm': lambda x: 23.3405*np.exp(-2.5627*np.square(x))+19.6095*np.exp(-0.202088*np.square(x))+
          13.1235*np.exp(-15.1009*np.square(x))+2.87516*np.exp(-132.721*np.square(x))+2.02876 ,
    'Pm3+': lambda x: 22.5527*np.exp(-2.4174*np.square(x))+20.1108*np.exp(-0.185769*np.square(x))+
            12.0671*np.exp(-13.1275*np.square(x))+2.07492*np.exp(-27.4491*np.square(x))+1.19499 ,
    'Sm': lambda x: 24.0042*np.exp(-2.47274*np.square(x))+19.4258*np.exp(-0.196451*np.square(x))+
          13.4396*np.exp(-14.3996*np.square(x))+2.89604*np.exp(-128.007*np.square(x))+2.20963 ,
    'Sm3+': lambda x: 23.1504*np.exp(-2.31641*np.square(x))+20.2599*np.exp(-0.174081*np.square(x))+
            11.9202*np.exp(-12.1571*np.square(x))+2.71488*np.exp(-24.8242*np.square(x))+0.954586 ,
    'Eu': lambda x: 24.6274*np.exp(-2.3879*np.square(x))+19.0886*np.exp(-0.1942*np.square(x))+
          13.7603*np.exp(-13.7546*np.square(x))+2.9227*np.exp(-123.174*np.square(x))+2.5745 ,
    'Eu2+': lambda x: 24.0063*np.exp(-2.27783*np.square(x))+19.9504*np.exp(-0.17353*np.square(x))+
            11.8034*np.exp(-11.6096*np.square(x))+3.87243*np.exp(-26.5156*np.square(x))+1.36389 ,
    'Eu3+': lambda x: 23.7497*np.exp(-2.22258*np.square(x))+20.3745*np.exp(-0.16394*np.square(x))+
            11.8509*np.exp(-11.311*np.square(x))+3.26503*np.exp(-22.9966*np.square(x))+0.759344 ,
    'Gd': lambda x: 25.0709*np.exp(-2.25341*np.square(x))+19.0798*np.exp(-0.181951*np.square(x))+
          13.8518*np.exp(-12.9331*np.square(x))+3.54545*np.exp(-101.398*np.square(x))+2.4196 ,
    'Gd3+': lambda x: 24.3466*np.exp(-2.13553*np.square(x))+20.4208*np.exp(-0.155525*np.square(x))+
            11.8708*np.exp(-10.5782*np.square(x))+3.7149*np.exp(-21.7029*np.square(x))+0.645089 ,
    'Tb': lambda x: 25.8976*np.exp(-2.24256*np.square(x))+18.2185*np.exp(-0.196143*np.square(x))+
          14.3167*np.exp(-12.6648*np.square(x))+2.95354*np.exp(-115.362*np.square(x))+3.58324 ,
    'Tb3+': lambda x: 24.9559*np.exp(-2.05601*np.square(x))+20.3271*np.exp(-0.149525*np.square(x))+
            12.2471*np.exp(-10.0499*np.square(x))+3.773*np.exp(-21.2773*np.square(x))+0.691967 ,
    'Dy': lambda x: 26.507*np.exp(-2.1802*np.square(x))+17.6383*np.exp(-0.202172*np.square(x))+
          14.5596*np.exp(-12.1899*np.square(x))+2.96577*np.exp(-111.874*np.square(x))+4.29728 ,
    'Dy3+': lambda x: 25.5395*np.exp(-1.9804*np.square(x))+20.2861*np.exp(-0.143384*np.square(x))+
            11.9812*np.exp(-9.34972*np.square(x))+4.50073*np.exp(-19.581*np.square(x))+0.68969 ,
    'Ho': lambda x: 26.9049*np.exp(-2.07051*np.square(x))+17.294*np.exp(-0.19794*np.square(x))+
          14.5583*np.exp(-11.4407*np.square(x))+3.63837*np.exp(-92.6566*np.square(x))+4.56796 ,
    'Ho3+': lambda x: 26.1296*np.exp(-1.91072*np.square(x))+20.0994*np.exp(-0.139358*np.square(x))+
            11.9788*np.exp(-8.80018*np.square(x))+4.93676*np.exp(-18.5908*np.square(x))+0.852795 ,
    'Er': lambda x: 27.6563*np.exp(-2.07356*np.square(x))+16.4285*np.exp(-0.223545*np.square(x))+
          14.9779*np.exp(-11.3604*np.square(x))+2.98233*np.exp(-105.703*np.square(x))+5.92046 ,
    'Er3+': lambda x: 26.722*np.exp(-1.84659*np.square(x))+19.7748*np.exp(-0.13729*np.square(x))+
            12.1506*np.exp(-8.36225*np.square(x))+5.17379*np.exp(-17.8974*np.square(x))+1.17613 ,
    'Tm': lambda x: 28.1819*np.exp(-2.02859*np.square(x))+15.8851*np.exp(-0.238849*np.square(x))+
          15.1542*np.exp(-10.9975*np.square(x))+2.98706*np.exp(-102.961*np.square(x))+1.63929 ,
    'Tm3+': lambda x: 27.3083*np.exp(-1.78711*np.square(x))+19.332*np.exp(-0.136974*np.square(x))+
            12.3339*np.exp(-7.96778*np.square(x))+5.38348*np.exp(-17.2922*np.square(x))+1.63929 ,
    'Yb': lambda x: 28.6641*np.exp(-1.9889*np.square(x))+15.4345*np.exp(-0.257119*np.square(x))+
          15.3087*np.exp(-10.6647*np.square(x))+2.98963*np.exp(-100.417*np.square(x))+7.56672 ,
    'Yb2+': lambda x: 28.1209*np.exp(-1.78503*np.square(x))+17.6817*np.exp(-0.15997*np.square(x))+
            13.3335*np.exp(-8.18304*np.square(x))+5.14657*np.exp(-20.39*np.square(x))+3.70983 ,
    'Yb3+': lambda x: 27.8917*np.exp(-1.73272*np.square(x))+18.7614*np.exp(-0.13879*np.square(x))+
            12.6072*np.exp(-7.64412*np.square(x))+5.47647*np.exp(-16.8153*np.square(x))+2.26001 ,
    'Lu': lambda x: 28.9476*np.exp(-1.90182*np.square(x))+15.2208*np.exp(-9.98519*np.square(x))+
          15.1*np.exp(-0.261033*np.square(x))+3.71601*np.exp(-84.3298*np.square(x))+7.97628 ,
    'Lu3+': lambda x: 28.4628*np.exp(-1.68216*np.square(x))+18.121*np.exp(-0.142292*np.square(x))+
            12.8429*np.exp(-7.33727*np.square(x))+5.59415*np.exp(-16.3535*np.square(x))+2.97573 ,
    'Hf': lambda x: 29.144*np.exp(-1.83262*np.square(x))+15.1726*np.exp(-9.5999*np.square(x))+
          14.7586*np.exp(-0.275116*np.square(x))+4.30013*np.exp(-72.029*np.square(x))+8.58154 ,
    'Hf4+': lambda x: 28.8131*np.exp(-1.59136*np.square(x))+18.4601*np.exp(-0.128903*np.square(x))+
            12.7285*np.exp(-6.76232*np.square(x))+5.59927*np.exp(-14.0366*np.square(x))+2.39699 ,
    'Ta': lambda x: 29.2024*np.exp(-1.77333*np.square(x))+15.2293*np.exp(-9.37046*np.square(x))+
          14.5135*np.exp(-0.295977*np.square(x))+4.76492*np.exp(-63.3644*np.square(x))+9.24354 ,
    'Ta5+': lambda x: 29.1587*np.exp(-1.50711*np.square(x))+18.8407*np.exp(-0.116741*np.square(x))+
            12.8268*np.exp(-6.31524*np.square(x))+5.38695*np.exp(-12.4244*np.square(x))+1.78555 ,
    'W': lambda x: 29.0818*np.exp(-1.72029*np.square(x))+15.43*np.exp(-9.2259*np.square(x))+
         14.4327*np.exp(-0.321703*np.square(x))+5.11982*np.exp(-57.056*np.square(x))+9.8875 ,
    'W6+': lambda x: 29.4936*np.exp(-1.42755*np.square(x))+19.3763*np.exp(-0.104621*np.square(x))+
           13.0544*np.exp(-5.93667*np.square(x))+5.06412*np.exp(-11.1972*np.square(x))+1.01074 ,
    'Re': lambda x: 28.7621*np.exp(-1.67191*np.square(x))+15.7189*np.exp(-9.09227*np.square(x))+
          14.5564*np.exp(-0.3505*np.square(x))+5.44174*np.exp(-52.0861*np.square(x))+10.472 ,
    'Os': lambda x: 28.1894*np.exp(-1.62903*np.square(x))+16.155*np.exp(-8.97948*np.square(x))+
          14.9305*np.exp(-0.382661*np.square(x))+5.67589*np.exp(-48.1647*np.square(x))+11.0005 ,
    'Os4+': lambda x: 30.419*np.exp(-1.37113*np.square(x))+15.2637*np.exp(-6.84706*np.square(x))+
            14.7458*np.exp(-0.165191*np.square(x))+5.06795*np.exp(-18.003*np.square(x))+6.49804 ,
    'Ir': lambda x: 27.3049*np.exp(-1.59279*np.square(x))+16.7296*np.exp(-8.86553*np.square(x))+
          15.6115*np.exp(-0.417916*np.square(x))+5.83377*np.exp(-45.0011*np.square(x))+11.4722 ,
    'Ir3+': lambda x: 30.4156*np.exp(-1.34323*np.square(x))+15.862*np.exp(-7.10909*np.square(x))+
            13.6145*np.exp(-0.204633*np.square(x))+5.82008*np.exp(-20.3254*np.square(x))+8.27903 ,
    'Ir4+': lambda x: 30.7058*np.exp(-1.30923*np.square(x))+15.5512*np.exp(-6.71983*np.square(x))+
            14.2326*np.exp(-0.167252*np.square(x))+5.53672*np.exp(-17.4911*np.square(x))+6.96824 ,
    'Pt': lambda x: 27.0059*np.exp(-1.51293*np.square(x))+17.7639*np.exp(-8.81174*np.square(x))+
          15.7131*np.exp(-0.424593*np.square(x))+5.7837*np.exp(-38.6103*np.square(x))+11.6883 ,
    'Pt2+': lambda x: 29.8429*np.exp(-1.32927*np.square(x))+16.7224*np.exp(-7.38979*np.square(x))+
            13.2153*np.exp(-0.263297*np.square(x))+6.35234*np.exp(-22.9426*np.square(x))+9.85329 ,
    'Pt4+': lambda x: 30.9612*np.exp(-1.24813*np.square(x))+15.9829*np.exp(-6.60834*np.square(x))+
            13.7348*np.exp(-0.16864*np.square(x))+5.92034*np.exp(-16.9392*np.square(x))+7.39534 ,
    'Au': lambda x: 16.8819*np.exp(-0.4611*np.square(x))+18.5913*np.exp(-8.6216*np.square(x))+
          25.5582*np.exp(-1.4826*np.square(x))+5.86*np.exp(-36.3956*np.square(x))+12.0658 ,
    'Au1+': lambda x: 28.0109*np.exp(-1.35321*np.square(x))+17.8204*np.exp(-7.7395*np.square(x))+
            14.3359*np.exp(-0.356752*np.square(x))+6.58077*np.exp(-26.4043*np.square(x))+11.2299 ,
    'Au3+': lambda x: 30.6886*np.exp(-1.2199*np.square(x))+16.9029*np.exp(-6.82872*np.square(x))+
            12.7801*np.exp(-0.212867*np.square(x))+6.52354*np.exp(-18.659*np.square(x))+9.0968 ,
    'Hg': lambda x: 20.6809*np.exp(-0.545*np.square(x))+19.0417*np.exp(-8.4484*np.square(x))+
          21.6575*np.exp(-1.5729*np.square(x))+5.9676*np.exp(-38.3246*np.square(x))+12.6089 ,
    'Hg1+': lambda x: 25.0853*np.exp(-1.39507*np.square(x))+18.4973*np.exp(-7.65105*np.square(x))+
            16.8883*np.exp(-0.443378*np.square(x))+6.48216*np.exp(-28.2262*np.square(x))+12.0205 ,
    'Hg2+': lambda x: 29.5641*np.exp(-1.21152*np.square(x))+18.06*np.exp(-7.05639*np.square(x))+
            12.8374*np.exp(-0.284738*np.square(x))+6.89912*np.exp(-20.7482*np.square(x))+10.6268 ,
    'Tl': lambda x: 27.5446*np.exp(-0.65515*np.square(x))+19.1584*np.exp(-8.70751*np.square(x))+
          15.538*np.exp(-1.96347*np.square(x))+5.52593*np.exp(-45.8149*np.square(x))+13.1746 ,
    'Tl1+': lambda x: 21.3985*np.exp(-1.4711*np.square(x))+20.4723*np.exp(-0.517394*np.square(x))+
            18.7478*np.exp(-7.43463*np.square(x))+6.82847*np.exp(-28.8482*np.square(x))+12.5258 ,
    'Tl3+': lambda x: 30.8695*np.exp(-1.1008*np.square(x))+18.3481*np.exp(-6.53852*np.square(x))+
            11.9328*np.exp(-0.219074*np.square(x))+7.00574*np.exp(-17.2114*np.square(x))+9.8027 ,
    'Pb': lambda x: 31.0617*np.exp(-0.6902*np.square(x))+13.0637*np.exp(-2.3576*np.square(x))+
          18.442*np.exp(-8.618*np.square(x))+5.9696*np.exp(-47.2579*np.square(x))+13.4118 ,
    'Pb2+': lambda x: 21.7886*np.exp(-1.3366*np.square(x))+19.5682*np.exp(-0.488383*np.square(x))+
            19.1406*np.exp(-6.7727*np.square(x))+7.01107*np.exp(-23.8132*np.square(x))+12.4734 ,
    'Pb4+': lambda x: 32.1244*np.exp(-1.00566*np.square(x))+18.8003*np.exp(-6.10926*np.square(x))+
            12.0175*np.exp(-0.147041*np.square(x))+6.96886*np.exp(-14.714*np.square(x))+8.08428 ,
    'Bi': lambda x: 33.3689*np.exp(-0.704*np.square(x))+12.951*np.exp(-2.9238*np.square(x))+
          16.5877*np.exp(-8.7937*np.square(x))+6.4692*np.exp(-48.0093*np.square(x))+13.5782 ,
    'Bi3+': lambda x: 21.8053*np.exp(-1.2356*np.square(x))+19.5026*np.exp(-6.24149*np.square(x))+
            19.1053*np.exp(-0.469999*np.square(x))+7.10295*np.exp(-20.3185*np.square(x))+12.4711 ,
    'Bi5+': lambda x: 33.5364*np.exp(-0.91654*np.square(x))+25.0946*np.exp(-0.39042*np.square(x))+
            19.2497*np.exp(-5.71414*np.square(x))+6.91555*np.exp(-12.8285*np.square(x))+6.7994 ,
    'Po': lambda x: 34.6726*np.exp(-0.700999*np.square(x))+15.4733*np.exp(-3.55078*np.square(x))+
          13.1138*np.exp(-9.55642*np.square(x))+7.02588*np.exp(-47.0045*np.square(x))+13.677 ,
    'At': lambda x: 35.3163*np.exp(-0.68587*np.square(x))+19.0211*np.exp(-3.97458*np.square(x))+
          9.49887*np.exp(-11.3824*np.square(x))+7.42518*np.exp(-45.4715*np.square(x))+13.7108 ,
    'Rn': lambda x: 35.5631*np.exp(-0.6631*np.square(x))+21.2816*np.exp(-4.0691*np.square(x))+
          8.0037*np.exp(-14.0422*np.square(x))+7.4433*np.exp(-44.2473*np.square(x))+13.6905 ,
    'Fr': lambda x: 35.9299*np.exp(-0.646453*np.square(x))+23.0547*np.exp(-4.17619*np.square(x))+
          12.1439*np.exp(-23.1052*np.square(x))+2.11253*np.exp(-150.645*np.square(x))+13.7247 ,
    'Ra': lambda x: 35.763*np.exp(-0.616341*np.square(x))+22.9064*np.exp(-3.87135*np.square(x))+
          12.4739*np.exp(-19.9887*np.square(x))+3.21097*np.exp(-142.325*np.square(x))+13.6211 ,
    'Ra2+': lambda x: 35.215*np.exp(-0.604909*np.square(x))+21.67*np.exp(-3.5767*np.square(x))+
            7.91342*np.exp(-12.601*np.square(x))+7.65078*np.exp(-29.8436*np.square(x))+13.5431 ,
    'Ac': lambda x: 35.6597*np.exp(-0.589092*np.square(x))+23.1032*np.exp(-3.65155*np.square(x))+
          12.5977*np.exp(-18.599*np.square(x))+4.08655*np.exp(-117.02*np.square(x))+13.5266 ,
    'Ac3+': lambda x: 35.1736*np.exp(-0.579689*np.square(x))+22.1112*np.exp(-3.41437*np.square(x))+
            8.19216*np.exp(-12.9187*np.square(x))+7.05545*np.exp(-25.9443*np.square(x))+13.4637 ,
    'Th': lambda x: 35.5645*np.exp(-0.563359*np.square(x))+23.4219*np.exp(-3.46204*np.square(x))+
          12.7473*np.exp(-17.8309*np.square(x))+4.80703*np.exp(-99.1722*np.square(x))+13.4314 ,
    'Th4+': lambda x: 35.1007*np.exp(-0.555054*np.square(x))+22.4418*np.exp(-3.24498*np.square(x))+
            9.78554*np.exp(-13.4661*np.square(x))+5.29444*np.exp(-23.9533*np.square(x))+13.376 ,
    'Pa': lambda x: 35.8847*np.exp(-0.547751*np.square(x))+23.2948*np.exp(-3.41519*np.square(x))+
          14.1891*np.exp(-16.9235*np.square(x))+4.17287*np.exp(-105.251*np.square(x))+13.4287 ,
    'U': lambda x: 36.0228*np.exp(-0.5293*np.square(x))+23.4128*np.exp(-3.3253*np.square(x))+
         14.9491*np.exp(-16.0927*np.square(x))+4.188*np.exp(-100.613*np.square(x))+13.3966 ,
    'U3+': lambda x: 35.5747*np.exp(-0.52048*np.square(x))+22.5259*np.exp(-3.12293*np.square(x))+
           12.2165*np.exp(-12.7148*np.square(x))+5.37073*np.exp(-26.3394*np.square(x))+13.3092 ,
    'U4+': lambda x: 35.3715*np.exp(-0.516598*np.square(x))+22.5326*np.exp(-3.05053*np.square(x))+
           12.0291*np.exp(-12.5723*np.square(x))+4.7984*np.exp(-23.4582*np.square(x))+13.2671 ,
    'U6+': lambda x: 34.8509*np.exp(-0.507079*np.square(x))+22.7584*np.exp(-2.8903*np.square(x))+
           14.0099*np.exp(-13.1767*np.square(x))+1.21457*np.exp(-25.2017*np.square(x))+13.1665 ,
    'Np': lambda x: 36.1874*np.exp(-0.511929*np.square(x))+23.5964*np.exp(-3.25396*np.square(x))+
          15.6402*np.exp(-15.3622*np.square(x))+4.1855*np.exp(-97.4908*np.square(x))+13.3573 ,
    'Np3+': lambda x: 35.7074*np.exp(-0.502322*np.square(x))+22.613*np.exp(-3.03807*np.square(x))+
            12.9898*np.exp(-12.1449*np.square(x))+5.43227*np.exp(-25.4928*np.square(x))+13.2544 ,
    'Np4+': lambda x: 35.5103*np.exp(-0.498626*np.square(x))+22.5787*np.exp(-2.96627*np.square(x))+
            12.7766*np.exp(-11.9484*np.square(x))+4.92159*np.exp(-22.7502*np.square(x))+13.2116 ,
    'Np6+': lambda x: 35.0136*np.exp(-0.48981*np.square(x))+22.7286*np.exp(-2.81099*np.square(x))+
            14.3884*np.exp(-12.33*np.square(x))+1.75669*np.exp(-22.6581*np.square(x))+13.113 ,
    'Pu': lambda x: 36.5254*np.exp(-0.499384*np.square(x))+23.8083*np.exp(-3.26371*np.square(x))+
          16.7707*np.exp(-14.9455*np.square(x))+3.47947*np.exp(-105.98*np.square(x))+13.3812 ,
    'Pu3+': lambda x: 35.84*np.exp(-0.484938*np.square(x))+22.7169*np.exp(-2.96118*np.square(x))+
            13.5807*np.exp(-11.5331*np.square(x))+5.66016*np.exp(-24.3992*np.square(x))+13.1991 ,
    'Pu4+': lambda x: 35.6493*np.exp(-0.481422*np.square(x))+22.646*np.exp(-2.8902*np.square(x))+
            13.3595*np.exp(-11.316*np.square(x))+5.18831*np.exp(-21.8301*np.square(x))+13.1555 ,
    'Pu6+': lambda x: 35.1736*np.exp(-0.473204*np.square(x))+22.7181*np.exp(-2.73848*np.square(x))+
            14.7635*np.exp(-11.553*np.square(x))+2.28678*np.exp(-20.9303*np.square(x))+13.0582 ,
    'Am': lambda x: 36.6706*np.exp(-0.483629*np.square(x))+24.0992*np.exp(-3.20647*np.square(x))+
          17.3415*np.exp(-14.3136*np.square(x))+3.49331*np.exp(-102.273*np.square(x))+13.3592 ,
    'Cm': lambda x: 36.6488*np.exp(-0.465154*np.square(x))+24.4096*np.exp(-3.08997*np.square(x))+
          17.399*np.exp(-13.4346*np.square(x))+4.21665*np.exp(-88.4834*np.square(x))+13.2887 ,
    'Bk': lambda x: 36.7881*np.exp(-0.451018*np.square(x))+24.7736*np.exp(-3.04619*np.square(x))+
          17.8919*np.exp(-12.8946*np.square(x))+4.23284*np.exp(-86.003*np.square(x))+13.2754 ,
    'Cf': lambda x: 36.9185*np.exp(-0.437533*np.square(x))+25.1995*np.exp(-3.00775*np.square(x))+
          18.3317*np.exp(-12.4044*np.square(x))+4.24391*np.exp(-83.7881*np.square(x))+13.2674 }
