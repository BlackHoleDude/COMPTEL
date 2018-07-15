# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

# Each input EVP event has the following structure:
#
#  Item               Type            Offset     Required Scaling
# 00  D1 x-location      INTEGER*2       0            / 32.0  = mm
# 01  D1 y-location      INTEGER*2       2            / 32.0  = mm
# 02  D1 lambda          INTEGER*2       4            / 256.0
# 03  D2 x-location      INTEGER*2       6            / 32.0  = mm
# 04  D2 y-location      INTEGER*2       8            / 32.0  = mm
# 05  D2 lambda          INTEGER*2       10           / 256.0 
# 06  PSD                INTEGER*2       12           / 128.0
# 07  TOF                INTEGER*2       14           / 128.0
# 08  MODCOM             INTEGER*2       16
# 09  REFLAG             INTEGER*2       18
# 10  VETO               INTEGER*2       20
# 11  TIMTAG - TJD       INTEGER*2       22
# 12  TIMTAG - tics      INTEGER*4       24
# 13  D1E                REAL*4          28
# 14  D2E                REAL*4          32
# 15  PHIBAR             REAL*4          36
# 16  Gal Lat            REAL*4          40
# 17  Gal Long           REAL*4          44
# 18  S/C azimuth        REAL*4          48
# 19  S/C zenith         REAL*4          52
# 20  Earth Horizon      REAL*4          56
# 21  Event Class        Char*8          60
#
# All angles in radians.
#


#
# INPUT PARAMETERS
#
#   1) event id
#   2) EVP filename
#   3) TJD
#   3) trigger time
#   4) T90
#   5) GRBlon
#   6) GRBlat
#   7) Emin
#   8) Emax
#   9) Modulation Factor
#   

       

import struct
import math
import os
import pandas
import scipy
from scipy.optimize import curve_fit
import numpy as np
import matplotlib.pyplot as plt
import astropy.coordinates as coords
from astropy import units as u
import astropy.time as ast
import astropy.visualization as aplt

# =============================================================================
# TEST CASE.
# Initialize the SkyCoord object 'grbloc'
# 
# EVP file = M50733.EVP
#
# TJD = 11579
# T1 = 69000
# T2 = 71000
# 
# FLR_RA = 318.75
# FLR_DEC = -15.95
#
# =============================================================================


evpfile   = input("Enter name of EVP File: ")
flr_ra  = float(input("Enter Right Ascension (RA) of Sun: "))
flr_dec = float(input("Enter Declination (DEC) of Sun: "))
stjd1, st1 = input("Enter Start TJD, Time(s) for data accumulation : ").split()
stjd2, st2 = input("Enter End   TJD, Time(s) for data accumulation : ").split()

tjd1 = float(stjd1)
tjd2 = float(stjd2)

t1 = float(st1)
t2 = float(st2)

ftjd1 = float(tjd1 + (t1/86400))
ftjd2 = float(tjd2 + (t2/86400))

tbin    = 10.0

flrloc = coords.SkyCoord(flr_ra * u.deg, flr_dec * u.deg, frame = 'icrs')

eventid = 'FLR TEST'

#fig1 = plt.figure()

#os.chdir('/Users/mark/Dropbox/Docs/Python')

params = ['D1X', 'D1Y', 'D1Z', 'D2X', 'D2Y', 'D2Z', 'PSD', 'TOF', 'MODCOM', \
          'REFLAG', 'VETO', 'TJD', 'TICS', 'D1E', 'D2E', 'PHIBAR', \
          'GLAT', 'GLON', 'AZI', 'ZEN', 'EHORA', 'CLASS', 'ETOT', 'PHI0', 'ARM']

#==============================================================================
# Open the EVP datafile, read in the data, and close the EVP datafile.
# The variable 'data' refers to the binary data froim the EVP datafile. 
#==============================================================================




###############################################################################
# Change this line to accomodate local file system
###############################################################################
ifile = '/Users/mark/Box/MyStuff/Projects/COMPTEL_Archive/COMPTEL_Data_Archive/MPE/EVP/' + evpfile
###############################################################################
###############################################################################



if os.path.exists(ifile):
    f = open(ifile, "rb")
    data = f.read()
    f.close()
else:
    print("The EVP file does not exist : " + ifile)
    exit()

#==============================================================================
# Determine the number of events in this file based on an event size of 
# 68 bytes.
#==============================================================================

nevents = int(len(data) / 68)

# =============================================================================
# We need to keep track of the offset of each event from the start of the 
# data array. Here we just initialize the offset, setting it to zero for 
# the first event.    
# =============================================================================

offset = 0

# =============================================================================
# Start the loop to process each event in the data array.    
# =============================================================================

ifirst = 0
offset = -68

for i in range(0,nevents) :
     
    #======================================================================
    # Increment offset for next event
    #======================================================================             
    offset = offset + 68    

    #==========================================================================
    # Unpack event from binary data
    # t is a tuple
    #==========================================================================

    t = struct.unpack('>hhhhhhHHHhHHIffffffff', data[offset+0:offset+60])
    evclass = data[offset+61:offset+68].decode('ascii')

    #==========================================================================
    # rec is a list.
    # It contains one event with the following parameters:
    #
    #     Item               Type            
    # 00  D1 x-location      short integer (2 bytes)
    # 01  D1 y-location      short integer (2 bytes)
    # 02  D1 lambda          short integer (2 bytes)
    # 03  D2 x-location      short integer (2 bytes)
    # 04  D2 y-location      short integer (2 bytes)
    # 05  D2 lambda          short integer (2 bytes)
    # 06  PSD                unsigned short integer (2 bytes)
    # 07  TOF                unsigned short integer (2 bytes)
    # 08  MODCOM             unsigned short integer (2 bytes)
    # 09  REFLAG             short integer (2 bytes)
    # 10  VETO               unsigned short integer (2 bytes)
    # 11  TIMTAG - TJD       unsigned short integer (2 bytes)
    # 12  TIMTAG - tics      unsigned integer (4 bytes)
    # 13  D1E                float (4 bytes)
    # 14  D2E                float (4 bytes)
    # 15  PHIBAR             float (4 bytes)
    # 16  Gal Lat            float (4 bytes)
    # 17  Gal Long           float (4 bytes)
    # 18  S/C azimuth        float (4 bytes)
    # 19  S/C zenith         float (4 bytes)
    # 20  Earth Horizon      float (4 bytes)
    # 21  Event Class        string 
    # 22  ETOT               float
    # 23  PHI0               float
    # 24  ARM = PHIBAR-PHI0  float
    #
    #==========================================================================

    rec = list(t)
    if (rec[11]+rec[12]/8000/86400.0 < ftjd1) :
        continue
    if (rec[11]+rec[12]/8000/86400.0 > ftjd2) :
        break
    rec.append(evclass)                    # append EVCLASS
    rec.append(rec[13]+rec[14])            # create and append ETOT

    #==========================================================================
    # Perform conversions as necessary for the event record values.
    #==========================================================================

    rec[0]    = rec[0] / 32.0
    rec[1]    = rec[1] / 32.0
    rec[2]    = rec[2] / 256.0
    rec[3]    = rec[3] / 32.0
    rec[4]    = rec[4] / 32.0
    rec[5]    = rec[5] / 256.0
    rec[6]    = rec[6] / 128.0
    rec[7]    = rec[7] / 128.0
    rec[11]   = float(rec[11])
    rec[15]   = math.degrees(rec[15])      # convert PHIBAR to degrees
    rec[16]   = math.degrees(rec[16])      # convert GLAT to degrees
    rec[17]   = math.degrees(rec[17])      # convert GLON to degrees
    rec[18]   = math.degrees(rec[18])      # convert AZI to degrees
    rec[19]   = math.degrees(rec[19])      # convert ZEN to degrees
    rec[20]   = math.degrees(rec[20])      # convert EHORA to degrees
    
    
    # =========================================================================
    # If there is a valid value for the galactic latitude of the scatter 
    # vector, then there is a valid scatter vector determination.  In that 
    # case, calculate PHI0 and ARM and append those to the event record (rec).
    #
    # If there is no valid value for the galactic latitude of the scatter
    # vector, then there is no valid scatter vector for this event. Set both 
    # PHI0 and ARM to -999 and append those values to the event record (rec).
    # =========================================================================

    if (rec[16] > -90.0) and (rec[16] < 90.0) :
        phigeo = flrloc.separation(coords.SkyCoord(rec[17],rec[16], frame = 'galactic', unit='deg'))
        rec.append(phigeo.degree)            # append PHI0
        rec.append(rec[15] - phigeo.degree)  # append ARM
    else :
        rec.append(math.nan)               # append PHI0
        rec.append(math.nan)               # append ARM

    
    #======================================================================
    # evts is a list
    #====================================================================== 

    if ifirst == 0 :
        evts = [rec]
        ifirst = 1
    else :
        evts.append(rec)
        
#    if (i == 100) :
#        break
    

# =============================================================================
# Create dataframe evtdf from list of events (evts)
# =============================================================================

evtdf = pandas.DataFrame.from_records(evts, columns = params)
evtdf = evtdf.assign(SECS=evtdf.TICS/8000)
evtdf = evtdf.assign(FTJD=evtdf.TJD + evtdf.SECS/86400)
evtdf = evtdf.assign(DT = ast.Time(evtdf.FTJD+2440000.5, format = 'jd'))


# =============================================================================
# Set up event filters.
# =============================================================================

D1E_min = 70.0
D1E_max = 20000.0

D2E_min = 650.0
D2E_max = 30000.0

TOF_min = 115.0
TOF_max = 130.0

PSD_min = 0.0
PSD_max = 110.0

ARM_min = -20.0
ARM_max = 20.0

PHIBAR_min = 0.0
PHIBAR_max = 36.0

ETOT_min = 0.75
ETOT_max = 10000.0


D1E_Limits    = [D1E_min, D1E_max]
D2E_Limits    = [D2E_min, D2E_max]
TOF_Limits    = [TOF_min, TOF_max]
PSD_Limits    = [PSD_min, PSD_max]
ARM_Limits    = [ARM_min, ARM_max]
PHIBAR_Limits = [PHIBAR_min, PHIBAR_max]
ETOT_Limits   = [ETOT_min, ETOT_max]

# =============================================================================
# Define all of the individual filters, one for each event parameter.
# =============================================================================

D1E_Filter  = '(evtdf.D1E  > D1E_Limits[0])   & (evtdf.D1E  < D1E_Limits[1])'
D2E_Filter  = '(evtdf.D2E  > D2E_Limits[0])   & (evtdf.D2E  < D2E_Limits[1])'
ETOT_Filter = '(evtdf.ETOT > ETOT_Limits[0])  & (evtdf.ETOT < ETOT_Limits[1])'
PSD_Filter  = '(evtdf.PSD  > PSD_Limits[0])   & (evtdf.PSD  < PSD_Limits[1])'
TOF_Filter  = '(evtdf.TOF  > TOF_Limits[0])   & (evtdf.TOF  < TOF_Limits[1])'
ARM_Filter  = '(evtdf.ARM  > ARM_Limits[0])   & (evtdf.ARM  < ARM_Limits[1])'
PHIBAR_Filter  = '(evtdf.PHIBAR  > PHIBAR_Limits[0])   & (evtdf.PHIBAR  < PHIBAR_Limits[1])'

# =============================================================================
# Combine all of the individual filters.
# =============================================================================

EVT_Filter = D1E_Filter  + " & " + \
             D2E_Filter  + " & " + \
             ETOT_Filter + " & " + \
             PSD_Filter  + " & " + \
             TOF_Filter  + " & " + \
             ARM_Filter  + " & " + \
             PHIBAR_Filter 


# =============================================================================
# Plot the GRB time history with all filters active.
# hvalues contains the histogram data.
# edges contaions the bin edge data.
# =============================================================================

hvalues, edges, patches = aplt.hist(evtdf.FTJD[eval(EVT_Filter)], range = (ftjd1, ftjd2), bins='freedman')
plt.xlabel('Fractional TJD')
plt.ylabel('Counts')
plt.minorticks_on()
plt.grid(which='major', axis='both')
plt.title('TJD ' + str(evtdf.TJD[0]))
plt.annotate('Total Cts = ' + str(sum(hvalues)), xy = (0.05, 0.90), xycoords='axes fraction')
plt.show()

#plt.plot_date(evtdf.DT.plot_date, hvalues)

# =============================================================================M16998
# Determine the bin midpoint values ('xvalues')
# =============================================================================

#lower = np.resize(edges, len(edges)-1)
#xvalues = lower + 0.5 * np.diff(edges)

# =============================================================================
# Extract the histogram data to fit the background time interval.
# The data are contained in arrays 'xvals' and 'yvals'.
# =============================================================================

#index = (xvalues < (trigtime-t90)) | (xvalues > (trigtime+2*t90))
#xvals = np.compress(index, xvalues)
#yvals = np.compress(index, hvalues)

# =============================================================================
# Define the background fit function.
# =============================================================================

def fit_fcn (x, a, b):
    return a*x + b

# =============================================================================
# 
# =============================================================================

#pfit, pcov = curve_fit(fit_fcn, xvals, yvals, p0=[0, 1])
#yfit = fit_fcn(xvalues, pfit[0], pfit[1])
#plt.plot(xvalues, yfit)
#plt.xlabel('Time (secs)')
#plt.ylabel('Counts')
#plt.minorticks_on()
#plt.grid(which='major', axis='both')
##plt.title(eventid)
##plt.annotate('Trigger Time = ' + str(trigtime), xy = (0.05, 0.90), xycoords='axes fraction')
##plt.annotate('T90 = ' + str(t90),      xy = (0.05, 0.85), xycoords='axes fraction')
#plt.show()
#

# =============================================================================
# loop through data from trigtime to (trigtime+t90)
# Accumulate differences between hvalues and yfit
# =============================================================================

#srctot = 0.0
#bgdtot = 0.0
#
#for i in range(0,len(xvalues)-1) :
#    if (xvalues[i] > trigtime) & (xvalues[i] < (trigtime+t90)):
#        srctot = srctot + (hvalues[i] - yfit[i])
#        bgdtot = bgdtot + yfit[i]
#
#print('Total Src Counts = ' + "%8.1f" % srctot)
#print('Total Bgd Counts = ' + "%8.1f" % bgdtot)
#plt.annotate('Src Cts = ' + "%8.1f" % srctot,      xy = (0.05, 0.80), xycoords='axes fraction')
#plt.annotate('Bgd Cts = ' + "%8.1f" % bgdtot,      xy = (0.05, 0.75), xycoords='axes fraction')
#



# srcdf = evtdf[(evtdf['TICS'] > trigtime*8000) & (evtdf['TICS'] < (trigtime+t90)*8000)]

# bgddf = evtdf[(evtdf['TICS'] < (trigtime-t90)*8000) | (evtdf['TICS'] > (trigtime+(2*t90))*8000)]



while True:
    print("\n\n")
    print("USER MENU")
    print("=========\n")

    print("Current Parameters :")
    print("     EVP File = ", evpfile)
    print("    TJD Range = ", tjd1, tjd2)
    print("   Time Range = ", t1, t2)
    print("   FTJD Range = ", ftjd1, ftjd2)
    print("    D1E Range = ", D1E_min, D1E_max)
    print("    D2E Range = ", D2E_min, D2E_max)
    print("    TOF Range = ", TOF_min, TOF_max)
    print("    PSD Range = ", PSD_min, PSD_max)
    print("    ARM Range = ", ARM_min, ARM_max)
    print(" PHIBAR Range = ",PHIBAR_min, PHIBAR_max)
    print("   ETOT Range = ",ETOT_min, ETOT_max)
    print(" Time Bin (s) = ", tbin)
    print("\n")
    print("Choose from one of the following:")
    print("(1) Set time interval")
    print("(2) Set D1E  limits")
    print("(3) Set D2E  limits")
    print("(4) Set ETOT limits")
    print("(5) Set TOF  limits")
    print("(6) Set PSD  limits")
    print("(7) Set ARM  limits")
    print("(8) Set PHIBAR limits")
    print("(9) Set time binning (secs)")
    print("\n")
    print("(11) Plot Time History")
    print("(12) Plot TOF Distribution")
    print("(13) Plot PSD Distribution")
    print("(14) Plot ARM Distribution")
    print("(15) Plot D1E Distribution")
    print("(16) Plot D2E Distribution")
    print("(17) Plot ETOT Distribution")
    print("\n")
    print("(99) EXIT Program")
    
    choice = input("Select Option >> ")
    
    if int(choice) == 1:
        t1 = float(input("Enter Start Time : "))
        t2 = float(input("Enter Stop  Time : "))
        
        D1E_Limits    = [D1E_min, D1E_max]
        D2E_Limits    = [D2E_min, D2E_max]
        TOF_Limits    = [TOF_min, TOF_max]
        PSD_Limits    = [PSD_min, PSD_max]
        ARM_Limits    = [ARM_min, ARM_max]
        PHIBAR_Limits = [PHIBAR_min, PHIBAR_max]
        ETOT_Limits   = [ETOT_min, ETOT_max]
    
    if int(choice) == 2:
        D1E_min = float(input("Enter D1E min : "))
        D1E_max = float(input("Enter D1E max : "))
        
        D1E_Limits    = [D1E_min, D1E_max]
        D2E_Limits    = [D2E_min, D2E_max]
        TOF_Limits    = [TOF_min, TOF_max]
        PSD_Limits    = [PSD_min, PSD_max]
        ARM_Limits    = [ARM_min, ARM_max]
        PHIBAR_Limits = [PHIBAR_min, PHIBAR_max]
        ETOT_Limits   = [ETOT_min, ETOT_max]
    
    if int(choice) == 3:
        D2E_min = float(input("Enter D2E min : "))
        D2E_max = float(input("Enter D2E max : "))
        
        D1E_Limits    = [D1E_min, D1E_max]
        D2E_Limits    = [D2E_min, D2E_max]
        TOF_Limits    = [TOF_min, TOF_max]
        PSD_Limits    = [PSD_min, PSD_max]
        ARM_Limits    = [ARM_min, ARM_max]
        PHIBAR_Limits = [PHIBAR_min, PHIBAR_max]
        ETOT_Limits   = [ETOT_min, ETOT_max]
    
    if int(choice) == 4:
        ETOT_min = float(input("Enter ETOT min : "))
        ETOT_max = float(input("Enter ETOT max : "))
        
        D1E_Limits    = [D1E_min, D1E_max]
        D2E_Limits    = [D2E_min, D2E_max]
        TOF_Limits    = [TOF_min, TOF_max]
        PSD_Limits    = [PSD_min, PSD_max]
        ARM_Limits    = [ARM_min, ARM_max]
        PHIBAR_Limits = [PHIBAR_min, PHIBAR_max]
        ETOT_Limits   = [ETOT_min, ETOT_max]
    
    if int(choice) == 5:
        TOF_min = float(input("Enter TOF min : "))
        TOF_max = float(input("Enter TOF max : "))
        
        D1E_Limits    = [D1E_min, D1E_max]
        D2E_Limits    = [D2E_min, D2E_max]
        TOF_Limits    = [TOF_min, TOF_max]
        PSD_Limits    = [PSD_min, PSD_max]
        ARM_Limits    = [ARM_min, ARM_max]
        PHIBAR_Limits = [PHIBAR_min, PHIBAR_max]
        ETOT_Limits   = [ETOT_min, ETOT_max]
    
    if int(choice) == 6:
        PSD_min = float(input("Enter PSD min : "))
        PSD_max = float(input("Enter PSD max : "))
        
        D1E_Limits    = [D1E_min, D1E_max]
        D2E_Limits    = [D2E_min, D2E_max]
        TOF_Limits    = [TOF_min, TOF_max]
        PSD_Limits    = [PSD_min, PSD_max]
        ARM_Limits    = [ARM_min, ARM_max]
        PHIBAR_Limits = [PHIBAR_min, PHIBAR_max]
        ETOT_Limits   = [ETOT_min, ETOT_max]
    
    if int(choice) == 7:
        ARM_min = float(input("Enter ARM min : "))
        ARM_max = float(input("Enter ARM max : "))
        
        D1E_Limits    = [D1E_min, D1E_max]
        D2E_Limits    = [D2E_min, D2E_max]
        TOF_Limits    = [TOF_min, TOF_max]
        PSD_Limits    = [PSD_min, PSD_max]
        ARM_Limits    = [ARM_min, ARM_max]
        PHIBAR_Limits = [PHIBAR_min, PHIBAR_max]
        ETOT_Limits   = [ETOT_min, ETOT_max]
    
    if int(choice) == 8:
        PHIBAR_min = float(input("Enter PHIBAR min : "))
        PHIBAR_max = float(input("Enter PHIBAR max : "))
        
        D1E_Limits    = [D1E_min, D1E_max]
        D2E_Limits    = [D2E_min, D2E_max]
        TOF_Limits    = [TOF_min, TOF_max]
        PSD_Limits    = [PSD_min, PSD_max]
        ARM_Limits    = [ARM_min, ARM_max]
        PHIBAR_Limits = [PHIBAR_min, PHIBAR_max]
        ETOT_Limits   = [ETOT_min, ETOT_max]

    if int(choice) == 9:
        tbin = float(input("Enter time bin size (secs) : "))
        
        D1E_Limits    = [D1E_min, D1E_max]
        D2E_Limits    = [D2E_min, D2E_max]
        TOF_Limits    = [TOF_min, TOF_max]
        PSD_Limits    = [PSD_min, PSD_max]
        ARM_Limits    = [ARM_min, ARM_max]
        PHIBAR_Limits = [PHIBAR_min, PHIBAR_max]
        ETOT_Limits   = [ETOT_min, ETOT_max]


    if int(choice) == 11:
        # =============================================================================
        # Define all of the individual filters, one for each event parameter.
        # =============================================================================
        
        D1E_Filter  = '(evtdf.D1E  > D1E_Limits[0])   & (evtdf.D1E  < D1E_Limits[1])'
        D2E_Filter  = '(evtdf.D2E  > D2E_Limits[0])   & (evtdf.D2E  < D2E_Limits[1])'
        ETOT_Filter = '(evtdf.ETOT > ETOT_Limits[0])  & (evtdf.ETOT < ETOT_Limits[1])'
        PSD_Filter  = '(evtdf.PSD  > PSD_Limits[0])   & (evtdf.PSD  < PSD_Limits[1])'
        TOF_Filter  = '(evtdf.TOF  > TOF_Limits[0])   & (evtdf.TOF  < TOF_Limits[1])'
        ARM_Filter  = '(evtdf.ARM  > ARM_Limits[0])   & (evtdf.ARM  < ARM_Limits[1])'
        PHIBAR_Filter  = '(evtdf.PHIBAR  > PHIBAR_Limits[0])   & (evtdf.PHIBAR  < PHIBAR_Limits[1])'
        
        # =============================================================================
        # Combine all of the individual filters.
        # =============================================================================
        
        EVT_Filter = D1E_Filter  + " & " + \
                     D2E_Filter  + " & " + \
                     ETOT_Filter + " & " + \
                     PSD_Filter  + " & " + \
                     TOF_Filter  + " & " + \
                     ARM_Filter  + " & " + \
                     PHIBAR_Filter 
        
        
        # =============================================================================
        # Plot the flare time history with all filters active.
        # =============================================================================
        
        plt.close()
        hvalues, edges, patches = plt.hist(evtdf.FTJD[eval(EVT_Filter)], range = (ftjd1, ftjd2), bins=int(((ftjd2-ftjd1)*86400)/tbin))
#        plt.hist(evtdf.SECS[eval(EVT_Filter)], range = (t1, t2), bins=int((t2-t1)/tbin))
        plt.xlabel('Fractional TJD')
        plt.ylabel('Counts')
        plt.minorticks_on()
        plt.grid(which='major', axis='both')
        plt.title('TJD ' + str(evtdf.TJD[0]))
        plt.annotate('Total Cts = ' + str(sum(hvalues)), xy = (0.05, 0.90), xycoords='axes fraction')
        plt.show()
#        fig1.canvas.draw()

    if int(choice) == 12:
        # =============================================================================
        # Define all of the individual filters, one for each event parameter.
        # =============================================================================
        
        D1E_Filter  = '(evtdf.D1E  > D1E_Limits[0])   & (evtdf.D1E  < D1E_Limits[1])'
        D2E_Filter  = '(evtdf.D2E  > D2E_Limits[0])   & (evtdf.D2E  < D2E_Limits[1])'
        ETOT_Filter = '(evtdf.ETOT > ETOT_Limits[0])  & (evtdf.ETOT < ETOT_Limits[1])'
        PSD_Filter  = '(evtdf.PSD  > PSD_Limits[0])   & (evtdf.PSD  < PSD_Limits[1])'
        TOF_Filter  = '(evtdf.TOF  > TOF_Limits[0])   & (evtdf.TOF  < TOF_Limits[1])'
        ARM_Filter  = '(evtdf.ARM  > ARM_Limits[0])   & (evtdf.ARM  < ARM_Limits[1])'
        PHIBAR_Filter  = '(evtdf.PHIBAR  > PHIBAR_Limits[0])   & (evtdf.PHIBAR  < PHIBAR_Limits[1])'
        
        # =============================================================================
        # Combine all of the individual filters.
        # =============================================================================
        
        EVT_Filter = D1E_Filter  + " & " + \
                     D2E_Filter  + " & " + \
                     ETOT_Filter + " & " + \
                     PSD_Filter  + " & " + \
                     TOF_Filter  + " & " + \
                     ARM_Filter  + " & " + \
                     PHIBAR_Filter 
        
        
        # =============================================================================
        # Plot the GRB time history with all filters active.
        # hvalues contains the histogram data.
        # edges contaions the bin edge data.
        # =============================================================================
        
        plt.close()
        hvalues, edges, patches  = plt.hist(evtdf.TOF[eval(EVT_Filter)], range = (0, 255), bins=256)
        plt.xlabel('TOF Channel')
        plt.ylabel('Counts')
        plt.minorticks_on()
        plt.grid(which='major', axis='both')
        plt.title('TJD ' + str(evtdf.TJD[0]))
        plt.annotate('Total Cts = ' + str(sum(hvalues)), xy = (0.05, 0.90), xycoords='axes fraction')
        plt.show()

    if int(choice) == 13:
        # =============================================================================
        # Define all of the individual filters, one for each event parameter.
        # =============================================================================
        
        D1E_Filter  = '(evtdf.D1E  > D1E_Limits[0])   & (evtdf.D1E  < D1E_Limits[1])'
        D2E_Filter  = '(evtdf.D2E  > D2E_Limits[0])   & (evtdf.D2E  < D2E_Limits[1])'
        ETOT_Filter = '(evtdf.ETOT > ETOT_Limits[0])  & (evtdf.ETOT < ETOT_Limits[1])'
        PSD_Filter  = '(evtdf.PSD  > PSD_Limits[0])   & (evtdf.PSD  < PSD_Limits[1])'
        TOF_Filter  = '(evtdf.TOF  > TOF_Limits[0])   & (evtdf.TOF  < TOF_Limits[1])'
        ARM_Filter  = '(evtdf.ARM  > ARM_Limits[0])   & (evtdf.ARM  < ARM_Limits[1])'
        PHIBAR_Filter  = '(evtdf.PHIBAR  > PHIBAR_Limits[0])   & (evtdf.PHIBAR  < PHIBAR_Limits[1])'
        
        # =============================================================================
        # Combine all of the individual filters.
        # =============================================================================
        
        EVT_Filter = D1E_Filter  + " & " + \
                     D2E_Filter  + " & " + \
                     ETOT_Filter + " & " + \
                     PSD_Filter  + " & " + \
                     TOF_Filter  + " & " + \
                     ARM_Filter  + " & " + \
                     PHIBAR_Filter 
        
        
        # =============================================================================
        # Plot the GRB time history with all filters active.
        # hvalues contains the histogram data.
        # edges contaions the bin edge data.
        # =============================================================================
        
        plt.close()
        hvalues, edges, patches  = plt.hist(evtdf.PSD[eval(EVT_Filter)], range = (0, 255), bins=256)
        plt.xlabel('PSD Channel')
        plt.ylabel('Counts')
        plt.minorticks_on()
        plt.grid(which='major', axis='both')
        plt.title('TJD ' + str(evtdf.TJD[0]))
        plt.annotate('Total Cts = ' + str(sum(hvalues)), xy = (0.05, 0.90), xycoords='axes fraction')
        plt.show()

    if int(choice) == 14:
        # =============================================================================
        # Define all of the individual filters, one for each event parameter.
        # =============================================================================
        
        D1E_Filter  = '(evtdf.D1E  > D1E_Limits[0])   & (evtdf.D1E  < D1E_Limits[1])'
        D2E_Filter  = '(evtdf.D2E  > D2E_Limits[0])   & (evtdf.D2E  < D2E_Limits[1])'
        ETOT_Filter = '(evtdf.ETOT > ETOT_Limits[0])  & (evtdf.ETOT < ETOT_Limits[1])'
        PSD_Filter  = '(evtdf.PSD  > PSD_Limits[0])   & (evtdf.PSD  < PSD_Limits[1])'
        TOF_Filter  = '(evtdf.TOF  > TOF_Limits[0])   & (evtdf.TOF  < TOF_Limits[1])'
        ARM_Filter  = '(evtdf.ARM  > ARM_Limits[0])   & (evtdf.ARM  < ARM_Limits[1])'
        PHIBAR_Filter  = '(evtdf.PHIBAR  > PHIBAR_Limits[0])   & (evtdf.PHIBAR  < PHIBAR_Limits[1])'
        
        # =============================================================================
        # Combine all of the individual filters.
        # =============================================================================
        
        EVT_Filter = D1E_Filter  + " & " + \
                     D2E_Filter  + " & " + \
                     ETOT_Filter + " & " + \
                     PSD_Filter  + " & " + \
                     TOF_Filter  + " & " + \
                     ARM_Filter  + " & " + \
                     PHIBAR_Filter 
        
        
        # =============================================================================
        # Plot the GRB time history with all filters active.
        # hvalues contains the histogram data.
        # edges contaions the bin edge data.
        # =============================================================================
        
        plt.close()
        hvalues, edges, patches = plt.hist(evtdf.ARM[eval(EVT_Filter)], range = (ARM_min, ARM_max), bins=256)
        plt.xlabel('ARM (degrees)')
        plt.ylabel('Counts')
        plt.minorticks_on()
        plt.grid(which='major', axis='both')
        plt.title('TJD ' + str(evtdf.TJD[0]))
        plt.annotate('Total Cts = ' + str(sum(hvalues)), xy = (0.05, 0.90), xycoords='axes fraction')
        plt.show()

    if int(choice) == 15:
        # =============================================================================
        # Define all of the individual filters, one for each event parameter.
        # =============================================================================
        
        D1E_Filter  = '(evtdf.D1E  > D1E_Limits[0])   & (evtdf.D1E  < D1E_Limits[1])'
        D2E_Filter  = '(evtdf.D2E  > D2E_Limits[0])   & (evtdf.D2E  < D2E_Limits[1])'
        ETOT_Filter = '(evtdf.ETOT > ETOT_Limits[0])  & (evtdf.ETOT < ETOT_Limits[1])'
        PSD_Filter  = '(evtdf.PSD  > PSD_Limits[0])   & (evtdf.PSD  < PSD_Limits[1])'
        TOF_Filter  = '(evtdf.TOF  > TOF_Limits[0])   & (evtdf.TOF  < TOF_Limits[1])'
        ARM_Filter  = '(evtdf.ARM  > ARM_Limits[0])   & (evtdf.ARM  < ARM_Limits[1])'
        PHIBAR_Filter  = '(evtdf.PHIBAR  > PHIBAR_Limits[0])   & (evtdf.PHIBAR  < PHIBAR_Limits[1])'
        
        # =============================================================================
        # Combine all of the individual filters.
        # =============================================================================
        
        EVT_Filter = D1E_Filter  + " & " + \
                     D2E_Filter  + " & " + \
                     ETOT_Filter + " & " + \
                     PSD_Filter  + " & " + \
                     TOF_Filter  + " & " + \
                     ARM_Filter  + " & " + \
                     PHIBAR_Filter 
        
        
        # =============================================================================
        # Plot the GRB time history with all filters active.
        # hvalues contains the histogram data.
        # edges contaions the bin edge data.
        # =============================================================================
        
        plt.close()
        hvalues, edges, patches  = plt.hist(evtdf.D1E[eval(EVT_Filter)], range = (D1E_min, D1E_max), bins=int((D1E_max - D1E_min) / 10))
        plt.xlabel('D1 Energy (keV)')
        plt.ylabel('Counts')
        plt.minorticks_on()
        plt.grid(which='major', axis='both')
        plt.title('TJD ' + str(evtdf.TJD[0]))
        plt.annotate('Total Cts = ' + str(sum(hvalues)), xy = (0.05, 0.90), xycoords='axes fraction')
        plt.show()

    if int(choice) == 16:
        # =============================================================================
        # Define all of the individual filters, one for each event parameter.
        # =============================================================================
        
        D1E_Filter  = '(evtdf.D1E  > D1E_Limits[0])   & (evtdf.D1E  < D1E_Limits[1])'
        D2E_Filter  = '(evtdf.D2E  > D2E_Limits[0])   & (evtdf.D2E  < D2E_Limits[1])'
        ETOT_Filter = '(evtdf.ETOT > ETOT_Limits[0])  & (evtdf.ETOT < ETOT_Limits[1])'
        PSD_Filter  = '(evtdf.PSD  > PSD_Limits[0])   & (evtdf.PSD  < PSD_Limits[1])'
        TOF_Filter  = '(evtdf.TOF  > TOF_Limits[0])   & (evtdf.TOF  < TOF_Limits[1])'
        ARM_Filter  = '(evtdf.ARM  > ARM_Limits[0])   & (evtdf.ARM  < ARM_Limits[1])'
        PHIBAR_Filter  = '(evtdf.PHIBAR  > PHIBAR_Limits[0])   & (evtdf.PHIBAR  < PHIBAR_Limits[1])'
        
        # =============================================================================
        # Combine all of the individual filters.
        # =============================================================================
        
        EVT_Filter = D1E_Filter  + " & " + \
                     D2E_Filter  + " & " + \
                     ETOT_Filter + " & " + \
                     PSD_Filter  + " & " + \
                     TOF_Filter  + " & " + \
                     ARM_Filter  + " & " + \
                     PHIBAR_Filter 
        
        
        # =============================================================================
        # Plot the GRB time history with all filters active.
        # hvalues contains the histogram data.
        # edges contaions the bin edge data.
        # =============================================================================
        
        plt.close()
        hvalues, edges, patches = plt.hist(evtdf.D2E[eval(EVT_Filter)], range = (D2E_min, D2E_max), bins=int((D2E_max - D2E_min) / 10))
        plt.xlabel('D2 Energy (keV)')
        plt.ylabel('Counts')
        plt.minorticks_on()
        plt.grid(which='major', axis='both')
        plt.title('TJD ' + str(evtdf.TJD[0]))
        plt.annotate('Total Cts = ' + str(sum(hvalues)), xy = (0.05, 0.90), xycoords='axes fraction')
        plt.show()

    if int(choice) == 17:
        # =============================================================================
        # Define all of the individual filters, one for each event parameter.
        # =============================================================================
        
        D1E_Filter  = '(evtdf.D1E  > D1E_Limits[0])   & (evtdf.D1E  < D1E_Limits[1])'
        D2E_Filter  = '(evtdf.D2E  > D2E_Limits[0])   & (evtdf.D2E  < D2E_Limits[1])'
        ETOT_Filter = '(evtdf.ETOT > ETOT_Limits[0])  & (evtdf.ETOT < ETOT_Limits[1])'
        PSD_Filter  = '(evtdf.PSD  > PSD_Limits[0])   & (evtdf.PSD  < PSD_Limits[1])'
        TOF_Filter  = '(evtdf.TOF  > TOF_Limits[0])   & (evtdf.TOF  < TOF_Limits[1])'
        ARM_Filter  = '(evtdf.ARM  > ARM_Limits[0])   & (evtdf.ARM  < ARM_Limits[1])'
        PHIBAR_Filter  = '(evtdf.PHIBAR  > PHIBAR_Limits[0])   & (evtdf.PHIBAR  < PHIBAR_Limits[1])'
        
        # =============================================================================
        # Combine all of the individual filters.
        # =============================================================================
        
        EVT_Filter = D1E_Filter  + " & " + \
                     D2E_Filter  + " & " + \
                     ETOT_Filter + " & " + \
                     PSD_Filter  + " & " + \
                     TOF_Filter  + " & " + \
                     ARM_Filter  + " & " + \
                     PHIBAR_Filter 
        
        
        # =============================================================================
        # Plot the GRB time history with all filters active.
        # hvalues contains the histogram data.
        # edges contaions the bin edge data.
        # =============================================================================
        
        plt.close()
        hvalues, edges, patches  = plt.hist(evtdf.ETOT[eval(EVT_Filter)], range = (ETOT_min, ETOT_max), bins=int((ETOT_max - ETOT_min) / 10))
        plt.xlabel('D1 Energy (keV)')
        plt.ylabel('Counts')
        plt.minorticks_on()
        plt.grid(which='major', axis='both')
        plt.title('TJD ' + str(evtdf.TJD[0]))
        plt.annotate('Total Cts = ' + str(sum(hvalues)), xy = (0.05, 0.90), xycoords='axes fraction')
        plt.show()

    if int(choice) == 99:
        break

        




