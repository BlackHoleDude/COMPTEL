# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

#
#==============================================================================
#
#  OAD DATA STRUCTURE
#  ------------------
#  Next, we open up the OAD file.  Here it is hard-coded for testing.
#  Note that the original EVP files were generated in big-endian format.
#  Here we are running on an intel processor, which is little-endian.
#
#   The following table indicates the correspondence between the arrays
#   used in this code (first column) and the arrays which are normally
#   referred to in the EVP documentation (and in the COMPASS code).
#
#     TIMTAG(1)  = TJD
#     TIMTAG(2)  = COMPASS tics
#     OAPKNO     = OA Packet Number
#     POSX       = S/C geocentric X-coordinate
#     POSY       = S/C geocentric Y-coordinate
#     POSZ       = S/C geocentric Z-coordinate
#     GCAZ       = Geocenter Azimuth
#     GCEL       = Geocenter Zenith
#     ZRASC      = Z-axis Right Ascension
#     ZDECL      = Z-axis Declination
#     XRASC      = X-axis Right Ascension
#     XDECL      = X-axis Declination
#     EHORA      = Earth Horizon Angle
#
#==============================================================================


import struct
import os
import sqlite3
import numpy
import pandas
import math
import matplotlib.pyplot as plt
import astropy.coordinates as coords
from astropy import units as u
from astropy.time import Time
from scipy import interpolate
#from aacgmv2 import convert
#from datetime import date
#from eci2geo import eci2geo


os.chdir('/Users/mark/Dropbox/Docs/Python')




#==============================================================================
# 
# EVP Data Structure
# ------------------
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
# 22  Etotal
# 23  SECS
# 24  POSX
# 25  POSY
# 26  POSZ
# 27  GCAZ
# 28  GCEL
# 29  ZRASC
# 30  ZDECL
# 31  XRASC
# 32  XDECL
# 33  EHORA
#
# All angles in radians.
#
#==============================================================================


# =============================================================================
# ifiles is a list
# =============================================================================

nevts = 0

obsper = "vp710.0"

os.chdir('/Users/mark/Dropbox/Docs/Projects/COMPTEL/EVP Database/')

print('\n\n')
print(obsper)

f = open("evp_file_list_" + obsper + ".txt",'rt')
ifiles = f.readlines()
f.close()

for line in ifiles :
    ifile = "/Users/mark/Box/MyStuff/Projects/COMPTEL_Archive/COMPTEL_Data_Archive/MPE/EVP/" + line.split()[0]
    qf = line.split()[1]
    print(ifile, qf)
    
    #==========================================================================
    # Open the EVP datafile, read in the data, and close the EVP datafile.
    # The variable 'data' refers to the binary data froim the EVP datafile. 
    #==========================================================================
    
    if os.path.exists(ifile):
        f = open(ifile, "rb")
        data = f.read()
        f.close()
    else:
        print("The EVP file does not exist : " + line.split()[0])
        continue
    
    #==========================================================================
    # Determine the number of events in this file based on an event size of 
    # 68 bytes.
    #==========================================================================
    
    nevents = int(len(data) / 68)
    
    # =========================================================================
    # We need to keep track of the offset of each event from the start of the 
    # data array. Here we just initialize the offset, setting it to zero for 
    # the first event.    
    # =========================================================================
    
    offset = 0
    
    # =========================================================================
    # Start the loop to process each event in the data array.    
    # =========================================================================
    
    for i in range(0,nevents) :
         
        #======================================================================
        # Unpack event from binary data into tuple t.
        # t is a tuple.
        #======================================================================
    
        t = struct.unpack('>hhhhhhHHHhHHIffffffff', data[offset+0:offset+60])
        evclass = data[offset+61:offset+68].decode('ascii')
    
        #======================================================================
        # rec is a list that starts out with the same content as t.
        # Several values are then converted (e.g., into mm or from radians
        # degrees). Tne values for EVCLASS and ETOT are then appended.
        #======================================================================
        rec = list(t)
        
        rec[0]    = rec[0] / 32.0              # convert D1X to mm
        rec[1]    = rec[1] / 32.0              # convert D1Y to mm
        rec[2]    = rec[2] / 256.0
        rec[3]    = rec[3] / 32.0              # convert D2X to mm
        rec[4]    = rec[4] / 32.0              # convert D2Y to mm
        rec[5]    = rec[5] / 256.0
        rec[6]    = rec[6] / 128.0             # convert PSD to chan no.
        rec[7]    = rec[7] / 128.0             # convert TOF to chan no.
        rec[15]   = math.degrees(rec[15])      # convert PHIBAR to degrees
        rec[16]   = math.degrees(rec[16])      # convert GLAT to degrees
        rec[17]   = math.degrees(rec[17])      # convert GLON to degrees
        rec[18]   = math.degrees(rec[18])      # convert AZI to degrees
        rec[19]   = math.degrees(rec[19])      # convert ZEN to degrees
        rec[20]   = math.degrees(rec[20])      # convert EHORA to degrees
        rec.append(evclass)                    # append EVCLASS
        rec.append(rec[13]+rec[14])            # create and append ETOT
    
        #======================================================================
        # evts is a list (of records)
        #====================================================================== 
        if i == 0 :
            evts = [rec]
        else :
            evts.append(rec)
            
        #======================================================================
        # Increment offset for next event
        #======================================================================             
        offset = offset + 68    
    
    #==========================================================================
    # Create a dataframe (evp) from list of events (evts).
    # params is a list of column names for the dataframe.
    #==========================================================================
    
    params = ['D1X', 'D1Y', 'D1Z', 'D2X', 'D2Y', 'D2Z', 'PSD', 'TOF', 'MODCOM', \
              'REFLAG', 'VETO', 'TJD', 'TICS', 'D1E', 'D2E', 'PHIBAR', \
              'GLAT', 'GLON', 'AZI', 'ZEN', 'EHORA', 'CLASS', 'ETOT']
    
    evp = pandas.DataFrame.from_records(evts, columns = params)
    
    # =========================================================================
    # 
    # =========================================================================
    
    tjdmin = int(evp.TJD.min())
    tjdmax = int(evp.TJD.max())
    if tjdmin == tjdmax :
        tjd = tjdmin
    else :
        print("EVP file extends beyond a single TJD.  Exiting...")
        exit()

    # =========================================================================
    # Now we want to add OAD data to each event...
    # Read in the OAD database and create a list (oaddata) of OAD records for 
    # this TJD. This is a list of tuples.
    # =========================================================================
    
    db = sqlite3.connect('gro_oad.db')
    cursor = db.cursor()
    cursor.execute("SELECT FTJD, POSX, POSY, POSZ, GCAZ, GCEL, \
                      ZRASC, ZDECL, XRASC, XDECL, EHORA, LON, LAT, ALT, MRIG \
                      FROM oad \
                      WHERE TJD = ?", (tjd,))
    oaddata = cursor.fetchall()                         
    db.close()
    
    # =========================================================================
    # Convert the OAD records (a list of tuples into a dataframe (oad).
    # If there is no OAD for this TJD, then we can't properly process these 
    # data.  So print out a message and discontinue the processing of this
    # EVP file.
    # =========================================================================
    
    oad = pandas.DataFrame(oaddata,columns = ['FTJD', 'POSX', 'POSY','POSZ','GCAZ','GCEL',\
                                         'ZRASC','ZDECL','XRASC','XDECL','SCEHORA',\
                                         'LON','LAT','ALT','MRIG'])
    if oad.empty:
        print("No OAD data for this TJD : " + str(tjd))
        continue
    
    
    #==========================================================================
    # Set up interpolation routines for each of the OAD parameters.
    # The interpolation of the longitude is determined from the separate 
    # interpolation of both the sine and cosine of the longitude, from which
    # the interpolated longitude is determined.
    #==========================================================================
    
    f_posx = interpolate.interp1d(oad.FTJD, oad.POSX, 'linear', fill_value='extrapolate')
    f_posy = interpolate.interp1d(oad.FTJD, oad.POSY, 'linear', fill_value='extrapolate')
    f_posz = interpolate.interp1d(oad.FTJD, oad.POSZ, 'linear', fill_value='extrapolate')
    f_gcaz = interpolate.interp1d(oad.FTJD, oad.GCAZ, 'linear', fill_value='extrapolate')
    f_gcel = interpolate.interp1d(oad.FTJD, oad.GCEL, 'linear', fill_value='extrapolate')
    f_zrasc = interpolate.interp1d(oad.FTJD, oad.ZRASC, 'linear', fill_value='extrapolate')
    f_zdecl = interpolate.interp1d(oad.FTJD, oad.ZDECL, 'linear', fill_value='extrapolate')
    f_xrasc = interpolate.interp1d(oad.FTJD, oad.XRASC, 'linear', fill_value='extrapolate')
    f_xdecl = interpolate.interp1d(oad.FTJD, oad.XDECL, 'linear', fill_value='extrapolate')
    f_scehora = interpolate.interp1d(oad.FTJD, oad.SCEHORA, 'linear', fill_value='extrapolate')
    f_coslon = interpolate.interp1d(oad.FTJD, numpy.cos(numpy.radians(oad.LON)), 'linear', fill_value='extrapolate')
    f_sinlon = interpolate.interp1d(oad.FTJD, numpy.sin(numpy.radians(oad.LON)), 'linear', fill_value='extrapolate')
    f_lat = interpolate.interp1d(oad.FTJD, oad.LAT, 'linear', fill_value='extrapolate')
    f_alt = interpolate.interp1d(oad.FTJD, oad.ALT, 'linear', fill_value='extrapolate')
    f_mrig = interpolate.interp1d(oad.FTJD, oad.MRIG, 'linear', fill_value='extrapolate')
    

    
    # ========================================================================
    # Add each interpolated parameter value to the DataFrame (evp).
    # ========================================================================
    
    evp = evp.assign(SECS=evp.TICS/8000)
    evp = evp.assign(FTJD=evp.TJD + (evp.SECS/86400))
    evp = evp.assign(POSX=f_posx(evp.FTJD))
    evp = evp.assign(POSY=f_posy(evp.FTJD))
    evp = evp.assign(POSZ=f_posz(evp.FTJD))
    evp = evp.assign(GCAZ=f_gcaz(evp.FTJD))
    evp = evp.assign(GCEL=f_gcel(evp.FTJD))
    evp = evp.assign(ZRASC=f_zrasc(evp.FTJD))
    evp = evp.assign(ZDECL=f_zdecl(evp.FTJD))
    evp = evp.assign(XRASC=f_xrasc(evp.FTJD))
    evp = evp.assign(XDECL=f_xdecl(evp.FTJD))
    evp = evp.assign(SCEHORA=f_scehora(evp.FTJD))
    evp = evp.assign(LON=numpy.degrees(numpy.arctan2(f_sinlon(evp.FTJD),f_coslon(evp.FTJD))))
    evp = evp.assign(LAT=f_lat(evp.FTJD))
    evp = evp.assign(ALT=f_alt(evp.FTJD))
    evp = evp.assign(MRIG=f_mrig(evp.FTJD))
    
    # =========================================================================
    # Add EVP file quality flag (QFLAG) and CGRO viewing period (VP)to
    # each event.
    # =========================================================================
    
    evp = evp.assign(QFLAG = qf)
    evp = evp.assign(VP = obsper)
    
    
    # =========================================================================
    # Now we determine the instrument mode for each event, using the TIM 
    # database...
    # Read TIM Database into a dataframe (timdata)
    # =========================================================================
    
    dbtim = sqlite3.connect('gro_tim.db')
    timdata = pandas.read_sql_query("select start_ftjd, stop_ftjd, mode from tim;", dbtim)
    dbtim.close()

    
    # =========================================================================
    # mode is a DataFrame
    # modes is a list
    # timdef is a dictionary
    # =========================================================================

    modes = []    
    for i in range(0,len(evp)) :
        mode = timdata.loc[(timdata['START_FTJD'] < evp.FTJD[i]) & (timdata['STOP_FTJD'] >= evp.FTJD[i]), ['MODE']]
        if mode.empty:
            print("There is no MODE data for FTJD = " + str(evp.FTJD[i]))
            exit()
        else:
            modes.append(mode.iloc[0]['MODE'])

 
    
# =============================================================================
#     
# =============================================================================
    
    timdef = { \
              'N/A'      : 0, \
              'NORMAL'   : 1, \
              'SAA'      : 2, \
              'AFTERSAA' : 3, \
              'PROTON'   : 4, \
              'SOLRNEUT' : 5, \
              'TRANSIT'  : 6, \
              'UNDEFINE' : 7, \
              'ALBEDONT' : 8, \
              'D1SINGLE' : 9, \
              'TEST'     : 10, \
              'SOLUNDEF' : 11, \
              'HVTEST'   : 12, \
              'D2SINGLE' : 13, \
              'ROBS'     : 14, \
              'LOW'      : 15, \
              'MODIFIED' : 16, \
              'STRAWMAN' : 17, \
              'SOLAR'    : 18, \
              'SOLAR80'  : 19, \
              }
    
    if len(modes) != len(evp) :
        continue
    evp = evp.assign(MODE = [timdef[x] for x in modes])
    
    
# =============================================================================
# 
# =============================================================================

    db = sqlite3.connect('gro_evp_' + obsper + '.db')
    cursor = db.cursor()
    cursor.execute('''\
                   CREATE TABLE IF NOT EXISTS evp(id INTEGER PRIMARY KEY, \
                   FTJD REAL, \
                   PSD REAL, TOF REAL, MODCOM INTEGER, REFLAG INTEGER, \
                   D1E REAL, D2E REAL, PHIBAR REAL, \
                   GCAZ REAL, GCEL REAL, AZI REAL, ZEN REAL, EHORA REAL, \
                   LON REAL, LAT REAL, ALT REAL, MRIG REAL, QFLAG INTEGER, MODE INTEGER)''')
    db.commit()
    for i in evp.index:
        cursor.execute('''\
                       INSERT INTO evp(FTJD, \
                       PSD, TOF, MODCOM, REFLAG, D1E, D2E, PHIBAR, \
                       GCAZ, GCEL, AZI, ZEN, EHORA, LON, LAT, ALT, MRIG, QFLAG, MODE) \
                       VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?, ?, ?)''', \
                       (evp.FTJD[i], \
                        evp.PSD[i], evp.TOF[i], int(evp.MODCOM[i]), int(evp.REFLAG[i]),\
                        evp.D1E[i], evp.D2E[i], evp.PHIBAR[i], \
                        evp.GCAZ[i], evp.GCEL[i], evp.AZI[i], evp.ZEN[i], evp.EHORA[i],\
                        evp.LON[i], evp.LAT[i], evp.ALT[i], evp.MRIG[i], int(evp.QFLAG[i]), int(evp.MODE[i])))
    db.commit()
    db.close()
    
    #
