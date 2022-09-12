import datetime
import earthpy as et
import earthpy.plot as ep
import importlib
import numpy as np
import os
import pandas as pd
import rasterio as rio
import scipy
import scipy.signal
import seaborn as sns 
import struct
import sys
import utm
import math
import unittest
from PIL import Image
from matplotlib import pyplot as plt 
from os import listdir
from os.path import isfile, join
from scipy.spatial.transform import Rotation as R
from scipy import interpolate

# add parent directory to the path for importing modules 
sys.path.insert(1, os.path.join(sys.path[0], '..'))

sys.path.append(os.path.join(sys.path[0], '../data'))

# objects for parsing raw DVL data 
import PathfinderDVL
import PathfinderEnsemble
import PathfinderTimeSeries

# objects for estimating ocean current velocities
import VelocityShearPropagation
import WaterColumn

# objects for controlling thruster to minimize transport cost 
import AdaptiveVelocityController

# objects for parsing flight and science computer log files
import SlocumFlightController
import SlocumScienceController
import dvl_plotter
import BathymetryMap
import MultiFactorTAN
import dvl_plotter_unit770_PR

# data for parsing seafloor bathymetry
# import bathy_meta_data
sns.set()

import warnings
warnings.simplefilter('ignore')

def reload_modules():
    importlib.reload(PathfinderDVL)
    importlib.reload(PathfinderEnsemble)
    importlib.reload(PathfinderTimeSeries)
    importlib.reload(VelocityShearPropagation)
    importlib.reload(WaterColumn)
    importlib.reload(AdaptiveVelocityController)
    importlib.reload(SlocumFlightController)
    importlib.reload(SlocumScienceController)
    importlib.reload(dvl_plotter)
    importlib.reload(dvl_plotter_unit770_PR)
#     importlib.reload(bathy_meta_data)
    importlib.reload(BathymetryMap)
    importlib.reload(MultiFactorTAN)

filepath = '/home/kraft/workspace/data/glider/BuzzBay/28JUN/'
directory = filepath+'dbd_parsed/'
ts_flight = SlocumFlightController.SlocumFlightController.from_directory(directory, save=False, verbose=False)

def get_utm_coords_from_glider_lat_lon(m_lat, m_lon):
    SECS_IN_MIN = 60
    MIN_OFFSET = 100
    lat_min  = math.fmod(m_lat, MIN_OFFSET) 
    lon_min  = math.fmod(m_lon, MIN_OFFSET) 
    lat_dec  = (m_lat - lat_min)/MIN_OFFSET + lat_min/SECS_IN_MIN
    lon_dec  = (m_lon - lon_min)/MIN_OFFSET + lon_min/SECS_IN_MIN
    try:
        utm_pos  = utm.from_latlon(lat_dec, lon_dec)
        easting  = round(utm_pos[0],2)
        northing = round(utm_pos[1],2)
        zone     = utm_pos[2]
        zone_letter  = utm_pos[3]
    except:
        easting = np.nan
        northing = np.nan
        zone = None
        zone_letter = None
    return(easting, northing, zone, zone_letter)

ctd_df = pd.read_csv(filepath+ 'rosbag_output/' + 'ctd-data.csv')
# "Zero" out CTD by determining the measured depth at which the glider spent the most time and under 6m
n, bins, patches = plt.hist(x=ctd_df.depth, bins= 'auto' )
bins
nbins = 0
for b in bins:
    if b > 6.0:
        break
    nbins += 1
#print(nbins)
target = n[0]
target_idx = 0
i = 0
for b in range(1,nbins):
    i = i+1
    if n[b] > target:
        target = n[b]
        target_idx = i
#print(target, target_idx, bins[target_idx])
est_surface_depth = bins[target_idx]
print('Zeroing out CTD Data. Est Surface Depth: ' + str(est_surface_depth))
ctd_df.depth = ctd_df.depth - est_surface_depth
# plt.figure()
# plt.plot(ctd_df.depth)
# plt.show()

# Use CTD to determine start and end points of dives
dive_dic = {}
dive_dic['dive1'] = [0]
count = 0
for i in range(1,len(ctd_df.Time)):
    dt = ctd_df.Time[i] - ctd_df.Time[i-1]
    if dt > 5.0:
        count += 1
        dive_dic['dive'+str(count)].append(i-1)
        dive_dic['dive'+str(count+1)] = [i]
dive_dic['dive'+str(count+1)].append(len(ctd_df.Time)-1)

# Create DateTime Column to make it easier to read
#ctd_df['Time'] = pd.to_datetime(ctd_df['header.stamp.secs'], unit='s')
ctd_df['Time_easy'] = (ctd_df['Time'] - ctd_df.Time[0])/60
# for key in dive_dic:
#     plt.figure()
#     plt.plot(ctd_df.Time_easy[dive_dic[key][0]:dive_dic[key][1]], ctd_df.depth[dive_dic[key][0]:dive_dic[key][1]]*-1)
#     plt.title(key)

selected_dive = 'dive3'

# ctd_start = ctd_df.Time[dive_dic[selected_dive][0]]           #################### TODO: for this dive set only
# ctd_end = ctd_df.Time[dive_dic[selected_dive][1]]
ctd_start = ctd_df.Time[dive_dic['dive3'][0]]
ctd_end = ctd_df.Time[dive_dic['dive5'][1]]
print(datetime.datetime.fromtimestamp(ctd_start), datetime.datetime.fromtimestamp(ctd_end))

# Trim Dive due to ABORT to make for cleaner data processing (only March 17 dive 6)
# ctd_start = ctd_df.Time[dive_dic[selected_dive][0]]
# ctd_end = ctd_df.Time[7200]

# time_zone_shift = 3600*5
time_zone_shift = 3600*(5-0.00925)

start_t = datetime.datetime.fromtimestamp(ctd_start+time_zone_shift)
end_t   = datetime.datetime.fromtimestamp(ctd_end+time_zone_shift)
dur     = end_t - start_t 
print("Duration:", dur)
print(start_t, end_t)

df_dbd  = ts_flight.df[str(start_t):str(end_t)].copy()

#JUST FOR MAR17 dive 6
# df_dbd  = pd.read_csv(filepath + 'df_dbd.csv')
# df_dbd.to_csv(filepath+'mar17_bad_dive_flight_data.csv', sep=',', index=False)

# plt.figure()
# plt.plot(df_dbd.m_depth*-1)
# plt.title('Depth from .dbd')
# plt.show()

# Interpolate AHRS and CTD data onto DVL timestamp. Simplest solution for post-processing.
# TODO consider how to trasnfer this to live application

time_dbd = df_dbd.time- time_zone_shift
# RAD_TO_DEG = 180/scipy.pi
RAD_TO_DEG = 180/scipy.pi
DEG_TO_RAD = scipy.pi/180.

try:
    ahrs_df = pd.read_csv(filepath + 'rosbag_output/' + 'devices-spartonm2-ahrs.csv')
    # ahrs_df.head()
    f_roll = scipy.interpolate.interp1d(ahrs_df.Time, ahrs_df.roll, 'nearest')
    f_pitch = scipy.interpolate.interp1d(ahrs_df.Time, ahrs_df.pitch, 'nearest')
    f_heading = scipy.interpolate.interp1d(ahrs_df.Time, ahrs_df['compass.heading'], 'nearest')
    print('Sparton AHRS Data Available and collected at 10 Hz')
except:
    # Mar 11 and Mar 14 - use TCM3
    f_roll = scipy.interpolate.interp1d(time_dbd, df_dbd.m_roll * RAD_TO_DEG, 'linear')
    f_pitch = scipy.interpolate.interp1d(time_dbd, df_dbd.m_pitch * RAD_TO_DEG, 'linear')
    f_heading = scipy.interpolate.interp1d(time_dbd, df_dbd.m_heading * RAD_TO_DEG, 'linear')
    print('TCM-3 IMU available and collected at 0.25 Hz')

# CTD
# f_depth = scipy.interpolate.interp1d(ctd_df.Time, ctd_df.depth, 'linear')
# f_temp = scipy.interpolate.interp1d(ctd_df.Time, ctd_df.temperature, 'linear')
# f_cond = scipy.interpolate.interp1d(ctd_df.Time, ctd_df.conductivity, 'linear')
f_depth = scipy.interpolate.interp1d(ctd_df.Time, ctd_df.depth, 'nearest')
f_temp = scipy.interpolate.interp1d(ctd_df.Time, ctd_df.temperature, 'nearest')
f_cond = scipy.interpolate.interp1d(ctd_df.Time, ctd_df.conductivity, 'nearest')

# Upload DVL rosbags
dvl_df        = pd.read_csv(filepath+ 'rosbag_output/' + 'devices-dvl-dvl.csv')
dvl_pd0_df    = pd.read_csv(filepath+ 'rosbag_output/' + 'devices-dvl-pd0.csv')
dvl_ranges_df = pd.read_csv(filepath+ 'rosbag_output/' + 'devices-dvl-ranges.csv')
dvl_raw_df    = pd.read_csv(filepath+ 'rosbag_output/' + 'devices-dvl-instrument-raw.csv')

dvl_raw_df_new = dvl_raw_df.set_index('Time')

start_dvl = dvl_raw_df_new.index.get_loc(ctd_start, method='nearest')
end_dvl = dvl_raw_df_new.index.get_loc(ctd_end, method='nearest')
print(start_dvl, end_dvl)

# Specific to when using DBD data as IMU
print('CTD Start: ', ctd_start, ' CTD end: ', ctd_end)
print('DBD Start: ', time_dbd[0], ' DBD END: ', time_dbd[-1])
print('Diff start: ',time_dbd[0] - ctd_start  , ' Diff end: ', time_dbd[-1] - ctd_end)

start_adjustment = int(np.ceil(time_dbd[0] - ctd_start))
end_adjustment = int(np.ceil(np.abs(time_dbd[-1] - ctd_end)))

# Initialize Timseries object
ts = PathfinderTimeSeries.PathfinderTimeSeries()
prev_ensemble = None
error_count = 0
print("initializing time series object...")
for i in range(start_dvl + start_adjustment, end_dvl-end_adjustment):
    ros_timestamp = dvl_raw_df.Time[i]
    roll = f_roll(ros_timestamp)
    pitch = f_pitch(ros_timestamp)
    heading = f_heading(ros_timestamp)
    depth = f_depth(ros_timestamp)
    temp = f_temp(ros_timestamp)
    cond = f_cond(ros_timestamp)
    ensemble_raw = dvl_raw_df.data[i][2:-1]
    ensemble_bytes = ensemble_raw.encode().decode('unicode_escape').encode("raw_unicode_escape")
    ensemble = PathfinderEnsemble.PathfinderEnsemble(ensemble_bytes, prev_ensemble, gps_fix=None, ros_time=ros_timestamp, \
                                                    ext_roll=roll, ext_pitch=pitch, ext_heading=heading, \
                                                    ext_depth=depth, ext_temp=temp, ext_cond=cond)
    ts.add_ensemble(ensemble)
    prev_ensemble = ensemble

                
ts.to_dataframe()
# ts.df.to_csv('DVL_DIVEXX.csv')
print("done with error count: {}".format(error_count))

#################### Compute Water Column Currents ########################################

btm_count = 0
# tuning parameters for working with DVL data 
pitch_bias           = 0
#pitch_bias           =  8    # [deg]   mounting pitch bias for the sonar
start_filter         =  0    # [bin #] avoid using the first number of bins
end_filter           =  1    # [bin #] avoid using the last number of bins 
voc_mag_filter       =  10.0  # [m/s]   filter out ocean current 
voc_delta_mag_filter =  0.5  # [m/s]   filter out deltas between layers
near_surface_filter  = 5   # [m]     ignore Vtw when near surface 
direction = 'descending'



# constants
DEG_TO_RAD = np.pi/180

# determine DVL parameters 
bin_len      = ts.df.depth_bin_length[0]
bin0_dist    = ts.df.bin0_distance[0]
bin_len      = bin_len
bin0_dist    = bin0_dist
max_range    = 40
# max_depth    = int(np.max(ts.df.ctd_depth)+max_range)
max_depth    = 14
x_beam       = 0
y_beam       = 1
sample_number = 20

# intialize water column
water_column = WaterColumn.WaterColumn(
    bin_len=bin_len, 
    bin0_dist=bin0_dist,
    max_depth=max_depth,
    start_filter=start_filter,
    end_filter=end_filter,
    voc_mag_filter=voc_mag_filter,
    voc_delta_mag_filter=voc_delta_mag_filter,
    voc_time_filter=10*60,
    sample_number=sample_number
)

#################### DVL Odometery ########################################
print("Starting DVL odometery...")


#TODO add in ability to synthetically remove bottom lock
# Since DVL data is at 1 Hz, bottom lock cutoff is essentially num of seconds since start of mission 
#    where use of bottom-lock in DVL-ODO (even if available) is not used
bottom_lock_cutoff = 0

#counters to see how algorithm works without bottom lock
count_vtw_est_w_current = 0
count_vtw_est = 0
count_voc_only = 0

# How long (in mins) will algorithm accept ocean current estimates i.e. forgetting factor
ocean_current_time_filter = 12.5 # mins
MIN_NUM_NODES = 12

# extract start_t position "origin" from the glider flight data 
for t in range(len(df_dbd)):
    if not np.isnan(df_dbd.m_x_lmc[t]):
        dbd_origin_x_lmc = df_dbd.m_x_lmc[t]
        dbd_origin_y_lmc = df_dbd.m_y_lmc[t]
        dbd_origin_m_lat = df_dbd.m_lat[t]
        dbd_origin_m_lon = df_dbd.m_lon[t]
        break

dbd_utm_x_origin, dbd_utm_y_origin, _, zone_letter = get_utm_coords_from_glider_lat_lon(
    dbd_origin_m_lat, 
    dbd_origin_m_lon
)

# initialize list for new odometry
rel_pos_x = [0]
rel_pos_y = [0]
rel_pos_x_no_BL = [0]
rel_pos_y_no_BL = [0]
rel_pos_x_vtw = [0]
rel_pos_y_vtw = [0]
rel_pos_z = [0]
delta_x_list = [0]
delta_y_list = [0]
vel_list_x = []
vel_list_y = []
u_list     = []
v_list     = []
# set flag for setting GPS updates
flag_gps_fix_at_surface = False
# set counter for synthetically removing bottom lock
bottom_lock_switch = 0

# iterate through the dive file to update odometry
for t in range(1,len(ts.df)):
    bottom_lock_switch = bottom_lock_switch + 1
    time         = ts.df.ros_timestamp[t]
    prev_x       = rel_pos_x[-1]
    prev_y       = rel_pos_y[-1]
    prev_x_no_BL = rel_pos_x_no_BL[-1]
    prev_y_no_BL = rel_pos_y_no_BL[-1]
    prev_x_vtw   = rel_pos_x_vtw[-1]
    prev_y_vtw   = rel_pos_y_vtw[-1]
    delta_t      = ts.df.delta_t[t]
    depth        = ts.df.ctd_depth[t]
    
    vog_u=0; vog_v=0; vtw_u=0; vtw_v=0
    # only use Vtw from pressure sensor when submerged 
    # depth = ts.df.ctd_depth[t]
    # if depth > near_surface_filter:
    #     vtw_u = ts.df.rel_vel_pressure_u[t]
    #     vtw_v = ts.df.rel_vel_pressure_v[t]
    #     flag_gps_fix_at_surface = False
    # # otherwise use the DVL to estimate the Vtw at the surface
    # else:
    #     vtw_u = ts.df.rel_vel_dvl_u[t]
    #     vtw_v = ts.df.rel_vel_dvl_v[t]

    if depth < 0.1:
        # use gps if no dvl bottom lock
        vog_u = ts.df.abs_vel_btm_u[t]
        vog_v = ts.df.abs_vel_btm_v[t]
        # vtw_u = 0
        # vtw_v = 0
        vtw_u = ts.df.rel_vel_dvl_u[t]
        vtw_v = ts.df.rel_vel_dvl_v[t]
    else:
        # retrieve over ground velocity from DVL in bottom track 
        vog_u = ts.df.abs_vel_btm_u[t]
        vog_v = ts.df.abs_vel_btm_v[t]

        vtw_u = ts.df.rel_vel_dvl_u[t]
        vtw_v = ts.df.rel_vel_dvl_v[t]

    #################################################################

    if not np.isnan(ts.df.abs_vel_btm_u[t]):
        vog_u = ts.df.abs_vel_btm_u[t]
        vog_v = ts.df.abs_vel_btm_v[t]
        voc_u = vog_u - vtw_u
        voc_v = vog_v - vtw_v
        voc_ref = WaterColumn.OceanCurrent(voc_u, voc_v, 0)
        # print(vog_u, vog_v)
        btm_count = btm_count +1
    #### else if: (at surface and gps is available )
    # calculate voc_ref using surface drift
    else:
        voc_ref = WaterColumn.OceanCurrent()

    # if depth < 10:
        # voc_ref = WaterColumn.OceanCurrent()
        
    # add shear nodes for each DVL depth bin that meet the filter criteria
    num_good_vel_bins = ts.df.num_good_vel_bins[t]
    if num_good_vel_bins > start_filter+end_filter:        
        
        # determine if glider ascending or descending
        delta_z = ts.df.delta_z[t] 
        if delta_z > 0:
            direction = 'descending'
        else:
            direction = 'ascending'

        # build list of velocity shears to add as ShearNode to water column
        delta_voc_u = []
        delta_voc_v = []

        # add all valid DVL bins to the shear list 
        #   + filtering of DVL bins will occur in the `add_new_shear` call
        for bin_num in range(int(num_good_vel_bins)):

            # retrieve the shear list from the DVL data 
            x_var = ts.get_profile_var_name('velocity', bin_num, x_beam)
            y_var = ts.get_profile_var_name('velocity', bin_num, y_beam)
            # print(x_var,y_var)
            dvl_x = -ts.df[x_var][t]
            dvl_y = -ts.df[y_var][t]

            # compute delta between dead-reckoned through-water velocity & DVL
            delta_voc_u.append(dvl_x - vtw_u)
            delta_voc_v.append(dvl_y - vtw_v)
            # print("shear norm: {}".format(math.sqrt(delta_voc_u[-1]**2+delta_voc_v[-1]**2)))

        shear_list = [WaterColumn.OceanCurrent(
                        delta_voc_u[i], 
                        delta_voc_v[i], 
                        0) 
                      for i in range(len(delta_voc_u))]

        # add shear node to the water column with shear list information 
        if len(shear_list):
            error_pose = water_column.add_new_shear(
                z_true=depth,
                t=time,
                shear_list=shear_list,
                voc_ref=voc_ref,
                direction=direction,
                pitch=pitch*DEG_TO_RAD,
                roll=roll*DEG_TO_RAD,
            )

    # add voc_ref measurement to the water column even if shear list is empty  
    elif not voc_ref.is_none():
        error_pose = water_column.add_new_shear(
            z_true=depth,
            t=time,
            shear_list=[],
            voc_ref=voc_ref,
            direction=direction,
            pitch=pitch*DEG_TO_RAD,
            roll=roll*DEG_TO_RAD,
        )

    # retrieve ocean current estimate from water column 
    voc_u = np.nan; voc_v = np.nan
    voc, count = water_column.get_voc_at_depth(depth, t)
    if not voc.is_none():
        voc_u = voc.u
        voc_v = voc.v
    u_list.append(voc_u)
    v_list.append(voc_v)

    #################################################################
    # initialize delta values to zero
    delta_x, delta_y = 0,0
    # CASE 1: use bottom track overground velocity if available
    if (not np.isnan(vog_u)):
        delta_x = vog_u*delta_t
        delta_y = vog_v*delta_t
        vel_list_x.append(vog_u)
        vel_list_y.append(vog_v)
    
    # CASE 2: use through water velocity and ocean current estimate if available
    elif (not np.isnan(vtw_u)) and (not np.isnan(voc_u)):
            delta_x = (vtw_u + voc_u)*delta_t
            delta_y = (vtw_v + voc_v)*delta_t
            vel_list_x.append(vtw_u + voc_u)
            vel_list_y.append(vtw_v + voc_v)
    # CASE 3: use through water velocity if available
    elif (not np.isnan(vtw_u)):
            delta_x = vtw_u*delta_t
            delta_y = vtw_v*delta_t
            vel_list_x.append(vtw_u)
            vel_list_y.append(vtw_v)
    # CASE 4: use ocean current estimate if available
    elif (not np.isnan(voc_u)):
            delta_x = voc_u*delta_t
            delta_y = voc_v*delta_t
            vel_list_x.append(voc_u)
            vel_list_y.append(voc_v)

    # set current position to DVL odometry result 
    cur_x = delta_x + prev_x
    cur_y = delta_y + prev_y
    
    ##################################################################
    # Synthetically remove Bottom-Lock

    # initialize delta values to zero
    delta_x_no_BL, delta_y_no_BL = 0,0
    if bottom_lock_switch < bottom_lock_cutoff:
        # CASE 1: use bottom track overground velocity if available
        if (not np.isnan(vog_u)):
            delta_x_no_BL = vog_u*delta_t
            delta_y_no_BL = vog_v*delta_t
        # CASE 2: use through water velocity and ocean current estimate if available
        elif (not np.isnan(vtw_u)) and (not np.isnan(voc_u)):
                delta_x_no_BL = (vtw_u + voc_u)*delta_t
                delta_y_no_BL = (vtw_v + voc_v)*delta_t
        # CASE 3: use through water velocity if available
        elif (not np.isnan(vtw_u)):
                delta_x_no_BL = vtw_u*delta_t
                delta_y_no_BL = vtw_v*delta_t
        # CASE 4: use ocean current estimate if available
        elif (not np.isnan(voc_u)):
                delta_x_no_BL = voc_u*delta_t
                delta_y_no_BL = voc_v*delta_t
    else:
        # CASE 2: use through water velocity and ocean current estimate if available
        if (not np.isnan(vtw_u)) and (not np.isnan(voc_u)):
                delta_x_no_BL = (vtw_u + voc_u)*delta_t
                delta_y_no_BL = (vtw_v + voc_v)*delta_t 
                count_vtw_est_w_current += 1
        # CASE 3: use through water velocity if available
        elif (not np.isnan(vtw_u)):
                delta_x_no_BL = vtw_u*delta_t
                delta_y_no_BL = vtw_v*delta_t
                count_vtw_est += 1
        # CASE 4: use ocean current estimate if available
        elif (not np.isnan(voc_u)):
                delta_x_no_BL = voc_u*delta_t
                delta_y_no_BL = voc_v*delta_t
                count_voc_only += 1

    # set current position to DVL odometry result 
    cur_x_no_BL = delta_x_no_BL + prev_x_no_BL
    cur_y_no_BL = delta_y_no_BL + prev_y_no_BL
    # overide currernt position with m_x_lmc/m_y_lmc when glider has been at surface for 10 seconds and then dives back down past 0.5m (last position given by m_x_lmc)
    # override current position if GPS fix is given 
    # if depth < near_surface_filter:
    #     cur_time = datetime.datetime.fromtimestamp(time+time_zone_shift)
    #     cur_dbd  = df_dbd[str(cur_time):].copy()
    #     if (len(cur_dbd.m_gps_x_lmc) != 0):
    #         if not np.isnan(cur_dbd.m_gps_x_lmc[0]):
    #             cur_x = cur_dbd.m_gps_x_lmc[0] - dbd_origin_x_lmc
    #             cur_y = cur_dbd.m_gps_y_lmc[0] - dbd_origin_y_lmc
    #             cur_x_no_BL = cur_x
    #             cur_y_no_BL  = cur_y
    #             flag_gps_fix_at_surface = True
    #             vel_list_x.append(cur_dbd.m_vx_lmc[0])
    #             vel_list_y.append(cur_dbd.m_vy_lmc[0])

    delta_x_vtw = vtw_u*delta_t
    delta_y_vtw = vtw_v*delta_t
    cur_x_vtw = delta_x_vtw + prev_x_vtw
    cur_y_vtw = delta_y_vtw + prev_y_vtw
    
    if depth < near_surface_filter:
        cur_time = datetime.datetime.fromtimestamp(time+time_zone_shift)
        cur_dbd  = df_dbd[str(cur_time):].copy()
        if (len(cur_dbd.m_gps_lat) != 0):
            if not np.isnan(cur_dbd.m_gps_lat[0]):
                cur_x, cur_y,_,_ = get_utm_coords_from_glider_lat_lon(cur_dbd.m_gps_lat[0],cur_dbd.m_gps_lon[0])
                cur_x = cur_x - dbd_utm_x_origin
                cur_y = cur_y - dbd_utm_y_origin
                cur_x_no_BL = cur_x
                cur_y_no_BL = cur_y
                cur_x_vtw = cur_x
                cur_y_vtw = cur_y
                flag_gps_fix_at_surface = True
                vel_list_x.append(cur_dbd.m_vx_lmc[0])
                vel_list_y.append(cur_dbd.m_vy_lmc[0])
    
    # update the odometry list of positions
    rel_pos_x.append(cur_x)
    rel_pos_y.append(cur_y)
    rel_pos_x_no_BL.append(cur_x_no_BL)
    rel_pos_y_no_BL.append(cur_y_no_BL)
    rel_pos_x_vtw.append(cur_x_vtw)
    rel_pos_y_vtw.append(cur_y_vtw)
    rel_pos_z.append(depth)
    delta_x_list.append(delta_x)
    delta_y_list.append(delta_y)
    
# add new odomety to the data frame
ts.df['rel_pos_x_no_BL'] = rel_pos_x_no_BL
ts.df['rel_pos_y_no_BL'] = rel_pos_y_no_BL
ts.df['rel_pos_x_vtw'] = rel_pos_x_vtw
ts.df['rel_pos_y_vtw'] = rel_pos_y_vtw
ts.df['rel_pos_x'] = rel_pos_x
ts.df['rel_pos_y'] = rel_pos_y
ts.df['rel_pos_z'] = rel_pos_z
ts.df['delta_x']   = delta_x_list
ts.df['delta_y']   = delta_y_list

print("> Finished Calculating Odometry!")

print("water column u:")
print(water_column.wc[:15,-20:,0])
print("water column v:")
print(water_column.wc[:15,-20:,1])
print("water column t:")
print(water_column.wc[:15,-20:,3])

# #################### Plot Results ########################################

# extract start_t position "origin" from the glider flight data 

for t in range(len(df_dbd)):
    if not np.isnan(df_dbd.m_x_lmc[t]):
        dbd_origin_x_lmc = df_dbd.m_x_lmc[t]
        dbd_origin_y_lmc = df_dbd.m_y_lmc[t]
        dbd_origin_m_lat = df_dbd.m_lat[t]
        dbd_origin_m_lon = df_dbd.m_lon[t]
        break

# dbd_origin_m_lat_DD, dbd_origin_m_lon_DD = decimal_minutes_to_decimal_degrees(dbd_origin_m_lat, dbd_origin_m_lon)

# Convert starting lat lon to UTM
dbd_origin_utm_x, dbd_origin_utm_y, zone_number, zone_letter = get_utm_coords_from_glider_lat_lon(
    dbd_origin_m_lat, 
    dbd_origin_m_lon
)

utm_dr_x = []
utm_dr_y = []
utm_gps_x = []
utm_gps_y = []
for i in range(0,len(df_dbd.m_lat)):
    utm_x, utm_y,_,_ = get_utm_coords_from_glider_lat_lon(df_dbd.m_lat[i], df_dbd.m_lon[i])
    utm_dr_x.append(utm_x)
    utm_dr_y.append(utm_y)

for i in range(0,len(df_dbd.m_gps_lat)):
    utm_x_gps, utm_y_gps,_,_ = get_utm_coords_from_glider_lat_lon(df_dbd.m_gps_lat[i], df_dbd.m_gps_lon[i])
    utm_gps_x.append(utm_x_gps)
    utm_gps_y.append(utm_y_gps)


# Generate DVL-ODO estimate of position in UTM coordinates
rel_pos_x_utm = (ts.df['rel_pos_x'] + dbd_origin_utm_x)
rel_pos_y_utm = (ts.df['rel_pos_y'] + dbd_origin_utm_y)

# UTM
df_dbd['utm_dr_x']      = utm_dr_x
df_dbd['utm_dr_y']      = utm_dr_y
df_dbd['utm_gps_x']     = utm_gps_x
df_dbd['utm_gps_y']     = utm_gps_y
ts.df['rel_pos_x_utm']  = rel_pos_x_utm
ts.df['rel_pos_y_utm']  = rel_pos_y_utm

# Convert DVL-ODO estimate of position to lat/lon
rel_pos_y_lat, rel_pos_x_lon = utm.to_latlon(ts.df.rel_pos_x_utm, ts.df.rel_pos_y_utm, zone_number, zone_letter)
m_lat_DD, m_lon_DD = utm.to_latlon(df_dbd.utm_dr_x, df_dbd.utm_dr_y, zone_number, zone_letter)
m_lat_GPS, m_lon_GPS = utm.to_latlon(df_dbd.utm_gps_x, df_dbd.utm_gps_y, zone_number, zone_letter)

# Lat/Lon in Decimal Degrees which matches how utm library converts utm to lat/lon
df_dbd['m_lat_DD']       = m_lat_DD
df_dbd['m_lon_DD']       = m_lon_DD
df_dbd['m_lat_gps_DD']   = m_lat_GPS
df_dbd['m_lon_gps_DD']   = m_lon_GPS
ts.df['rel_pos_lon']     = rel_pos_x_lon
ts.df['rel_pos_lat']     = rel_pos_y_lat

print("Saving CSV")
ts.save_as_csv(name="dataframe")
ts_flight.save_as_csv(name="fs_dataframe")

# fig, ax = plt.subplots(figsize=(12,12))
# sns.set(font_scale = 1.5)
# sns.scatterplot(df_dbd.utm_dr_x,df_dbd.utm_dr_y)

################################### plots #############################################################

fig, ax = plt.subplots(figsize=(12,12))
sns.set(font_scale = 1.5)
linewidth = 8
plt_bg = False

sns.scatterplot(
    ts.df.rel_pos_x, # - dbd_origin_x_lmc, 
    ts.df.rel_pos_y,# - dbd_origin_y_lmc, 
    linewidth=0, 
    color='orange', 
    label='DVL-Odo',
    s=linewidth, 
    zorder=2,
)
odos=1

sns.scatterplot(
    ts.df.rel_pos_x_no_BL,
    ts.df.rel_pos_y_no_BL,
    color='blue', 
    label='NO-BL',
    linewidth=0,
    s=linewidth, 
    zorder=4,
)
odos=4

sns.scatterplot(
    ts.df.rel_pos_x_vtw,
    ts.df.rel_pos_y_vtw,
    color = 'green', 
    label='VTW',
    linewidth=0,
    s=linewidth, 
    zorder=5,
)
odos=5

sns.scatterplot(
    x=df_dbd.utm_dr_x - dbd_origin_utm_x,
    y=df_dbd.utm_dr_y - dbd_origin_utm_y,
    color='purple',
    label='DR-DACC',
    linewidth=0,
    s=linewidth,
    data=df_dbd,
    zorder=1,
)
odos=2

sns.scatterplot(
    x=df_dbd.utm_gps_x - dbd_origin_utm_x, 
    y=df_dbd.utm_gps_y - dbd_origin_utm_y,
    marker='X',
    color='tab:red', 
    s=200,
    label='GPS Fix',
    data=df_dbd,
    zorder=3,
)
odos=3

lgnd = plt.legend(loc='lower left', fontsize='small')
for i in range(odos):
    lgnd.legendHandles[i]._sizes = [100]

plt.axis('equal')
xlim=ax.get_xlim()
ylim=ax.get_ylim()

ax.set_xlim(xlim)
ax.set_ylim(ylim)
plt.xlabel('X Position [m]')
plt.ylabel('Y Position [m]')
plt.title('DVL-ODO', fontweight='bold')

voc_u_list,voc_v_list,voc_w_list, z_list = water_column.compute_averages()
print("> Finished Estimating Water Column Currents!")
dvl_plotter.plot_water_column_currents(voc_u_list, voc_v_list, voc_w_list, z_list)

plt.show()
