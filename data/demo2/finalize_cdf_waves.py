"""
The workflow here produces data that work in ncBrowse, Panoply

1. fix time
2. trim to in water times
3. update depth coordinate using mean water surface
4. trim to depth using pressure following
5. fix metadata
"""
# for local testing
import sys
sys.path.append('e:\\python\\ADCPy\\')
from adcpy.EPICstuff.EPICmisc import check_fill_value_encoding
from adcpy.EPICstuff.EPICmisc import fix_missing_time
import os
import glob
import datetime as dt
import numpy as np
import matplotlib.pyplot as plt
import netCDF4 as nc
from netCDF4 import num2date
import xarray as xr
import stglib

# -------------- user settings
datapath = 'E:\\data\\Matanzas\\V23857\\python\\'
# output from repopulate EPIC
infileroot = '11121whVwaves00repo.cdf'
finalfile = '11121whVwaves-cal.cdf'
var = 'Pressure'  # a convenient variable to plot to check things
# the nominal DELTA_T for this time series
delta_t = 3600  # in seconds, what the inter-burst time interval should be
initial_instrument_height = 2.423
orientation = "UP"
WATER_DEPTH = 14.9

# dates to trim to - this syntax is probably important
deployed = "2018-01-24 16:49:00"
recovered = "2018-04-12 20:00:00"  # this is the date that cuts off bad data

# variables we do not need to keep
# in this particular case, SDP_850 was always 0
vars2drop = {'SDP_850'}

# user metadata not in the file already
atts_to_update = {
    'DELTA_T': str(delta_t),
    'initial_instrument_height': initial_instrument_height,
    'orientation': orientation,
    'WATER_DEPTH': WATER_DEPTH
}

# TODO
# atts_to_remove = {}


# -------------- begin operations

# Check time variable and fix missing time stamps
# chunking is important, this is a very large file
ds = xr.open_dataset(datapath+infileroot, chunks={'time': 24}, drop_variables=vars2drop)

# look for the missing time values
print('Time is from {} to {}'.format(ds['time'][0].data, ds['time'][-1].data))
# identify where the bad times might be
bad_bursts = np.asarray(list(map(lambda x: True if np.isnat(x) else False, ds['time'][:].values)))
# bads = np.asarray(list(map(lambda x: True if x == np.nan else False, ds['time'][:].data)))
# get the index
bad_burst_indices = np.where(bad_bursts is True)
# find the last good time - does it match the out of water time we know?
print('Time is from {} to {}'.format(ds['time'][0].data, ds['time'][-1].data))
print('There are {} missing current bursts'.format(len(bad_burst_indices[0])))
print(bad_burst_indices)
print(len(bad_burst_indices[0]))

# fix the missing time values
ds_new, number_filled = fix_missing_time(ds, delta_t)
print("{} time intervals were changed from NaT to a time stamp".format(number_filled))
print('ds time is from {} to {}'.format(ds['time'][0].data, ds['time'][-1].data))
print('ds_new time is from {} to {}'.format(ds_new['time'][0].data, ds_new['time'][-1].data))

# Update metadata for this time adjustment
ds_new = stglib.utils.add_start_stop_time(ds_new)
ds_new = stglib.utils.insert_history(ds_new, '; time gaps filled')

# always have to reset _FillValue from NaN back to CF compliant
ds_new, nonan_encoding = check_fill_value_encoding(ds_new)

dt = ds_new.time.diff(dim='time') / 1000000000  # result is in sec
s = dt.mean().data
# command fo showing plot
dt.plot()
ds_new.attrs['DELTA_T'] = int(s)  # needs to be an integer
print('DELTA_T = mean(diff(time)) = {}'.format(ds_new.attrs['DELTA_T']))
print('DELTA_T ranges from {} to {}'.format(dt.max().data, dt.min().data))

# make sure time's type is now EVEN or UNEVEN
# dependent on variaton found in the burst time stamps, this may vary if there are gaps
ds_new['time'].attrs['type'] = "UNEVEN"

# save this stage of the work
s = infileroot.split('.')
newfileroot = s[0] + '-fixt.' + s[1]
ds_new.to_netcdf(datapath + newfileroot, encoding=nonan_encoding)
ds.close()
ds_new.close()
print(datapath + newfileroot)

# # Trim to in water time
ds = xr.open_dataset(datapath + newfileroot, chunks={'time': 24})
ds[var][0].plot()
# always have to reset _FillValue from NaN back to CF compliant
ds_new, nonan_encoding = check_fill_value_encoding(ds)
ds.attrs['Deployment_date'] = deployed
ds.attrs['Recovery_date'] = recovered
# try stglib first
ds_new = stglib.core.utils.clip_ds(ds_new)
# dsnew = clip_ds(dsnew)
# update metadata
ds_new = stglib.utils.add_start_stop_time(ds_new)
ds_new[var].plot()
s = newfileroot.split('.')
newfileroot = s[0] + '-trmt.' + s[1]
ds_new.to_netcdf(datapath + newfileroot, encoding=nonan_encoding)
ds.close()
ds_new.close()
print(datapath + newfileroot)

# # update depth coordinate variable using water_depth
# This MUST be done with direct netCDF calls, xarray will not allow changes to coordinate variable data
# TODO see if we can use stglib methods

df = nc.Dataset(datapath + newfileroot, mode='r+')
df.history = df.history + '; recompute depth information'
for attname in df.ncattrs():
    if 'water' in attname.lower():
        print('{} : {}'.format(attname, df.getncattr(attname)))
d = df['depth'][:].data
print('input depth from {} to {}'.format(d[0], d[-1]))


def update_depth_variable(df, initial_instrument_height, orientation, WATER_DEPTH):
    # these are not in the raw file attributes
    # print('initial_instrument_height : {}'.format(df.initial_instrument_height))
    # print('orientation : {}'.format(df.orientation))
    # print('WATER_DEPTH : {}'.format(df.WATER_DEPTH))

    d = df['depth'][:].data
    print('depth is found to be from {} to {}'.format(d[0], d[-1]))
    b = df['bindist'][:].data
    print('bindist from {} to {}'.format(b[0], b[-1]))

    if 'UP' in orientation.upper():
        depths = WATER_DEPTH - initial_instrument_height - b[:]
    else:
        depths = -1 * (WATER_DEPTH - initial_instrument_height + b[:])

    print('depth is now from {} to {}'.format(depths[0], depths[-1]))
    df['depth'][:] = depths[:]
    # ds.assign_coords(depth=depths[:])

    return df


df = update_depth_variable(df, initial_instrument_height, orientation, WATER_DEPTH)
d = df['depth'][:].data
print('depth from {} to {}'.format(d[0], d[-1]))
df.close()
print(datapath + newfileroot)


# # Trim to surface
# TODO see if we can use the stglib method
# modified from https://github.com/dnowacki-usgs/stglib/aqd/qaqc.py trim_vel for Aquadopp
# to work on any variable that contains the depth dimension
def trim_bins(*args, **kwargs):
    """
    Trim profile data to water level determined by pressure
    ds = trim_bins(ds, pressure_name='Pressure')

    pressure_name = name of the pressure variable to use
    pad = extra depth to add to extend or reduce the trim, for instance to account for sidelobes

    Note that returned data set will have NaNs as the fill, for EPIC and CF compliance,
    make sure the encoding _FillValue is not NaN before using to_netcdf

    """
    ds = args[0]

    if 'pressure_name' in kwargs.keys():
        # user is overriding or doing a separate process
        pvarname = kwargs['pressure_name']
    else:
        pvarname = None
    if 'pad' in kwargs.keys():
        # user is overriding or doing a separate process
        pad = kwargs['pad']
    else:
        pad = None

    if (pvarname is not None) and (pvarname in ds):
        # generate a list of the variables with depth coordinate
        vars2trim = list(map(lambda t: t[0] if 'depth' in ds[t[0]].coords else None, ds.variables.items()))
        vars2trim = list(filter(lambda v: v != None, vars2trim))
        vars2trim = list(filter(lambda v: v not in {'depth', 'bindist'}, vars2trim))
        print('trimming {}'.format(vars2trim))
        print('using pressure from variable named {}, mean = {}'.format(
            pvarname, ds[pvarname].mean().values))

        # try to get the instrument height, which we need to get water level
        if 'initial_instrument_height' in ds.attrs:
            inst_height = ds.attrs['initial_instrument_height']
        elif 'transducer_offset_from_bottom' in ds.attrs:
            inst_height = ds.attrs['transducer_offset_from_bottom']
        else:
            inst_height = 0

        print('using instrument height of {}'.format(inst_height))

        if pad != None:
            print('a pad of {} was provided'.format(pad))
        else:
            pad = 0.0

        # TODO account for units of pressure as db and decapasals
        # in this case it is decapascals
        WL = ds[pvarname][:] / 1000 + inst_height + pad

        print(WL.mean().values)
        print(ds[pvarname].mean().values)
        print(ds['bindist'].mean().values)

        for var in vars2trim:
            print('\tTrimming {}'.format(var))
            ds[var] = ds[var].where(ds['bindist'] < WL)

        # find first bin that is all bad values
        # Dan Nowacki comments: there might be a better way to do this using xarray and named
        # dimensions, but this works for now
        lastbin = np.argmin(np.all(np.isnan(ds[vars2trim[0]].values), axis=0) == False)

        # this trims so there are no all-nan rows in the data
        ds = ds.isel(depth=slice(0, lastbin))

        histtext = 'Data clipped using {}'.format(pvarname)

        ds = stglib.utils.insert_history(ds, histtext)

    else:
        print('Did not trim data')

    return ds


ds = xr.open_dataset(datapath + newfileroot, chunks={'time': 24})
print(datapath + newfileroot)
ds['att1'][:, 0, :].plot()

# Here we create a working copy of the Dataset and back up the encoding from the original data prior to
# the trim operation. Some of the encoding seemed to get lost. check_FillValue_encoding announces when it finds NaNs
ds_new, nonan_encoding = check_fill_value_encoding(ds)
ds_new = trim_bins(ds_new, pressure_name='Pressure')
ds_new['att1'][:, 0, :].plot()
s = newfileroot.split('.')
newfileroot = s[0] + '-trms.' + s[1]
ds_new.to_netcdf(datapath + newfileroot, encoding=nonan_encoding)
ds.close()
ds_new.close()
print(datapath + newfileroot)

# # Fix up metadata
# touch up various metadata issues per: https://cmgsoft.repositoryhosting.com/trac/cmgsoft_m-cmg/wiki/preBBVcheck%20
ds = xr.open_dataset(datapath + newfileroot, chunks={'time': 24})
print(datapath + newfileroot)
ds_new = ds
# use the user supplied dictionary to update metadata
for att in atts_to_update.items():
    ds_new.attrs[att[0]] = att[1]

# These are things checked by BBV_meta_check.m
gatts = ds_new.attrs
if 'VAR_FILL' not in gatts.keys():
    print('VAR_FILL missing, being inserted as 1E35')
    ds_new.attrs['VAR_FILL'] = 1e35
else:
    print('VAR_FILL is {}'.format(gatts['VAR_FILL']))
print('history is {}'.format(gatts['history']))
att = 'serial_number'
if att not in gatts.keys():
    print('{} missing'.format(att))
else:
    print('{} is {}'.format(att, gatts[att]))

# check that all the variables have the serial number, as all are attached to this ADCP
varlist = ds_new.variables.keys()

for var in varlist:
    attlist = ds_new[var].attrs
    if 'serial_number' not in attlist:
        print('\tserial_number added to {}'.format(var))
        ds_new[var].attrs['serial_number'] = ds_new.attrs['serial_number']

ds_new = stglib.utils.add_min_max(ds_new)

# update water_depth information based on trimmed pressure
ds_new = stglib.utils.create_water_depth(ds_new)
# we don't know if we need a wter_depth variable, this is already in the global WATER_DEPTH
# dsnew = stglib.utils.create_water_depth_var(dsnew)

# all done, do some checking
# ds_new.attrs
# ds_new['Pressure']
# if len(dsnew['Pressure'].shape) > 1:
# ds_new['Pressure'][:].plot()

# and finally check the encoding one more time
ds_new, nonan_encoding = check_fill_value_encoding(ds_new)

ds_new.to_netcdf(datapath+finalfile, encoding=nonan_encoding)

ds.close()
ds_new.close()
print(datapath + newfileroot)

# # We are done!!
