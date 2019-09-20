"""
The workflow here prepares a final, Best Basic Version current velocity profile data file of rotated current data
that will work in ncBrowse and Panoply

Input is data that have been converted from raw binary pd0, rotated to earth coordinates, reshaped into
bursts and samples redistributed by time and time has 1 dimension.

1. fix time
2. trim to in water times
3. update depth coordinate using mean water surface
4. trim to depth using pressure following
5. fix metadata
"""
# for local testing
import os
import sys
sys.path.append('e:\\python\\ADCPy\\')
from adcpy.EPICstuff.EPICmisc import check_fill_value_encoding
from adcpy.EPICstuff.EPICmisc import fix_missing_time
from adcpy.EPICstuff.EPICmisc import display_encoding
import numpy as np
import netCDF4 as nc
import xarray as xr
import stglib

# ------------- user settings
data_path = 'E:\\data\\Matanzas\\V23857\\python\\'
# output from repopulate EPIC
input_file = '11121whVwaves02repo.nc'
final_file = '11121whVwaves02-cal.nc'
var = 'P_1'  # a convenient variable to plot to check things
# the nominal DELTA_T for this time series
delta_t = 3600  # in seconds, what the inter-burst time interval should be

# dates to trim to - this syntax is probably important
deployed = "2018-01-24 16:49:00"
recovered = "2018-04-12 20:00:00"  # this is the date that cuts off bad data

# variables we do not need to keep
# in this particular case, SDP_850 was always 0
vars2drop = {'SDP_850'}

# user metadata not in the file already
atts_to_update = {'DELTA_T': str(delta_t)}

delete_interim_files = True

# atts_to_remove = {}

# -------------- begin operations

interim_files = []  # we will delete these later
# --------------- helper functions


# modified from https://github.com/dnowacki-usgs/stglib/aqd/qaqc.py trim_vel for Aquadopp
# to work on any variable that contains the depth dimension
def trim_bins(*args, **kwargs):
    """
    Trim profile data to water level determined by pressure
    ds = trim_bins(ds_to_trim, pressure_name='P_1')

    pressure_name = name of the pressure variable to use
    pad = extra depth to add to extend or reduce the trim, for instance to account for sidelobes

    Note that returned data set will have NaNs as the fill, for EPIC and CF compliance,
    make sure the encoding _FillValue is not NaN before using to_netcdf

    """
    ds_to_trim = args[0]

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

    if (pvarname is not None) and (pvarname in ds_to_trim):
        # generate a list of the variables with depth coordinate
        vars2trim = list(map(lambda t: t[0] if 'depth' in ds_to_trim[t[0]].coords else None,
                             ds_to_trim.variables.items()))
        vars2trim = list(filter(lambda v: v is not None, vars2trim))
        vars2trim = list(filter(lambda v: v not in {'depth', 'bindist'}, vars2trim))
        print('trimming {}'.format(vars2trim))
        print('using pressure from variable named {}, mean = {}'.format(
            pvarname, ds_to_trim[pvarname].mean().values))

        # try to get the instrument height, which we need to get water level
        if 'initial_instrument_height' in ds_to_trim.attrs:
            inst_height = ds_to_trim.attrs['initial_instrument_height']
        elif 'transducer_offset_from_bottom' in ds_to_trim.attrs:
            inst_height = ds_to_trim.attrs['transducer_offset_from_bottom']
        else:
            inst_height = 0

        print('using instrument height of {}'.format(inst_height))

        if pad is not None:
            print('a pad of {} was provided'.format(pad))
        else:
            pad = 0.0

        water_level = ds_to_trim[pvarname][:] + inst_height + pad

        print(water_level.mean().values)
        print(ds_to_trim[pvarname].mean().values)
        print(ds_to_trim['bindist'].mean().values)

        for variable in vars2trim:
            print('\tTrimming {}'.format(variable))
            ds_to_trim[variable] = ds_to_trim[variable].where(ds_to_trim['bindist'] < water_level)

        # find first bin that is all bad values
        # Dan Nowacki comments: there might be a better way to do this using xarray and named
        # dimensions, but this works for now
        last_bin = np.argmin(np.all(np.isnan(ds_to_trim[vars2trim[0]].values), axis=0) is False)

        # this trims so there are no all-nan rows in the data
        ds_trimmed = ds_to_trim.isel(depth=slice(0, last_bin))

        history_text = f'Data clipped using {pvarname}'
        print(history_text)

        ds_trimmed = stglib.utils.insert_history(ds_trimmed, history_text)

    else:
        print('Did not trim data')

    return ds_trimmed


def update_depth_variable(df):
    print('initial_instrument_height : {}'.format(df.initial_instrument_height))
    print('orientation : {}'.format(df.orientation))
    print('WATER_DEPTH : {}'.format(df.WATER_DEPTH))

    depth_data = df['depth'][:].data
    print('depth is found to be from {} to {}'.format(depth_data[0], depth_data[-1]))
    b = df['bindist'][:].data
    print('bindist from {} to {}'.format(b[0], b[-1]))

    if 'UP' in df.orientation.upper():
        depths = df.WATER_DEPTH - df.initial_instrument_height - b[:]
    else:
        depths = -1 * (df.WATER_DEPTH - df.initial_instrument_height + b[:])

    print('depth is now from {} to {}'.format(depths[0], depths[-1]))
    df['depth'][:] = depths[:]
    # ds.assign_coords(depth=depths[:])

    return df


# Check time variable and fix missing time stamps
# chunking is important, this is a very large file
ds = xr.open_dataset(data_path + input_file, chunks={'time': 24}, drop_variables=vars2drop)
print(f'encoding for input file {data_path + input_file}')
display_encoding(ds, '_FillValue')

# look for the missing time values
print('Time is from {} to {}'.format(ds['time'][0].data, ds['time'][-1].data))
# identify where the bad times might be
# TODO should use np.where, is this line redundant?
bad_bursts = np.asarray(list(map(lambda x: True if np.isnat(x) else False, ds['time'][:].values)))
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

dt = ds_new.time.diff(dim='time') / 1000000000  # result is in sec
s = dt.mean().data
# command fo showing plot
# dt.plot()
ds_new.attrs['DELTA_T'] = int(s)  # needs to be an integer
print('DELTA_T = mean(diff(time)) = {}'.format(ds_new.attrs['DELTA_T']))
print('DELTA_T ranges from {} to {}'.format(dt.max().data, dt.min().data))

# make sure time's type is now EVEN or UNEVEN
# dependent on variaton found in the burst time stamps, this may vary if there are gaps
ds_new['time'].attrs['type'] = "UNEVEN"

# save this stage of the work
s = input_file.split('.')
new_file_root = s[0] + '-fixt.' + s[1]
# ds_new.to_netcdf(data_path + new_file_root, encoding=no_nan_encoding)
print(f'encoding for ds_new, before check_fill_value_encoding')
display_encoding(ds_new, '_FillValue')
ds_new = check_fill_value_encoding(ds_new)
print(f'encoding for ds_new, after check_fill_value_encoding')
display_encoding(ds_new, '_FillValue')
ds_new.to_netcdf(data_path + new_file_root)
ds.close()
ds_new.close()
print(f'finished {data_path + new_file_root}')
interim_files.append(data_path + new_file_root)

# # Trim to in water time
ds = xr.open_dataset(data_path + new_file_root, chunks={'time': 24})
# ds[var][0].plot()
# always have to reset _FillValue from NaN back to CF compliant
ds_new, no_nan_encoding = check_fill_value_encoding(ds)
ds.attrs['Deployment_date'] = deployed
ds.attrs['Recovery_date'] = recovered
# try stglib first
ds_new = stglib.core.utils.clip_ds(ds_new)
# ds_new = clip_ds(ds_new)
# update metadata
ds_new = stglib.utils.add_start_stop_time(ds_new)
# ds_new[var].plot()
s = new_file_root.split('.')
new_file_root = s[0] + '-trmt.' + s[1]
# ds_new.to_netcdf(data_path + new_file_root, encoding=no_nan_encoding)
display_encoding(ds_new, '_FillValue')
ds_new.to_netcdf(data_path + new_file_root)
ds.close()
ds_new.close()
print(f'finished {data_path + new_file_root}')
interim_files.append(data_path + new_file_root)

# update depth coordinate variable using WATER_DEPTH
# This MUST be done with direct netCDF calls, xarray will not allow changes to coordinate variable data
# TODO see if we can use stglib methods

cdf = nc.Dataset(data_path + new_file_root, mode='r+')
cdf.history = cdf.history + '; recompute depth information'
for attname in cdf.ncattrs():
    if 'water' in attname.lower():
        print('{} : {}'.format(attname, cdf.getncattr(attname)))
d = cdf['depth'][:].data
print('input depth from {} to {}'.format(d[0], d[-1]))

cdf = update_depth_variable(cdf)
d = cdf['depth'][:].data
print('depth from {} to {}'.format(d[0], d[-1]))
cdf.close()
print(data_path + new_file_root)

ds = xr.open_dataset(data_path + new_file_root, chunks={'time': 24})
print(f'trimming bins for {data_path + new_file_root}')
# ds['AGC_1202'][:, 0, :, 0, 0].plot()

# Here we create a working copy of the Dataset and back up the encoding from the original data prior to
# the trim operation. Some of the encoding seemed to get lost. check_FillValue_encoding announces when it finds NaNs
# ds_new, no_nan_encoding = check_fill_value_encoding(ds)
ds_new = trim_bins(ds_new, pressure_name='P_1')

# ds_new['AGC_1202'][:, 0, :, 0, 0].plot()
s = new_file_root.split('.')
new_file_root = s[0] + '-trmb.' + s[1]
print(data_path + new_file_root)
#ds_new['time'].encoding['_FillValue'] = False
#ds_new['depth'].encoding['_FillValue'] = False
#ds_new['lat'].encoding['_FillValue'] = False
#ds_new['lon'].encoding['_FillValue'] = False
#ds_new['sample'].encoding['_FillValue'] = False

# ds_new.to_netcdf(data_path + new_file_root, encoding=no_nan_encoding)
display_encoding(ds_new, '_FillValue')
# this write is failing here and not in Jupyter
ds_new.to_netcdf(data_path + new_file_root)
ds.close()
ds_new.close()
print(f'finished {data_path + new_file_root}')
interim_files.append(data_path + new_file_root)

# # Fix up metadata
# touch up various metadata issues per: https://cmgsoft.repositoryhosting.com/trac/cmgsoft_m-cmg/wiki/preBBVcheck%20
ds = xr.open_dataset(data_path + new_file_root, chunks={'time': 24})
print(data_path + new_file_root)
ds_new = ds
# use the user supplied dictionary to update metadata
for att in atts_to_update.items():
    ds_new.attrs[att[0]] = att[1]

# These are things checked by BBV_meta_check.m
global_attributes = ds_new.attrs
print('lat att {} and lat var {}'.format(global_attributes['latitude'], ds['lat'][0].values))
if global_attributes['latitude'] != ds['lat'][0].values:
    print('\t *** latitude mismatch')
print('lon att {} and lon var {}'.format(global_attributes['longitude'], ds['lon'][0].values))
if global_attributes['longitude'] != ds['lon'][0].values:
    print('\t *** longitude mismatch')
if 'VAR_FILL' not in global_attributes.keys():
    print('VAR_FILL missing, being inserted as 1E35')
    ds_new.attrs['VAR_FILL'] = 1e35
else:
    print('VAR_FILL is {}'.format(global_attributes['VAR_FILL']))
print('history is {}'.format(global_attributes['history']))
att = 'serial_number'
if att not in global_attributes.keys():
    print('{} missing'.format(att))
else:
    print('{} is {}'.format(att, global_attributes[att]))

# check that all the variables have the serial number, as all are attached to this ADCP
variable_list = ds_new.variables.keys()

for var in variable_list:
    attlist = ds_new[var].attrs
    if 'serial_number' not in attlist:
        print('\tserial_number added to {}'.format(var))
        ds_new[var].attrs['serial_number'] = ds_new.attrs['serial_number']

if type(ds.attrs['MOORING']) != 'str':
    print('Mooring {} is not a string'.format(ds_new.attrs['MOORING']))
    ds_new.attrs['MOORING'] = str(int(ds_new.attrs['MOORING']))
    print('MOORING is {} type {}'.format(ds_new.attrs['MOORING'], type(ds_new.attrs['MOORING'])))

ds_new = stglib.utils.add_min_max(ds_new)

# update water_depth information based on trimmed pressure
ds_new = stglib.utils.create_water_depth(ds_new)
# we don't know if we need a wter_depth variable, this is already in the global WATER_DEPTH
# dsnew = stglib.utils.create_water_depth_var(dsnew)

# all done, do some checking
# ds_new.attrs
# ds_new['P_1']
# if len(dsnew['P_1'].shape) > 1:
# ds_new['P_1'][:, 0, 0].plot()

# and finally check the encoding one more time
ds_new, no_nan_encoding = check_fill_value_encoding(ds_new)

ds_new.to_netcdf(data_path + final_file, encoding=no_nan_encoding)

ds.close()
ds_new.close()
print(data_path + new_file_root)

# get rid of the interim files
if delete_interim_files is True:
    for file in interim_files:
        print(f'removing interim file {file}')
        # os.remove(file)

# # We are done!!
