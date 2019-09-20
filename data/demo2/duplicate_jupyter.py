datapath = 'E:\\data\\Matanzas\\V23857\\python\\'
# output from repopulate EPIC
# works on these files
#infileroot = '11121whVwaves01repo.nc'
#finalfile = '11121whVwaves01-cal.nc'
infileroot = '11121whVwaves00repo.nc'
finalfile = '11121whVwaves00-cal.nc'
infileroot = '11121whVwaves02repo-fixt-trmt.nc'
var = 'P_1' # a convenient variable to plot to check things
delta_t = 3600 # in seconds, what the inter-burst time interval should be

# dates to trim to - this syntax is probably important
deployed = "2018-01-24 16:49:00"
recovered = "2018-04-12 20:00:00" # this is the date that cuts off bad data

# variables we do not need to keep
# in this particular case, SDP_850 was always 0
vars2drop = {'SDP_850'}

# the nominal DELTA_T for this time series
atts_to_update = {'DELTA_T': str(delta_t)}

# TODO
# atts_to_remove = {}

import os
import sys
import glob
import datetime as dt
import numpy as np
import matplotlib.pyplot as plt
# %matplotlib inline
import netCDF4 as nc
from netCDF4 import num2date
import xarray as xr
import stglib

ds = xr.open_dataset(datapath+infileroot,chunks={'time':24},drop_variables=vars2drop)

def display_encoding(ds, item):
    for var in ds.variables.items():
        if item in ds[var[0]].encoding:
            print('encoding for {} {} is {}'.format(var[0], item, ds[var[0]].encoding[item]))
        else:
            print('encoding for {} does not have {}'.format(var[0], item))

display_encoding(ds, '_FillValue')

ds['time'].encoding


def make_encoding_dict(ds):
    """
    prepare encoding dictionary for writing a netCDF file later

    :param ds: xarray Dataset
    :return: dict with encoding prepared for xarray.to_netcdf to EPIC/CF conventions
    """
    encoding_dict = {}

    for item in ds.variables.items():
        # print(item)
        var_name = item[0]
        var_encoding = ds[var_name].encoding

        encoding_dict[var_name] = var_encoding

        # print('encoding for {} is {}'.format(var_name, encoding_dict[var_name]))

        # is it a coordinate?
        if var_name in ds.coords:
            # coordinates do not have a _FillValue
            if '_FillValue' in encoding_dict[var_name]:
                print(f'encoding {var_name} fill value to False')
            else:
                print(f'encoding {var_name} is missing fill value, now added and set to False')

            encoding_dict[var_name]['_FillValue'] = False

        else:
            # _FillValue cannot be NaN and must match the data type
            # so just make sure it matches the data type.
            if '_FillValue' in encoding_dict[var_name]:
                print('{} fill value is {}'.format(var_name, encoding_dict[var_name]['_FillValue']))
                if np.isnan(encoding_dict[var_name]['_FillValue']):
                    print(f'NaN found in _FillValue of {var_name}, correcting')
                    if 'float' in encoding_dict[var_name]['dtype']:
                        encoding_dict[var_name]['_FillValue'] = 1E35
                    elif 'int' in encoding_dict[var_name]['dtype']:
                        encoding_dict[var_name]['_FillValue'] = 32768
                elif encoding_dict[var_name]['_FillValue'] is None:
                    print(f'None found in _FillValue of {var_name}, correcting')
                    if 'float' in encoding_dict[var_name]['dtype']:
                        encoding_dict[var_name]['_FillValue'] = 1E35
                    elif 'int' in encoding_dict[var_name]['dtype']:
                        encoding_dict[var_name]['_FillValue'] = 32768
                else:
                    print('encoding found in _FillValue of {} remains {}'.format(var_name,
                                                                                 encoding_dict[var_name]['_FillValue']))

    return encoding_dict

encoding_dict = make_encoding_dict(ds)

ds.to_netcdf('junk.nc', encoding=encoding_dict)

ds.close()
