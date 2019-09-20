# this script will spawn a browser tab using
# panel serve browse_burst.py --show

import xarray as xr
import pandas as pd
import numpy as np
import holoviews as hv
from holoviews import streams # do I use this?
import hvplot.xarray
import panel as pn

# initializations
pn.extension()  # initializes the panel back end


# get a list of the variable names in the file that are not ordinate
def get_var_name_list(ds):
    coordnames = []
    for cname in ds.coords:
        coordnames.append(cname)

    varnames = []
    for vname in ds.variables:
        # eliminate any coordinates
        if vname not in coordnames:
            varnames.append(vname)

    return varnames


infile = '11121whVwaves02repo.nc'
# infile = '11121whVwaves02repo.cdf'
# infile = '11121whVwaves01repo-fixt-trmt-trms-met.nc'

ds = xr.open_dataset(infile)

# I use the helper function defined above get_var_name_list to make a list of variable names that make sense to plot
select_variable = pn.widgets.Select(options=get_var_name_list(ds),name='Pick a variable')

select_variable # test
# TODO name is not in the docs as something that can be set - is ther a generic set of thigns that can be set in all widgets?

# depth_slider = pn.widgets.FloatSlider(start=ds['depth'].values.min(), end=ds['depth'].values.max(), value=ds['depth'][:].values.mean())
depth_spinner = pn.widgets.Spinner(name='Set depth to view', start=ds['depth'].values.min(), end=ds['depth'].values.max(),
                                   value=ds['depth'][:].values.mean(), step=1)
# burst_slider = pn.widgets.IntSlider(start=0, end=10, value=5)
# note that get_date_input depends on this widget
burst_spinner = pn.widgets.Spinner(name='Set burst index to view', start=0, end=len(ds['time']),
                                   value=len(ds['time'])/2, step=1)


@pn.depends(burst_num=burst_spinner.param.value,
            var_name=select_variable.param.value,
            depth=depth_spinner.param.value)
def create_plot_obj(burst_num, var_name='Pressure', depth=0, **kwargs):
    depth_cell = np.min(np.where(ds['depth'] >= depth))  # find the equivalent depth
    burst_num = int(burst_num)  # the spinner will return a float, we need an index
    time_from_burst_selection = ds['time'][burst_num]

    data_slice = ds.sel(time=time_from_burst_selection)[var_name]

    # build a title for the plot
    plot_title = '{} at {}, index {}'.format(data_slice.long_name, burst_num, time_from_burst_selection)

    # TODO check for lat and lon or ordinate dimensions other than time, depth and sample and squash them

    plot_width = 700

    if len(data_slice.shape) == 1:
        the_plot_object = hv.Curve(data_slice)
        the_plot_object.opts(width=plot_width, title=plot_title)

    if len(data_slice.shape) == 2:

        if 'vel' in var_name:
            cmap = 'seismic'
        elif 'att' in var_name:
            cmap = 'PiYG'
        else:
            cmap = 'Viridis'

        y_label = '{}, {}'.format(data_slice.depth.long_name, data_slice.depth.units)
        the_plot_object = hv.Image((data_slice['sample'].values, np.flipud(data_slice['depth'].values),
                                    data_slice.values.transpose()),
                                   kdims=['sample', y_label])

        the_plot_object.opts(colorbar=True, cmap=cmap, width=plot_width, title=plot_title)

    return the_plot_object


pnobj = pn.Row(pn.Column(select_variable, burst_spinner, depth_spinner, width=180), create_plot_obj)

# https://panel.pyviz.org/getting_started/index.html
pnobj.servable()

ds.close()