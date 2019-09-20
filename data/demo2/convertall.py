# script to process raw data to netcdf using python module
# at the anaconda prompt in the data directory, with IOOS3 or equivalent activated, run this script as
# E:\data\Matanzas\V23857\python>python converall.py > convertalloutput.txt

import adcpy.TRDIstuff.TRDIpd0tonetcdf as pd0
import adcpy.EPICstuff.ADCPcdf2ncEPIC as cdf2nc
import datetime as dt

input_path = 'E:\\data\\Matanzas\\V23857\\'
output_path = 'E:\\data\\Matanzas\\V23857\\python\\'
pd0File = input_path + 'raw\\11121whV23857.pd0'
attFile = 'E:\\data\\Matanzas\\glob_att1112.txt'
serial_number = '23857'
delta_t = "3600"
testing = False
do_part_one = False
do_part_two = True

if not testing:
    cdfFile = output_path + '11121whV.cdf'
    ncFile = output_path + '11121whV.nc'
    ensembles_to_convert = [0, -1]  # -1 is for all
else:
    # test subset
    cdfFile = output_path + '11121whVsubset.cdf'
    ncFile = output_path + '11121whVsubset.nc'
    ensembles_to_convert = [0, 10000]

timetype = 'CF'

settings = {
    'good_ensembles': ensembles_to_convert,
    'orientation': 'UP',  # uplooking ADCP, for downlookers, use DOWN
    'transducer_offset_from_bottom': 2.423,  # meters
    'transformation': 'EARTH',  # | BEAM | INST
    # if file is not trimmed, best to use False here
    'use_pressure_for_WATER_DEPTH': False,
}

print(pd0File)

all_start_time = dt.datetime.now()

if do_part_one:
    # this command will display basics about the raw binary data
    maximum_ensembles, ensLen, ensData, startOfData = pd0.analyzepd0file(pd0File, True)
    
    # will print out two layers of nested dictionary
    print('data found in file ', pd0File)
    for key, value in sorted(ensData.items()):
        value_type = type(ensData[key])
        if value_type == dict:
            print('\n', key, ':')
            for key1, value1 in sorted(ensData[key].items()):
                print(key1, ':', value1)
            else:
                print(key, ':', value)
        else:
            print('\nData ---> %s' % key)
    
    start_time = dt.datetime.now()
    print("start binary to raw cdf conversion at %s" % start_time)
    # when using this module this way, the start and end ensembles are required.
    # use Inf to indicate all ensembles.
    print('Converting from %s\n to %s\n' % (pd0File, cdfFile))
    pd0.convert_pd0_to_netcdf(pd0File, cdfFile, ensembles_to_convert, serial_number, timetype, delta_t)
    end_time = dt.datetime.now()
    print("finished binary to raw cdf conversion at %s" % start_time)
    print("processing time was %s" % (end_time - start_time))

if do_part_two:
    settings['timetype'] = timetype
    
    start_time = dt.datetime.now()
    print("start raw cdf to EPIC conversion at %s" % start_time)
    # when using this module this way, the start and end ensembles are required.
    # use Inf to indicate all ensembles.
    cdf2nc.doEPIC_ADCPfile(cdfFile, ncFile, attFile, settings)
    end_time = dt.datetime.now()
    print("finished raw cdf to EPIC conversion at %s" % end_time)
    print("processing time was %s" % (end_time - start_time))

all_end_time = dt.datetime.now()
print("For all the operations, processing time was %s" % (all_end_time - all_start_time))
