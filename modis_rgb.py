#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 11:06:57 2019

@author: Chris Tracy
"""

"""
This module contains functions for reading in and plotting HDF5 swath data from the 
Moderate-Resolution Imaging SpectroRadiometer (MODIS) with 36 different spectral bands. 
An RGB is formed through separate band extraction and regridding of the relevant bands. 
More information on the instrument is available at http://modis.gsfc.nasa.gov/data/ . 
Data is available at https://ladsweb.modaps.eosdis.nasa.gov/search/ .

Routine Listing
-------
read_modis_file : Read in the MODIS files, re-scales the lats/lons to match band data
scale_rgb : Mask and histogram the red, green, and blue input bands
plot_true_rgb : Create an image of re-sampled RGB data from the MODIS file
"""

import numpy as np

def read_modis_file(file, nointerp = False):
    """
    This function reads in a MODIS HDF file, re-scales the data, and interpolate
    the lats and lons to match the shape of the color bands. The bands, lats, and 
    lons are then returned from the function.
    
    Parameters
    ------------
    file: The filepath string of the MODIS file to be read in
    nointerp: A boolean parameter. If False (default), an interpolation is done on
    the data.
    
    Returns
    ------------
    bands: A dictionary holding each of the scaled/offset color bands
    lats: A 2D array of the latitudes of the MODIS file (interpolated or not)
    lons: A 2D array of the longitudes of the MODIS file (interpolated or not)
    """
    
    from netCDF4 import Dataset
    
    #Open the file. Make a dictionary to hold the band data.
    data = Dataset(file)
    bands = dict()
    
    #This reads in bands 1-2
    ev250 = data.variables['EV_250_Aggr1km_RefSB']
    
    #Extract each of the bands and apply a scale/offset. Add them to the dictionary.
    for idx, b in enumerate(np.arange(1, 3)):
        data1 = ev250[idx, :, :]
        bands[b] = (data1 * ev250.reflectance_scales[idx]) + ev250.reflectance_offsets[idx]
    
    #This reads in bands 3-7
    ev500 = data.variables['EV_500_Aggr1km_RefSB']
    
    #Extract each of the bands and apply a scale/offset. Add them to the dictionary.
    for idx, b in enumerate(np.arange(3, 8)):
        data1 = ev500[idx, :, :]
        bands[b] = (data1 * ev500.reflectance_scales[idx]) + ev500.reflectance_offsets[idx]
    
    #Read in the lats/lons
    lats = data.variables['Latitude'][:]
    lons = data.variables['Longitude'][:]
    
    if not nointerp:
        
        #If above condition is met, interpolate the lats/lons to match the shape of the bands.
        #Use the shape ratio between the band data and the lats as an interpolation "zoom" factor.
        from scipy.ndimage import zoom
        
        factor = np.array(ev500[0, :, :].shape)/np.array(lats.shape)
        lats = zoom(lats, factor)
        lons = zoom(lons, factor)
    
    #Close the file
    data.close()
    
    return bands, lats, lons

def scale_rgb(rgb, min_input = 0.0, max_input = 1.1):
    """
    This function scales RGB (red, green, blue) band data from the MODIS file. 
    Here, the RGB data is put into a dictionary with the keys r, g, b. A mask 
    for each color is made, and the input is scaled appropriately for binary purposes. 
    The data is then put into a histogram and an RGB copy, which is outputted from the function.
    
    Parameters
    ------------
    rgb: A dictionary of assigned color bands
    min_input: The minimum value of the data, optional parameter (default = 0.0)
    max_input: The maximum value of the data, optional parameter (default = 1.1)
    
    Returns
    ------------
    _rgb: The scaled RGB data dictionary
    """
    
    from skimage.exposure import equalize_hist
    
    #Make an RGB copy to protect the original input data.
    _rgb = rgb.copy()
    
    #Build the RGB binary mask. Flip the mask so that the unmasked data is kept.
    mask = rgb['r'].mask | rgb['g'].mask | rgb['b'].mask
    mask = ~mask
    
    for key, vals in _rgb.items():
    
        #Scale the RGB data. The values in the scaling are based on info here:
        #http://gis-lab.info/docs/modis_true_color.pdf
        #Note: this will effectively un-mask the masked array.
        data = np.interp(vals, (min_input, max_input), (0, 255))
        
        #Apply the RGB mask. Equalize/histogram the data.
        data = data * mask
        data = equalize_hist(data)
        
        #The read_modis1b function takes 8-bit numbers and returns
        #an array of floats in the range 0-1. Make sure there are 8-bit numbers
        #scaled from 0-255 for the RGB image. Add the re-scaled color band to the
        #dictionary.
        data = np.uint8(data * 255)
        _rgb[key] = data
    
    return _rgb

def plot_true_rgb(r, g, b, lats, lons, 
                  lat_bounds = None, lon_bounds = None, 
                  binx = 15000, biny = 15000, noscale = False):
    """
    This function plots the true color RGB image from the MODIS file data.
    Make note that r, g, and b should be masked arrays, and that the user can
    provide map bounds in the form of either a tuple, list, or array. The input bins
    will be passed to pyresample.area_config.create_area_def.
    
    Parameters
    -----------
    r: A NumPy masked array containing values of the "red" channel.
    g: A NumPy masked array containing values of the "green" channel.
    b: A NumPy masked array containing values of the "blue" channel.
    lon_bounds: The optional (min, max) tuple of the longitude bounds of the output map. 
             Default is to use min/max of the lons.
    lat_bounds: The optional (min, max) tuple of the latitude bounds of the output map. 
                Default is to use min/max of the lats.
    binx: Input resolution in the x-direction (default = 15000 meters or 15 km)
    biny: Input resolution in the y-direction (default = 15000 meters or 15 km)
    noscale: A boolean signifying whether to scale the RGB data. Default = False (scale the data)
    
    Returns
    ------------
    img: The final gridded image
    """
    
    import matplotlib.pyplot as plt
    import cartopy.crs as ccrs
    import pyresample
    
    #Pack the channels into a dictionary. Scale the data with scale_rgb.
    rgb = {'r': r, 'g': g, 'b': b}
    if not noscale:
        rgb = scale_rgb(rgb)
    
    #Map projections
    map_proj = ccrs.Mercator()
    ll_proj = ccrs.PlateCarree()
    
    #Next, the data is resampled onto a regular grid.
    #Get the extent of the data in the map coordinate system.
    if lon_bounds is None:
        lon_bounds = lons.min(), lons.max()   
    if lat_bounds is None:
        lat_bounds = lats.min(), lats.max()
        
    #Convert these to map coordinates. Define the extent of the map in projection coordinates.
    ll_mapped = map_proj.transform_points(ll_proj, np.array(lon_bounds), np.array(lat_bounds))
    x_map = ll_mapped[:, 0]
    y_map = ll_mapped[:, 1]
    map_extent = (*x_map, *y_map) #Tuple of the map bounds
    
    #Define the target grid
    grid = pyresample.area_config.create_area_def('null', map_proj.proj4_params,
                                                  units = 'meters',
                                                  resolution = (binx, biny),
                                                  area_extent = map_extent)
    
    #Get the swath grid (aka "source grid"). Resample each band using nearest neighbor.
    #To do so, define a radius of influence (ROI, 5000 meters here) and the grid dictionary.
    swath = pyresample.SwathDefinition(lons, lats)
    radius_influence = 5000
    rgb_grid = dict()
    
    #Resample each color band using a "nearest" method with the specified ROI. Add
    #the resampled band to the rgb_grid dictionary.
    for key, val in rgb.items():
        
        data = pyresample.kd_tree.resample_nearest(swath, val, grid,
                                                   radius_of_influence = radius_influence)
        rgb_grid[key] = data
        
    #For the image, stack the RGB values into a 3D array.
    rgb_grid = np.dstack(list(rgb_grid.values()))
    
    #Finally, make the map (Mercator projection). Set the extent, using the variables 
    #associated with the projection.
    fig, ax = plt.subplots(subplot_kw = {'projection': map_proj})
    ax.set_extent(map_extent, map_proj)
    
    #Set the image using the variables associated with the gridded data.
    img = ax.imshow(rgb_grid, transform= grid.to_cartopy_crs(),
                    extent = grid.to_cartopy_crs().bounds, origin = 'upper')
    
    #Additional map features
    ax.coastlines()
    ax.gridlines(draw_labels=True, linestyle=':')
    
    return img

# if __name__ == '__main__':
    
#     file = 'C:/Users/ctracy/Downloads/MOD021KM.A2005240.1700.061.2017185042936.hdf'
#     bands, lat, lon = read_modis_file(file)
#     img = plot_true_rgb(bands[1], bands[4], bands[3], lat, lon)
