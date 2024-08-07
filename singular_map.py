# Import Libraries
import os
import geopandas as gpd
import glob
import shutil
from glob import glob
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import numpy as np
import shapely
from shapely import wkt
from descartes.patch import PolygonPatch
import mapclassify

# Standard Paths
data_path = os.getenv('DATA','/data')
inputs_path = os.path.join(data_path,'inputs')
outputs_path = os.path.join(data_path,'outputs')
if not os.path.exists(outputs_path):
    os.mkdir(outputs_path)

# Define and create individual input paths
#grid_path = os.path.join(inputs_path,'grid')
boundary_path = os.path.join(inputs_path,'boundary')
parameters_path = os.path.join(inputs_path,'parameters')
udm_para_in_path = os.path.join(inputs_path, 'udm_parameters')

# Define output path for parameters
parameters_out_path=os.path.join(outputs_path,'parameters')
if not os.path.exists(parameters_out_path):
    os.mkdir(parameters_out_path)

# Find the parameter file
parameter_file = glob(parameters_path + "/*.csv", recursive = True)
print('parameter_file:', parameter_file)

if len(parameter_file) != 0 :
    all_parameters = pd.concat(map(pd.read_csv,parameter_file),ignore_index=True)
    print(all_parameters)
    if 'LOCATION' in all_parameters.values:
        location_row = all_parameters[all_parameters['PARAMETER']=='LOCATION']
        location=location_row['VALUE'].values[0]
        print('LOCATION:',location)
    else:    
        location = int(os.getenv('LOCATION'))
    if 'PROJECTION' in all_parameters.values:
        projection_row = all_parameters[all_parameters['PARAMETER']=='PROJECTION']
        projection=projection_row['VALUE'].values[0]
        print('projection:',projection)
    else:
        projection = int(os.getenv('PROJECTION'))

# Find the scenario datasets, and boundary
scenarios = glob(inputs_path + "/1km_data*.csv", recursive = True)
print('Scenarios:', scenarios)
boundary = glob(boundary_path + "/*.gpkg", recursive = True)
print('Boundary:', boundary)

dst_crs = 'epsg:'+projection
# Read in the boundary file and set the crs
if len(boundary) != 0:
    boundary1 = gpd.read_file(boundary[0])
    boundary1.set_crs(dst_crs)

# Create empty panda dataframes for each of the parameters of interest
filename=[]
filename=['xx' for n in range(len(scenarios))]
tot_count=[]
tot_count = pd.DataFrame()
tot_count['index']=[0 for n in range(1000)]
damages=[]
damages = pd.DataFrame()
damages['index']=[0 for n in range(1000)]
results=[]
results=pd.DataFrame(results)

# For each file, read in the data for each parameter, in this instance total building count and damages
# The code uses a loop, in case future modifications on the code wish to compare multiple scenarios
for i in range(0,len(scenarios)):
    test = scenarios[i]
    file_path = os.path.splitext(test)
    filename[i]=file_path[0].split("/")
    unit_name = filename[i][-1]
    parameters_1 = pd.read_csv(os.path.join(inputs_path, unit_name + '.csv'))

    # Move to outputs
    src =scenarios[i]
    dst = os.path.join(outputs_path,unit_name + '.csv')
    shutil.copy(src,dst)
    
    tot_count[unit_name] = parameters_1['Total_Building_Count']
    damages[unit_name] = parameters_1['Damage_Rank']

tot_count.pop('index')
damages.pop('index')

# Identify the maximum and minimum values from each array (if multiple scenarios are considered it will look for
# the max and min across all to create a uniform colourbar between all maps)
results['Total_Buildings_max'] = tot_count.max()
results['Total_Damages_max'] = damages.max()
results['Total_Buildings_min'] = tot_count.min()
results['Total_Damages_min'] = damages.min()

# Remove NaN values from the list of damages and building counts
damages= damages.dropna()
tot_count = tot_count.dropna()

damages = damages.loc[(damages!=0).any(axis=1)]
tot_count = tot_count.loc[(tot_count!=0).any(axis=1)]

# Calculate the 10 quartiles for the list of damages
damage_quartiles=[]
damage_quartiles=[0 for n in range(10)]
for i in range(10):
    damage_quartiles[i] = np.percentile(damages,i*10)

# #damage_quartiles=['{:.0f}'.format(elem) for elem in damage_quartiles]
print('x:',damage_quartiles)

# Calculate the 10 quartiles for the list of buildings affected
buildings_quartiles=[]
buildings_quartiles=[0 for n in range(10)]
for i in range(10):
    buildings_quartiles[i] = np.percentile(tot_count,i*10)

#buildings_quartiles=['{:.0f}'.format(elem) for elem in buildings_quartiles]
print('x:',buildings_quartiles)

# Replace any nan values with zero
tot_count.replace(0, np.nan, inplace = True)
damages.replace(0, np.nan, inplace = True)

results.index.names=['Scenario']
results = results.reset_index()

for i in range (0,len(results)):
    results.Scenario[i]=results.Scenario[i].split(location + '_')[-1]

## WORK OUT THE YEAR ##
results['year'] = 1
for i in range (0,len(results)):
    results.year[i] = results.Scenario[i].split('_')[1] 

## WORK OUT THE SSP ##
results['ssp'] = 1
for i in range (0,len(results)):
    results.ssp[i] = results.Scenario[i].split('_')[0] 

## WORK OUT THE DEPTH ##
results['depth'] = 1
for i in range (0,len(results)):
    results.depth[i] = results.Scenario[i].split('_')[2]

damages_min = results.agg({'Total_Damages_min':['min']}).unstack()
damages_max = results.agg({'Total_Damages_max':['max']}).unstack()
build_min = results.agg({'Total_Buildings_min':['min']}).unstack()
build_max = results.agg({'Total_Buildings_max':['max']}).unstack()

# Create a dataframe listing of all the scenario results
dataframes_list=[]
for i in range(0,len(scenarios)):
    temp_df = pd.read_csv(scenarios[i])
    dataframes_list.append(temp_df)

# Identify the geometry column from the csv files to map the data to the os grid cells
# gdf is a dataframe containing all of the datasets
gdf=[]
for i in range(0,len(scenarios)):
    temp_df = dataframes_list[i]
    #temp_df['geometry'] = temp_df['geometry'].apply(wkt.loads)
    temp_gdf = gpd.GeoDataFrame(temp_df)
    temp_gdf['geometry'] = temp_gdf['geometry_x'].apply(wkt.loads)
    temp_gdf.pop('geometry_x')
    temp_gdf.set_geometry('geometry',crs=dst_crs)
    gdf.append(temp_gdf)


# if there is only one scenario to view:
if len(scenarios) == 1:

    ##### BUILDINGS #####

    # Plot the boundary of the city
    fig,axarr = plt.subplots(figsize = (16,8))
    pcm = boundary1.boundary.plot(edgecolor = 'black', lw = 0.5, ax=axarr)
    
    # Read in the data from the gdf database ready to clip to the boundary
    gdf_clip = gdf[0]
    gdf_clip.crs = boundary1.crs

    # Clip the output data to the boundary
    city_clipped = gpd.clip(gdf_clip,boundary1)

    # Plot the clipped data, add a title and x-labels
    pcm = city_clipped.plot(column = "Total_Building_Count",ax=axarr,scheme = 'quantiles', k=20, edgecolor = 'black',lw = 0.2,cmap='Greys')#vmin=build_min[0],vmax=build_max[0],

    # Work out the scenario, year and depth of each run
    depth_1 = str(results['depth'][0])
    ssp_1 = str(results['ssp'][0])
    year_1 = str(results['year'][0])

    
    
    pcm.set_title(location + '_' + ssp_1 + '_' + year_1 + '_' + depth_1 + '_BUILDINGS', fontsize=12)
    plt.setp(pcm.get_xticklabels(), horizontalalignment='right') #rotation=30,

    # Add a colourbar to the figure
    cmap = mpl.cm.Greys

    # bounds = [0, 200, 400, 600, 800, 1000, 1200, 1400]

    # print('bounds:',bounds)
    # print('buildings_quartiles:', buildings_quartiles)

    norm = mpl.colors.BoundaryNorm(buildings_quartiles, cmap.N)
    cax = fig.add_axes([0.79, 0.15, 0.02, 0.7])
    cb3 = mpl.colorbar.ColorbarBase(cax, cmap=cmap,
                    norm = norm,
                    boundaries =[0]+buildings_quartiles+[1200],
                    extend='both',
                    extendfrac='auto',
                    ticks = buildings_quartiles,
                    format='%.0f',
                    spacing='uniform',
                    orientation='vertical')
    cb3.set_label('# Buildings flooded')
 
# # if there are two scenarios to view
# if len(scenarios) == 2:
#     # Create a subplot
#     fig,axarr = plt.subplots(1,2,figsize = (16,8),sharex = True, sharey = True)

#     # Plot the boundary of the city for both subplots
#     for i in range(0,2):
#         if len(boundary_path) != 0:
#             pcm = boundary1.boundary.plot(edgecolor = 'black', lw = 0.5,ax=axarr[i])#,vmin=build_min[0],vmax=build_max[0])
    
#     for i in range(0,len(scenarios)):
#         # Read in the data from the gdf database ready to clip to the boundary
#         gdf_clip = gdf[i]
#         gdf_clip.crs = boundary1.crs

#         # Clip the output data to the boundary
#         city_clipped = gpd.clip(gdf_clip,boundary1)

#         # Plot the clipped data, add a title and x-labels
#         pcm = city_clipped.plot(column = "Total_Building_Count",ax=axarr[i],vmin=build_min[0],vmax=build_max[0],edgecolor = 'black',lw = 0.2,cmap='Greys')

#         # Work out the scenario, year and depth of each run
#         depth_1 = results['depth'][i]
#         ssp_1 = results['scenario'][i]
#         year_1 = results['year'][i]

#         axarr[i].set_title(location + '_'+ ssp_1 + '_'+ year_1 + '_' + depth_1 + '_BUILDINGS', fontsize=12)
#         plt.setp(axarr[i].get_xticklabels(), rotation=30, horizontalalignment='right')

#     # Add a colourbar to the figure
#     fig = pcm.get_figure()
#     cax = fig.add_axes([0.95, 0.15, 0.03, 0.7])
#     sm = plt.cm.ScalarMappable(norm=plt.Normalize(vmin=build_min[0],vmax=build_max[0]),cmap='Greys')
#     sm._A = []
#     fig.colorbar(sm, cax=cax)

# # # if there are four scenarios to view
# # if len(scenarios) == 4:
#     # Create a subplot
#     fig,axarr = plt.subplots(2,2,figsize = (16,16),sharex = True, sharey = True)
#     m=0

#     # Plot the boundary of the city for both subplots
#     for i in range(0,2):
#         for j in range(0,2):
#             if len(boundary_path) != 0:
#                 pcm = boundary1.boundary.plot(edgecolor = 'black', lw = 0.5,ax=axarr[i,j],vmin=build_min[0],vmax=build_max[0])

#     for i in range(0,2):
#         for j in range(0,2):
#             # Read in the data from the gdf database ready to clip to the boundary
#             gdf_clip = gdf[m]
#             gdf_clip.crs = boundary1.crs

#             # Clip the output data to the boundary
#             city_clipped = gpd.clip(gdf_clip,boundary1)

#             # Plot the clipped data, add a title and x-labels
#             pcm = city_clipped.plot(column = "Total_Building_Count",ax=axarr[i,j],vmin=build_min[0],vmax=build_max[0],edgecolor = 'black',lw = 0.2,cmap='Greys')

#                 # Work out the scenario, year and depth of each run
#             depth_1 = results['depth'][m]
#             ssp_1 = results['scenario'][m]
#             year_1 = results['year'][m]

#             axarr[i,j].set_title(location + '_'+ ssp_1 + '_'+ year_1 + '_' + depth_1 + '_BUILDINGS', fontsize=12)
#             plt.setp(axarr[i].get_xticklabels(), rotation=30, horizontalalignment='right')
#             m=m+1

#     # Add a colourbar to the figure
#     fig = pcm.get_figure()
#     cax = fig.add_axes([0.95, 0.15, 0.03, 0.7])
#     sm = plt.cm.ScalarMappable(norm=plt.Normalize(vmin=build_min[0],vmax=build_max[0]),cmap='Greys')
#     sm._A = []
#     fig.colorbar(sm, cax=cax)

# Save the figure to the output path
plt.savefig(os.path.join(outputs_path, location +'_Buildings.png'), bbox_inches='tight' ,dpi=600)

# if there is only one scenario to view:
if len(scenarios) == 1:

    ##### DAMAGES #####

    # Plot the boundary of the city
    fig,axarr = plt.subplots(figsize = (16,8))
    pcm1 = boundary1.boundary.plot(edgecolor = 'black', lw = 0.5, ax=axarr)
    
    # Read in the data from the gdf database ready to clip to the boundary
    gdf_clip = gdf[0]
    gdf_clip.crs = boundary1.crs

    # Clip the output data to the boundary
    city_clipped = gpd.clip(gdf_clip,boundary1)

    # Plot the clipped data, add a title and x-labels
    pcm1 = city_clipped.plot(column = "Damage_Rank",ax=axarr,scheme = 'quantiles', k=20,edgecolor = 'black',lw = 0.2,cmap='GnBu') #vmin=damages_min[0],vmax=damages_max[0]

    # Work out the scenario, year and depth of each run
    depth_1 = str(results['depth'][0])
    ssp_1 = str(results['ssp'][0])
    year_1 = str(results['year'][0])
    
    pcm1.set_title(location + '_' + ssp_1 + '_' + year_1 + '_' + depth_1 + '_DAMAGES', fontsize=12)
    plt.setp(pcm.get_xticklabels(), rotation=30, horizontalalignment='right')

    # Add a colourbar to the figure
    cmap = mpl.cm.GnBu

    norm = mpl.colors.BoundaryNorm(damage_quartiles, cmap.N)
    cax = fig.add_axes([0.79, 0.15, 0.02, 0.7])
    cb3 = mpl.colorbar.ColorbarBase(cax, cmap=cmap,
                    norm = norm,
                    boundaries =[-10]+damage_quartiles+[10],
                    extend='both',
                    extendfrac='auto',
                    ticks = damage_quartiles,
                    format='%.0f',
                    spacing='uniform',
                    orientation='vertical')
    cb3.set_label('Damages Incurred (£)')

# # if there are two scenarios to view
# if len(scenarios) == 2:
#     # Create a subplot
#     fig,axarr = plt.subplots(1,2,figsize = (16,8),sharex = True, sharey = True)

#     # Plot the boundary of the city for both subplots
#     for i in range(0,2):
#         if len(boundary_path) != 0:
#             pcm = boundary1.boundary.plot(edgecolor = 'black', lw = 0.5,ax=axarr[i])
    
#     for i in range(0,len(scenarios)):
#         # Read in the data from the gdf database ready to clip to the boundary
#         gdf_clip = gdf[i]
#         gdf_clip.crs = boundary1.crs

#         # Clip the output data to the boundary
#         city_clipped = gpd.clip(gdf_clip,boundary1)

#         # Plot the clipped data, add a title and x-labels
#         pcm = city_clipped.plot(column = "Damage",ax=axarr[i],vmin=damages_min[0],vmax=damages_max[0],edgecolor = 'black',lw = 0.2,cmap='GnBu')

#         # Work out the scenario, year and depth of each run
#         depth_1 = results['depth'][i]
#         ssp_1 = results['scenario'][i]
#         year_1 = results['year'][i]

#         axarr[i].set_title(location + '_'+ ssp_1 + '_'+ year_1 + '_' + depth_1 + '_DAMAGES', fontsize=12)
#         plt.setp(axarr[i].get_xticklabels(), rotation=30, horizontalalignment='right')

#     # Add a colourbar to the figure
#     fig = pcm.get_figure()
#     cax = fig.add_axes([0.95, 0.15, 0.03, 0.7])
#     sm = plt.cm.ScalarMappable(norm=plt.Normalize(vmin=damages_min[0],vmax=damages_max[0]),cmap='GnBu')
#     sm._A = []
#     fig.colorbar(sm, cax=cax)

# # if there are four scenarios to view
# if len(scenarios) == 4:
#     # Create a subplot
#     fig,axarr = plt.subplots(2,2,figsize = (16,16),sharex = True, sharey = True)
#     m=0

#     # Plot the boundary of the city for both subplots
#     for i in range(0,2):
#         for j in range(0,2):
#             if len(boundary_path) != 0:
#                 pcm = boundary1.boundary.plot(edgecolor = 'black', lw = 0.5,ax=axarr[i,j],vmin=damages_min[0],vmax=damages_max[0])

#     for i in range(0,2):
#         for j in range(0,2):
#             # Read in the data from the gdf database ready to clip to the boundary
#             gdf_clip = gdf[m]
#             gdf_clip.crs = boundary1.crs

#             # Clip the output data to the boundary
#             city_clipped = gpd.clip(gdf_clip,boundary1)

#             # Plot the clipped data, add a title and x-labels
#             pcm = city_clipped.plot(column = "Damage",ax=axarr[i,j],vmin=damages_min[0],vmax=damages_max[0],edgecolor = 'black',lw = 0.2,cmap='GnBu')

#                 # Work out the scenario, year and depth of each run
#             depth_1 = results['depth'][m]
#             ssp_1 = results['scenario'][m]
#             year_1 = results['year'][m]

#             axarr[i,j].set_title(location + '_'+ ssp_1 + '_'+ year_1 + '_' + depth_1 + '_DAMAGES', fontsize=12)
#             #plt.setp(axarr[i].get_xticklabels(), rotation=30, horizontalalignment='right')
#             m=m+1

#     # Add a colourbar to the figure
#     fig = pcm.get_figure()
#     cax = fig.add_axes([0.95, 0.15, 0.03, 0.7])
#     sm = plt.cm.ScalarMappable(norm=plt.Normalize(vmin=build_min[0],vmax=build_max[0]),cmap='GnBu')
#     sm._A = []
    # fig.colorbar(sm, cax=cax)

# Save the figure to the output path
plt.savefig(os.path.join(outputs_path, location +'_Damages.png'), bbox_inches='tight' ,dpi=600)


# If linked to UDM results, pass the udm details through to the outputs
udm_para_out_path = os.path.join(outputs_path, 'udm_parameters')
if not os.path.exists(udm_para_out_path):
    os.mkdir(udm_para_out_path)

meta_data_txt = glob(udm_para_in_path + "/**/metadata.txt", recursive = True)
meta_data_csv = glob(udm_para_in_path + "/**/metadata.csv", recursive = True)
attractors = glob(udm_para_in_path + "/**/attractors.csv", recursive = True)
constraints = glob(udm_para_in_path + "/**/constraints.csv", recursive = True)

if len(meta_data_txt)==1:
    src = meta_data_txt[0]
    dst = os.path.join(udm_para_out_path,'metadata.txt')
    shutil.copy(src,dst)

if len(meta_data_csv)==1:
    src = meta_data_csv[0]
    dst = os.path.join(udm_para_out_path,'metadata.csv')
    shutil.copy(src,dst)

if len(attractors)==1:
    src = attractors[0]
    dst = os.path.join(udm_para_out_path,'attractors.csv')
    shutil.copy(src,dst)

if len(constraints)==1:
    src = constraints[0]
    dst = os.path.join(udm_para_out_path,'constraints.csv')
    shutil.copy(src,dst)


# Move the parameter file to the outputs folder
if len(parameter_file) != 0 :
    for i in range (0, len(parameter_file)):
        file_path = os.path.splitext(parameter_file[i])
        filename=file_path[0].split("/")
    
        src = parameter_file[i]
        dst = os.path.join(parameters_out_path,filename[-1] + '.gpkg')
        shutil.copy(src,dst)
