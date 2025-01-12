"""
  To slice the vti file centered around a given center point (x0, y0, z0)
    in the volumn defined by the vti file.  
  The range is given along x, y, and z direction as dx, dy, and dz, with 
    minimum of 1 slice included.  
    
Author:

  Zhan Zhang (zhanzhang@anl.gov)
    
Details:
       
  The 2-D slice in xy, xz and yz plane are first given to check the center 
    and range (plot as boxes).  Each 2-D slice is over the full range of volume
    in the in-plane direction, and summed over the given range of the 3rd 
    (perpendicular) direction. For example, the xz cut is a sum of all slices 
    within the range of z0-dz/2 <= z <= z0+dz/2.  
  The 1-D plot is obtained by summing over 1 of the 2 dimensions of the 2-D slice. 
    Each 2-D slice generates two 1-D plots.  The plot would be same along each axis
    from different summations.  
  Effort is taken to normalize out points without intensity, which is better than
    doing nothing but not really able to handle raw data taken with heavy filters. 
    
Change list:

  2023-03-24 (ZZ):     
    - Adopted from the Jupyter Notebook.  likely need to go through the 
        cells there to figure out the proper slice center and the range. 
    - Use Json file as input parameters    
    - cycling through multiple vti files    
        
"""
import sys
import os
import time
import re
import numpy as np
import json
import argparse
from spec2nexus import spec

import vtk
from vtk.util import numpy_support
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# finding index in an array with values between value1 and value2
def find_indice(array, value1, value2):
    array = np.asarray(array)
    idx1 = (np.abs(array - value1)).argmin()
    idx2 = (np.abs(array - value2)).argmin()
    #return np.array( range(min(idx1, idx2), max(idx1, idx2)+1) )
    return min(idx1, idx2), max(idx1, idx2)+1

# functions to parse the json file.  
def parseArgs():
    parser = argparse.ArgumentParser(description='Default VTI slice script. Expects a json config file pointed at the data.')
    parser.add_argument('configPath', 
                        nargs='?', 
                        default=os.path.join(os.getcwd(), 'vtiSliceconfig.json'),
                        help='Path to config file. If none supplied, directs to a config.json located in CWD')
    return parser.parse_args()

def generateScanLists(inputScanLists):
    scanListTop = []
    for inputScanList in inputScanLists:
        # How many conditions/cycles in total
        num_cycles = inputScanList["cycles"]
        # Total number of scans at one condition, say each temperature
        scans_in_1_cycle = inputScanList["scans_per_cycle"]
        SetsOfRSM = inputScanList["rsm_sets"]
        
        for i in range(0, num_cycles):
            for oneSetRSM in SetsOfRSM:
                scan_s = oneSetRSM["start"]
                scan_e = oneSetRSM["end"]
                scans_in_1_rsm = scan_e - scan_s + 1
                scanListTop = scanListTop + \
                    ( [[f"{x}" if scans_in_1_rsm==1 else f"{x}-{x+scans_in_1_rsm-1}"] for \
                    x in range(scan_s+i*scans_in_1_cycle, scan_e+i*scans_in_1_cycle+1, scans_in_1_rsm)] )
    return scanListTop

# do 3-panel line plot
def lineplots(plot_qx, line_x, plot_qy, line_y, plot_qz, line_z, \
        subtitles, xlabels, ylabels):
    subp = []
    fig, axes = plt.subplots(1, 3, sharey=False, figsize=(15, 5))

    subp.append( axes[0].plot(plot_qx,  line_x,  'b.-', label= 'Summed over YZ from Y slice') )
    #subp.append( axes[0].plot(plot_qy,  line_y,  'r.-', label= 'Summed over YZ from X slice') )
    subp.append( axes[1].plot(plot_qy,  line_y,  'b.-', label= 'Summed over YZ from X slice') )
    subp.append( axes[2].plot(plot_qz,  line_z,  'b.-', label= 'Summed over XY from X slice') )

    for subplot, ax, subtitle, xlabel, ylabel in zip(subp, axes.ravel(), subtitles, xlabels, ylabels):
        ax.set(title = subtitle, xlabel = xlabel, ylabel = ylabel)
        if plot_logscale:
            ax.set_yscale("log")

    fig.tight_layout()  
    if show_plots:
        plt.show()
    else:
        #plt.show(block = False)
        #time.sleep(2)
        plt.close()
    
# write data to file
def writedatafile(output_filename, write_mode, tmp_str, \
        header, headerfmt, output_data, outfmt):
    outfile = open(output_filename, write_mode)
    outfile.write(tmp_str)
    outfile.write('#\n')
    outfile.write(headerfmt % header)
    np.savetxt(outfile, output_data, fmt=outfmt)
    outfile.close()

 
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
# Get the input from the input file
args = parseArgs()
with open(args.configPath, 'r') as config_f:
    config = json.load(config_f)

# workpath and config file path
projectDir = config["project_dir"]
                      
# flags for verbose, plot_logscale, and show_plots
verbose = config["verbose"]
plot_logscale = config["plot_logscale"]
show_plots = config["show_plots"]

# plot axes label
#x_axis = config["x_axis"]
#y_axis = config["y_axis"]
#z_axis = config["z_axis"]

# Output flags
output_all = config["output_all"]
output_x   = config["output_x"]
output_y   = config["output_y"]
output_z   = config["output_z"]

# input vti files 
datasets = config["datasets"]

#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$

#=====================================
# Outer-most loop, iterate over different base_files (folders)

for idx1, dataset in enumerate(datasets, 1):
    # base file name and scan list number
    base_filename = dataset["base_filename"]
    scanListTop = dataset["scan_list"]
    vti_catag = dataset["vti_catagory"]
    # Assign axes label based on the catagoray value here.
    if(vti_catag.lower()=="hkl"):
        x_axis, y_axis, z_axis = ["h", "k", "l"]
    else:
        x_axis, y_axis, z_axis = ["Qx", "Qy", "Qz"]
        
    slice_settings = dataset["slice_settings"]
    
    if scanListTop == None:
        inputScanLists = dataset["scan_range"]
        scanListTop = generateScanLists(inputScanLists)

    # SPEC file for getting some motor values
    spec_file = os.path.join(projectDir, base_filename+'.spec')
    specData = spec.SpecDataFile(spec_file)
    # Check if SPEC file exists at the location defined. 
    if verbose:
        print(spec_file)
    if(not os.path.isfile(spec_file)):
        if verbose:
            print(" doesn't exist! Moving on...")
        continue
    
    # Inner-loop, iterate over each vti file
    for idx2, scanList in enumerate(scanListTop, 1):
        
        # Depending on the data file structure, this file path/name
        #  might need to be changed.  
        data_dir = os.path.join(projectDir, "analysis_runtime", base_filename)
        filebase = f"{base_filename}_{scanList[0]}_{vti_catag}"
        vti_file = os.path.join(data_dir, filebase+'.vti')
        # setup some strings for output file header
        tmp_str = f"# VTI file: {vti_file}\n"

        # Check if the vti file exists at the location defined. 
        if verbose:
            print(tmp_str)
        if(not os.path.isfile(vti_file)):
            if verbose:
                print(" doesn't exist! Moving on...")
            continue
        
        # get scan numbers: srange() from rsMap3D package can be used if its there
        # Try this simplified way
        scan_nums = [int(s) for s in re.findall(r'\d+', scanList[0]) if s.isdigit()]
        if verbose:
            print(f"Scan numbers in this set: {scan_nums}")
        """
        # This is the part experiment specific
        for scan_num in scan_nums:
            # read motor ksamx and ksamy positions for each scan
            scan = specData.getScan(scan_num)
            scan.interpret()
            ksamx = scan.positioner['ksamx']
            ksamy = scan.positioner['ksamy']
            # adding to the header string
            tmp_str += f"# Scan {scan_num}: ksamx = {ksamx}, ksamy = {ksamy}\n"
        """
        
        # read vti file
        vti_reader = vtk.vtkXMLImageDataReader()
        vti_reader.SetFileName(vti_file)
        vti_reader.Update()
        vti_data = vti_reader.GetOutput()
        vti_point_data = vti_data.GetPointData()
        vti_array_data = vti_point_data.GetScalars()
        array_data = numpy_support.vtk_to_numpy(vti_array_data)

        dim = vti_data.GetDimensions()
        steps = vti_data.GetSpacing()
        origin = vti_data.GetOrigin()
        print(f"dimensions are: {dim}")

        # generate 3 orthogonal axes 
        qx = np.linspace(origin[0], origin[0]+dim[0]*steps[0], dim[0])
        qy = np.linspace(origin[1], origin[1]+dim[1]*steps[1], dim[1])
        qz = np.linspace(origin[2], origin[2]+dim[2]*steps[2], dim[2])

        array_data0 = np.reshape(array_data, dim[::-1])
        array_data0 = np.transpose(array_data0)

        qx_min, qx_max = qx.min(), qx.max()
        qy_min, qy_max = qy.min(), qy.max()
        qz_min, qz_max = qz.min(), qz.max()

        if verbose:
            print('Coordinate vectors:')
            print(f'  {x_axis}: {np.shape(qx)}, min = {qx_min}, max = {qx_max}' )
            print(f'  {y_axis}: {np.shape(qy)}, min = {qy_min}, max = {qy_max}' )
            print(f'  {z_axis}: {np.shape(qz)}, min = {qz_min}, max = {qz_max}' )
            print('Data array:')
            print('  Dimensions: ', dim)
            print('  array shape: ', np.shape(array_data0) )


        #define the slice location and 'thickness'
        qx_pos, qx_delta = [ slice_settings["x_cen"], slice_settings["x_range"]]
        qy_pos, qy_delta = [ slice_settings["y_cen"], slice_settings["y_range"]]
        qz_pos, qz_delta = [ slice_settings["z_cen"], slice_settings["z_range"]]

        qx_ind1, qx_ind2 = find_indice(qx, qx_pos-qx_delta/2, qx_pos+qx_delta/2)
        qy_ind1, qy_ind2 = find_indice(qy, qy_pos-qy_delta/2, qy_pos+qy_delta/2)
        qz_ind1, qz_ind2 = find_indice(qz, qz_pos-qz_delta/2, qz_pos+qz_delta/2)
        print(f"qz_ind1 = {qz_ind1}; qz_ind2 = {qz_ind2}")

        data_block_qx_range = array_data0[qx_ind1:qx_ind2, :, :]
        data_block_qy_range = array_data0[:, qy_ind1:qy_ind2, :]
        data_block_qz_range = array_data0[:, :, qz_ind1:qz_ind2]

        if verbose:
            print('Sliced Data array:')
            print('  array shape for data_block_qx_range: ', np.shape(data_block_qx_range) )
            print('  array shape for data_block_qy_range: ', np.shape(data_block_qy_range) )
            print('  array shape for data_block_qz_range: ', np.shape(data_block_qz_range) )

        data_slice_yz = data_block_qx_range.sum(axis=0)
        data_slice_xz = data_block_qy_range.sum(axis=1)
        data_slice_xy = data_block_qz_range.sum(axis=2)

        """
        # Also have errorbar calculated, not sure what use of it here yet.
        data_slice_yz_error =np.sqrt(data_slice_yz)
        data_slice_xz_error =np.sqrt(data_slice_xz)
        data_slice_xy_error =np.sqrt(data_slice_xy)

        # Here need to figure out how many points are summed are non-zero data on each of the pixel.
        #  Later the normalization needs it to level the field.
        data_slice_yz_stat = np.count_nonzero(data_block_qx_range, axis = 0)
        data_slice_xz_stat = np.count_nonzero(data_block_qy_range, axis = 1)
        data_slice_xy_stat = np.count_nonzero(data_block_qz_range, axis = 2)

        data_slice_yz_stat[data_slice_yz_stat==0] = 1
        data_slice_xz_stat[data_slice_xz_stat==0] = 1
        data_slice_xy_stat[data_slice_xy_stat==0] = 1

        # Scale the summed slice with actual number of point with data
        data_slice_yz = data_slice_yz/data_slice_yz_stat*np.amax(data_slice_yz_stat)
        data_slice_xz = data_slice_xz/data_slice_xz_stat*np.amax(data_slice_xz_stat)
        data_slice_xy = data_slice_xy/data_slice_xy_stat*np.amax(data_slice_xy_stat)

        # Same scaling for the errorbar -- if they are needed at this point
        data_slice_yz_error = data_slice_yz_error/data_slice_yz_stat*np.amax(data_slice_yz_stat)
        data_slice_xz_error = data_slice_xz_error/data_slice_xz_stat*np.amax(data_slice_xz_stat)
        data_slice_xy_error = data_slice_xy_error/data_slice_xy_stat*np.amax(data_slice_xy_stat)
        """
        # I have to switch the dimension to plot the first on horizontal 
        #  with going up and right are positive.  
        plot_slice_yz = np.transpose(data_slice_yz.copy(), (1, 0))
        plot_slice_xz = np.transpose(data_slice_xz.copy(), (1, 0))
        plot_slice_xy = np.transpose(data_slice_xy.copy(), (1, 0))

        if plot_logscale:
            plot_slice_yz[plot_slice_yz<1] = 1
            plot_slice_xz[plot_slice_xz<1] = 1
            plot_slice_xy[plot_slice_xy<1] = 1

            plot_slice_yz = np.log10( plot_slice_yz )
            plot_slice_xz = np.log10( plot_slice_xz )
            plot_slice_xy = np.log10( plot_slice_xy )

        if verbose:
            print('Summed Images:')
            print('  array shape for plot_slice_yz: ', np.shape(plot_slice_yz) )
            print('  array shape for plot_slice_xz: ', np.shape(plot_slice_xz) )
            print('  array shape for plot_slice_xy: ', np.shape(plot_slice_xy) )

        ###################################
        # image plot here for summed slices, all three of them
        subp = []
        subtitles = [f'{x_axis} = {qx_pos:.3f}+/-{qx_delta/2:.4f} slice', 
                     f'{y_axis} = {qy_pos:.3f}+/-{qy_delta/2:.4f} slice', 
                     f'{z_axis} = {qz_pos:.3f}+/-{qz_delta/2:.4f} slice']
        xlabels =[y_axis, x_axis, x_axis]
        ylabels =[z_axis, z_axis, y_axis]
        interp = 'bilinear'

        fig, axes = plt.subplots(1, 3, sharey=False, figsize =(15, 5))
        subp.append( axes[0].imshow(plot_slice_yz, origin='lower', #interpolation=interp,
                cmap = 'hsv',
                extent= (qy_min, qy_max, qz_min, qz_max), 
                clim =(plot_slice_yz.min()+0, plot_slice_yz.max()-0 ) ) )
        subp.append( axes[1].imshow(plot_slice_xz, origin='lower', #interpolation=interp, 
                cmap = 'hsv',
                extent= (qx_min, qx_max, qz_min, qz_max),
                clim =(plot_slice_xz.min()+0, plot_slice_xz.max()-0 ) ) )
        subp.append( axes[2].imshow(plot_slice_xy, origin='lower', #interpolation=interp, 
                cmap = 'hsv',
                extent= (qx_min, qx_max, qy_min, qy_max),
                clim =(plot_slice_xy.min()+0, plot_slice_xy.max()-0 ) ) )

        # Create a Rectangle patch
        rect = patches.Rectangle((qy_min, qz_pos-qz_delta/2),
                                 qy_max-qy_min, qz_delta,
                                 linewidth=1, edgecolor='g', facecolor='none')
        # Add the patch to the Axes
        axes[0].add_patch(rect)
        # Create a Rectangle patch
        rect = patches.Rectangle((qy_pos-qy_delta/2, qz_min ),
                                 qy_delta, qz_max-qz_min, 
                                 linewidth=1, edgecolor='g', facecolor='none')
        # Add the patch to the Axes
        axes[0].add_patch(rect)

        # Create a Rectangle patch
        rect = patches.Rectangle((qx_min, qz_pos-qz_delta/2),
                                 qx_max-qx_min, qz_delta,
                                 linewidth=1, edgecolor='k', facecolor='none')
        axes[1].add_patch(rect)
        # Create a Rectangle patch
        rect = patches.Rectangle((qx_pos-qx_delta/2, qz_min ),
                                 qx_delta, qz_max-qz_min, 
                                 linewidth=1, edgecolor='g', facecolor='none')
        # Add the patch to the Axes
        axes[1].add_patch(rect)

        # Create a Rectangle patch
        rect = patches.Rectangle((qx_min, qy_pos-qy_delta/2),
                                 qx_max-qx_min, qy_delta,
                                 linewidth=1, edgecolor='b', facecolor='none')
        axes[2].add_patch(rect)
        rect = patches.Rectangle((qx_pos-qx_delta/2, qy_min),
                                 qx_delta, qy_max-qy_min,
                                 linewidth=1, edgecolor='w', facecolor='none')
        axes[2].add_patch(rect)

        for subplot, ax, subtitle, xlabel, ylabel in zip(subp, axes.ravel(), subtitles, xlabels, ylabels):
            fig.colorbar(subplot, ax = ax)
            ax.set(title = subtitle, xlabel = xlabel, ylabel = ylabel)

        fig.tight_layout()  
        if show_plots:
            plt.show()
        else:
            #plt.show(block = False)
            #time.sleep(2)
            plt.close()

        # Done slice plot
        ###################################

        #**********************************
        # sum one dimension from the slices to get 1-D plot.  
        line_x = data_slice_xz[:,qz_ind1:qz_ind2].sum(axis=1)
        line_y = data_slice_yz[:,qz_ind1:qz_ind2].sum(axis=1)
        line_z = data_slice_yz[qy_ind1:qy_ind2,:].sum(axis=0)

        line_x_error = np.sqrt(line_x)
        line_y_error = np.sqrt(line_y)
        line_z_error = np.sqrt(line_z)

        # add another plot for the original line data.
        plot_qx = qx
        plot_qy = qy
        plot_qz = qz

        # line plot
        subtitles = [f'Intensity vs. {x_axis}' , 
                     f'Intensity vs. {y_axis}' ,
                     f'Intensity vs. {z_axis}' ]
        xlabels =[x_axis, y_axis, z_axis]
        ylabels =['Intensity', 'Intensity', 'Intensity']
        
        lineplots(plot_qx, line_x, plot_qy, line_y, plot_qz, line_z, \
            subtitles, xlabels, ylabels)
        #**********************************
 
        #__________________________________
        # normalize to number of points with real data, and remove those points having no data.  
        line_x_stat = np.count_nonzero( array_data0[:, qy_ind1:qy_ind2, qz_ind1:qz_ind2], axis=(1, 2) )
        line_y_stat = np.count_nonzero( array_data0[qx_ind1:qx_ind2, :, qz_ind1:qz_ind2], axis=(0, 2) )
        line_z_stat = np.count_nonzero( array_data0[qx_ind1:qx_ind2, qy_ind1:qy_ind2, :], axis=(0, 1) )

        temp = line_x_stat!=0
        #plot_qx = qx[temp]-qx_pos
        plot_qx = qx[temp]
        #line_x = line_x[temp]/line_x_stat[temp]*np.amax(line_x_stat)
        #line_x_error = line_x_error[temp]/line_x_stat[temp]*np.amax(line_x_stat)
        line_x = line_x[temp]/line_x_stat[temp]
        line_x_error = line_x_error[temp]/line_x_stat[temp]

        temp = line_y_stat!=0
        #plot_qy = qy[temp]-qy_pos
        plot_qy = qy[temp]
        line_y = line_y[temp]/line_y_stat[temp]
        line_y_error = line_y_error[temp]/line_y_stat[temp]

        temp = line_z_stat!=0
        plot_qz = qz[temp]
        line_z = line_z[temp]/line_z_stat[temp]
        line_z_error = line_z_error[temp]/line_z_stat[temp]
        #__________________________________

        #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        # line plot again
        lineplots(plot_qx, line_x, plot_qy, line_y, plot_qz, line_z, \
            subtitles, xlabels, ylabels)
        #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        ###################################
        # output the line-cut to a text file, 
        #  all is for all three lines in one file, separated by a comment line
        headerfmt  = '#%11s%13s%13s\n'    
        outfmt  = ' %10.6f %10.6e %10.6e '
        tmp_str += f'# {x_axis}_pos, {x_axis}_delta = {qx_pos:.4f}, {qx_delta:.4f}\n'
        tmp_str += f'# {y_axis}_pos, {y_axis}_delta = {qy_pos:.4f}, {qy_delta:.4f}\n' 
        tmp_str += f'# {z_axis}_pos, {z_axis}_delta = {qz_pos:.4f}, {qz_delta:.4f}\n' 

        if output_all:
            output_filename = os.path.join(data_dir, f"{filebase}.txt")
            if verbose:
                print(f"All output saved to: {output_filename}")
            # in case dimensions are different, write 3 line-cuts separately
            header  = (x_axis, "I_yzSum", "Err_yzSum")
            output_data= np.column_stack((plot_qx, line_x, line_x_error))
            writedatafile(output_filename=output_filename, write_mode='w+', tmp_str=tmp_str, \
                header=header, headerfmt=headerfmt, \
                output_data=output_data, outfmt=outfmt)

            header  = (y_axis, "I_xzSum", "Err_xzSum")
            output_data= np.column_stack((plot_qy, line_y, line_y_error))
            writedatafile(output_filename=output_filename, write_mode='a+', tmp_str='', \
                header=header, headerfmt=headerfmt, \
                output_data=output_data, outfmt=outfmt)       

            header  = (z_axis, "I_xySum", "Err_xySum")
            output_data= np.column_stack((plot_qz, line_z, line_z_error))
            writedatafile(output_filename=output_filename, write_mode='a+', tmp_str='', \
                header=header, headerfmt=headerfmt, \
                output_data=output_data, outfmt=outfmt)      

        # Save each line cut as a separated file according to the flag settings
        if output_x:
            # do a .xye file for H direction
            output_filename = os.path.join(data_dir, f"{filebase}_{x_axis.lower()}.xye") 
            if verbose:
                print(f"Line profile along y-axis saved to: {output_filename}")
            header  = (x_axis, "Intensity", "Err")
            output_data= np.column_stack((plot_qx, line_x, line_x_error))
            writedatafile(output_filename=output_filename, write_mode='w+', tmp_str=tmp_str, \
                header=header, headerfmt=headerfmt, \
                output_data = output_data, outfmt=outfmt)

        if output_y:
            # do a .xye file for along K direction
            output_filename = os.path.join(data_dir, f"{filebase}_{y_axis.lower()}.xye") 
            if verbose:
                print(f"Line profile along x-axis saved to: {output_filename}")
            header  = (y_axis, "Intensity", "Err")
            output_data= np.column_stack((plot_qy, line_y, line_y_error))
            writedatafile(output_filename=output_filename, write_mode='w+', tmp_str=tmp_str, \
                header=header, headerfmt=headerfmt, \
                output_data = output_data, outfmt=outfmt)

        if output_z:
            # do a .xye file for along L direction
            output_filename = os.path.join(data_dir, f"{filebase}_{z_axis.lower()}.xye") 
            if verbose:
                print(f"Line profile along z-axis saved to: {output_filename}")
            header  = (z_axis, "Intensity", "Err")
            output_data= np.column_stack((plot_qz, line_z, line_z_error))
            writedatafile(output_filename=output_filename, write_mode='w+', tmp_str=tmp_str, \
                header=header, headerfmt=headerfmt, \
                output_data = output_data, outfmt=outfmt)
        ###################################

