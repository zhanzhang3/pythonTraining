
# playing with read a single scan out of the spec file and save it to a new text file
#   utilizing the spec2nexus package

import os
import numpy as np
from spec2nexus import spec

# flags
# write output to data file(s)
#  write_file == 1 : copy the single scan with all the comments to a separate file
#  write_file == 2 : copy the data with scanCmd, scanTime, and column label
#  write_file == 3 : copy the specific columns from the data to a separate file.
#  write_file == 99: special case, write the data file with one column label/values replaced.
write_file = 99
# verbose mode on/off
verbose = 1

# SPEC file path, absolute, not the format 
#  For Linux system
#specdatapath = os.path.join("/home/33id/", "data/hui/20180607/ternaryalloy/")
#  For Windows system, Note the double \\
#specdatapath = os.path.join("Z:\\", "data\\hui\\20180607\\ternaryalloy\\")
specdatapath = os.path.join("C:\\", "work\\Users\\YCao\LSFO\\")
sampleName = "LSFO_001_Cryo"
specdatafile = os.path.join(specdatapath, "%s.spec") % sampleName
specData = spec.SpecDataFile(specdatafile)

# list of scans need to integrate, for example:
#scans = [12, ]
scans = list( range(28,37) ) 
# selected column label or positioner label.
col_label = ["TwoTheta", "L", "seconds", "I00", "trans", "corrdet", "Energy", "I0", "imroi1", "ksamx", "ksamy"]

for scannum in scans:
  scan = specData.getScan(scannum)
  scan.interpret()
  output_scann = scan.scanNum
  data_pnts = len( scan.data[ scan.L[0] ] )
  output_data = []

  if write_file >= 1:
    #output_filename = os.path.join(specdatapath, "analysis_runtime", sampleName, "%s_S%04d.txt") % (sampleName, scannum)
    #output_filename = os.path.join("D:\\UniWork\\Users\\HuiJie\\", sampleName, "%s_S%04d.dat") % (sampleName, scannum)
    output_filename = os.path.join(specdatapath, "%s_S%04d.dat") % (sampleName, scannum)

    # generate the directory if not exist yet.
    if not os.path.exists(os.path.dirname(output_filename)):
      try:
        os.makedirs(os.path.dirname(output_filename))
      except OSError as exc: # Guard against race condition
        if exc.errno != errno.EEXIST:
          raise

    outfile = open(output_filename, 'w+')
    #outfile.write('## SPECfile = %s\n' % specdatafile)

  if verbose: 
    print("Scan# = ", scannum, ": ", scan.scanCmd)
  
  if write_file == 1:
    outfile.write(scan.raw)
  elif write_file == 2:
    outfile.write("#S %s  %s\n" % (output_scann, scan.scanCmd) )
    outfile.write("#D %s\n" % scan.date)
    outfile.write("#L ")
    for label in scan.L:
      outfile.write("%s  " % label)
    outfile.write("\n")
    for line in scan.data_lines:
      outfile.write("%s\n" % line)
  elif write_file == 3:
    outfile.write("#S %s  %s\n" % (output_scann, scan.scanCmd) )
    outfile.write("#D %s\n" % scan.date)
    outfile.write("#L ")
    outfmt = "   "
    for label in col_label:
      if label in scan.L:
        outfile.write("%14s  " % label)
        output_data.append( np.array(scan.data[label]) )
        outfmt = outfmt + "%15.4f "
      elif label in scan.positioner:
        outfile.write("%14s  " % label)
        output_data.append(scan.positioner[label]*np.ones(data_pnts))
        outfmt = outfmt + "%15.4f "
    outfile.write("\n")
    output_data = np.transpose(output_data)
    np.savetxt(outfile, output_data, fmt = outfmt)
  elif write_file == 99:
    # replace the data column
    if scannum in [29, 32, 35]:
        scan.data["Temp_sa"]=[0.000037]*len(scan.data["Temp_sa"])
    else:
        scan.data["Temp_sa"]=[0.0234]*len(scan.data["Temp_sa"])
    outfmt = ""
    for label in scan.L:
        output_data.append(np.array(scan.data[label]) )
        outfmt += "%.6g "
    # replacee the label
    ind = scan.L.index("Temp_sa")
    scan.L[ind] = "trans"
    out_label = f"#L " + "  ".join(_str for _str in scan.L) +"\n"
    outfile.write(out_label)
    output_data = np.transpose(output_data)
    np.savetxt(outfile, output_data, fmt = outfmt)
    
  if write_file >= 1:
    outfile.close()
    if verbose:
      print("Data written to " + output_filename + ".")
