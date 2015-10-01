#!/bin/bash
# ----------------------------------------
# Documentation
: <<'STARTDOC'
EVM Script to Run
Assumes that the script is run at the root directory of
the downloaded source file of Motion Amplification

Note that for this script to run
the file run_evm.sh and evm binary
  (i) should be located in the same folder,
  (ii) and that folder should be included in system path
      [if you want to do (ii) automatically
      A. cd ~
      B. <some_editor> .bashrc
      C. at the end of file, add this line
            export PATH=$PATH:<directory_of_executable>
      D. Reload the terminal

MCR variable value should be set using the same method
as described for (ii), add the following at the end of .bashrc
      export MCR='/path/to/MCR/dir/v<version_num>/'

Alternately, in terminal, type MCR='path/to/mcr/directory
Then run this script

[In summary, ~/.bashrc should have two lines of code added,
one line for including bash directory to PATH
another for the MCR]


rootdir\ (this is the directory where evm.sh, evm, and reproduceResults shouldbe)
    data\ (this directory contains the sample videos)
    Resultsv<MCRVerNum>\ (this directory contains generated results)
       ex: Resultsv80\, if you have 2012b linux MCR

STARTDOC

#configure environment
LD_LIBRARY_PATH=/home/jasf/Programs/MATLAB/MCR/v80/runtime/glnxa64:/home/jasf/Programs/MATLAB/MCR/v80/bin/glnxa64:/home/jasf/Programs/MATLAB/MCR/v80/sys/os/glnxa64:/home/jasf/Programs/MATLAB/MCR/v80/sys/java/jre/glnxa64/jre/lib/amd64/native_threads:/home/jasf/Programs/MATLAB/MCR/v80/sys/java/jre/glnxa64/jre/lib/amd64/server:/home/jasf/Programs/MATLAB/MCR/v80/sys/java/jre/glnxa64/jre/lib/amd64

XAPPLRESDIR=/home/jasf/Programs/MATLAB/MCR/v80/X11/app-defaults

MCR=/home/jasf/Programs/MATLAB/MCR/v80




# ----------------------------------------
# Shell Function

print_fun() {
    # printString should be re-initalized after each inFile change
    echo
    printString='echo "Processing $inFile" && echo'
    eval $printString    
}

# ----------------------------------------
# Actual script

f='./run_evm.sh'

# S=Source, R=Results, MCR=Matlab Compiler Runtime
SDIR='./data'
RDIR="ResultsSIGGRAPH2012"
mkdir $RDIR
if [[ -z "$MCR" ]]; then
    echo "MCR directory not specified! To specify, type: MCR='path/to/MATLAB_Compiler_Runtime/v80'" 
    echo "Also you MUST export MCR variable in command line"

    echo "Using default MCR installation path"
    unixOS=$(uname -s)
    macOS="Darwin"
    if [[ "$unixOS" = "$macOS" ]]; then
	echo "OS is Mac: default path set to"
	echo "/Applications/MATLAB/MATLAB_Compiler_Runtime/v80"
	MCR="/Applications/MATLAB/MATLAB_Compiler_Runtime/v80"
    else
	echo "OS is UNIX: default path set to"
	echo "/usr/local/MATLAB/MATLAB_Compiler_Runtime/v80"
	MCR='/usr/local/MATLAB/MATLAB_Compiler_Runtime/v80'
    fi
fi

#------------------------------------------------------------
# baby2.mp4 with 'ideal' filter
inFile="$SDIR/baby2.mp4"
print_fun

$f $MCR $inFile $RDIR 30 'color' 140/60 160/60 150 'ideal' 1 6

#------------------------------------------------------------
# camera.mp4, with butterworth filter
inFile="$SDIR/camera.mp4"
print_fun

$f $MCR $inFile $RDIR 300 'motion' 45 100 150 'butter' 0 20


#------------------------------------------------------------
#subway.mp4, with 'butter'worth filter
inFile="$SDIR/subway.mp4"
print_fun

$f $MCR $inFile $RDIR 30 'motion' 3.6 6.2 60 'butter' 0.3 90

#------------------------------------------------------------
# shadow.mp4, with 'motion' 'butter'worth
inFile="$SDIR/shadow.mp4"
print_fun

$f $MCR $inFile $RDIR 30 'motion' 0.5 10 5 'butter' 0 48

#------------------------------------------------------------
# guitar.mp4, with two 'ideal' filters
# beware, 'ideal' filters require at least 5GB of RAM
inFile="$SDIR/guitar.mp4"
print_fun

# amplify E
$f $MCR $inFile $RDIR 600 'motion' 72 92 50 'ideal' 0 10

# amplify A
$f $MCR $inFile $RDIR 600 'motion' 100 120 100 'ideal' 0 10

#------------------------------------------------------------
# face.mp4, with 'ideal' 'color' filter
inFile="$SDIR/face.mp4"
print_fun

$f $MCR $inFile $RDIR 30 'color' 50/60 60/60 50 'ideal' 1 4

#------------------------------------------------------------
#face2.mp4, with 'butter'worth 'motion' filter and 'color'
inFile="$SDIR/face2.mp4"
print_fun

#'Motion'
$f $MCR $inFile $RDIR 30 'motion' 0.5 10 20 'butter' 0 80

#'Color'
$f $MCR $inFile $RDIR 30 'color' 50/60 60/60 50 'ideal' 1 6
