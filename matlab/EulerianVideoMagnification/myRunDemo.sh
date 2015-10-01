#! /bin/bash
#configure environment
#As definições a seguir são necessárias para configurar o Matlab Compiler RUNTIME
LD_LIBRARY_PATH=/home/jasf/Programs/MATLAB/MCR/v80/runtime/glnxa64:/home/jasf/Programs/MATLAB/MCR/v80/bin/glnxa64:/home/jasf/Programs/MATLAB/MCR/v80/sys/os/glnxa64:/home/jasf/Programs/MATLAB/MCR/v80/sys/java/jre/glnxa64/jre/lib/amd64/native_threads:/home/jasf/Programs/MATLAB/MCR/v80/sys/java/jre/glnxa64/jre/lib/amd64/server:/home/jasf/Programs/MATLAB/MCR/v80/sys/java/jre/glnxa64/jre/lib/amd64

XAPPLRESDIR=/home/jasf/Programs/MATLAB/MCR/v80/X11/app-defaults

#esse será o path onde o programa procurará pelo MCR
MCR=/home/jasf/Programs/MATLAB/MCR/v80

#adiciona o path onde estão instalados os scripts 
PATH=$PATH:/home/jasf/hot/research/matlab/EulerianVideoMagnification/

inFile='./input/inputTest.avi'
RDIR="output"
f='./run_evm.sh'

#./run_evm.sh MCR_PATH inputVideoFile outputDir samplingRate magType lo hi alpha filterType [magnification_parameters]

#------------------------------------------------------------
# same parameter as in baby2.mp4 with 'ideal' filter
$f $MCR $inFile $RDIR 30 'color' 140/60 160/60 150 'ideal' 1 6

#------------------------------------------------------------
# same parameter as in  camera.mp4, with butterworth filter
$f $MCR $inFile $RDIR 300 'motion' 45 100 150 'butter' 0 20


#------------------------------------------------------------
# same parameter as in subway.mp4, with 'butter'worth filter

$f $MCR $inFile $RDIR 30 'motion' 3.6 6.2 60 'butter' 0.3 90

#------------------------------------------------------------
# same parameter as in  shadow.mp4, with 'motion' 'butter'worth

$f $MCR $inFile $RDIR 30 'motion' 0.5 10 5 'butter' 0 48

#------------------------------------------------------------
# same parameter as in  guitar.mp4, with two 'ideal' filters
# beware, 'ideal' filters require at least 5GB of RAM

# same parameter as in  amplify E
$f $MCR $inFile $RDIR 600 'motion' 72 92 50 'ideal' 0 10

# same parameter as in  amplify A
$f $MCR $inFile $RDIR 600 'motion' 100 120 100 'ideal' 0 10

#------------------------------------------------------------
# same parameter as in  face.mp4, with 'ideal' 'color' filter

$f $MCR $inFile $RDIR 30 'color' 50/60 60/60 50 'ideal' 1 4

#------------------------------------------------------------
# same parameter as in face2.mp4, with 'butter'worth 'motion' filter and 'color'
# same parameter as in 'Motion'
$f $MCR $inFile $RDIR 30 'motion' 0.5 10 20 'butter' 0 80
# same parameter as in 'Color'
$f $MCR $inFile $RDIR 30 'color' 50/60 60/60 50 'ideal' 1 6
