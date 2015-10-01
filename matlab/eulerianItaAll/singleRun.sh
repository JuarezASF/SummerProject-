#! /bin/bash
#configure environment
LD_LIBRARY_PATH=/home/jasf/Programs/MATLAB/MCR/v80/runtime/glnxa64:/home/jasf/Programs/MATLAB/MCR/v80/bin/glnxa64:/home/jasf/Programs/MATLAB/MCR/v80/sys/os/glnxa64:/home/jasf/Programs/MATLAB/MCR/v80/sys/java/jre/glnxa64/jre/lib/amd64/native_threads:/home/jasf/Programs/MATLAB/MCR/v80/sys/java/jre/glnxa64/jre/lib/amd64/server:/home/jasf/Programs/MATLAB/MCR/v80/sys/java/jre/glnxa64/jre/lib/amd64

XAPPLRESDIR=/home/jasf/Programs/MATLAB/MCR/v80/X11/app-defaults

MCR=/home/jasf/Programs/MATLAB/MCR/v80

PATH=$PATH:/home/jasf/hot/research/matlab/EulerianVideoMagnification/

#primeiro argumento: arquivo video de entrada
#segundo argumento: diretorio de saida
inFile=$1
RDIR=$2
f='../EulerianVideoMagnification/run_evm.sh'

#------------------------------------------------------------
# same parameter as in face2.mp4, with 'butter'worth 'motion' filter and 'color'
# same parameter as in 'Motion'
$f $MCR $inFile $RDIR 30 'motion' 0.5 10 20 'butter' 0 80
