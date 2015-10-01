import sys
import pathlib
import subprocess
import os

inputDir = './input'
inputFiles = [f for f in pathlib.Path(inputDir).iterdir() if f.is_file() and f.suffix == '.avi']

command = './singleRun.sh'
done = 0
for f in inputFiles:
    fileName = f.as_posix()


    print 'running on file', f.name
    subprocess.call([command, fileName, 'output'])

    done += 1
    print len(inputFiles) - done, 'to go'
