import cv2
import sys
sys.path.append('../')
sys.path.append('../flowExperiment')

import jasf
from jasf import jasf_cv
import  flowUtil
import numpy as np

from flowSubtraction import FlowDiffComputer



if __name__ == "__main__":
    cam = cv2.VideoCapture('./input/inputTest-butter-from-0.5-to-10-alpha-20-lambda_c-80-chromAtn-0.avi')

    #get first frame so we set width and whigth of the flow computer
    ret, frame = cam.read()
    flowComputer = flowUtil.FlowComputer()
    width, height = frame.shape[1], frame.shape[0]
    grid = flowUtil.getGrid(0,0, width-1, height-1, 1,1) 
    flowComputer.setGrid(grid)
    #initialize flowComputer with first frame(at this point, we have only one image and no flow is computer)
    flowComputer.apply(frame)

    #initialize flow diff computer
    diffComputer = FlowDiffComputer(flowComputer)


    allBlack = np.zeros((height, width), dtype=np.uint8)


    #find optimal size for fourrier transform
    rows,cols,k = frame.shape
    nrows = cv2.getOptimalDFTSize(rows)
    ncols = cv2.getOptimalDFTSize(cols)


    modeDemoSpectrum = False
    displayResults = True
    writeMode = True
    writeDFT = True
    writeFlowDiff=True

    flowDiffWriter = object()
    DFTWriter = object()

    if writeFlowDiff:
        #get video writer to write flow diff output
        flowDiff_h, flowDiff_w = allBlack.shape
        videoFrameRate = cam.get(cv2.CAP_PROP_FPS)
        flowDiffWriter = cv2.VideoWriter('./output/flowDiff.avi', cv2.VideoWriter_fourcc(*'XVID'), videoFrameRate,
                (flowDiff_w, flowDiff_h), isColor = False)

    if writeDFT:
        #get video writer to write DFT of flowDiff
        DFTWriter = cv2.VideoWriter('./output/flowDiff_DFT.avi', cv2.VideoWriter_fourcc(*'XVID'), videoFrameRate,
                (ncols, nrows), isColor=False)

    if displayResults:
        #initialize windows to be used
        jasf.cv.getManyWindows(['input',  'flowDiffMag','flowDiffMagSpectrum'])


    #run spectral analysis in only one image gotten from the video
    if modeDemoSpectrum:
        gray = jasf_cv.convertBGR2Gray(frame)
        img = gray.copy()

        dft = cv2.dft(np.float32(img),flags = cv2.DFT_COMPLEX_OUTPUT)
        dft_shift = np.fft.fftshift(dft)
     
        magnitude_spectrum = 20*np.log(cv2.magnitude(dft_shift[:,:,0],dft_shift[:,:,1]))

        maxM =  magnitude_spectrum.max()
        minM =  magnitude_spectrum.min()

        magnitude_spectrum = (magnitude_spectrum - minM)*255.0/maxM

        magnitude_spectrum = magnitude_spectrum.astype(np.uint8)

        print magnitude_spectrum.max()
        print magnitude_spectrum.min()

        #show input frame and frame with flow arrows drawn
        cv2.imshow('input', gray)
        cv2.imshow('flowDiffMagSpectrum', magnitude_spectrum)

        ch = jasf.cv.waitKey(0)

        quit()

    else:
        counter = 0
        while True:
            #quit if 'q' is pressed
            ch = jasf.cv.waitKey(5)
            if ch == ord('q'):
                break

            #get new frame and stop if we're not able to read
            ret, frame = cam.read()
            counter += 1
            if ret == False:
                break
            
            #get difference in flow from this to the previous flow
            flowP, flowMag = diffComputer.apply(frame)

            #paint a black frame with the flow mag in the points where there was flow to compare
            output = allBlack.copy()
            output[flowP[:,1], flowP[:,0]] = 10*flowMag

            img = output.copy()

            #altera image para tamanho optimo
            nimg = np.zeros((nrows,ncols))
            nimg[:rows,:cols] = img

            dft = cv2.dft(np.float32(nimg),flags = cv2.DFT_COMPLEX_OUTPUT)
            dft_shift = np.fft.fftshift(dft)
         
            magnitude_spectrum = 20*np.log(cv2.magnitude(dft_shift[:,:,0],dft_shift[:,:,1]) + 0.00001)

            maxM =  magnitude_spectrum.max()
            minM =  magnitude_spectrum.min()


            magnitude_spectrum = np.interp(magnitude_spectrum, [minM, maxM], [0, 255])
            #magnitude_spectrum = (magnitude_spectrum - minM)*255.0/maxM

            magnitude_spectrum = magnitude_spectrum.astype(np.uint8)

            #show input frame and frame with flow arrows drawn

            if writeMode:
                if writeDFT:
                    DFTWriter.write(magnitude_spectrum)
                    cv2.imwrite('./output/DFT/mag_%04d.jpg'%counter, magnitude_spectrum)
                if writeFlowDiff:
                    flowDiffWriter.write(output)
                    cv2.imwrite('./output/flowDiff/diff_%04d.jpg'%counter, output)
            if displayResults:
                cv2.imshow('input', frame)
                cv2.imshow('flowDiffMag', output)
                cv2.imshow('flowDiffMagSpectrum', magnitude_spectrum)
        if writeDFT:
            DFTWriter.release()
        if writeFlowDiff:
            flowDiffWriter.release()

    cv2.destroyAllWindows()
    cam.release()
