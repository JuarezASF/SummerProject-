#Mouse Seizure Detection

## Objective

We aim to develop a method to detect mice seizure events based on low quality video recordings.

## Introduction

For some diseases, epilepsy being one of them, it is usual to induce the same disease into mice and study 
its behavior on this simpler form of life. In the case of epilepsy one is concerned about the duration
of every event and the frequency of events. It's useful to learn how those two parameters evolve as the
mouse gets older and older.

The input to our problem is a video containing recordings of two mice, and we want to train a classifier
that is able to tell whether there is a seizure event happing at each instant(or window) of time.

I started working on this project during the Summer of 2015 under the supervision of Dr. Devika Subramanina. By this time 
I was studying at Rice University on a exchange year program. I've now returned to Universidade de Brasília
and I keep working the project, but now under the joint supervision of Devika and Dr. Flávio B. Vidal.

## What we've got so far.

We have covered a lot of ground, and one thing we've learned is that this is no easy task. Low quality videos really
limit the usability of many computer vision tasks. I divide the path we've covered into the following sub-items:

* Tracking of the mouse

    This is currently being done on a color based tracking. The mouse is quite white and we can use it, but there are other
white objects in the background. The mouse, however, is the only one inside the cage. Thus, first we segment the 
noisy texture of the cage, and then apply color based threshold inside it. We use Otsu threshold to find the optimal
value. Of course, there are some morphological operations involved to clean the result and some decision taking in order
to decide which contour corresponds to the mouse.

* Trying to figure out what is the mouse doing

    The basic assumption is that there is a frequency component that is characteristic of the seizure event. But
    the frequency component of what? We've trying looking at the optical flow. But again, optical flow at which point?
    We've tried looking at the frequency spectrum of the mean optical flow of the points belonging to the mouse. 
    This turned out to be not a stable feature.


    Right now, we're taking a different approach. We compute the magnitude of the flow difference at each point. The
    flow difference is the difference between the flow vectors at consecutive frames.
