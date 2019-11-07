# Robot Tracking
Code and documentation for the task of tracking and localizing a robot on a grid track.
(We were the only team who used Computer Vision to perform the task.)

--------------------------------
### Goal
The goal of this project is to track the trajectory of a [Lego Mindstorm NXT](https://robots.ieee.org/robots/mindstorms/) on a custom track and measure the errors in the final state over 20 rollouts.

--------------------------------
### Setup

To track the robot on the track, aruco markers with different IDs are used.
Two markers are used, one on the front and the other on the back, to precisely know the orientation of the robot. (This can also be done using a single marker)   

<img src="/images/collage.jpg" width="400">

Markers of different IDs are added to the 4 edges of the map.
The markers are aligned such that the first point of the marker that is detected directly coincides with the edge of the track. 

<img src="/images/map_small.png" width="400">

Camera is used to take a video or a photo of the robot at its initial and the final positions. 

<img src="/images/apparatus_front_view.png" width="300">

--------------------------------
### Procedure
The complete procedure and the setup is given here.

<img src="/images/pipeline.png" width="500">

--------------------------------
### Results

<img src="/images/all_images.jpg" width="500">


