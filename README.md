## Frame Based Event Simulator Rotation Estimation with Global Contrast Maximization

Using ECDS dataset[3] as ground-truth event data.

* The `/experiments` folder include several test experiments, such as:
1. `DISFlow_param`: DIS optical flow parameter selection.
2. `eventSim_param`: Event simulation parameter tuning.
3. `eventSim_EPS`: EPS comparison between simulated events and ground-truth.
4. `eventSim_check`: Visualization of interpolated directions in event simulation.

* `eventFromDataset`: Performs rotation estimation with global contrast maximization using dataset images and ground-truth timestamps.
* `eventFromVideo`: Performs rotation estimation using video frames and their timestamps.
* `eventFromStream`: Simulates events from a live camera stream (without performing rotation estimation).




## reference:
[1] A. Ziegler, D. Teigland, J. Tebbe, T. Gossard and A. Zell, "Real-time event simulation with frame-based cameras," 2023 IEEE International Conference on Robotics and Automation (ICRA), London, United Kingdom, 2023, pp. 11669-11675, doi: 10.1109/ICRA48891.2023.10160654. https://github.com/cogsys-tuebingen/event_simulator.git  
[2] H. Kim and H. J. Kim, "Real-Time Rotational Motion Estimation With Contrast Maximization Over Globally Aligned Events," in IEEE Robotics and Automation Letters, vol. 6, no. 3, pp. 6016-6023, July 2021, doi: 10.1109/LRA.2021.3088793. https://github.com/Haram-kim/Globally_Aligned_Events.git  
[3] Mueggler E, Rebecq H, Gallego G, Delbruck T, Scaramuzza D. The event-camera dataset and simulator: Event-based data for pose estimation, visual odometry, and SLAM. The International Journal of Robotics Research. 2017;36(2):142-149. doi:10.1177/0278364917691115 https://rpg.ifi.uzh.ch/davis_data.html  
