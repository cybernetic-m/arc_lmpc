# Autonomous Car Racing control via LMPC
Learning Model Predictive Control for Autonomous Car Racing (ARC). 

1. **Planning and Modelling**: In the planning phase we have used two different techniques (**Polynomial Interpolation** and **Beziér Curves**) to plan the trajectory. We have created three different tracks: **Water Fall Track**, **Tornado Circuit** and **Monza Circuit**. In the modelling phase we have model the ARC using the **kinematic model of the bicycle**.

2. **Control**: we have used two different techniques of **LMPC** control described in [1] (**Q Table**) and [3] (**Convex Hull**).

# Installation
1. Clone the repository:  
 ```sh 
 git clone "https://github.com/cybernetic-m/arc_lmpc.git"
 cd arc_lmpc
 ```

2. Install the dependencies:  
```sh 
pip install -r requirements.txt
```

# Project Structure 

```sh 
arc_lmpc
├── LICENSE
├── README.md
├── controller 
│   ├── lmpc_control.py => LMPC controller with convex hull (Reference [3])
│   ├── qlmpc_control.py => LMPC controller with Q Table (Reference [1])
│   └── trivial_control.py => Path following controller at constant velocity 
├── gif_images
│   ├── gif_race.gif
│   ├── lap_gif.gif
│   ├── track1traj.png
│   ├── track2traj.png
│   └── track3traj.png
├── main.py => Main to execute the LMPC with convex hull
├── main_q.py => Main to execute the LMPC with Q Table
├── requirements.txt
└── utils
    ├── animation.py => File to generate the best lap animation and the race animations
    ├── images
    │   ├── car.png
    │   └── start_flag.png
    ├── model.py => File that contain the kinematic model of the robot and all important quantities
    ├── track.py => File that contain the track with polynomial interpolation and its definitions
    └── track_bez.py =>File that contain the track with Bezier Curves and its definitions 
 ```

# Usage
To use the project you can run:
1. LMPC Control with Convex Hull
```sh 
python3 main.py
```
2. LMPC Control with Q Table
```sh 
python3 main_q.py
```

# Race
In the race it's possible to see that the LMPC with convex hull at lap=30 perform better with respect to the previous iterations. 
![Demo](gif_images/gif_race.gif)

# Best Lap
In best lap we're plotting the lap=30 
![Demo](gif_images/lap_gif.gif)

# Water Fall Track
In this plots it's possible to see the evolution of the trajectories in the Water Fall Track.
![Alt Text](gif_images/track1traj.png)

# Tornado Circuit
In this plots it's possible to see the evolution of the trajectories in the Tornado Circuit.
![Alt Text](gif_images/track2traj.png)

# Monza Circuit
In this plots it's possible to see the evolution of the trajectories in the real Monza Circuit.
![Alt Text](gif_images/track3traj.png)

# References
[1]. [U. Rosolia and F. Borrelli: Learning Model Predictive Control for Iterative
Tasks. A Data-Driven Control Framework](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8039204).

[2]. [H. Xue et al. : Learning Model Predictive Control with Error Dynamics Regression for Autonomous Racing](https://arxiv.org/pdf/2309.10716).

[3].  [U. Rosolia and F. Borrelli : Learning How to Autonomously Race a Car: A Predictive Control Approach](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8896988).



