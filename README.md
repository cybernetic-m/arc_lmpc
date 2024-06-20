# Autonomous Car Racing control via LMPC
Learning Model Predictive Control for Autonomous Car Racing (ARC) based on method explained in

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
│   ├── lmpc_control.py
│   ├── qlmpc_control.py
│   └── trivial_control.py
├── gif_images
│   ├── gif_race.gif
│   ├── lap_gif.gif
│   ├── track1traj.png
│   ├── track2traj.png
│   └── track3traj.png
├── main.py
├── main_q.py
├── requirements.txt
└── utils
    ├── animation.py
    ├── images
    │   ├── car.png
    │   └── start_flag.png
    ├── model.py
    ├── track.py
    └── track_bez.py
 ```

# Usage
1. 

# Race
![Demo](gif_images/gif_race.gif)

# Best Lap
![Demo](gif_images/lap_gif.gif)

# Water Fall Track
![Alt Text](gif_images/track1traj.png)

# Tornado Circuit
![Alt Text](gif_images/track2traj.png)

# Monza Circuit
![Alt Text](gif_images/track3traj.png)

# References
[1]. [U. Rosolia and F. Borrelli: Learning Model Predictive Control for Iterative
Tasks. A Data-Driven Control Framework](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8039204).

[2]. [H. Xue et al. : Learning Model Predictive Control with Error Dynamics Regression for Autonomous Racing](https://arxiv.org/pdf/2309.10716).

[3].  [U. Rosolia and F. Borrelli : Learning How to Autonomously Race a Car: A Predictive Control Approach](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8896988).



