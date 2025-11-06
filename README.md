<!--
  README.md - Pose Estimation Tool for Robotics Lab
-->

<div align="center">
  <h1>3D Pose Estimation Tool</h1>
  <h3>Developed during a student research job at the Robotics Lab, Vrije Universiteit Brussel</h3>

  <!-- Main project preview image -->
  <img src="images/aruco_detection_scene.jpg" width="450" alt="Aruco marker detection in the lab" />
</div>

---

## 📖 Project Overview

This repository contains a **Python-based 3D pose estimation software** designed to support bachelor, master theses, and research projects at the Vrije Universiteit Brussel (VUB). The project was developed during **August 2024 and August 2025** as a student initiative under the supervision of **Prof. Dr. Jan Lemeire**.

The software estimates the 3D **position and orientation (pose)** of objects using standard cameras and two types of fiducial markers:

- **ArUco markers** (robust, but slower detection)
- **Colored dots** (fast, requires controlled environment)

All code is fully implemented in Python using **NumPy** for matrix computations and **OpenCV** for computer vision tasks.  

<div align="center">
  <img src="images/colored_dots_drone.jpg" width="350" alt="Example of colored dots setup on drone frame" />
  <img src="images/aruco_pose_estimation_screenshot.jpg" width="450" alt="Screenshot of ArUco pose estimation algorithm running" />
</div>

---

## 🗂️ Repository Structure
### Repository Structure

Each marker type has its dedicated folder:
project_root/
├── aruco/                # ArUco marker detection
│   ├── 2D/               # 2D localization scripts
│   └── 3D/               # 3D pose estimation scripts
├── colored_dots/         # Colored dots detection
│   ├── 2D/               # 2D localization scripts
│   └── 3D/               # 3D pose estimation scripts
├── calibration/          # Intrinsic & extrinsic camera calibration
│   ├── chessboard_images/ 
│   └── calibration_scripts/
├── paper/                # PDF and LaTeX source of the formal research paper
└── README.md

### Calibration

- **Intrinsic calibration**: Using chessboard patterns to determine focal lengths, optical center, and distortion parameters.
- **Extrinsic calibration**: Defines the position and orientation of each camera in the world frame.

---

## ⚙️ Features

- Detects object markers and calculates 3D pose in real time
- Supports multiple cameras, selecting the best two for pose estimation
- Implements **iterative algorithms** for error minimization
- Estimates pose uncertainty
- Fully configurable for different marker sizes, colors, and camera setups

---

## 📄 Research Paper

A formal research paper accompanies this project, detailing:

- Mathematical derivation of the pose estimation algorithm
- Comparison between ArUco and colored dot methods
- Multi-camera integration and optimization
- Uncertainty estimation

Preview of the paper:  

<div align="center">
  <img src="images/paper_preview.png" width="400" alt="Preview of LaTeX paper" />
</div>

The full paper is available in the `paper/` folder as a PDF.  

---

## 🛠️ Technologies & Libraries

- ![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
- ![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)
- ![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)
- Image processing, linear algebra, and camera calibration tools included

---

## 📷 Example Usage

```bash
# Run ArUco 3D pose estimation
python aruco/3D/pose_estimation.py --camera-config calibration/camera_params.yaml

# Run colored dots 3D pose estimation
python colored_dots/3D/pose_estimation.py --camera-config calibration/camera_params.yaml

Each marker type has its dedicated folder:

