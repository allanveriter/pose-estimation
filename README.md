<!--
  README.md - 3D Pose Estimation Tool
-->

<div align="center">
  <h1>3D Pose Estimation Tool</h1>
  <h3>Developed during a student research job at the Robotics Lab, Vrije Universiteit Brussel</h3>
</div>

---

## 📄 Overview

<img src="images/aruco_detection_scene.jpg" width="400" align="right" />

This repository contains a **Python-based 3D pose estimation tool** developed to support bachelor and master theses, as well as research projects, at the Vrije Universiteit Brussel (VUB).  

A **formal research paper** documenting the full algorithm, mathematical derivation, and results is included in this repository as [`paper.pdf`](paper.pdf).

The software estimates the 3D **position and orientation** of objects using multiple cameras and two types of fiducial markers:

- **ArUco markers** – robust but slower detection  
- **Colored dots** – fast detection, requires controlled environment  

---

## 🖼️ Images & Examples

<figure>
  <img src="images/colored_dots_drone.jpg" width="350"/>
  <figcaption>Example setup of a drone frame with four colored dots for detection.</figcaption>
</figure>

<figure>
  <img src="images/aruco_pose_estimation_screenshot.jpg" width="450"/>
  <figcaption>Screenshot of ArUco pose estimation in progress. Four cameras are tracking the marker; pose and error values are printed for iterative refinement.</figcaption>
</figure>

---
---

## ⚙️ Features

- Real-time 3D pose estimation  
- Automatic selection of best two cameras for tracking  
- Iterative error minimization and uncertainty estimation  
- Configurable for different marker types, sizes, and camera setups  
- Includes intrinsic and extrinsic camera calibration scripts  

---

## 🛠️ Technologies & Libraries

- ![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)  
- ![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)  
- ![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)  

---
