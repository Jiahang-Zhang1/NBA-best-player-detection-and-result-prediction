# NBA-best-player-detection-and-result-prediction
NBA best player detection and result prediction

Contributors：-1. Name: Jiahang Zhang, NYU Student ID: jz7581
              -2. Name: Minghao Wang, NYU Student ID: 

# NBA Statistics ML Project

This repository contains a machine-learning project (Project A: **NBA statistics data**) with two goals:

1) **Outstanding player mining** via outlier detection + clustering  
2) **Game outcome prediction** via supervised classification  
:contentReference[oaicite:1]{index=1}

---

## 1. Project Overview

NBA provides rich historical statistics for players and teams. This project applies machine learning to:
- detect **outlier players** (both exceptional and poor performers) and then identify the **truly outstanding** group, and
- predict **win/loss outcomes** from match-level and team-level features.  
:contentReference[oaicite:2]{index=2}

---

## 2. Task A — Outstanding Player Detection (Unsupervised)

### Problem Formulation
Selecting “outstanding players” is treated as an **unsupervised clustering problem** because the dataset has **no labels** for “best player.”  
The pipeline first detects **outliers**, then clusters them to separate **excellent** vs **poor** players.  
:contentReference[oaicite:3]{index=3}

### Data & Features
The project uses **career-level** player statistics (regular season and playoffs) because multi-year averages better reflect true ability than a single season. :contentReference[oaicite:4]{index=4}

Engineered features include:
- minutes per game, points per game, turnovers per game, rebounds per game, assists per game,
- blocks per game, field-goal %, free-throw %  
:contentReference[oaicite:5]{index=5}

Important data handling:
- remove players with very low playing time (e.g., minutes/game < 3) to reduce noise, :contentReference[oaicite:6]{index=6}
- account for missing historical stats: before ~1970, steals/blocks/turnovers were not recorded, so the dataset is split based on whether these fields are all zero to improve robustness. :contentReference[oaicite:7]{index=7}

### Methods
**Outlier detection**
- LOF (Local Outlier Factor) and Isolation Forest are both used to detect outliers. :contentReference[oaicite:8]{index=8}

**Outlier clustering**
- Outliers contain both “very good” and “very bad” players, so **K-Means** is applied to outliers.
- To represent overall excellence (position-agnostic), clustering focuses on **points/game, minutes/game, and shooting %**. :contentReference[oaicite:9]{index=9}
- The number of clusters is chosen by the **elbow method** (K=5). :contentReference[oaicite:10]{index=10}
- The “outstanding” cluster is selected as the cluster with the **highest average points/game**, and results from regular season & playoffs are merged; the intersection identifies players who are exceptional in both contexts. :contentReference[oaicite:11]{index=11}

### Notes on Results (from the report)
- With the same outlier ratio, Isolation Forest finds **more** outliers than LOF (stricter definition) and its outstanding-player set includes LOF’s results. :contentReference[oaicite:12]{index=12}
- LOF is **faster**, while Isolation Forest is **more reasonable** for this project according to the report. :contentReference[oaicite:13]{index=13}
- The outlier ratio parameter was tested at 0.1 and 0.2 to adjust strictness. :contentReference[oaicite:14]{index=14}

---

## 3. Task B — Game Outcome Prediction (Supervised)

### Data Source
Match data (preseason + regular season + playoffs) for **2015–2017** seasons is collected from basketball-reference.com, along with team capability/ranking metrics. :contentReference[oaicite:15]{index=15}

### Feature Engineering (high level)
From match results and schedules, features include:
- home/away info and final scores → win/loss label (HW = home win), net score margin, and points, :contentReference[oaicite:16]{index=16}
- recent form features: each team’s **last 3 games** win/loss indicators (hm1–hm3, vm1–vm3), :contentReference[oaicite:17]{index=17}
- weekly aggregated averages from cumulative net score / weekly stats to reflect team state. :contentReference[oaicite:18]{index=18}
- team ranking features: MOV/A, ORtg/A, DRtg/A, NRtg/A are added into the match dataset. :contentReference[oaicite:19]{index=19}

Data processing:
- normalize/standardize features to improve convergence and accuracy, :contentReference[oaicite:20]{index=20}
- remove highly correlated features (> 0.9) to reduce redundant dimensions (examples include (vMOV,vNRtg), (hMOV,hNRtg), (hSco_avew,vSco_avew)). :contentReference[oaicite:21]{index=21}

### Models
- **SVM** and **Logistic Regression** are trained and compared. :contentReference[oaicite:22]{index=22}

### Notes on Results (from the report)
- SVM and Logistic Regression achieve similar accuracy; Logistic Regression runs faster. :contentReference[oaicite:23]{index=23}
- Reported prediction accuracy reaches about **70%**. :contentReference[oaicite:24]{index=24}

---

## 4. How to Run (example)

### Environment
- Python 3.x
- Recommended packages: `numpy`, `pandas`, `scikit-learn`, `matplotlib`

```bash
pip install numpy pandas scikit-learn matplotlib



