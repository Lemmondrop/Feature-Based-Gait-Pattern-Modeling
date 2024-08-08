# Feature-Based-Gait-Pattern-Modeling

Analyze gait by collecting real-time current consumption as a person walks on a treadmill.
---
## Overview

This project utilized an adapter sensor that can collect real-time current consumption on a treadmill to collect data when the subject walks on the treadmill, confirming that unlike existing treadmill walking methods, it is possible to analyze gait by simply attaching a sensor without modifying the treadmill.

project has been published in SCIE(MDPI, Electronics)

* __DOI: https://doi.org/10.3390/electronics12204201__

## Notice

Note that the code in this git is __not fully written, Different users will have different file paths and image filenames, so you'll need to modify those before using it__. 

and The adapter sensor used in this project to analyze gait was collected using Elegaiter provided by CiKLux, a Korean company, and if you want to use your own adapter sensor to analyze gait, the results may be slightly different from this study.

## Manual

It consists of 3 python files in total, and the code is python. ( all of files have been converted ipynb to py )

The order of use of the files is
- [Raw Data Preprocessing](https://github.com/Lemmondrop/Feature-Based-Gait-Pattern-Modeling/blob/main/Raw%20Data%20Preprocessing.py)

- [Gait Points Extraction](https://github.com/Lemmondrop/Feature-Based-Gait-Pattern-Modeling/blob/main/Gait%20Points%20Extraction.py)

- [Data Analysis(statistic analysis)](https://github.com/Lemmondrop/Feature-Based-Gait-Pattern-Modeling/blob/main/Data%20Analysis(statistic%20analysis).py)

If you went through the above files in order

The Raw Data Preprocessing code runs through the Flow process shown in the image below, resulting in two data files: an enveloped gait data Excel file and a gait data Excel file that has been smoothed by applying a Kalman-Filter.

![image](https://github.com/user-attachments/assets/fb01a6e0-7e5c-4ce8-aeae-f8dc6d8dbbd4)

Then, apply the Gait Points Extraction code file to extract the gait points, and the three gait points are located as shown in the following image,

<p align="center"><img src="https://github.com/user-attachments/assets/df12b3e3-d5b1-4eb8-84cb-9e3ecadc34ee" height="500px" width="700px"></p>

with P1 being the start of the gait, P2 being the point where the gait data is at its lowest value, and P3 being the point where the heel of the opposite foot starts to fall off after some time from P2.

Finally, we import an Excel file that organizes the results of P1, P2, and P3 by subject to proceed with the gait analysis.
Inside, we have implemented each method and written code for Boxplot and visualization.

Below is an image of the results when visualized using this code.
![Fig 8  BMI groups gait points average](https://github.com/user-attachments/assets/4648a7a8-ef2d-48cb-9b27-6a1880a6fe0e)

