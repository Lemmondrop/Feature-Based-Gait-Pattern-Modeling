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

You can use the Raw Data Preprocessing code to preprocess the data of the real-time current consumption to get the gait data file with the Envelope process applied and the gait data file with the Smoothing process applied using the Kalman Filter.

![image](https://github.com/user-attachments/assets/fb01a6e0-7e5c-4ce8-aeae-f8dc6d8dbbd4)

Then, in the Gait Points Extraction process, you can get the data of gait points P1, P2, and P3 for each subject, and it is recommended to organize each result in a separate excel file.
Finally, we import an Excel file that organizes the results of P1, P2, and P3 by subject to proceed with the gait analysis.
Inside, we have implemented each method and written code for Boxplot and visualization.
