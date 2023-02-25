# CV-Paper-Statistics

A simple tool to visualize the main keywords of accepted papers for the recent Computer Vision conferences (CVPR 2019-2022、ICCV 2019 & 2021).

Inspired by [`ICLR2019-OpenReviewData`](https://github.com/shaohua0116/ICLR2019-OpenReviewData) and [`CVPR-2019-Paper-Statistics`](https://github.com/hoya012/CVPR-2019-Paper-Statistics)

## Prerequisites

[pandas](https://pandas.pydata.org/)
[plotly](https://plotly.com/python/)
[dash](https://dash.plotly.com/)
[nltk](https://www.nltk.org/install.html)

## CVPR Paper Keywords statistics

Keywords are sorted in descending order:     

<p align="center">
  <img width="1000" src="./demo.gif">
</p>

## Usage
Step 1. Clone the project and install relevant packages:
```
git clone git@github.com:ycmin95/CVPaperStatistics.git
cd ./CVPaperStatistics
pip install -r requirements.txt
```
Step 2. Run the app:
```
python app.py
``` 
Step 3. Connect to the Plotly-Dash app 127.0.0.1:8050 in the browser (or ServerIP:SperificePort)