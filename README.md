## HuBMAP

- To segment instances of micro-vascular structures from healthy human kidney tissue slides.
- Utilized semi-supervised learning (SSL) for training a YOLOv8x model in instance segmentation with only 20% labeled data. Implemented self-training to harness the combined power of labelled and unlabelled data, effectively optimizing model performance.
- Achieved substantial enhancement in instance segmentation, raising overall mAP50-95 score from 0.47 to a remarkable 0.591

- This is the port of my original notebook on kaggle: https://www.kaggle.com/code/sohithbandari/hubmap-yolov8-semi-supervised/

- Download the dataset from kaggle: https://www.kaggle.com/competitions/hubmap-hacking-the-human-vasculature/data
- Unzip data

Run main.py

- Metrics after 10 iterations: 0.471 to 0.591
