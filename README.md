# Traffic Sign Classification Project

This project implements a Sillnet and Convolutional Neural Network (CNN) to extract semantic feature and classify traffic signs using the **German Traffic Sign Recognition Benchmark (GTSRB)** dataset.

---

![image](https://github.com/user-attachments/assets/d09ebb1d-9a2b-4ed3-a934-6c45a1433a7d)

## Dataset

Download the dataset from Kaggle:

[GTSRB - German Traffic Sign Dataset](https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign)

After downloading, extract the dataset into the directory:  
`./data/gtsrb/`

---

## Requirements

Ensure Python 3.9+ is installed. Install the required dependencies using:

```bash
pip install -r requirements.txt

python train.py #training

python main.py #inference
