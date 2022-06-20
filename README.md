# Support-Vector-Machine
Pattern Recognition homework4 in NYCU.  

This project is to implement the cross-validation and grid search using only NumPy and train the SVM model.

The sample code can be download in this [link](https://github.com/NCTU-VRDL/CS_AT0828/tree/main/HW4).

## Requirement
```bash
$ conda create --name PR python=3.8 -y
$ conda activate PR
$ conda install matplotlib pandas scikit-learn -y
```

## Training & Evaluation
You can use the following command to train SVM model and find the best parameters. After training the model, the program will automatically evaluate the model.

```bash
python 310551031_HW4.py
```

<p float="left">
  <img src="https://user-images.githubusercontent.com/44439517/174540562-80b546b5-0e07-458b-9968-3aa822a81884.png" width="60%" height="60%"/>
</p>

## Result
This model can achieves 0.9010 on testing data.
