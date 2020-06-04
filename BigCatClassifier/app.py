# -*- coding: utf-8 -*-
"""
Created on Tue May  5 02:50:38 2020

@author: rjsod
"""
from __future__ import division, print_function
#coding=utf-8
import sys
import os
import glob
import re
from pathlib import Path
from ipywidgets import FloatProgress


#From the fastai library
from fastai import *
from torchvision.models import *

from fastai.vision import *



from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename

#Defining the Flask App
app=Flask(__name__)

path=Path("path")
classes=['tigris','uncia','pardes']

#Creating ImageDataBunch

imgData=ImageDataBunch.single_from_classes(path,classes,ds_tfms=get_transforms(),size=224).normalize(imagenet_stats)

learn=create_cnn(imgData,models.resnet34)
learn.load('stage-2')


def model_predict(img_path):
    
    img = open_image(img_path)
    pred_class,pred_idx,outputs= learn.predict(img)
    return pred_class




@app.route('/',methods=['GET','POST'])
def upload():
    if request.method=='POST':
        f=request.files['file']
        
        basepath=os.path.dirname(__file__)
        file_path=os.path.join(
                basepath,'uploads',secure_filename(f.filename))
        f.save(file_path)
        
        #make Prediction 
        preds =  model_predict(file_path)
        return preds
    else:
        return None

if __name__=="__main__":
    app.run()



