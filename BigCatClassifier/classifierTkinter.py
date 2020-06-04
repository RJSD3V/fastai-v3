# -*- coding: utf-8 -*-
"""
Created on Wed May 20 11:53:03 2020

@author: rjsod---Raajas Sode
For Google Collab Notebook: https://github.com/RJSD3V/fastai-v3/blob/master/Classifying_Spotted_Big_Cats.ipynb
"""

from fastai import *
from fastai.vision import *


from tkinter import *
from PIL import ImageTk, Image
from tkinter import filedialog
import os


model_path='D:\\Proposal Footage\\PythonLesson1\\BigCatClassifier\\path\\models'
root=Tk()
root.geometry("750x700")
root.resizable(width=True,height=True)
learn=load_learner(Path(model_path),'export.pkl')


def openfn():
    global filename
    filename=filedialog.askopenfilename(title='open')
    
    return filename

def open_img():
    x=openfn()
    print(x)
    img=Image.open(x)
    img=img.resize((224,224), Image.ANTIALIAS)
    img=ImageTk.PhotoImage(img)
    panel=Label(root,image=img)
    panel.image=img
    panel.pack()
    
def predict():
      img=open_image(filename)
      pred_class,pred_idx,outputs=learn.predict(img)
      
      output=''
      if(str(pred_class)=='tigris'):
          output="Its a Tiger!"
      elif(str(pred_class)=='pardus'):
          output="Its a Leopard !!"
      elif(str(pred_class)=='uncia'):
          output='Its a Snow Leopard !!'
      label=Label(text="Species : "+str(pred_class))
      label.pack()
      prediction=Label(text='My Analysis : '+output)
      prediction.pack()
      
      
      
      
    
title=Label(root,text="BIG CAT CLASSIFIER", font=('Helvetica',30)).pack()
subtitle=Label(root,text="Click and upload a photo-- I'll tell you what cat it is").pack()
warning=Label(root,text="I can currently tell the difference between A Leopard(Pantera pardus), a Tiger(Panthera tigris) and a Snow Leopard (Panthera uncia)").pack()
button=Button(root,text='open image',command=open_img).pack()
button1=Button(root,text='Predict',command=predict).pack()

credit=Label(root,text="Made By: Raajas Sode").pack()



root.mainloop()