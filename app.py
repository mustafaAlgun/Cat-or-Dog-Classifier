import gradio as gr
from fastai.vision.all import *


def is_cat(x): return x[0].isupper()


learn = load_learner('model.pkl')

categories = ('Dog', 'Cat')


def classify_image(img):
    pred, ids, probs = learn.predict(img)
    return dict(zip(categories, map(float, probs)))


image = gr.inputs.Image(shape=(192, 192))
label = gr.outputs.Label()
examples = ['dog.jpg', 'cat.jpg', 'confusing.jfif']
enable_queue=True
interpretation = 'default'
title = "Cat or Dog Classifier"
description = "A cat or dog classifier trained with fastai on photos obtained Duck Duck Go photos. Created as a demo for Gradio and HuggingFace Spaces."


iface = gr.Interface(fn=classify_image, inputs=image, outputs=label, title=title, examples=examples, description=description, interpretation=interpretation, enable_queue=enable_queue).launch(share=True)
iface.launch(inline=False)
