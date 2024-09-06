import gradio as gr
from fastai.vision.all import *

def is_cat(x):
    return x[0].isupper()

learn = load_learner('model.pkl')

categories = ("Dog", "Cat")

def classify_image(img):
    pred, idx, probs = learn.predict(img)
    return dict(zip(categories, map(float, probs)))

# Remove 'shape' argument from gr.Image()
image = gr.Image(type="pil")  # You can specify the image type here, like 'pil' for PIL images
label = gr.Label()#apahgnakpalababibutofucku 

intf = gr.Interface(fn=classify_image, inputs=image, outputs=label)
intf.launch(inline=False)









# import gradio as gr

# def greet(name):
#     return "Hello " + name + "!!"

# demo = gr.Interface(fn=greet, inputs="text", outputs="text")
# demo.launch()