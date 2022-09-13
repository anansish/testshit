from PyQt5.QtGui import QImage
import os, io
import random
import threading
import argparse, os, sys, glob
sys.path.append(".")

from generation_manager import Generator
from flask import Flask, request, send_file, make_response
from flask_lt import run_with_lt

app = Flask(__name__)
run_with_lt(app)

@app.route("/img2img", methods=['POST'])
def img2img():
    image_data = request.data
    image=QImage(image_data, int(request.args.get("w")), int(request.args.get("h")), QImage.Format_ARGB32)
    flags=request.args.to_dict()
    if 'seed' in flags.keys():
        if (int(flags['seed'])==0):
            flags['seed']=random.randint(0,100000)
    else:
        flags['seed'] = random.randint(0, 100000)
    generation_thread=threading.Thread(target=generator.img2img, args=(flags, image,))
    generation_thread.start()
    #image, _ = generator.img2img(flags, image)
    #img_byte_arr = io.BytesIO()
    #image.save(img_byte_arr, format='PNG')
    #img_byte_arr.seek(0)
    return "OK" #send_file(img_byte_arr,"image/png")

@app.route("/inpaint", methods=['POST'])
def img2imgInpainting():
    image_data = request.data
    image=QImage(image_data, int(request.args.get("w")), int(request.args.get("h")), QImage.Format_ARGB32)
    flags=request.args.to_dict()
    if 'seed' in flags.keys():
        if (int(flags['seed'])==0):
            flags['seed']=random.randint(0,100000)
    else:
        flags['seed'] = random.randint(0, 100000)
    image, _ = generator.img2imgInpainting(flags, image)
    image.show()
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format='PNG')
    img_byte_arr.seek(0)
    return send_file(img_byte_arr,"image/png")

@app.route("/progress")
def progress():
    image, progress = generator.get_progress()
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format='PNG')
    img_byte_arr.seek(0)
    response=make_response(send_file(img_byte_arr,"image/png"))
    response.headers['X-Progress'] = progress
    return response

@app.route("/txt2img")
def txt2img():
    flags = request.args.to_dict()
    if 'seed' in flags.keys():
        if (int(flags['seed'])==0):
            flags['seed']=random.randint(0,100000)
    else:
        flags['seed'] = random.randint(0, 100000)
    image, _ = generator.generate(flags)
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format='PNG')
    img_byte_arr.seek(0)
    return send_file(img_byte_arr,"image/png")

generator=Generator()
generator.load_model()

app.run(host="0.0.0.0")
