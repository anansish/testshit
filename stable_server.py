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
    return "OK"

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
    generation_thread = threading.Thread(target=generator.img2imgInpainting, args=(flags, image,))
    generation_thread.start()
    return "OK"

@app.route("/progress")
def progress():
    args=request.args.to_dict()
    visual=False
    if 'visual' in args.keys():
        visual=args['visual']=='True'
    image, progress = generator.get_progress(visual)
    response=None
    if progress==0 or not visual:
        response=make_response()
        response.headers['X-Progress'] = progress
    if (visual or progress==100) and (progress > 0):
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
    generation_thread = threading.Thread(target=generator.generate, args=(flags,))
    generation_thread.start()
    return "OK"

generator=Generator()
generator.load_models()

app.run(host="0.0.0.0")
