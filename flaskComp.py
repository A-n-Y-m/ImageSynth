from flask import Flask,render_template,request,jsonify,send_file,url_for

from flask_cors import CORS
import demo
import json
import base64
from PIL import Image
from io import BytesIO
import os
import time
def downres(img,qual):
  output_buffer = BytesIO()
  img.save(output_buffer, format='JPEG',quality=int(qual)) 
  output_buffer.seek(0)
  img_str = base64.b64encode(output_buffer.getvalue())
  img_str=img_str.decode("utf-8")
  return img_str


print(os.getcwd())

app = Flask(__name__)
CORS(app, resources={r'/*': {'origins': '*'}})


@app.route('/data', methods = ['POST', 'GET'])
def data():
    if request.method == 'GET':
        start=time.time()
        image = demo.generateImage(request.args.get('inptext'))
        img_str=downres(image,request.args.get('quality'))
        end=time.time()
        print("Image Generation time taken = ",end-start,"seconds")
        return img_str


@app.route ('/<input>')
def uploaded_image(filename,input):
    return send_file(filename,mimetype='image/png')


if __name__=="__main__":
    app.run(host='localhost', port=5000)