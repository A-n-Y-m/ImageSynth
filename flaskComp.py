from flask import Flask,render_template,request,jsonify,send_file,url_for

from flask_cors import CORS
import demo
import json
import base64
from PIL import Image
from io import BytesIO
import os



print(os.getcwd())

app = Flask(__name__)
print("asd")
CORS(app, resources={r'/*': {'origins': '*'}})


# def serve_pil_image(pil_img):
#     img_io = BytesIO()
#     pil_img.save(img_io, 'PNG', quality=70)
#     img_io.seek(0)
#     return send_file(img_io,as_attachment=False, mimetype='image/png',cache_timeout=0)

# @app.route('/data', methods = ['POST', 'GET'])
# def data():
#     if request.method == 'GET':
#         image = demo.generateImage(request.args.get('inptext'))
#         return serve_pil_image(image)
@app.route('/data', methods = ['POST', 'GET'])
def data():
    if request.method == 'GET':
        image = demo.generateImage(request.args.get('inptext'))
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        buffered.seek(0)
        img_str = base64.b64encode(buffered.getvalue())
        img_str=img_str.decode("utf-8")
        return img_str


@app.route ('/<input>')
def uploaded_image(filename,input):
    print("in this") 
    return send_file(filename,mimetype='image/png')


if __name__=="__main__":
    app.run(host='localhost', port=5000, debug=True)