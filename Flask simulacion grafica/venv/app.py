from flask import Flask , render_template,request
from predicter import *

app = Flask(__name__)

@app.route("/")

def home():
    return render_template('index.html')

@app.route("/submit", methods=['POST'])
def submit_form():
    file = request.files['imageInput']
    # Process the file or perform other actions
    
    # Example: Save the file to a folder
    filePathStatic = 'static/uploads/img/' + file.filename
    
    file.save(filePathStatic)
    message = predicter(filePathStatic)
    # ../static/vendor/swiper/swiper-bundle.min.css
    #uploads/ProfilerEnvidaNsight.png
    filePathStatic = "../" + filePathStatic
    print(filePathStatic)
    return render_template('Proccesed.html',filename=file.filename,filePath=filePathStatic, message=message)