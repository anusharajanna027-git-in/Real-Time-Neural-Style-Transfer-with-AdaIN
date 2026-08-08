import os
import torch
from flask import Flask, render_template, request, redirect, url_for, send_from_directory
from flask_wtf import FlaskForm
from flask_bootstrap import Bootstrap
from werkzeug.utils import secure_filename
from wtforms import FileField, SubmitField, FloatField, HiddenField
from wtforms.validators import InputRequired
from PIL import Image, ImageFile
from torchvision import transforms
import io

ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None

# Import your existing AdaIN code
from utils.model import VGGEncoder, Decoder
from utils.utils import adaptive_instance_normalization, calc_mean_std


app = Flask(__name__)
app.config['SECRET_KEY'] = 'supersecretkey'
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}
Bootstrap(app)

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
  # form for to taking the input from the user=>content image, style image, alpha value and submit button
class UploadForm(FlaskForm):
    content = FileField('Content Image')
    style = FileField('Style Image')
    content_path = HiddenField() # paths to store the uploaded images in the server
    style_path = HiddenField() # paths to store the uploaded images in the server
    alpha = FloatField('Alpha', default=1.0) # alpha value to control the strength of the style transfer
    submit = SubmitField('Transfer Style')#submit button


#next we load out model so we set the device first
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


#next we load the encoder and decoder models
encoder = VGGEncoder('vgg_normalised.pth').to(device)
decoder = Decoder().to(device)

decoder_path = 'decoder.pth'
if os.path.exists(decoder_path):
    decoder.load_state_dict(torch.load(decoder_path, map_location=device))
    print(f"Loaded pretrained decoder from {decoder_path}")
elif os.path.exists('experiment/experiment1/decoder_1.pth'):
    decoder.load_state_dict(torch.load('experiment/experiment1/decoder_1.pth', map_location=device))
    print("Loaded decoder from experiment/experiment1/decoder_1.pth")
else:
    print("Warning: Decoder checkpoint not found, running with initial weights.")

# we set both models to evaluation mode because we are not training the model we are just using it for inference
encoder.eval()
decoder.eval()


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

# here we implement same pytorch operations means what we builded the  NST model we use here inside the function style_transform()
def style_transfer(content_image, style_image, encoder, decoder, alpha, device):
    content_transform = transforms.Compose([
        transforms.Resize(512),  # incase the size is differ from  user uploaded image so we resize to 512 =>this is like precautions we take beacause we train the mdoel od size 512 sooo
        transforms.ToTensor()
    ])

    style_transform = transforms.Compose([
        transforms.Resize(512),
        transforms.ToTensor()
    ])
    #transform the  oth images content and style
    content_image = content_transform(content_image).unsqueeze(0).to(device)# suppose user input image size is [3,512,512,512] it must be become [1,3,512,512] of 4 dim basically we add ust batch_size=1
    style_image = style_transform(style_image).unsqueeze(0).to(device)

    with torch.no_grad():
        #passing the both inside encoder
        content_feats = encoder(content_image, is_test=True) # because is _test=true means at last layer output we need soo
        style_feats = encoder(style_image, is_test=True)

        # then we pass  both through AdaIN layer
        stylized_feats = adaptive_instance_normalization(content_feats, style_feats)

        stylized_feats = alpha * stylized_feats + (1 - alpha) * content_feats

        stylized_image = decoder(stylized_feats)

    return stylized_image


def save_image(image, path):
    image = image.cpu().clone()
    image = image.squeeze(0)
    image = image.clamp(0, 1)
    image = transforms.ToPILImage()(image)
    image.save(path)



@app.route('/', methods=['GET', 'POST'])
def index():
    form = UploadForm()
    result_image = None
    content_filename = None
    style_filename = None
    error = None

    if form.validate_on_submit():
        if form.content.data and form.content.data.filename:
            if allowed_file(form.content.data.filename):
                content_filename = secure_filename(form.content.data.filename)
                form.content.data.save(os.path.join(app.config['UPLOAD_FOLDER'], content_filename))
                form.content_path.data = content_filename
        else:
            content_filename = form.content_path.data

        if form.style.data and form.style.data.filename:
            if allowed_file(form.style.data.filename):
                style_filename = secure_filename(form.style.data.filename)
                form.style.data.save(os.path.join(app.config['UPLOAD_FOLDER'], style_filename))
                form.style_path.data = style_filename
        else:
            style_filename = form.style_path.data

        if content_filename and style_filename:
            content_path = os.path.join(app.config['UPLOAD_FOLDER'], content_filename)
            style_path = os.path.join(app.config['UPLOAD_FOLDER'], style_filename)
            
            try:
                content_image = Image.open(content_path).convert('RGB')
                style_image = Image.open(style_path).convert('RGB')

                alpha = float(form.alpha.data)
                stylized_image = style_transfer(content_image, style_image, encoder, decoder, alpha, device)

                result_filename = 'stylized_' + content_filename
                result_path = os.path.join(app.config['UPLOAD_FOLDER'], result_filename)
                save_image(stylized_image, result_path)
                
                result_image = result_filename
            except Exception as e:
                error = str(e)
        else:
            if not content_filename and not style_filename:
                error = 'Please upload content and style images'
            elif not content_filename:
                error = 'Please upload content image'
            elif not style_filename:
                error = 'Please upload style image'
    elif request.method == 'POST':
        content_filename = form.content_path.data
        style_filename = form.style_path.data
        if not content_filename and not style_filename:
            error = 'Please upload both content and style images'
        elif not content_filename:
            error = 'Please upload content image'
        elif not style_filename:
            error = 'Please upload style image'

    return render_template('index.html', form=form, result_image=result_image, content_image=content_filename,
                           style_image=style_filename, error=error)


@app.route('/uploads/<filename>')
def send_image(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


@app.route('/examples/<path:filename>')
def send_example(filename):
    return send_from_directory('examples', filename)


if __name__ == '__main__':
    from werkzeug.serving import run_simple
    run_simple('localhost', 5000, app, use_reloader=True, use_debugger=True)