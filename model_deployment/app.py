
from flask import Flask, render_template, request , jsonify
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
app = Flask(__name__)

@app.route('/', methods = ['GET'])
def hello_world():
    return render_template('index.html')

@app.route('/',methods = ['POST'])

def predict():
    h = 100
    w = 100
    PATH = './savemodel/'
    imagefile = request.files['imagefile']
    image_path = "./images/" + imagefile.filename
    imagefile.save(image_path)
    img= Image.open(imagefile)
    img = img.resize((h,w))
    loader = transforms.Compose([transforms.Resize(96),
                                 transforms.ToTensor(),
                                 transforms.Normalize((0, 0, 0),
                                                      (1, 1, 1))])
    image = loader(img)
    x = torch.unsqueeze(image, 0)

    class Model(nn.Module):
        def __init__(self, in_channels=3, numb_classes=4):
            super(Model, self).__init__()
            self.conv1 = nn.Conv2d(
                in_channels=in_channels,
                out_channels=8,
                kernel_size=(3, 3),
                stride=(1, 1),
                padding=(1, 1),
            )
            self.pool1 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
            self.conv2 = nn.Conv2d(
                in_channels=8,
                out_channels=16,
                kernel_size=(3, 3),
                stride=(1, 1),
                padding=(1, 1),
            )
            self.pool2 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
            self.conv3 = nn.Conv2d(
                in_channels=16,
                out_channels=32,
                kernel_size=(3, 3),
                stride=(1, 1),
                padding=(1, 1),
            )

            self.pool3 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
            self.fc1 = nn.Linear(32 * 12 * 12, numb_classes)

        def forward(self, x):
            x = F.relu(self.conv1(x))
            x = self.pool1(x)
            x = F.relu(self.conv2(x))
            x = self.pool2(x)
            x = F.relu(self.conv3(x))
            x = self.pool3(x)
            x = x.reshape(x.shape[0], -1)
            x = self.fc1(x)
            return x

    model = Model(in_channels=3, numb_classes=4)
    states = [torch.load(PATH+f'model_fold{fold}_best.path') for fold in range(5)]
    for state in states:
        model.load_state_dict(state['model'])
        model.eval()
        with torch.no_grad():
            y = model(x)
            _,result = y.max(1)
            predicted_idx = str(result.item())
    return render_template('index.html', prediction =f'ဤအရွက်သည် အရွက်အရောင်ဇယားတွင် အဆင့် {predicted_idx} ဖြစ်သည်။')
if __name__ == '__main__':
    app.run(port=100, debug=True)
