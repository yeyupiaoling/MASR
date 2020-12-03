import argparse
import functools
import time
import torch
import torch.nn.functional as F
from ctcdecode import CTCBeamDecoder
from flask import request, Flask, render_template
from flask_cors import CORS
from utils import data
from data.utility import add_arguments, print_arguments

app = Flask(__name__, template_folder="templates", static_folder="static", static_url_path="/static")
# 允许跨越访问
CORS(app)

parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
parser.add_argument("--model_path",
                    default="save_model/model.pth",
                    type=str,
                    help="trained model path. (default: %(default)s)")
parser.add_argument("--lm_path",
                    default="lm/zh_giga.no_cna_cmn.prune01244.klm",
                    type=str,
                    help="language model path. (default: %(default)s)")
parser.add_argument("--host",
                    default="localhost",
                    type=str,
                    help="server host. (default: %(default)s)")
parser.add_argument("--port",
                    default=5000,
                    type=int,
                    help="server port. (default: %(default)s)")
args = parser.parse_args()
print_arguments(args)

alpha = 0.8
beta = 0.3
cutoff_top_n = 40
cutoff_prob = 1.0
beam_width = 32
num_processes = 4
blank_index = 0

model = torch.load(args.model_path)
model = model.cuda()
model.eval()

decoder = CTCBeamDecoder(model.vocabulary,
                         args.lm_path,
                         alpha,
                         beta,
                         cutoff_top_n,
                         cutoff_prob,
                         beam_width,
                         num_processes,
                         blank_index)


def translate(vocab, out, out_len):
    return "".join([vocab[x] for x in out[0:out_len]])


def predict(wav_path):
    wav = data.load_audio(wav_path)
    spec = data.spectrogram(wav)
    spec.unsqueeze_(0)
    with torch.no_grad():
        spec = spec.cuda()
        y = model.cnn(spec)
        y = F.softmax(y, 1)
    y_len = torch.tensor([y.size(-1)])
    y = y.permute(0, 2, 1)  # B * T * V
    print("decoding...")
    out, score, offset, out_len = decoder.decode(y, y_len)
    return translate(model.vocabulary, out[0][0], out_len[0][0])


# 语音识别接口
@app.route("/recognition", methods=['POST'])
def recognition():
    f = request.files['audio']
    if f:
        # 临时保存路径
        file_path = "dataset/upload" + "." + f.filename.split('.')[-1]
        f.save(file_path)
        try:
            start = time.time()
            text = predict(file_path)
            end = time.time()
            print("识别时间：%dms，识别结果：%s" % (round((end - start) * 1000), text))
            result = str({"code": 0, "msg": "success", "result": text}).replace("'", '"')
            return result
        except:
            return str({"error": 1, "msg": "audio read fail!"})
    return str({"error": 3, "msg": "audio is None!"})


@app.route('/')
def home():
    return render_template("index.html")


if __name__ == '__main__':
    app.run(host=args.host, port=args.port)
