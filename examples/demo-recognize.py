from models.conv import GatedConv

model = GatedConv.load("save_model/model_1.pth")

text = model.predict("dataset/test.wav")

print("")
print("识别结果:")
print(text)
