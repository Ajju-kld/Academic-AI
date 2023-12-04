from doctr.io import DocumentFile
from doctr.models import ocr_predictor

model = ocr_predictor(det_arch='db_resnet50', reco_arch='crnn_vgg16_bn', pretrained=True)
doc = DocumentFile.from_pdf("pdf/ktu.pdf")
result = model(doc)
result.show(pages=[4])