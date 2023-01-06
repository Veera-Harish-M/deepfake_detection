from PIL import Image, ImageChops
import extract_face
import matplotlib.pyplot as plt
import torch
import effnet
from torch.utils.model_zoo import load_url
from torchvision.transforms import ToPILImage
import utils

net_model = 'EfficientNetAutoAttB4'
train_db = 'FFPP'

weight_url = {
'EfficientNetAutoAttB4ST_DFDC':'https://f002.backblazeb2.com/file/icpr2020/EfficientNetAutoAttB4ST_DFDC_bestval-4df0ef7d2f380a5955affa78c35d0942ac1cd65229510353b252737775515a33.pth',
'EfficientNetAutoAttB4ST_FFPP':'https://f002.backblazeb2.com/file/icpr2020/EfficientNetAutoAttB4ST_FFPP_bestval-ddb357503b9b902e1b925c2550415604c4252b9b9ecafeb7369dc58cc16e9edd.pth',
'EfficientNetAutoAttB4_DFDC':'https://f002.backblazeb2.com/file/icpr2020/EfficientNetAutoAttB4_DFDC_bestval-72ed969b2a395fffe11a0d5bf0a635e7260ba2588c28683630d97ff7153389fc.pth',
'EfficientNetAutoAttB4_FFPP':'https://f002.backblazeb2.com/file/icpr2020/EfficientNetAutoAttB4_FFPP_bestval-b0c9e9522a7143cf119843e910234be5e30f77dc527b1b427cdffa5ce3bdbc25.pth',
'EfficientNetB4ST_DFDC':'https://f002.backblazeb2.com/file/icpr2020/EfficientNetB4ST_DFDC_bestval-86f0a0701b18694dfb5e7837bd09fa8e48a5146c193227edccf59f1b038181c6.pth',
'EfficientNetB4ST_FFPP':'https://f002.backblazeb2.com/file/icpr2020/EfficientNetB4ST_FFPP_bestval-ccd016668071be5bf5fff68e446d055441739ec7113fb1a6eee998f08396ae92.pth',
'EfficientNetB4_DFDC':'https://f002.backblazeb2.com/file/icpr2020/EfficientNetB4_DFDC_bestval-c9f3663e2116d3356d056a0ce6453e0fc412a8df68ebd0902f07104d9129a09a.pth',
'EfficientNetB4_FFPP':'https://f002.backblazeb2.com/file/icpr2020/EfficientNetB4_FFPP_bestval-93aaad84946829e793d1a67ed7e0309b535e2f2395acb4f8d16b92c0616ba8d7.pth',
'Xception_DFDC':'https://f002.backblazeb2.com/file/icpr2020/Xception_DFDC_bestval-e826cdb64d73ef491e6b8ff8fce0e1e1b7fc1d8e2715bc51a56280fff17596f9.pth',
'Xception_FFPP':'https://f002.backblazeb2.com/file/icpr2020/Xception_FFPP_bestval-bb119e4913cb8f816cd28a03f81f4c603d6351bf8e3f8e3eb99eebc923aecd22.pth',
}

device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')


import glob
cv_img = []
faces_coll=[]
fake_faces=[]
for img in glob.glob("fakeme/*.jpg"):
    im_test = Image.open(img)
    im_faces = extract_face.process_image(img=im_test)
    fake_faces.append(im_faces)
    faces_coll.append(im_faces['faces'][0])

faces_coll_reals=[]
real_faces=[]
for img in glob.glob("reals/*.jpg"):
    im_test_reals = Image.open(img)
    im_faces_reals = extract_face.process_image(img=im_test_reals)
    real_faces.append(im_faces_reals)
    faces_coll_reals.append(im_faces_reals['faces'][0])


model_url = weight_url['{:s}_{:s}'.format(net_model,train_db)]
net = getattr(effnet,net_model)().eval().to(device)
net.load_state_dict(load_url(model_url,map_location=device,check_hash=True))

face_policy = 'scale'
face_size = 224
transf = utils.get_transformer(face_policy, face_size, net.get_normalizer(), train=False)
faces_t = torch.stack( [ transf(image=im)['image'] for im in faces_coll ] )

with torch.no_grad():
    faces_pred = torch.sigmoid(net(faces_t.to(device))).cpu().numpy().flatten()

print("\n\n----------------Fakes Time------------------\n\n")
yTrue=[]
yScore=[]
YTRUE=[]

TrueFakes=0
FalseFakes=0
for i in faces_pred:
    YTRUE.append(1)
    if i>0.2:
        TrueFakes+=1
        yTrue.append(1)
    else:
        FalseFakes+=1
        yTrue.append(0)
    yScore.append(i)
    print('Score for Fake Image: {:.4f}'.format(i))
print("\n\n----------------Reals Time------------------\n\n")

faces_reals_t = torch.stack( [ transf(image=im)['image'] for im in faces_coll_reals ] )

with torch.no_grad():
    faces_pred_reals = torch.sigmoid(net(faces_reals_t.to(device))).cpu().numpy().flatten()

FalseReals=0
TrueReals=0

for i in faces_pred_reals:
    YTRUE.append(1)
    if i<0.2:
        TrueReals+=1
        yTrue.append(1)
    else:
        FalseReals+=1
        yTrue.append(0)
    yScore.append(i)
    print('Score for Real Images: {:.4f}'.format(i))



fig,ax = plt.subplots(4,5)
j=0
ax=ax.flatten()
for i in ax:
    if (j<10):
        i.imshow(faces_coll[j]) 
        i.set_title('Fakes')
    else: 
        i.imshow(faces_coll_reals[j])
        i.set_title('Reals')
    j+=1



import numpy as np
from sklearn.metrics import roc_auc_score
y_true = np.array(yTrue)
y_scores = np.array(yScore)

# print(f"ROC Score: {roc_auc_score(y_true, y_scores)}")
# Confusion Matrix
# from sklearn.metrics import confusion_matrix
# print("")
# print(f"Confusion Matrix:\n    NO Yes\nNo {confusion_matrix(YTRUE, yTrue)[0]}\nYes{confusion_matrix(YTRUE, yTrue)[1]}")



# # Accuracy
from sklearn.metrics import accuracy_score

print(f"No of reals:{len(faces_pred_reals)}")
print(f"No of Real images classified as Real:{TrueReals}")
print(f"No of Real Images classified as Fake:{FalseReals}")
print(f"No of Fakes:{len(faces_pred)}")
print(f"No of Fake images classified as Real:{FalseFakes}")
print(f"No of Fake Images classified as Fake:{TrueFakes}")
print(f"Accuracy:{accuracy_score(YTRUE, yTrue)}")



# from sklearn.metrics import roc_curve
# fpr, tpr, thresholds = roc_curve(y_true, y_scores)

# plt.plot([0, 1], [0, 1], 'k--')
# plt.plot(fpr, tpr)
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('ROC Curve')
# plt.show()
# # Recall
# from sklearn.metrics import recall_score
# print(f"Recall score {recall_score(YTRUE, yTrue, average=None)}")
# # Precision
# from sklearn.metrics import precision_score
# print(f"Precision: {precision_score(YTRUE, yTrue, average=None)}")


plt.show()