from keras.models import load_model
import numpy as np
import argparse
import numpy
import pandas
from sklearn.metrics import confusion_matrix, accuracy_score,classification_report

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required=True,
	help="path to trained model model")
args = vars(ap.parse_args())
# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)
dataframe = pandas.read_csv("test_data.csv", header=None) #itc_248.csv working
dataset = dataframe.values
X = dataset[:,0:248].astype(float)        #.astype(float)
Y_ = dataset[:,248]
Y_ = Y_.reshape(-1, 1) # Convert data to a single column

#categories of the types of traffic
encode_categories = [np.array(['ATTACK', 'DATABASE', 'FTP-CO2TROL', 'FTP-DATA', 'FTP-PASV',
        'MAIL', 'MULTIMEDIA', 'P2P', 'SERVICES', 'WWW'], dtype=object)]


print("[INFO] loading network...")
model = load_model(args["model"])

res = model.predict(X)

print("The result is:")
i = 0
res_ult = np.empty((0,0))
for r in res:
    ind = np.where(r==1)
    print("actual network:~"+Y_[i]+" -------- "+"predicted network:~"+encode_categories[0][(ind[0])])
    res_ult = np.append(res_ult, encode_categories[0][(ind[0])])
    print("\n")
    i = i+1

con_mat = confusion_matrix(Y_, res_ult)
print("Confusion matrix: \n", con_mat)
acc = accuracy_score(Y_, res_ult)
print("Accuracy of the model: ", acc)
class_report = classification_report(Y_, res_ult)
print("Full report: ", class_report)


