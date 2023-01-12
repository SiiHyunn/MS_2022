import matplotlib.pyplot as plt
import pandas as pd
import glob

df = pd.read_csv("./modelAccuracy.csv")

# epoch,train_loss,val_loss,train_acc,val_acc

loss = df["train_loss"].tolist()
acc = df["train_acc"].tolist()
val_loss = df["val_loss"].tolist()
val_acc = df["val_acc"].tolist()
epochs = range(1,len(acc)+1)

plt.subplot(2, 1, 1)
plt.plot(epochs, acc, 'b',label ="Training acc")
plt.plot(epochs, val_acc, 'r',label ="Validation acc")
plt.title("Training and Validation accuracy")
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(epochs, loss, 'b',label ="Training loss")
plt.plot(epochs, val_loss, 'r',label ="Validation loss")
plt.title("Training and Validation loss")
plt.legend()

plt.tight_layout()
plt.show()