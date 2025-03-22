from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

def save_confusion(cm,path,title):
    fig,ax=plt.subplots(figsize=(4,4))
    ConfusionMatrixDisplay(confusion_matrix=np.asarray(cm),display_labels=['Truth','Lie']).plot(ax=ax,cmap='Blues',colorbar=False)
    ax.set_title(title);fig.tight_layout();fig.savefig(path,dpi=140);plt.close(fig)
