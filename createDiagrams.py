import matplotlib.pyplot as plt

def createGraphValTrainLoss(history, epochs, name):
    '''
    Erstellt ein Diagramm des Losses. 

    '''
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    
    epochs = range(1, len(loss) + 1)
    
    plt.figure()
    
    plt.plot(epochs, loss, 'r-', label="Training loss")
    plt.plot(epochs, val_loss, 'b', label="Validation loss")
    plt.title("Training and validation loss")
    plt.legend()
    
    plt.savefig(name)
    plt.show()
    
def createGraphValTrainAcc(history, epochs, name):
    '''
    Erstellt ein Diagramm der Genauigkeiten. 
    '''
    loss = history.history['acc']
    val_loss = history.history['val_acc']
    
    epochs = range(1, len(loss) + 1)
    
    plt.figure()
    
    plt.plot(epochs, loss, 'r-', label="Training accuracy")
    plt.plot(epochs, val_loss, 'b', label="Validation accuracy")
    plt.title("Training and validation accuracy")
    plt.legend()
    
    plt.savefig(name)
    plt.show()
    