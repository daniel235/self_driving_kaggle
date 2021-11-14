from data_prepare import Preprocessing
from models import Models
#TODO: create data reader as a module (preprocessing, resizing, compressing)
data = Preprocessing("data/driving_log.csv")
data.get_data()


#TODO: create networks (different models with same inputs for modularity)  Also implement torch for gpus as an option
myModel = Models()
cnn = myModel.cnn()


#TODO: test network (evaluation on test data set)
cnn = myModel.optimization(cnn)
print(data.train_y)
myModel.train(cnn, [data.train, data.train_y], [data.test, data.test_y])
myModel.plot(cnn)
