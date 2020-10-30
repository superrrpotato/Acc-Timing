#import pycuda.driver as cuda


class learningStat():
    def __init__(self):
        self.lossSum = 0
        self.numSamples = 0
        self.minloss = None
        self.lossLog = []
        self.bestLoss = False
    def reset(self):
        self.lossSum = 0
        self.numSamples = 0
    def loss(self):
        if self.numSamples > 0:
            return self.lossSum/self.numSamples
        else:
            return None
    def accuracy(self):
        pass
    def update(self):
        currentLoss = self.loss()
        self.lossLog.append(currentLoss)
        if self.minloss is None:
            self.minloss = currentLoss
        else:
            if currentLoss < self.minloss:
                self.minloss = currentLoss
                self.bestLoss = True
            else:
                self.bestLoss = False
    def displayString(self):
        loss = self.loss()
        minloss = self.minloss
        if loss is None:
            return 'No testing results'
        elif minloss is None:
            return 'loss = %-11.5g'%(loss)
        else:
            return 'loss = %-11.5g (min = %-11.5g)'%(loss, minloss)
class learningStats():
    def __init__(self):
        self.linesPrinted = 0
        self.training = learningStat()
        self.testing = learningStat()
    def update(self):
        self.training.update()
        self.training.reset()
        self.testing.update()
        self.testing.reset()
    def print(self, epoch, iter=None, timeElapsed=None, lr=None, header=None,\
            footer=None):
        print('\033[%dA' % (self.linesPrinted))
        self.linesPrinted = 1
        epochStr = 'Epoch : %10d' % (epoch)
        iterStr = '' if iter is None else '(i = %7d)' % (iter)
        profileStr = '' if timeElapsed is None else ', %12.4f s elapsed' %\
            timeElapsed
        lrStr = '' if lr is None else '    LearningRate : %3.8f' % lr
        if header is not None:
            for h in header:
                print('\033[2K' + str(h))
                self.linesPrinted += 1
        print(epochStr + iterStr + profileStr + lrStr)
        print(self.training.displayString())
        #print(self.testing.displayString())
        self.linesPrinted += 3
        if footer is not None:
            for f in footer:
                print('\033[2K' + str(f))
                self.linesPrinted += 1
    def plot(self, saveFig=False, path='./'):
        plt.figure()
        plt.cla()
        if len(self.training.lossLog) > 0:
            plt.semilogy(self.training.lossLog, label='Training')
        if len(self.testing.lossLog) > 0:
            plt.semilogy(self.testing.lossLog, label='Testing')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        if saveFig is True:
            plt.savefig(path + 'loss.png')

    def save(self, filename=''):
        with open(filename + 'loss.txt', 'wt') as loss:
            loss.write('#%11s %11s\r\n' % ('Train', 'Test'))
            for i in range(len(self.training.lossLog)):
                loss.write('%12.6g %12.6g \r\n' % (self.training.lossLog[i],\
                    self.testing.lossLog[i]))
    def load(self, filename='', numEpoch=None, modulo=1):
        saved = {}
        saved['loss'] = np.loadtxt(filename + 'loss.txt')
        if numEpoch is None:
            saved['epoch'] = saved['loss'].shape[0] // modulo * modulo + 1
        else:
            saved['epoch'] = numEpoch
        self.training.lossLog = saved['loss'][:saved['epoch'], 0].tolist()
        self.testing.lossLog = saved['loss'][:saved['epoch'], 1].tolist()
        self.training.minloss = saved['loss'][:saved['epoch'], 0].min()
        self.testing.minloss = saved['loss'][:saved['epoch'], 1].min()
        return saved['epoch']

class aboutCudaDevices():
     def __init__(self):
        pass
     def info(self):
        """
        Class representation as number of devices connected and about
        them.
        """
        num = cuda.Device.count()
        string = ""
        string += ("%d device(s) found:\n" % num)
        for i in range(num):
            string += ("    %d) %s (Id: %d)\n" % ((i + 1),\
                cuda.Device(i).name(), i))
            string += ("          Memory: %.2f GB\n" %\
                    (cuda.Device(i).total_memory() / 1e9))
            return string
