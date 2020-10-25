import pycuda.driver as cuda

class aboutCudaDevices():
     def __init__(self):
         pass
     def info(self):
         """Class representation as number of devices connected and about
         them."""
         num = cuda.Device.count()
         string = ""
         string += ("%d device(s) found:\n" % num)
         for i in range(num):
             string += ("    %d) %s (Id: %d)\n" % ((i + 1),\
                 cuda.Device(i).name(), i))
             string += ("          Memory: %.2f GB\n" %\
                     (cuda.Device(i).total_memory() / 1e9))
             return string
