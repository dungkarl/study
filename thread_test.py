from threading import Thread
import threading
import time


class myThread(Thread):
 	"""docstring for myThread"""
 	def __init__(self, name, counter, delay):
 		super(myThread, self).__init__()
 		self.name= name
 		self.counter=counter
 		self.delay=delay

 	def run(self):
 		print ("san sang chay" + self.name)
 		while self.counter:
 			time.sleep(self.delay)
 			print ("%s: %s" % (self.name, time.ctime(time.time())))
 			self.counter -= 1
 		print ("ket thuc vong lap", self.name)


try:
    while True:
        thread1 = myThread("thread 1", 10, 1)
        thread2 = myThread("thread 2", 10, 2)
        thread1.start()
        #thread1.daemon = True
        thread2.start()
        thread1.join()
        thread2.join()
        
        #print(threading.activeCount())
except:
    print ("Error")
