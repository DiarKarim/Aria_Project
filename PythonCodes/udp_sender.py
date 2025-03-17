import socket
import time 
import numpy as np

UDP_IP = "127.0.0.1"
UDP_PORT = 8899
MESSAGE = b"Hellow!"
sock = socket.socket(socket.AF_INET, # Internet
                     socket.SOCK_DGRAM) # UDP

elapsedTime = 0
startTime = time.time()
k = 0 
duration = 10.0

while(elapsedTime<duration):
    
    elapsedTime=time.time()-startTime
    
    Word = "Hello: " + str(k) 
    MESSAGE = Word.encode("utf-8")
    sock.sendto(MESSAGE, (UDP_IP, UDP_PORT))
    k=k+1
    print("Time remaining: " + str(np.round(duration-elapsedTime)) + " Message: " + str(MESSAGE))