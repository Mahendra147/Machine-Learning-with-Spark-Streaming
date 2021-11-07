import socket
import sys
import time
import csv

def send_data_to_spark(tcp_connection, reader):
    count = 0
    for row in reader:
        finstr =''
        for j in row:
            finstr = finstr+j
        finstr = finstr + '\n'
        count = count + 1
        tcp_connection.send(finstr.encode())
        if(count==2000):
            tcp_connection.send(finstr.encode())
            time.sleep(2)
            print('In sleep')
            count = 0
    tcp_connection.close()

TCP_IP = 'localhost'
TCP_PORT = 9009
conn=None
s=socket.socket(socket.AF_INET,socket.SOCK_STREAM)
s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
s.bind((TCP_IP,TCP_PORT))

csvfile = open('','r')
fieldnames = (" ", " ", " ")
reader=csv.reader(csvfile,fieldnames)
time.sleep(2)

s.listen(1)
print("Waiting for connection...")

conn,addr=s.accept()
print("Connected")

send_data_to_spark(conn,reader)