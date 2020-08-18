from obspy import read
import os
import ftan as ft

streamfiles=os.listdir()
GFs=[]

for streamfile in streamfiles:
   x = streamfile.split(".")
   if x[-1]=="SAC":
       st=read(streamfile)
       print(x[0])
       filename=ft.ftan(streamfile,freqmin=1/15,freqmax=1)
       ft.plot(streamfile, group_speeds=True)