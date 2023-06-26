from ultralytics import YOLO 
import cv2 

class Kancil:
    def __init__(self,lokasimodel):
        self.model=YOLO(lokasimodel)
        
    def deteksi(self,gambar):
        hasil=self.model.predict(gambar,verbose=False)
        kelas = hasil[0].names
        hasil=hasil[0].boxes.data
        return hasil,kelas
    
    def visualis(self,gambar,bbox,kelas=None,color=None):
        for x1,y1,x2,y2,konf,ic in bbox:
            x1= int(x1)
            y1= int(y1)
            x2= int(x2)
            y2= int(y2)
            ic= int(ic)
            
            if color is None : 
                color = (255,0,0)   
                
            else:
                color = color[ic]

            if kelas is not None :
                ic= kelas[ic]
                
            cv2.rectangle(gambar,(x1,y1),(x2,y2),color,1)
            cv2.putText(gambar,str(ic),(x1,y1),1,1,color,1)
            
            return gambar
        
        
