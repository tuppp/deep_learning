import FunctionLibraryExtended as library
import numpy as np


class Getdata:
    def __init__(self):
        pass
        self.val_id = 0
        self.train = []
        self.train_labels = []
        self.val = []
        self.val_labels = []
        self.fold1=None
        self.fold2=None
        self.fold3=None
        self.fold4=None
        self.fold5=None
        self.fold6=None
        self.res1=None
        self.res2=None
        self.res3=None
        self.res4=None
        self.res5=None
        self.res6=None

    def fold_shuffle(self):
        if self.val_id==1:
            self.train=np.append(self.fold2,self.fold3,axis=0)
            self.train=np.append(self.train,self.fold4, axis=0)
            self.train = np.append(self.train, self.fold5, axis=0)
            self.train = np.append(self.train, self.fold6, axis=0)
            self.val=self.fold1
            self.val_labels=self.res1
            self.train_labels = np.append(self.res2, self.res3, axis=0)
            self.train_labels = np.append(self.train_labels, self.res4, axis=0)
            self.train_labels = np.append(self.train_labels, self.res5, axis=0)
            self.train_labels = np.append(self.train_labels, self.res6, axis=0)
        if self.val_id==2:
            self.train=np.append(self.fold1,self.fold3,axis=0)
            self.train=np.append(self.train,self.fold4, axis=0)
            self.train = np.append(self.train, self.fold5, axis=0)
            self.train = np.append(self.train, self.fold6, axis=0)
            self.val=self.fold2
            self.val_labels=self.res2
            self.train_labels = np.append(self.res1, self.res3, axis=0)
            self.train_labels = np.append(self.train_labels, self.res4, axis=0)
            self.train_labels = np.append(self.train_labels, self.res5, axis=0)
            self.train_labels = np.append(self.train_labels, self.res6, axis=0)
        if self.val_id==3:
            self.train=np.append(self.fold1,self.fold2,axis=0)
            self.train=np.append(self.train,self.fold4, axis=0)
            self.train = np.append(self.train, self.fold5, axis=0)
            self.train = np.append(self.train, self.fold6, axis=0)
            self.val=self.fold3
            self.val_labels=self.res3
            self.train_labels = np.append(self.res1, self.res2, axis=0)
            self.train_labels = np.append(self.train_labels, self.res4, axis=0)
            self.train_labels = np.append(self.train_labels, self.res5, axis=0)
            self.train_labels = np.append(self.train_labels, self.res6, axis=0)
        if self.val_id==4:
            self.train=np.append(self.fold2,self.fold3,axis=0)
            self.train=np.append(self.train,self.fold1, axis=0)
            self.train = np.append(self.train, self.fold5, axis=0)
            self.train = np.append(self.train, self.fold6, axis=0)
            self.val=self.fold4
            self.val_labels=self.res4
            self.train_labels = np.append(self.res2, self.res3, axis=0)
            self.train_labels = np.append(self.train_labels, self.res1, axis=0)
            self.train_labels = np.append(self.train_labels, self.res5, axis=0)
            self.train_labels = np.append(self.train_labels, self.res6, axis=0)
        if self.val_id==5:
            self.train=np.append(self.fold2,self.fold3,axis=0)
            self.train=np.append(self.train,self.fold4, axis=0)
            self.train = np.append(self.train, self.fold1, axis=0)
            self.train = np.append(self.train, self.fold6, axis=0)
            self.val=self.fold5
            self.val_labels=self.res5
            self.train_labels = np.append(self.res2, self.res3, axis=0)
            self.train_labels = np.append(self.train_labels, self.res4, axis=0)
            self.train_labels = np.append(self.train_labels, self.res1, axis=0)
            self.train_labels = np.append(self.train_labels, self.res6, axis=0)
        if self.val_id==6:
            self.train=np.append(self.fold2,self.fold3,axis=0)
            self.train=np.append(self.train,self.fold4, axis=0)
            self.train = np.append(self.train, self.fold5, axis=0)
            self.train = np.append(self.train, self.fold1, axis=0)
            self.val=self.fold6
            self.val_labels=self.res6
            self.train_labels = np.append(self.res2, self.res3, axis=0)
            self.train_labels = np.append(self.train_labels, self.res4, axis=0)
            self.train_labels = np.append(self.train_labels, self.res5, axis=0)
            self.train_labels = np.append(self.train_labels, self.res1, axis=0)

    def start(self):
        tmp=library.getConnectionDWD()
        #Anfrage PLZ
        # result=tmp[1].execute("""SELECT station_id,postcode, station_name, count(*)
        #                         FROM dwd
        #                          GROUP BY station_name
        #                          """)
        #result=tmp[1].execute("""SELECT station_id,postcode, station_name,measure_date
        #                        FROM dwd
        #                         Where station_name='Berlin-Tegel'
        #                         ORDER BY measure_date ASC
        #                         """)
        #
        # result=tmp[1].execute("""SELECT m1.measure_date, COUNT(m1.measure_date) as c FROM dwd m1
        #                             Where (m1.station_name='BRAUNLAGE' OR m1.station_name='POTSDAM' OR
        #                          m1.station_name='BREMEN' OR m1.station_name='TRIER-PETRISBERG' OR m1.station_name='FICHTELBERG') AND
        #                          m1.measure_date>=19480101
        #                         GROUP BY m1.measure_date
        #                          """)

        result=tmp[1].execute("""SELECT m1.station_name,m1.measure_date,m1.average_temp,m2.c
                                FROM dwd m1
                                join 
                               (SELECT measure_date, COUNT(measure_date) as c FROM dwd
                               Where (station_name='BRAUNLAGE' OR station_name='POTSDAM' OR
                                 station_name='BREMEN' OR station_name='TRIER-PETRISBERG' OR station_name='FICHTELBERG') AND
                                 measure_date>=19480101
                                 GROUP BY measure_date) m2
                                 on (m1.measure_date = m2.measure_date)
                                 Where (m1.station_name='BRAUNLAGE' OR m1.station_name='POTSDAM' OR
                                 m1.station_name='BREMEN' OR m1.station_name='TRIER-PETRISBERG' OR m1.station_name='FICHTELBERG') AND
                                 m1.measure_date>=19480101 and m2.c=5
                                 """)
        fold=[[] for i in range(6)]
        res=[[] for i in range(6)]
        tmp=[[]for i in range(5)]
        tmpres=[]
        result=sorted(result, key=lambda element: ( element[1]))
        for line in result:
            if(len(tmp[4])==7):
                fold_num=np.random.randint(6, size=1)[0]
                if len(fold[fold_num])==0:
                    fold[fold_num]=np.array([tmp])
                else:
                    fold[fold_num]=np.append(fold[fold_num],np.array([tmp]),axis=0)
                for i in range(5):
                    tmp[i]=tmp[i][1:]
                res[fold_num].append(line[2])
            if "Bremen" in line[0]:
                tmp[0].append(line[2])
            if "Braunlage" in line[0]:
                tmp[2].append(line[2])
            if "Potsdam" in line[0]:
                tmp[3].append(line[2])
            if "Fichtelberg" in line[0]:
                tmp[1].append(line[2])
            if "Trier" in line[0]:
                tmp[4].append(line[2])

        for i in range(6):
            res[i]=np.array(res[i])

        self.res1=res[0]
        self.res2=res[1]
        self.res3=res[2]
        self.res4=res[3]
        self.res5=res[4]
        self.res6=res[5]
        self.fold1=fold[0].reshape((fold[0].shape[0],5,7,1))
        self.fold2=fold[1].reshape((fold[1].shape[0],5,7,1))
        self.fold3=fold[2].reshape((fold[2].shape[0],5,7,1))
        self.fold4=fold[3].reshape((fold[3].shape[0],5,7,1))
        self.fold5=fold[4].reshape((fold[4].shape[0],5,7,1))
        self.fold6=fold[5].reshape((fold[5].shape[0],5,7,1))



        #print(fold[0])
        #print(res[0])


getdata=Getdata()
getdata.start()
getdata.val_id=3
getdata.fold_shuffle()
print(getdata.train.shape)
print(getdata.train_labels.shape)
print(getdata.val.shape)
print(getdata.val_labels.shape)
