#Copyright 2018 Mishanov Fedor

import pandas as pd
import matplotlib as plt
import numpy as np
import seaborn as sns
from scipy import stats

fontsize=40


required_columns={"Model",
                  "Release date",
                  "Max resolution",
                  "Low resolution",
                  "Effective pixels",
                  "Zoom wide (W)",
                  "Zoom tele (T)",
                  "Normal focus range",
                  "Macro focus range",
                  "Storage included",
                  "Weight (inc. batteries)",
                  "Dimensions",
                  "Price"
                 }



class Lab1Worker:
    def __init__(self,filename):
        #Читаем файл
        self.data=pd.read_csv(filename)
        print("File reading OK")
        missing=[]
        for c in required_columns:
            if not c in self.data.columns:
                missing.append(c)
                
        if len(missing)>0:
            string="Missing columns:"
            for i in missing:
                string+=i+";"
            raise ValueError(string) #Ругаемся на отсутствующие столбцы, если таковые окажутся
        print("Required columns OK")
        #Убираем названия типов
        self.data.drop([0],inplace=True)
        #Приводим данные к нужным типам
        self.data["Release date"]=self.data["Release date"].astype("int")
        self.data["Max resolution"]=self.data["Max resolution"].astype("double")
        self.data["Low resolution"]=self.data["Low resolution"].astype("double")
        self.data["Normal focus range"]=self.data["Normal focus range"].astype("double")
        self.data["Zoom wide (W)"]=self.data["Zoom wide (W)"].astype("double")
        self.data["Effective pixels"]=self.data["Effective pixels"].astype("double")
        self.data["Dimensions"]=self.data["Dimensions"].astype("double")
        self.data["Weight (inc. batteries)"]=self.data["Weight (inc. batteries)"].astype("double")
        self.data["Price"]=self.data["Price"].astype("double")
        self.data["Release date"]=self.data["Release date"].astype("double")
        self.data["Storage included"]=self.data["Storage included"].astype("double")
        self.data["Zoom tele (T)"]=self.data["Zoom tele (T)"].astype("double")
    def records_count(self):
        return self.data.shape[0]
    def remove_zeroes(self):
        self.data.replace(0.0,np.nan,inplace=True)
        self.data.dropna(inplace=True)
    def heatmap(self):
        table=self.data.pivot_table(index = "Release date",
                        columns = "Max resolution",
                        values = "Price",
                        aggfunc = "mean").fillna(0).applymap(float)
        plt.pyplot.figure(figsize=(150,75))
        p = sns.heatmap(table, annot=True, fmt=".1f", linewidths=.15,square=False,xticklabels=True,yticklabels=True,annot_kws={"size":fontsize})
        cbar_ax=p.figure.axes[-1]
        cbar_ax.tick_params(labelsize=fontsize,rotation=0)
        plt.pyplot.xticks(fontsize=fontsize)
        plt.pyplot.xlabel("Max resolution",fontsize=fontsize)
        plt.pyplot.yticks(fontsize=fontsize,rotation=0)
        plt.pyplot.ylabel("Release year",fontsize=fontsize,rotation=90)
        return p
    def pearsonr(self):
        return stats.pearsonr(self.data["Price"],self.data["Max resolution"])
    def spearmanr(self):
        return stats.spearmanr(self.data["Price"],self.data["Max resolution"], nan_policy="omit")
    def priceplot(self):
        return sns.distplot(self.data["Price"])
    def resplot(self):
        return sns.distplot(self.data["Max resolution"])
    def price_by_year(self):
        means=self.data.groupby(["Release date"])["Price"].mean()
        return means.plot()
    def add_multiply_column(self,name,col1,col2):
        self.data[name]=self.data[col1]*self.data[col2]
    def max_price_items(self):
        return self.data[self.data["Price"]==self.data["Price"].max()][["Model","Release date","Price"]]
    def min_price_items(self):
        return self.data[self.data["Price"]==self.data["Price"].min()][["Model","Release date","Price"]]
    def max_new_models(self):
        gd=self.data.groupby(["Release date"])["Model"].count()
        return (gd.idxmax(),gd.max())
    def most_present_firm(self):
        
        firms= self.data["Model"].map(lambda m: m.split(' ')[0])
        u,p,cs = np.unique(firms,return_inverse=True,return_counts=True)
        mx=cs.argmax()
        return (u[mx],cs[mx])
