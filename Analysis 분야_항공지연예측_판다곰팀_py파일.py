import pandas as pd
from time import strftime
import datetime
import time
import numpy as np
import os
import xgboost as xgb
from sklearn.metrics import f1_score

os.chdir("../flight/flight")

#predict 데이터 (AFSNT_DLY)을 위한 함수는 모두 _predict로 끝남


def read_df(df):
  schedule=list()
  y=df['SDT_YY'].tolist()
  m=list(map(lambda x: str(x) if x>=10 else str(0)+str(x), df['SDT_MM'].tolist()))
  d=list(map(lambda x: str(x) if x>=10 else str(0)+str(x), df['SDT_DD'].tolist()))
  t=list(map(lambda x : int(str(x)[:1]), df['STT'].tolist()))
  t=list(map(lambda x: str(x) if x>=10 else str(0)+str(x), t))
  print(len(df), len(d))
  for i in range(len(df)):
    p=str(y[i])+m[i]+d[i]+str(t[i])
    schedule.append(p)
  df['SCHEDULE']=schedule
  df['SCHEDULE']=df['SCHEDULE'].apply(int)
  df['DATE']=list(map(lambda x : str(x)[:8], df['SCHEDULE'].tolist()))
  df['DATE']=df['DATE'].astype(str)
  df['ATT_TIME']=df['DATE']+"-"+df['ATT']
  df['ATT_TIME'] = pd.to_datetime(df['ATT_TIME'],format='%Y%m%d-%H:%M')
  df['STT_TIME']=df['DATE']+"-"+df['STT']
  df['STT_TIME'] = pd.to_datetime(df['STT_TIME'],format='%Y%m%d-%H:%M')
  diff=[]
  for i in range(len(df)):
    diff.append((df['ATT_TIME'][i]-df['STT_TIME'][i]).seconds)
  df['DIFF']=diff
  df=df.drop(["STT_TIME","ATT_TIME"],axis=1)
  return df


def read_df_predict(df):
  schedule=list()

  m=list(map(lambda x: str(x) if x>=10 else str(0)+str(x), df['SDT_MM'].tolist()))
  d=list(map(lambda x: str(x) if x>=10 else str(0)+str(x), df['SDT_DD'].tolist()))
  t=list(map(lambda x : int(str(x)[:1]), df['STT'].tolist()))
  t=list(map(lambda x: str(x) if x>=10 else str(0)+str(x), t))
  print(len(df), len(d))
  for i in range(len(df)):
    p=str(2018)+m[i]+d[i]+str(t[i])
    schedule.append(p)
  df['SCHEDULE']=schedule
  df['SCHEDULE']=df['SCHEDULE'].apply(int)
  df['DATE']=list(map(lambda x : str(x)[4:8], df['SCHEDULE'].tolist()))
  df['DATE']=df['DATE'].astype(str)
  df=df.drop(['DLY', 'DLY_RATE'], axis=1)
  return df

df=pd.read_csv("AFSNT.CSV", encoding="CP949")
from sklearn.preprocessing import LabelEncoder
le1 = LabelEncoder().fit(df['REG'].tolist())
le2 = LabelEncoder().fit(df['FLO'].tolist())
le3 = LabelEncoder().fit(df['FLT'].tolist())
le4 = LabelEncoder().fit(df['DLY'].tolist())
le5 = LabelEncoder().fit(df['CNL'].tolist())
le6 = LabelEncoder().fit(df['ARP'].tolist())
le7 = LabelEncoder().fit(df['ODP'].tolist())
le8 = LabelEncoder().fit(df['SDT_DY'].tolist())
le9 = LabelEncoder().fit(df['AOD'].tolist())
le10 = LabelEncoder().fit(df['IRR'].tolist())
le11= LabelEncoder().fit(df['DRR'].tolist())



def airport_organize(data,airport):
  data1=data[(data["ARP"]==airport)&(data["AOD"]=="D")]
  data2=data[(data["ODP"]==airport)&(data["AOD"]=="A")]
  data=data1.append(data2)


  data['REG']=list(le1.fit_transform(data['REG'].tolist()))
  data['FLO']=list(le2.fit_transform(data['FLO'].tolist()))
  data['FLT']=list(le3.fit_transform(data['FLT'].tolist()))
  data['DLY']=list(le4.fit_transform(data['DLY'].tolist()))
  data['CNL']=list(le5.fit_transform(data['CNL'].tolist()))
  data['ARP']=list(le6.fit_transform(data['ARP'].tolist()))
  data['ODP']=list(le7.fit_transform(data['ODP'].tolist()))
  data['SDT_DY']=list(le8.fit_transform(data['SDT_DY'].tolist()))
  data['AOD']=list(le9.fit_transform(data['AOD'].tolist()))
  data['IRR']=list(le10.fit_transform(data['IRR'].tolist()))
  data['DRR']=list(le11.fit_transform(data['DRR'].tolist()))
  data['STT_TM']=list(map(lambda x : int(str(x)[:1]), data['STT'].tolist()))

  return data


def airport_organize_predict(data,airport):
  data1=data[(data["ARP"]==airport)&(data["AOD"]=="D")]
  data2=data[(data["ODP"]==airport)&(data["AOD"]=="A")]
  data=data1.append(data2)


from sklearn.preprocessing import LabelEncoder
  le = LabelEncoder()
  data['FLO']=list(le2.fit_transform(data['FLO'].tolist()))
  data['FLT']=list(le3.fit_transform(data['FLT'].tolist()))
  data['ARP']=list(le6.fit_transform(data['ARP'].tolist()))
  data['ODP']=list(le7.fit_transform(data['ODP'].tolist()))
  #data['SDT_DY']=list(le8.fit_transform(data['SDT_DY'].tolist()))
  data['AOD']=list(le9.fit_transform(data['AOD'].tolist()))
  return data


def add_weather(a,weather_code):
  weather_code['TM'] = weather_code['TM'].apply(int)
  weather_code=pd.DataFrame(list(zip(weather_code['TM'].tolist(), weather_code['WD'].tolist(), weather_code['VIS'].tolist(), weather_code['WC'].tolist(), weather_code['RN'].tolist(), weather_code['CA_TOT'].tolist())), columns=["TM", "WD", "VIS", "WC", "RN", "CA_TOT"])
  output=weather_code.merge(a, right_on = "SCHEDULE", left_on = "TM")
  t=list(map(lambda x : int(str(x)[:8]), output['TM'].tolist()))
  output['TM']=t
  output.fillna(0)
  return output

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import xgboost as xgb



def add_prob(origin_data, a):

  data=origin_data.groupby(["DATE"]).mean()
  data=data.drop(["SDT_YY","SDT_MM","SDT_DD","SDT_DY","ARP","ODP","FLO","FLT","REG","AOD","IRR","CNL","RN"],axis=1)
  data=data.fillna(0).astype('int')

  delay=[]
  for i in range(len(data)):
    if data["DLY"].iloc[i]>=a:
      delay.append(1)
    else:
      delay.append(0)
  data["delay"]=delay



  model = xgb.XGBClassifier(n_estimators=1024, max_depth=512, learning_rate=0.2, subsample=0.4)
  y=data["delay"]
  x=data.drop(["delay"], axis=1)
  my_model_classifier = model.fit(x,y)
  y_predict = my_model_classifier.predict_proba(x)
  data['Probability']=y_predict[:,1]

  model1 = xgb.XGBRegressor(n_estimators=1024, max_depth=512, learning_rate=0.2, subsample=0.4)
  y=data["DIFF"]
  x=data.drop(["DIFF"], axis=1)
  my_model_regressor = model1.fit(x,y)
  y_predict1 = my_model_regressor.predict(x)
  data['DIFF_predict']=y_predict1

  t=data.index.tolist()
  output=pd.DataFrame(list(zip(t, data['Probability'], data["DIFF_predict"])), columns=['DATE', 'Probability',"DIFF_predict"])
  output['DATE']=output['DATE'].apply(int)
  origin_data['DATE']=origin_data['DATE'].apply(int)
  output1=pd.merge(output, origin_data, on='DATE')
  output1=output1.drop(["STT","ATT","CNR","CNL","SCHEDULE","TM"],axis=1)
  output1.fillna(0)

  output['DATE']=list(map(lambda x : str(x)[4:], output['DATE'].tolist()))
  output['DATE']=output['DATE'].astype('str')


  return output1, output



#기본 데이터에 weather 넣기
def org_arp(arp, weather, prob):
  data=add_weather(arp, weather)
  data, predictresult=add_prob(data, prob)
  data=data.drop(['DATE', 'IRR', "DRR","REG", "DIFF"], axis=1)
  data=data[['Probability','DIFF_predict','WD','VIS','WC','RN','CA_TOT','SDT_YY','SDT_MM','SDT_DD','SDT_DY','ARP','ODP','FLO','FLT','AOD','STT_TM','DLY']]
  data=data.fillna(0)
  return data, predictresult


#예측할 데이터에 weather 넣기
def org_arp_predict(arp, weather, predict):
  data=add_weather(arp, weather)
  data=pd.merge(data, predict, on="DATE")
  data=data.drop(['TM', 'SCHEDULE', 'DATE'], axis=1)
  data=data[['Probability','DIFF_predict','WD','VIS','WC','RN','CA_TOT','SDT_YY','SDT_MM','SDT_DD','SDT_DY','ARP','ODP','FLO','FLT','AOD','STT_TM',"STT"]]
  return data


#기본적으로 xgb 모형 돌리는 함수
def do_xgb(data, data_predict, estimator, max, rate, sample):

  model=xgb.XGBClassifier(n_estimators=estimator, max_depth=max, learning_rate=rate, subsample=sample, nthread = 10)
  y=data["DLY"]
  x=data.drop(["DLY"], axis=1)
  my_model = model.fit(x, y)
  data_predict=data_predict.drop(["STT"],axis=1)
  y_predict = my_model.predict(data_predict)
  y_probability=my_model.predict_proba(data_predict)
  return y_predict, y_probability[:,1]


#xgb모형 앙상블
def xgb_pred(data, data_predict, estimator, max, rate, sample):
  from operator import add
  predict_1, probability_1=do_xgb(data, data_predict, estimator, max, rate, sample)
  predict_2, probability_2=do_xgb(data, data_predict, estimator, max, rate, sample)
  predict_3, probability_3=do_xgb(data, data_predict, estimator, max, rate, sample)
  predict_4, probability_4=do_xgb(data, data_predict, estimator, max, rate, sample)
  predict_5, probability_5=do_xgb(data, data_predict, estimator, max, rate, sample)


  predict=list(map(add, predict_1,predict_2))
  predict=list(map(add, predict, predict_3))
  predict=list(map(add, predict, predict_4))
  predict=list(map(add, predict, predict_5))

  probability=list(map(add, probability_1,probability_2))
  probability=list(map(add, probability, probability_3))
  probability=list(map(add, probability, probability_4))
  probability=list(map(add, probability, probability_5))

  result=list()
  for i in range(len(predict)):
    predict[i]=predict[i]/5
    probability[i]=probability[i]/5
    if predict[i]>=0.3:
      result.append(1)
    else:
      result.append(0)

  data_predict['DLY']=result
  data_predict['DLY_RATE']=probability
  return data_predict



def inverse_transform(data):
  data['FLO']=list(le2.inverse_transform(data['FLO'].tolist()))
  data['FLT']=list(le3.inverse_transform(data['FLT'].tolist()))
  data['ARP']=list(le6.inverse_transform(data['ARP'].tolist()))
  data['ODP']=list(le7.inverse_transform(data['ODP'].tolist()))
  data['AOD']=list(le9.inverse_transform(data['AOD'].tolist()))
#   data['SDT_DY']=list(le8.inverse_transform(data['SDT_DY'].tolist()))
  data=data[['SDT_YY','SDT_MM','SDT_DD','SDT_DY','ARP','ODP','FLO','FLT','AOD',"STT",'DLY', 'DLY_RATE','Probability','DIFF_predict','WD','VIS','WC','RN','CA_TOT']]

  return data



rkpc=pd.read_csv("RKPC_air.csv", encoding="CP949")
rkjb=pd.read_csv("RKJB_air.csv", encoding="CP949")
rkpu=pd.read_csv("RKPU_air.csv", encoding="CP949")
rkjy=pd.read_csv("RKJY_air.csv", encoding="CP949")
rkny=pd.read_csv("RKNY_air.csv", encoding="CP949")
rksi=pd.read_csv("RKSI_air.csv", encoding="CP949")
rkss=pd.read_csv("RKSS_air.csv", encoding="CP949")

#the weather in Sep 2018
rkpc_18=pd.read_csv("RKPC_air_stcs201809.csv",encoding="CP949")
rkjb_18=pd.read_csv("RKJB_air_stcs201809.csv",encoding="CP949")
rkpu_18=pd.read_csv("RKPU_air_stcs201809.csv",encoding="CP949")
rkjy_18=pd.read_csv("RKJY_air_stcs201809.csv",encoding="CP949")
rkny_18=pd.read_csv("RKNY_air_stcs201809.csv",encoding="CP949")
rksi_18=pd.read_csv("RKSI_air_stcs201809.csv",encoding="CP949")
rkss_18=pd.read_csv("RKSS_air_stcs201809.csv",encoding="CP949")


df=pd.read_csv("AFSNT.CSV", encoding="CP949")
df=read_df(df)

arp1_data=airport_organize(df, "ARP1") #김포
arp2_data=airport_organize(df, "ARP2") #김해
arp3_data=airport_organize(df, "ARP3") #제주
arp4_data=airport_organize(df, "ARP4") #대구
arp5_data=airport_organize(df, "ARP5") #울산
arp6_data=airport_organize(df, "ARP6") #청주
arp7_data=airport_organize(df, "ARP7") #무안
arp8_data=airport_organize(df, "ARP8") #광주
arp9_data=airport_organize(df, "ARP9") #여수
arp10_data=airport_organize(df, "ARP10") #양양
arp11_data=airport_organize(df, "ARP11") #포항
arp12_data=airport_organize(df, "ARP12") #사천
arp13_data=airport_organize(df, "ARP13") #군산
arp14_data=airport_organize(df, "ARP14") #원주
arp15_data=airport_organize(df, "ARP15") #인천

arp1_data, arp1_predict=org_arp(arp1_data, rkss, 0.3)
arp2_data, arp2_predict=org_arp(arp2_data, rkpu, 0.3)
arp3_data, arp3_predict=org_arp(arp3_data, rkpc, 0.3)
arp4_data, arp4_predict=org_arp(arp4_data, rkpu, 0.3)
arp5_data, arp5_predict=org_arp(arp5_data, rkpu, 0.3)
arp6_data, arp6_predict=org_arp(arp6_data, rkss, 0.3)
arp7_data, arp7_predict=org_arp(arp7_data, rkjb, 0.3)
arp8_data, arp8_predict=org_arp(arp8_data, rkjb, 0.3)
arp9_data, arp9_predict=org_arp(arp9_data, rkss, 0.3)
arp10_data, arp10_predict=org_arp(arp10_data, rkny, 0.3)
arp11_data, arp11_predict=org_arp(arp11_data, rkpu, 0.3)
arp12_data, arp12_predict=org_arp(arp12_data, rkjy, 0.3)
arp13_data, arp13_predict=org_arp(arp13_data, rkjb, 0.3)
arp14_data, arp14_predict=org_arp(arp14_data, rkny, 0.3)
arp15_data, arp15_predict=org_arp(arp15_data, rksi, 0.3)


dataset=pd.read_csv("AFSNT_DLY.CSV", encoding="CP949")
dataset=read_df_predict(dataset)


arp1_data_predict=airport_organize_predict(dataset, "ARP1") #김포
arp2_data_predict=airport_organize_predict(dataset, "ARP2") #김해
arp3_data_predict=airport_organize_predict(dataset, "ARP3") #제주
arp4_data_predict=airport_organize_predict(dataset, "ARP4") #대구
arp5_data_predict=airport_organize_predict(dataset, "ARP5") #울산
arp6_data_predict=airport_organize_predict(dataset, "ARP6") #청주
arp7_data_predict=airport_organize_predict(dataset, "ARP7") #무안
arp8_data_predict=airport_organize_predict(dataset, "ARP8") #광주
arp9_data_predict=airport_organize_predict(dataset, "ARP9") #여수
arp10_data_predict=airport_organize_predict(dataset, "ARP10") #양양
arp11_data_predict=airport_organize_predict(dataset, "ARP11") #포항
arp12_data_predict=airport_organize_predict(dataset, "ARP12") #사천
arp13_data_predict=airport_organize_predict(dataset, "ARP13") #군산
arp14_data_predict=airport_organize_predict(dataset, "ARP14") #원주
arp15_data_predict=airport_organize_predict(dataset, "ARP15") #인천


arp1_data_predict=org_arp_predict(arp1_data_predict, rkss_18,arp1_predict)
arp2_data_predict=org_arp_predict(arp2_data_predict, rkpu_18,arp2_predict)
arp3_data_predict=org_arp_predict(arp3_data_predict, rkpc_18,arp3_predict)
arp4_data_predict=org_arp_predict(arp4_data_predict, rkpu_18,arp4_predict)
arp5_data_predict=org_arp_predict(arp5_data_predict, rkpu_18,arp5_predict)
arp6_data_predict=org_arp_predict(arp6_data_predict, rkss_18,arp6_predict)
arp7_data_predict=org_arp_predict(arp7_data_predict, rkjb_18,arp7_predict)
arp8_data_predict=org_arp_predict(arp8_data_predict, rkjb_18,arp8_predict)
arp9_data_predict=org_arp_predict(arp9_data_predict, rkss_18,arp9_predict)
arp10_data_predict=org_arp_predict(arp10_data_predict, rkny_18,arp10_predict)
arp11_data_predict=org_arp_predict(arp11_data_predict, rkpu_18,arp11_predict)
arp12_data_predict=org_arp_predict(arp12_data_predict, rkjy_18,arp12_predict)
arp13_data_predict=org_arp_predict(arp13_data_predict, rkjb_18,arp13_predict)
arp14_data_predict=org_arp_predict(arp14_data_predict, rkny_18,arp14_predict)
arp15_data_predict=org_arp_predict(arp15_data_predict, rksi_18,arp15_predict)


arp1=xgb_pred(arp1_data, arp1_data_predict, 1024, 512, 0.2, 0.4)
arp1=inverse_transform(arp1)

arp2=xgb_pred(arp2_data, arp2_data_predict, 1024, 512, 0.2, 0.4)
arp2=inverse_transform(arp2)

arp3=xgb_pred(arp3_data, arp3_data_predict, 1024, 512, 0.2, 0.4)
arp3=inverse_transform(arp3)

arp4=xgb_pred(arp4_data, arp4_data_predict, 1024, 512, 0.25, 0.4)
arp4=inverse_transform(arp4)

arp5=xgb_pred(arp5_data, arp5_data_predict, 1024, 1024, 0.3, 0.4)
arp5=inverse_transform(arp5)

arp6=xgb_pred(arp6_data, arp6_data_predict, 1024, 512, 0.3, 0.4)
arp6=inverse_transform(arp6)

arp7=xgb_pred(arp7_data, arp7_data_predict, 1024, 512, 0.3, 0.4)
arp7=inverse_transform(arp7)

arp8=xgb_pred(arp8_data, arp8_data_predict, 1024, 512, 0.25, 0.4)
arp8=inverse_transform(arp8)

arp9=xgb_pred(arp9_data, arp9_data_predict, 1024, 1024, 0.25, 0.4)
arp9=inverse_transform(arp9)

arp11=xgb_pred(arp11_data, arp11_data_predict, 1024, 128, 0.2, 0.4)
arp11=inverse_transform(arp11)

arp12=xgb_pred(arp12_data, arp12_data_predict, 1024, 512, 0.3, 0.4)
arp12=inverse_transform(arp12)

arp13=xgb_pred(arp13_data, arp13_data_predict, 2048, 256, 0.25, 0.4)
arp13=inverse_transform(arp13)

arp14=xgb_pred(arp14_data, arp14_data_predict, 1024, 256, 0.2, 0.4)
arp14=inverse_transform(arp14)

arp15=xgb_pred(arp15_data, arp15_data_predict, 2048, 512, 0.3, 0.4)
arp15=inverse_transform(arp15)

result_list=[arp1, arp2, arp3, arp4, arp5, arp6, arp7, arp8, arp9, arp11, arp12, arp13, arp14, arp15]
result=pd.concat(result_list, ignore_index=True)
result.to_csv("result.csv",index=False)
