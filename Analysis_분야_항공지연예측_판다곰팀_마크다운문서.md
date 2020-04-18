


아래에 있는 함수 중 predict로 끝나는 함수들은 테스트 데이터(AFSNT_DLY)를 위한 함수입니다. 트레이닝데이터(AFSNT)와 구성하는 컬럼이 달라 따로 함수를 만들었습니다.


```python
import pandas as pd
from time import strftime
import datetime
import time
import numpy as np
import os
import xgboost as xgb
from sklearn.metrics import f1_score


```

전처리에 필요한 모듈을 불러옵니다.


```python
def read_df():
  schedule=list()
  y=df['SDT_YY'].tolist()
  m=list(map(lambda x: str(x) if x>=10 else str(0)+str(x), df['SDT_MM'].tolist()))
  d=list(map(lambda x: str(x) if x>=10 else str(0)+str(x), df['SDT_DD'].tolist()))
  t=list(map(lambda x : int(str(x)[:1]), df['STT'].tolist()))
  t=list(map(lambda x: str(x) if x>=10 else str(0)+str(x), t))
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
```

- read_df()
운항실적 데이터인 AFSNT파일 열들을 전처리하는 함수입니다.
항공편 운항 날짜 정보인 년(SDT_YY),월(SDT_MM),일(SDT_DD),계획시각(STT)로 나눠져있는 정보를 년,월,일, 시각을 포함한 하나의 열으로 표현합니다. 
예를 들어 YY가 2017 MM이 1 DD가 1 STT가 10:05 인 데이터가 있다면 이를 합쳐 2017010110로 나타낸 p를 만들어줍니다. 이 p가 모여 'SCHEDULE' 컬럼을 완성합니다.
계획시각(STT)뿐만아니라 실제시각(ATT)도 년,월,일을 포함한 형식으로 만들어 줍니다.
이후, 해당항공기가 계획시각에 비해 얼마나 빠르게 혹은 늦게 출발하였는지 파악하기 위해 계획 시간과 실제 시각의 차이를 구해 'DIFF' 칼럼을 만듭니다.


```python
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
```


-read_df_predict()
TEST 데이터(AFSNT_DLY)에 대해 read_df()와 동일한 작업수행해주는 read_df_predict 함수 생성합니다
다만 test데이터에는 ATT 컬럼이 없기 떄문에 계획된 비행기 출발/도착 시각(STT)와 실제 출/도착 시간(ATT)의 차는 구하지 않습니다. 


```python
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

```


-트레이닝 데이터(AFSNT)의 범주형 변수들을 숫자형으로 바꾸기 위한 코드입니다. 각 열은 지연 예측이 모두 끝난 뒤 본래의 값으로 되돌아가야하기 때문에 열마다 다른 LabelEncoder()를 적용시킵니다.


```python
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
  
```


- airport_organize()
출발공항을 기준으로 지연을 파악하기 위해, AOD가 D일때는 ARP를 기준으로, AOD가 A일때는 ORP를 기준으로 데이터를 정리합니다.
예를들어 ARP가 김포이고 ODP가 김해이며 AOD가 D(도착)이라면 출발공항인 김해공항을 기준으로 변경하여줍니다.
그 후에 위에서 작성한 각 컬럼별 sklearn.LabelEncoder을 이용하여 범주형 변수를 가진 'REG','FLO','FLT','DLY','CNL','ARP','ODP','SDT_DY','AOD','IRR','DRR' 열
을 수치화 합니다. 
'STT_TM' 컬럼은 계획된 출발 시각을 나타냅니다. STT가 15:55였다면 STT_TM은 15입니다.
```


```python
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
```


-airport_organize_predict()
TEST 데이터(AFSNT_DLY)에 대해 airport_organize()와 동일한 작업수행해주는 airport_organize_predict 함수를 생성합니다
TEST 데이터에는 'REG', 'DLY', 'CNL', 'IRR', 'DRR' 의 컬럼이 존재하지 않아 수치화 작업을 하지않습니다
```


```python
def add_weather(a,weather):
  weather['TM'] = weather['TM'].apply(int)
  weather=pd.DataFrame(list(zip(weather['TM'].tolist(), weather['WD'].tolist(), weather['VIS'].tolist(), weather['WC'].tolist(), weather['RN'].tolist(), weather['CA_TOT'].tolist())), columns=["TM", "WD", "VIS", "WC", "RN", "CA_TOT"])
  output=weather.merge(a, right_on = "SCHEDULE", left_on = "TM")
  t=list(map(lambda x : int(str(x)[:8]), output['TM'].tolist()))
  output['TM']=t
  output.fillna(0)
  return output    
```


- add_weather(공항데이터, 기상데이터)
해당 공항의 운항실적 데이터와 항공기상데이터 일부를 합쳐주는 add_weather 함수를 생성합니다.
이때 항공기상데이터 중 유의미하게 결항여부에 영향을 미친다고 생각한 'WD','VIS','WC','RN','CA_TOT'를 뽑아왔습니다.
여기서 'WD'는 풍향, 'VIS'는 시정(가시거리), 'WC'는 일기(날씨), 'RN'은 강수량, 'CA_TOT'는 전운량을 의미합니다.


```python
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import xgboost as xgb 
```


이후 결항여부를 예측하고, 검정하기위한 모듈들을 불러줍니다.


```python
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

#예측값의 확률
  model = xgb.XGBClassifier(n_estimators=1024, max_depth=512, learning_rate=0.2, subsample=0.4)

  y=data["delay"]
  x=data.drop(["delay"], axis=1)
  my_model = model.fit(x, y)
  y_predict = my_model.predict_proba(x)
  data['Probability']=y_predict[:,1]
  model1 = xgb.XGBRegressor(n_estimators=1024, max_depth=512, learning_rate=0.2, subsample=0.4)

#차이
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
```


-add_prob()
운항실적과 기상데이터를 바탕으로 지연여부(DLY)와 시각차이(DIFF)의 예측치를 구해주는 add_prob함수를 만듭니다.
가장 먼저 이 함수는 날짜를 기준으로 운항실적과 기상데이터를 모두 평균을 낸 새로운 데이터셋 data를 만듭니다.
이 데이터 셋 각 날짜의 DLY의 평균이 0.3 이상 이면 1(지연), 0.3 이하이면 0(지연아님)으로 두고 delay컬럼을 만듭니다.
이후 다른 변수들을 고려했을때,각 날짜에서 실제로 지연이 될 확률을 구한 prob 칼럼을 생성하고, 운항실적과 기상데이터를 바탕으로 DIFF(STT와 ATT의 차이)의 예측치를 구합니다.


```python
def org_arp(arp, weather, prob):
  data=add_weather(arp, weather)
  data, predictresult=add_prob(data, prob)
  data=data.drop(['DATE', 'IRR', "DRR","REG", "DIFF"], axis=1)
  data=data[['Probability','DIFF_predict','WD','VIS','WC','RN','CA_TOT','SDT_YY','SDT_MM','SDT_DD','SDT_DY','ARP','ODP','FLO','FLT','AOD','STT_TM','DLY']]
  data=data.fillna(0)
  return data, predictresult
```


-org_arp()
운항실적 데이터와 항공기상데이터를 합쳐주는 org_arp() 함수를 만듭니다. 위에서 작성한 add_weather(), add_prob() 함수를 이용합니다. 
이때, 모든공항에대한 누적 항공기상 데이터가 존재하지 않아,
항공 기상 데이터가 없는 공항에는 기상 데이터가 있는 직선거리 상 가장 가까운 공항들의 데이터를 배정했습니다.
각각의 항공기상 데이터를 RKJB 무안공항의 데이터를 광주공항, 군산공항이, RKPU 울산공항의 데이터를 김해공항, 대구공항, 포항공항이, RKJY 여수공항의 데이터를 사천공항이, RKNY 양양공항의 데이터를 원주공항이, RKSS 김포공항의 데이터를 청주공항이 이용합니다.
RKSI 인천공항, RKPC 제주공항은 다른 공항과 공유 없이 독자적으로 데이터를 이용합니다.
```


```python
#예측할 데이터에 weather 넣기
def org_arp_predict(arp, weather, predict):
  data=add_weather(arp, weather)
  data=pd.merge(data, predict, on="DATE")
  data=data.drop(['TM', 'SCHEDULE', 'DATE'], axis=1)
  data=data[['Probability','DIFF_predict','WD','VIS','WC','RN','CA_TOT','SDT_YY','SDT_MM','SDT_DD','SDT_DY','ARP','ODP','FLO','FLT','AOD','STT_TM',"STT"]]
  return data
  
```



-org_arp_predcit()
테스트데이터(AFSNT_DLY)에 대해 동일한 작업 실행하는 org_arp_predict 함수를 생성합니다. 테스트 데이터와 트레이닝 데이터의 컬럼 순서를 맞춰줍니다.


```python
#learning rate 0.05단위로 조정, 0.3까지
def do_xgb(data, estimator, max, rate, sample):
 model=xgb.XGBClassifier(n_estimators=estimator, max_depth=max, learning_rate=rate, subsample=sample, nthread = 10)
  y=data["DLY"]
  x=data.drop(["DLY"], axis=1)
  my_model = model.fit(x, y)
  data_predict=data_predict.drop(["STT"],axis=1)
  y_predict = my_model.predict(data_predict)
  y_probability=my_model.predict_proba(data_predict)
  return y_predict, y_probability[:,1]


```


-do_xgb()
전처리한 데이터들을 바탕으로 모델을 만들고, 테스트 셋에대한 검정력을 확인하는 do_xgb 함수를 만듭니다.
xgb 모듈의 classifier를 이용했습니다.


```python
#xgb모형 앙상블
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

```


-xgb_pred()
xgb모형 5개를 앙상블 시켜 최종 결과를 구합니다. 이용 시 각 공항에 대한 최적화된 파라미터들을 입력한 뒤, 총 5개의 xgb 모형만들어 각각의 모형에 test를 합니다.
각각의 모형에서 나온 5개의 결과를 기준을 평균을 내 지연인지/아닌지 판단, DLY_RATE를 구합니다.
5개의 모형에서 나온 지연(DLY)값들의 평균이 0.3 이상이면 지연이라 판단해 1로 뒀습니다. 반대로 평균이 0.3 미만이면 지연이 아니라고 판단해 0이라고 뒀습니다.


```python
def inverse_transform(data):
  data['FLO']=list(le2.inverse_transform(data['FLO'].tolist()))
  data['FLT']=list(le3.inverse_transform(data['FLT'].tolist()))
  data['ARP']=list(le6.inverse_transform(data['ARP'].tolist()))
  data['ODP']=list(le7.inverse_transform(data['ODP'].tolist()))
  data['AOD']=list(le9.inverse_transform(data['AOD'].tolist()))
#   data['SDT_DY']=list(le8.inverse_transform(data['SDT_DY'].tolist()))
  data=data[['SDT_YY','SDT_MM','SDT_DD','SDT_DY','ARP','ODP','FLO','FLT','AOD',"STT",'DLY', 'DLY_RATE','Probability','DIFF_predict','WD','VIS','WC','RN','CA_TOT']]
  
  return data
```

-inverse_transform(data)
이전에 수치화한 범주형 변수를 원상복귀하는 변수입니다. 또한 최종적 데이터프레임을 출력하기 전, 컬럼의 순서를 AFSNT_DLY와 같게끔 바꿉니다.
AFSNT_DLY 데이터프레임은 'SDT_YY','SDT_MM','SDT_DD','SDT_DY','ARP','ODP','FLO','FLT','AOD',"STT",'DLY', 'DLY_RATE' 컬럼만 갖고 있지만, 지연 예측을 하는 과정에서 나온 'Probability','DIFF_predict','WD','VIS','WC','RN','CA_TOT'까지 뒤에 추가했습니다. 


아래는 실제 데이터입력, 모형을 돌리는 코드입니다.


```python


rkpc=pd.read_csv("RKPC_air.csv",encoding="CP949")
rkjb=pd.read_csv("RKJB_air.csv",encoding="CP949")
rkpu=pd.read_csv("RKPU_air.csv",encoding="CP949")
rkjy=pd.read_csv("RKJY_air.csv",encoding="CP949")
rkny=pd.read_csv("RKNY_air.csv",encoding="CP949")
rksi=pd.read_csv("RKSI_air.csv",encoding="CP949")
rkss=pd.read_csv("RKSS_air.csv",encoding="CP949")


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




```



날씨데이터를 불러오고, 19년 9월 기상예측을 위하여 18년 9월 자료를 사용합니다. 이때 모든 공항에 대한 누적 날씨 정보가 존재하지 않아, 직선 거리를 기준으로 가까운 공항끼리묶어서 사용하였습니다.



```python
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
```


지연여부 예측데이터를 저장시킬 AFSNT_DLY를 불러오고, 예측을 위한 데이터들을 만든 함수를 이용하여 정리해줍니다.


```python
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

```


위에서 작성한 xgb_pred()함수를 이용하여 지연여부를 예측합니다. 앞에서 구한 결과 arp10은 데이터가 없는 데이터셋임이 알려져서 arp10은 구하지 않았습니다. 공항별 지연 예측을 하면, 모두 합쳐서 하나의 csv파일로 출력합니다.
