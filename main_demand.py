from pandas import DataFrame, Series
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.tsa.stattools as ts
from statsmodels.graphics.api import qqplot
from statsmodels.sandbox.stats.diagnostic import acorr_ljungbox
import random
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error,mean_absolute_error

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.sans-serif'] = ['SimHei']
random.seed(10)

def boxpierce_test(x):
	qljungbox, pval, qboxpierce, pvalbp = acorr_ljungbox(x, boxpierce=True)
	fig, axes = plt.subplots(1, 2)
	axes[0].plot(qljungbox, label='LB统计量');
	axes[0].set_ylabel('真实-Q')
	axes[0].plot(qboxpierce, label='Q统计量')
	axes[1].plot(pval, label='LB统计量');
	axes[1].set_ylabel('P-Value')
	axes[1].plot(pvalbp, label='Q统计量')
	axes[0].legend()
	axes[1].legend()
	plt.show()

# 差分操作
def diff_ts(ts, d):
	global shift_ts_list
	#  动态预测第二日的值时所需要的差分序列
	global last_data_shift_list
	shift_ts_list = []
	last_data_shift_list = []
	tmp_ts = ts
	for i in d:
		last_data_shift_list.append(tmp_ts[-i])
		print(last_data_shift_list)
		shift_ts = tmp_ts.shift(i)
		shift_ts_list.append(shift_ts)
		tmp_ts = tmp_ts - shift_ts
	tmp_ts.dropna(inplace=True)
	return tmp_ts

# 还原操作
def predict_diff_recover(predict_value, d):
	if isinstance(predict_value, float):
		tmp_data = predict_value
		for i in range(len(d)):
			tmp_data = tmp_data + last_data_shift_list[-i - 1]
	elif isinstance(predict_value, np.ndarray):
		tmp_data = predict_value[0]
		for i in range(len(d)):
			tmp_data = tmp_data + last_data_shift_list[-i - 1]
	else:
		tmp_data = predict_value
		for i in range(len(d)):
			try:
				tmp_data = tmp_data.add(shift_ts_list[-i - 1])
			except:
				raise ValueError('What you input is not pd.Series type!')
		tmp_data.dropna(inplace=True)
	return tmp_data

def proper_model(data_ts, maxLag):
	init_bic = sys.maxint
	init_p = 0
	init_q = 0
	init_properModel = None
	for p in np.arange(maxLag):
		for q in np.arange(maxLag):
			model = ARMA(data_ts, order=(p, q))
			try:
				results_ARMA = model.fit(disp=-1, method='css')
			except:
				continue
			bic = results_ARMA.bic
			if bic < init_bic:
				init_p = p
				init_q = q
				init_properModel = results_ARMA
				init_bic = bic
	return init_bic, init_p, init_q, init_properModel

# 读取文件
path = "D:\\停车\\研一第一学期论文\\新华医院\\新华医院ARIMA总表.xlsx"
ori_data =pd.read_excel(path,sheet_name="周一",index_col="index")
test = pd.read_excel(path,sheet_name="周一测试",index_col="index")
test_data = test["需求比"]

data_demand_240= ori_data["需求比"].diff(33).astype("float64")
data_demand_240.dropna(inplace=True)
print("===================================================================================")

# plt.plot(data_demand_240)

# ADF检验
adf_result = ts.adfuller(data_demand_240,1,regresults=True)
print(adf_result)
print("====================================================")

# Q统计和LB统计
# boxpierce_test(data_demand_240)

# 绘制出自相关图与偏自相关图
fig = plt.figure(figsize=(12,8))
ax1=fig.add_subplot(211)
fig = sm.graphics.tsa.plot_acf(data_demand_240,lags=20,ax=ax1,alpha=0.05,title="自相关系数")
ax2 = fig.add_subplot(212)
fig = sm.graphics.tsa.plot_pacf(data_demand_240,lags=20,ax=ax2,alpha=0.05,title="偏自相关系数")
plt.show()

a=(sm.tsa.arma_order_select_ic(data_demand_240,max_ar=6,max_ma=6,ic='aic'))['aic_min_order']
p=a [0]
q=a[1]
print(a)

# ARMA(1,0)
arma_mod_auto = sm.tsa.ARMA(data_demand_240,(2,1)).fit()
# print(arma_mod_auto.aic,arma_mod_auto.bic,arma_mod_auto.hqic)
# print("=========================================================ARMA(1,0)")


#残差序列的检验
resid = arma_mod_auto.resid
# fig = plt.figure(figsize=(12,8))
# ax1 = fig.add_subplot(211)
# fig = sm.graphics.tsa.plot_acf(resid.values.squeeze(), lags=20, ax=ax1,title="残差自相关系数")
# ax2 = fig.add_subplot(212)
# fig = sm.graphics.tsa.plot_pacf(resid, lags=20, ax=ax2,title="残差偏自相关系数")
# plt.show()

# 检验是否符合正太分布
# fig =plt.figure(figsize=(12,8))
# ax = fig.add_subplot(111)
# fig = qqplot(resid,line="q",ax=ax,fit=True)
# plt.show()
# print(stats.normaltest(resid))

# 检验DW值
arma_mod_1_1_dw=sm.stats.durbin_watson(arma_mod_auto.resid.values)
print(arma_mod_1_1_dw)

#残差检验
r,q,p = sm.tsa.acf(resid.values.squeeze(), qstat=True)
data = np.c_[range(1,41), r[1:], q, p]
table = pd.DataFrame(data, columns=['lag', "AC", "Q", "Prob(>Q)"])
print(table.set_index('lag'))

# 时间序列拟合的误差
predict_ts = arma_mod_auto.predict()
diff_shift_ts = ori_data["需求比"].shift(33)
diff_recover_1 = predict_ts.add(diff_shift_ts)
plt.figure(facecolor='white')
diff_recover_1[33:].plot(color='blue', label='Predict')
ori_data["需求比"][33:].plot(color='red', label='Original')
plt.legend(loc='best')
plt.title('RMSE: %.4f'% np.sqrt(sum((diff_recover_1[33:]-ori_data["需求比"][33:])**2)/ori_data["需求比"][33:].size))
plt.show()
print("拟合误差")
print("MSE=%.8f"% mean_squared_error(diff_recover_1[33:],ori_data["需求比"][33:]))
print("RMSE=%.8f"% +np.sqrt(mean_squared_error(diff_recover_1[33:],ori_data["需求比"][33:])))
print("MAE=%.8f"%+mean_absolute_error(diff_recover_1[33:],ori_data["需求比"][33:]))
# hebing = DataFrame({"预测值":predict_ts,"实际值":ori_data["需求比"]})
# hebing.to_excel("D:\\停车\\研一第一学期论文\\ARIMA\\拟合误差.xlsx")


# 预测与实际的差值
forcast = arma_mod_auto.forecast(33)[1]
original_data= ori_data["需求比"][462:]
predict_data = forcast+original_data
predict_data.index = test_data.index
print(predict_data.index)
fig = plt.figure(figsize=(12,8))
plt.plot(predict_data.index,predict_data)
predict_data.plot(color='blue', label='Predict')
test_data.plot(color='red', label='Original')
plt.title('RMSE: %.4f'% np.sqrt(sum((predict_data-test_data)**2)/(test_data.size)))
plt.show()
print("预测误差")
print("MSE=%.8f"%+mean_squared_error(predict_data,test_data))
print("RMSE=%.8f"%+np.sqrt(mean_squared_error(predict_data,test_data)))
print("MAE=%.8f"%+mean_absolute_error(predict_data,test_data))
predict_to_excel =DataFrame({"预测":predict_data,"实际":test_data})
predict_to_excel.to_excel("D:\\停车\\研一第一学期论文\\高新国际\\yuce.xlsx")

# yuce_data = DataFrame({"预测值":forcast,"实际":ori_data["需求比"][462:]})
# yuce_data.to_excel("D:\\停车\\研一第一学期论文\\ARIMA\\第二周预测误差.xlsx")



# def  huadong(data):
# 	i=1
# 	a = []
# 	arma_mod_auto = sm.tsa.ARMA(np.asarray(data),(7, 6)).fit()
# 	cur_forcast = arma_mod_auto.forecast()[1]
# 	print(a)
# 	a.append(cur_forcast)
# 	data["%f"%i] = cur_forcast
# 	i+=1
# 	if  i<34:
# 		huadong(data)
# 	print(data)
	
	
# huadong(data_demand_240)




