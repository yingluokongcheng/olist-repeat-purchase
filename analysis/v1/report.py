from pathlib import Path
import json, html, shutil, hashlib
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

O=Path('outputs/olist-logistic-2026-10-16')
r=json.loads((O/'results.json').read_text(encoding='utf-8'))
m=pd.read_csv(O/'model_metrics.csv'); t=m[m.stage=='test'].set_index('model')
cal=pd.read_csv(O/'calibration.csv'); cases=pd.read_csv(O/'descriptive_cases.csv')
coef=pd.read_csv(O/'coefficients.csv')
cluster=next(b for b in r['bootstrap'] if b['unit']=='state_cluster')['intervals']
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'figure.facecolor':'white'})
colors={'M1':'#6b7c93','M2':'#207c9d','M3':'#b66337'}
fig,axes=plt.subplots(1,2,figsize=(12,4.8),layout='constrained')
ax=axes[0]
for i,model in enumerate(['M1','M2','M3']):
    point=t.loc[model,'AP']; b=cluster['AP_'+model]
    ax.errorbar(point*100,i,xerr=[[100*(point-b['low'])],[100*(b['high']-point)]],fmt='o',capsize=4,color=colors[model],markersize=7)
ax.axvline(t.loc['M0','AP']*100,color='#888888',ls='--',label='Constant-score baseline')
ax.set(yticks=[0,1,2],yticklabels=['M1  Background','M2  + Fulfillment','M3  + Regional mix'],xlabel='Average precision (%)',title='Ranking: small gains, wide uncertainty')
ax.invert_yaxis();ax.legend(fontsize=8,loc='upper center',bbox_to_anchor=(.5,-.17));ax.grid(axis='x',alpha=.18)
ax=axes[1]
for model in ['M1','M2','M3']:
    g=cal[cal.model==model]
    ax.plot(g.predicted*100,g.observed*100,'o-',label=model,color=colors[model],alpha=.8)
lim=max(cal.observed.max(),cal.predicted.max())*100*1.12
ax.plot([0,lim],[0,lim],'--',color='#888888',label='Perfect calibration')
ax.set(xlabel='Predicted probability (%)',ylabel='Observed repeat rate (%)',title='Probabilities are systematically too low',xlim=(0,lim),ylim=(0,lim))
ax.legend(fontsize=8);ax.grid(alpha=.18)
fig.suptitle('Olist logistic regression | Final time-held-out test: 7,893 customers, 87 repeats',fontsize=12)
fig.savefig(O/'model_comparison.png',dpi=170);plt.close(fig)

fig,axes=plt.subplots(1,2,figsize=(11,4.7),layout='constrained')
names={'bed_bath_table':'Bed, bath & table','sports_leisure':'Sports & leisure'}
for ax,cat in zip(axes,names):
    g=cases[(cases.category==cat)&(cases.dimension=='late')].sort_values('group')
    ys=g.rate.to_numpy()*100
    ax.errorbar([0,1],ys,yerr=[ys-g.low.to_numpy()*100,g.high.to_numpy()*100-ys],fmt='o',capsize=5,color='#207c9d')
    ax.set(xticks=[0,1],xticklabels=['On time','Late'],ylabel='90-day repeat rate (%)',title=names[cat],xlim=(-.5,1.5),ylim=(0,3))
    for i,(_,row) in enumerate(g.iterrows()): ax.annotate(f'{int(row.repeat90)} / {int(row.customers):,}',(i,ys[i]),xytext=(8,0),textcoords='offset points',fontsize=9)
    ax.grid(axis='y',alpha=.18)
fig.suptitle('Descriptive only: full cohort, unadjusted rates with Wilson 95% intervals',fontsize=11)
fig.savefig(O/'case_delivery.png',dpi=170);plt.close(fig)
fig,axes=plt.subplots(1,2,figsize=(11,4.7),layout='constrained')
for ax,cat in zip(axes,names):
    g=cases[(cases.category==cat)&(cases.dimension=='match_group')].sort_values('group')
    ys=g.rate.to_numpy()*100
    ax.errorbar(np.arange(4),ys,yerr=[ys-g.low.to_numpy()*100,g.high.to_numpy()*100-ys],fmt='o-',capsize=4,color='#b66337')
    ax.set(xticks=np.arange(4),xticklabels=['Q1 (low)','Q2','Q3','Q4 (high)'],ylabel='90-day repeat rate (%)',title=names[cat],ylim=(0,3.6),xlabel='Within-category regional relative-share quartile')
    ax.grid(axis='y',alpha=.18)
fig.suptitle('Descriptive regional mix: no consistent monotonic pattern in these two cases',fontsize=11)
fig.savefig(O/'case_regional.png',dpi=170);plt.close(fig)

def tab(df): return df.to_html(index=False,border=0,classes='data',float_format=lambda v:f'{v:.6f}',escape=True)
def pct(v): return f'{100*v:.2f}%'
table_rows=[]
labels={'M0':'固定训练期平均值','M1':'首购背景','M2':'加履约体验','M3':'再加地区消费结构'}
for model in labels:
    a=t.loc[model]
    table_rows.append({'模型':model+' '+labels[model],'AP ↑':f'{a.AP:.5f}','Brier ↓':f'{a.Brier:.6f}','Log loss ↓':f'{a.log_loss:.5f}','ROC-AUC ↑':f'{a.ROC_AUC:.3f}','平均预测概率':pct(a.mean_prediction),'前790人复购数':'无排序能力' if model=='M0' else str(int(a.top10_hits))})
interval_rows=[]
for pair in ['M2-M1','M3-M2']:
    new,old=pair.split('-');b=cluster['dAP_'+pair]
    interval_rows.append({'比较':pair,'AP差值':f'{t.loc[new,"AP"]-t.loc[old,"AP"]:.6f}','按州重采样95%区间':f'[{b["low"]:.6f}, {b["high"]:.6f}]'})
case_table=cases[cases.dimension.isin(['overall','late'])].copy()
case_table['品类']=case_table.category.map({'bed_bath_table':'床浴桌用品','sports_leisure':'运动休闲'})
case_table['分组']=[('全部' if d=='overall' else ('准时' if str(g)=='0' else '延误')) for d,g in zip(case_table.dimension,case_table.group)]
case_table['复购率']=case_table.rate.map(pct)
case_table['95%区间']=[f'{pct(lo)}—{pct(hi)}' for lo,hi in zip(case_table.low,case_table.high)]
case_table=case_table[['品类','分组','customers','repeat90','复购率','95%区间']].rename(columns={'customers':'客户','repeat90':'复购'})
val=m[m.stage=='validation'][['model','AP','Brier','mean_prediction','top10_hits']].copy()
fit_checks=pd.DataFrame(r['fit_checks']).drop(columns='warnings')
shift=pd.read_csv(O/'prediction_shift_decomposition.csv')
month_shift=float(shift[shift.feature.isin(['num__month_sin','num__month_cos'])].test_minus_refit_logit_contribution.sum())
source=r['feature_checks']['source_zip_sha256']
body=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>Olist 逻辑回归结果</title>
<style>body{{font-family:"Microsoft YaHei",system-ui,sans-serif;background:#eef3f7;color:#193047;line-height:1.8;margin:0}}main{{max-width:1120px;margin:28px auto;padding:34px;background:white;border-radius:12px}}h1{{font-size:28px}}h2{{font-size:21px;margin-top:34px}}.lead{{background:#e7f1f5;border-left:5px solid #207c9d;padding:18px}}.warn{{background:#fff2de;padding:18px}}table{{border-collapse:collapse;width:100%;font-size:14px;margin:15px 0}}th,td{{text-align:right;border-bottom:1px solid #dce4ec;padding:10px}}th:first-child,td:first-child{{text-align:left}}th{{background:#eef3f7}}.scroll{{overflow-x:auto}}img{{width:100%;height:auto;border:1px solid #e4eaf0;border-radius:6px}}a{{color:#20718b}}code{{word-break:break-all}}small{{color:#526579}}@media(max-width:700px){{main{{margin:0;padding:16px}}}}@media print{{body{{background:white}}main{{margin:0;padding:0}}}}</style><main>
<h1>Olist：履约体验与地区消费结构的增量预测价值</h1>
<p>2026-10-16 · 三组整体逻辑回归 · 两个品类描述案例 · 未运行 LightGBM</p>
<div class="lead"><b>结果：</b>履约信息带来小幅排序改善，但主要指标的差值区间包含零。地区消费结构的额外贡献接近零。模型平均复购概率约0.54%，实际为1.10%，概率校准明显不足。</div>
<h2>1. 实验设置</h2>
<p>沿用已审计的51,981位客户、567次90天复购。初始训练6,459人/71次复购，验证6,862人/60次复购；固定设置后使用2017年底前已成熟的25,780人/303次复购重新训练，测试2018-04-01至05-02送达的7,893人/87次复购。训练与验证、最终训练与测试均满足90天标签成熟要求。</p>
<p>先保存实验方案，再训练。三组均使用L2逻辑回归（C=0.1、lbfgs、最大3000次迭代）；不调参、不做类别重采样或权重平衡。缺失值中位数、标准化和类别编码均只在各自训练集学习。出现不足200次的品类/州在编码中归入低频组。验证集只用于诊断，没有根据其分数选模型设置。代码的特征白名单不包含标签、复购日期或客户标识。</p>
<ul><li><b>M1：</b>首购金额、件数、重量、体积、买卖双方近似距离、多卖家标记、缺失标记、月份正余弦、品类和州。</li><li><b>M2：</b>M1加送达耗时、是否延误、延误天数、运费/商品金额。连续长尾变量先log1p再标准化。</li><li><b>M3：</b>M2加首购品类的当地平滑占比，以及当地/全国占比的对数。多品类订单按品类平均。</li></ul>
<p>地区特征使用首购前180天且当时已经审批的订单，排除自身；每笔历史订单在品类间分配总权重1。全国品类占比使用每类0.5的固定平滑，州级占比向全国收缩100笔订单等价量。两个参数事前固定，没有用验证或测试结果优化。地区特征只反映Olist观察到的成交结构，不等同真实需求或文化。</p>
<h2>2. 最终测试结果</h2><p>测试实际复购率为 <b>1.102%</b>。AP为Average Precision，是非插值的PR曲线汇总指标；不是ROC-AUC，也不是概率准确率。此处不使用梯形积分PR-AUC。↑越高越好，↓越低越好。</p>
<div class="scroll">{tab(pd.DataFrame(table_rows))}</div>
<p>M2筛选前790人识别19位复购客户，命中率2.405%、覆盖全部复购客户的21.84%，约为随机筛选预期命中率的2.18倍；M1识别17位。M3与M2同为19位。这是排序结果，不是营销带来的增量复购。</p>
<img src="model_comparison.png" alt="三个模型的AP区间及概率校准曲线">
<h2>3. 增量是否稳定</h2>{tab(pd.DataFrame(interval_rows))}
<p>客户级与州级各进行1000次配对bootstrap，使用同一批固定测试预测比较模型。表中为州级重采样95%百分位区间。两次增量AP区间均包含零，不能声称有稳定改善。州级样本仅27组且大小不均，区间也只能作为不确定性诊断。bootstrap没有重新训练模型，未包含模型训练不稳定性、时间漂移或渠道覆盖误差。</p>
<p>M2的Brier相比M1略降；客户级区间不含零，但州级区间包含零。不能只挑一种重采样方式宣称稳健提升。M3相对M2在Brier和log loss上几乎相同。</p>
<h2>4. 更重要的问题：概率偏低</h2>
<div class="warn">M2/M3平均预测约<b>0.54%</b>，实际为<b>1.10%</b>。直接使用训练期平均复购率的M0，Brier和log loss反而更好。因此当前模型有一定排序能力，但不能把输出概率直接用于利润计算或精细运营决策。</div>
<p>固定模型的输入贡献分解显示，从最终训练池到测试期，月份正余弦合计使平均log-odds变化约 <b>{month_shift:.3f}</b>，是概率下移的主要模型内贡献。最终训练主要覆盖2017年下半年，测试位于次年春季，季节项在有限月份上的外推不可靠。这是模型预测的数学分解，不是已经证明真实季节效应，也不是对全部校准误差的因果解释。</p>
<p>首次验证也已出现低估：实际0.874%，M1/M2/M3分别平均预测约0.255%/0.214%/0.212%。按事先固定比较，本轮没有看到测试结果后再删月份变量、重新校准或换参数。后续若修改，需要把它标为新的探索版本，并避免继续把本测试集称作全新的独立证据。</p>
<details><summary>验证集详细指标</summary>{tab(val)}</details>
<h2>5. 模型系数能解释什么</h2>
<p>M3中，送达耗时和运费负担的标准化系数为负；当地品类占比和相对占比系数接近零。延误标记与延误天数同时存在，系数分别为负和正，必须联合理解，不能据此说“延误越久越好”。所有系数都经过正则化，且只表示控制已纳入变量后的预测关联；本轮不报告因果效应或显著性结论。</p>
<h2>6. 两个品类仅做描述</h2>{tab(case_table)}
<img src="case_delivery.png" alt="两个品类准时及延误客户的复购率和区间">
<p>两个品类中，延误组的原始复购率都较低，但事件仅5个和3个，区间很宽。这些比较未调整金额、地区或商品差异，不能据此认定改善配送会提高复购。</p>
<img src="case_regional.png" alt="两个品类按地区相对品类占比四分位分组的描述复购率">
<p>按各品类内部的地区相对占比分成四组后，两者都没有一致的单调上升形态。这里使用全审计队列，仅为描述，不能充当独立验证或反过来调整模型。区间为客户级Wilson区间，未处理州内相关性。</p>
<h2>7. 当前建议</h2>
<p>保留M1—M3这次完整比较，如实报告“履约信息有弱信号、地区消费结构未显示稳定的额外线性预测价值”。这不证明地区因素不重要：州和品类已在M1中，M3检验的是两项特定历史结构指标在其上的额外贡献，且受稀少复购和模型形式限制。</p>
<p>下一步优先讨论季节项外推与概率校准，再决定是否加入LightGBM。复杂算法不会自动增加复购事件或解决观察数据的因果识别问题。本轮止于逻辑回归与描述案例，没有自动运行LightGBM或发布GitHub。</p>
<h2>8. 复核、文件与来源</h2>
<p>三组模型在两个训练阶段共拟合6次，全部收敛。复购标签沿用前次逐客户复核结果；30条固定随机样本的地区加权历史经直接过滤核对一致；AP和Brier独立重算一致；训练与测试客户不重叠、标签成熟边界检查通过。</p>
<details><summary>拟合检查</summary>{tab(fit_checks)}</details>
<p><a href="results.json">完整机器可读结果</a> · <a href="model_metrics.csv">模型指标</a> · <a href="descriptive_cases.csv">品类描述表</a> · <a href="coefficients.csv">系数</a> · <a href="calibration.csv">校准分组</a> · <a href="state_diagnostics.csv">州级诊断</a> · <a href="protocol.json">固定实验设置</a> · <a href="REPRODUCE.txt">复现说明</a></p>
<p>数据：<a href="https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce">Olist发布页</a>。方法：<a href="https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html">scikit-learn逻辑回归</a>、<a href="https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html">Average Precision定义</a>。使用scikit-learn {r['versions']['sklearn']}。</p>
<small>源ZIP SHA-256：<code>{source}</code>。本地报告，不代表商业部署或因果验证。</small>
</main></html>'''
(O/'逻辑回归结果与解释.html').write_text(body,encoding='utf-8')
shutil.copy2('work/dataset-scout/sample_audit.py',O/'sample_audit.py')
shutil.copy2('work/olist-logistic/report.py',O/'report.py')
shutil.copy2('work/olist-logistic/run_models.py',O/'run_models.py')
versions=r['versions']
(O/'requirements.txt').write_text('\n'.join([f'pandas=={versions["pandas"]}',f'numpy=={versions["numpy"]}',f'scipy=={versions["scipy"]}',f'scikit-learn=={versions["sklearn"]}',f'matplotlib=={matplotlib.__version__}','joblib==1.6.0','threadpoolctl==3.7.0'])+'\n',encoding='utf-8')
(O/'REPRODUCE.txt').write_text('''Olist 逻辑回归复现说明
本目录是已完成结果与源码快照，未包含原始数据、模型pickle或客户级预测。
在原工作目录中，使用 work/olist-model-env/Scripts/python.exe 按以下顺序运行：
1. work/dataset-scout/sample_audit.py
2. work/olist-logistic/features.py
3. work/olist-logistic/run_models.py
4. work/olist-logistic/report.py

在新目录复现时：
- 建立 work/dataset-scout、work/olist-logistic 和 outputs 文件夹。
- 将本目录的 sample_audit.py 放到 work/dataset-scout/。
- 将 features.py、run_models.py、report.py、protocol.json 放到 work/olist-logistic/。
- 从 Olist 原始发布页下载ZIP，保存为 work/dataset-scout/olist.zip；核对results.json中的SHA256。
- 新建Python环境，安装requirements.txt中的依赖；从新目录根运行上述脚本。
- sample_audit.py 会重建审计队列缓存，features.py 重建时点特征，run_models.py 训练并生成指标。
- run_models.py 同时输出 prediction_shift_decomposition.csv，report.py 读取固定预测的诊断结果并生成报告。

AP使用Average Precision，不是梯形插值PR-AUC。
验证阶段仅诊断；C=0.1预先固定，未优化其他超参数。
最终训练含验证期，但其90天结果在测试期开始前已成熟。
只有整体逻辑回归，无LightGBM。品类使用全队列做探索描述，不做模型或因果估计。
''',encoding='utf-8')
print(str(O/'逻辑回归结果与解释.html'))
