from pathlib import Path
import json, shutil, hashlib, math
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

W=Path('work/olist-calibration');O=Path('outputs/olist-calibration-v2-2026-10-16')
r=json.loads((O/'results.json').read_text(encoding='utf-8'))
m=pd.read_csv(O/'metrics.csv');ev=m[m.stage=='evaluation'].set_index('model');cal=m[m.stage=='calibration'].set_index('model')
bins=pd.read_csv(O/'calibration_bins.csv');eb=bins[bins.stage=='evaluation']
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
colors={'A':'#8a8294','B':'#167a9b','C':'#b96935'}
fig,axs=plt.subplots(1,2,figsize=(12,4.8),layout='constrained')
for i,n in enumerate(['A','B','C']):
    axs[0].bar(i,ev.loc[n,'mean_prediction']*100,color=colors[n],width=.6)
    axs[0].text(i,ev.loc[n,'mean_prediction']*100+.025,f'{ev.loc[n,"mean_prediction"]*100:.3f}%',ha='center')
axs[0].axhline(ev.loc['A','observed_rate']*100,color='#33495d',ls='--',label='Observed 1.102%')
axs[0].set(xticks=[0,1,2],xticklabels=['A: month terms','B: no month terms','C: B + calibration'],ylabel='Average probability (%)',ylim=(0,1.35),title='Removing month terms helps; calibration overshoots downward')
axs[0].legend(loc='upper left',fontsize=8);axs[0].grid(axis='y',alpha=.12)
for n in ['A','B','C']:
    g=eb[eb.model==n]
    axs[1].plot(g.predicted*100,g.observed*100,'o-',label=n,color=colors[n])
mx=1.8
axs[1].plot([0,mx],[0,mx],'--',color='#888',label='Perfect calibration')
axs[1].set(xlabel='Predicted probability (%)',ylabel='Observed repeat rate (%)',xlim=(0,mx),ylim=(0,mx),title='Five equal-count groups; B and C have identical membership')
axs[1].legend(fontsize=8);axs[1].grid(alpha=.12)
fig.suptitle('Fixed A/B/C revision | Already-viewed evaluation: 7,893 customers, 87 repeats',fontsize=12)
fig.savefig(O/'calibration_comparison.png',dpi=170);plt.close(fig)

def tab(df):return df.to_html(index=False,border=0,escape=True,classes='data')
labels={'P_train':'训练期固定均值','P_cal':'校准期固定均值','A':'A 保留月份','B':'B 去掉月份','C':'C 去月份＋校准'}
rows=[]
for n in ['A','B','C','P_train','P_cal']:
    q=ev.loc[n]
    rows.append({'版本':labels[n],'平均预测':f'{q.mean_prediction*100:.3f}%','Log loss ↓':f'{q.log_loss:.6f}','Brier ↓':f'{q.Brier:.8f}','AP ↑':f'{q.AP:.6f}','ROC-AUC ↑':f'{q.ROC_AUC:.4f}','前790人复购数':'不排序' if n.startswith('P_') else str(int(q.top10_hits))})
cluster=next(b for b in r['evaluation_bootstrap'] if b['unit']=='state_cluster')['intervals']
introws=[]
for pair in ['B-A','C-B','B-P_train','C-P_cal']:
    n,o=pair.split('-')
    for metric in ['log_loss','Brier','AP']:
        ci=cluster[metric+'_'+pair]
        introws.append({'比较':pair,'指标':metric,'差值':f'{ev.loc[n,metric]-ev.loc[o,metric]:.8f}','按州重采样95%区间':f'[{ci["low"]:.8f}, {ci["high"]:.8f}]'})
delta_rows=[]
for b in r['calibrator_bootstrap']:
    ci=b['intervals']['delta'];p=b['intervals']['evaluation_mean_prediction']
    delta_rows.append({'重采样单位':b['unit'],'delta 95%区间':f'[{ci["low"]:.3f}, {ci["high"]:.3f}]','评估平均预测区间':f'{p["low"]*100:.3f}%—{p["high"]*100:.3f}%'})
prob=pd.DataFrame([{'时期':'2017年12月校准期','实际复购率':f'{cal.loc["B","observed_rate"]*100:.3f}%','B平均预测':f'{cal.loc["B","mean_prediction"]*100:.3f}%','C平均预测':f'{cal.loc["C","mean_prediction"]*100:.3f}%'},{'时期':'2018年4—5月修订评估期','实际复购率':f'{ev.loc["B","observed_rate"]*100:.3f}%','B平均预测':f'{ev.loc["B","mean_prediction"]*100:.3f}%','C平均预测':f'{ev.loc["C","mean_prediction"]*100:.3f}%'}])
before=json.loads((W/'v1_before.json').read_text(encoding='utf-8'))
assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in before.items())
body=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>Olist A/B/C 固定修订实验</title>
<style>body{{font-family:"Microsoft YaHei",system-ui,sans-serif;background:#eef3f7;color:#183247;line-height:1.8;margin:0}}main{{max-width:1150px;margin:28px auto;padding:34px;background:white;border-radius:12px}}h1{{font-size:28px}}h2{{font-size:21px;margin-top:34px}}.lead{{background:#e7f1f5;border-left:5px solid #167a9b;padding:18px}}.warn{{background:#fff2de;padding:18px}}table{{border-collapse:collapse;width:100%;font-size:14px;margin:15px 0}}th,td{{padding:9px;border-bottom:1px solid #dce4ec;text-align:right}}th:first-child,td:first-child{{text-align:left}}th{{background:#eef3f7}}.scroll{{overflow-x:auto}}img{{width:100%;height:auto}}a{{color:#1d6385}}code{{word-break:break-all}}small{{color:#506575}}@media(max-width:700px){{main{{margin:0;padding:16px}}}}@media print{{body{{background:white}}main{{margin:0;padding:0}}}}</style><main>
<h1>Olist：去月份与概率校准的固定修订实验</h1><p>2026-10-16 · A/B/C 一次固定比较 · 保留全部上一版结果</p>
<div class="lead"><b>结论：</b>去掉月份项明显缓解整体概率低估；排序没有改善。再用12月数据做截距校准后，后续评估的概率表现反而略变差。校准前后所有客户排序完全一致。没有追加其他模型、参数或校准器。</div>
<p class="warn"><b>证据边界：</b>本实验由上一版已查看的结果触发，评估仍使用同一批客户，属于探索性修订，不能声称新的独立验证。以下不确定区间不能消除这种选择与信息复用带来的偏差。</p>
<h2>1. 冻结方案与公平比较</h2>
<p>底层模型训练：2017年7—8月，6,459人/71次复购。校准：2017年12月，6,862人/60次复购。修订评估：2018-04-01至05-02，7,893人/87次复购。两个时间边界均满足90天结果已经成熟。模型、校准器及评估人群互不重叠，底层模型校准后不再重训。</p>
<ul><li><b>A：</b>保留月份正余弦的M2，首购背景＋履约体验。</li><li><b>B：</b>仅删除月份正余弦，其余输入、预处理和参数与A一致。</li><li><b>C：</b>冻结B，仅在12月拟合一个截距修正量 delta，斜率固定为1。</li></ul>
<p>统一使用L2逻辑回归C=0.1、lbfgs、最大3000次迭代。中位数填补、标准化、低频类别合并只在训练集拟合。不调参、不改变排除条件，不增加品类/地区交互。不训练LightGBM。</p>
<p>上一版最终模型使用25,780人/303次复购重训；本轮只有6,459人/71次复购。A/B/C可相互比较，但不能把本轮与上一版最终模型的差值全部归因于去月份或校准。A在12月的预测已复核与上一版初始M2一致。</p>
<h2>2. 修订评估结果</h2><p>实际复购率 <b>1.102%</b>。Log loss与Brier越小越好；AP与ROC-AUC越大越好。AP使用Average Precision，不是梯形PR面积。</p>
<div class="scroll">{tab(pd.DataFrame(rows))}</div>
<img src="calibration_comparison.png" alt="ABC平均预测及校准曲线">
<p>B把平均概率从A的0.139%提高到0.942%，相对实际1.102%明显更接近，Log loss和Brier也改善。AP从0.015916略降至0.015668，前790人命中从12人变为11人，不能说排序能力提高。ROC-AUC小幅升高但与其他排序指标方向不完全一致，不选择性强调。</p>
<p>C平均预测进一步降至0.819%，低估扩大，Log loss和Brier相对B点估计变差。其差值区间包含零，因此描述为“本次评估没有收益、点估计略差”，不声称稳定变差。</p>
<p>训练期固定均值1.099%恰好接近本次实际1.102%，Log loss优于B；Brier几乎相同，B仅有极小点估计优势。B相对这一简单基准的概率指标差值区间均包含零，因此不能称为概率预测已经可靠。</p>
<h2>3. 为什么校准没有继续变好</h2>{tab(prob)}
<p>12月B预测1.006%，实际0.874%，因此校准器估计 delta=<b>{r['delta']:.6f}</b>，相当于把预测胜算乘以<b>{r['odds_multiplier']:.4f}</b>，整体下调。到了后续评估，B原本已经低于实际，继续下调使偏差扩大。这与两个时期的发生率和预测偏差不同一致，不能仅凭本次数据断言已经识别某种真实时间漂移机制。</p>
<p>公式：p_C = sigmoid(logit(p_B) + delta)。12月校准后的平均预测恰好等于实际，这是拟合方程本身的性质，不是校准成功的独立证据。这个方法只能整体平移概率，不能解决分组形状错误或保证跨时期稳定。</p>
<h2>4. 排序核验</h2>
<ul><li>评估集7,893位客户：B、C完整排序逐项相同，改变位置数为0。</li><li>前10%名单790人：重合790人，复购均为11人。</li><li>AP、ROC-AUC精确相同；校准期也通过同样检查。</li><li>B/C的11人命中率为1.392%，覆盖87位复购客户的12.64%，约为随机筛选预期命中率的1.26倍。</li></ul>
<h2>5. 不确定性：不能只看一个分数</h2>
<p>以下差值均为“前者减后者”。Log loss/Brier负值为改善，AP正值为改善。对同一批固定预测做客户级与州级各1000次配对bootstrap；表中展示州级结果。</p>
<div class="scroll">{tab(pd.DataFrame(introws))}</div>
<p>B相对A的Log loss、Brier改善在两种重采样中方向一致，区间都不含零；AP差异区间包含零。C相对B的概率指标差异区间包含零；AP差异严格为零。此处不包括底层模型训练不稳定性，也不构成新的独立确认。</p>
<p>校准只有60个复购事件。另对校准集进行两种各1000次重采样、重新估计delta，底层模型和评估样本固定：</p>{tab(pd.DataFrame(delta_rows))}
<p>delta区间包含零，州级区间尤其宽，说明整体修正量本身也不稳定。以上校准不确定性与前面的评估抽样不确定性分别估计，不能把任一个当作包含所有来源的总区间。</p>
<h2>6. 本轮停止位置</h2>
<p>支持“有限月份训练下，去掉周期项能缓解本次外推问题”；不支持“去月份提高了排序”，也不支持“用12月截距校准能改善未来概率”。B仍低估实际，且概率指标没有明显优于简单历史均值，因此月份与校准问题尚未全部解决。</p>
<p>本轮按预先约定到此停止。未删除其他变量、未更换校准月份、未尝试其他校准方法或超参数，也没有根据评估结果追加LightGBM。下一步需要重新讨论验证证据和用途，而非在同一批已知结果上继续追分。</p>
<h2>7. 保存与核验</h2>
<p>上一版输出、模型、预测及压缩包共37个文件在运行前后SHA-256全部相同。仅训练A/B两套底层模型并拟合一个校准截距，两套模型均收敛。AP、Brier另行计算核对一致，时间成熟边界与人群不重叠检查通过。</p>
<p>注册时间（UTC）：{r['registration']['registered_utc']}。设置哈希：<code>{r['registration']['protocol_sha256']}</code>。该注册发生于本次拟合前，不是V1测试查看前的预注册。</p>
<p><a href="results.json">全部结果与区间</a> · <a href="metrics.csv">指标表</a> · <a href="calibration_bins.csv">概率分组</a> · <a href="protocol.json">固定实验设置</a> · <a href="v1_preservation.json">上一版保留核验</a> · <a href="REPRODUCE.txt">复现说明</a></p>
<p>源数据与标签沿用上一版，数据来源：<a href="https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce">Olist公开数据</a>。此次未更新GitHub。</p>
</main></html>'''
(O/'ABC固定实验结果与解释.html').write_text(body,encoding='utf-8')
shutil.copy2(W/'report.py',O/'report.py')
shutil.copy2(W/'v1_before.json',O/'v1_before.json')
shutil.copy2('outputs/olist-logistic-2026-10-16/requirements.txt',O/'requirements.txt')
(O/'REPRODUCE.txt').write_text('''本轮为查看V1结果后的固定探索修订，不能当作新的独立验证。
运行依赖上一版已审计的 work/olist-logistic/model_frame.pkl 和 validation_predictions.pkl。
上一版完整数据准备与特征源码保留在 outputs/olist-logistic-2026-10-16/。
本轮源文件位于 work/olist-calibration/：protocol.json、registration.json、v1_before.json、run_fixed.py、report.py。
使用同一环境，从原工作目录执行：
work/olist-model-env/Scripts/python.exe work/olist-calibration/run_fixed.py
work/olist-model-env/Scripts/python.exe work/olist-calibration/report.py
run_fixed.py 会校验注册设置、上一版37个文件哈希，固定拟合A/B，并只校准B的截距。
结果写入 outputs/olist-calibration-v2-2026-10-16/；模型与客户级预测仅写入 work/olist-calibration/。
本结果包不包含原始数据、模型pickle或客户级预测。
跨机器复制时需先复现V1并建立对应路径。原始v1_before.json用于本机保存验证；新机器如文件字节变化，必须另存新的保存记录，而非绕过哈希断言。
''',encoding='utf-8')
print(ev[['mean_prediction','log_loss','Brier','AP','top10_hits']].to_string(float_format=lambda v:f'{v:.10f}'))
print(O/'ABC固定实验结果与解释.html')
