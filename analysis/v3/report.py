from pathlib import Path
import json, hashlib, shutil
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

W=Path('work/olist-lightgbm');O=Path('outputs/olist-lightgbm-v3-2026-10-16')
r=json.loads((O/'results.json').read_text(encoding='utf-8'))
m=pd.read_csv(O/'metrics.csv');ev=m[m.stage=='evaluation'].set_index('model');dev=m[m.stage=='development'].set_index('model')
curve=pd.read_csv(O/'learning_curves.csv');bins=pd.read_csv(O/'calibration_bins.csv')
pred=pd.read_pickle(W/'evaluation_predictions.pkl')
ties=[]
for name in ['LR','G1','G2']:
    p=pred[name].to_numpy();y=pred.y90.to_numpy();idx=np.argsort(-p,kind='stable');threshold=p[idx[789]];above=p>threshold;tied=p==threshold;slots=790-int(above.sum());e=int(y[tied].sum())
    ties.append({'model':name,'strictly_above':int(above.sum()),'boundary_tie_customers':int(tied.sum()),'boundary_tie_events':e,'boundary_slots':slots,'expected_hits_if_uniform_tie_break':float(y[above].sum()+slots*e/tied.sum()),'observed_fixed_order_hits':int(y[idx[:790]].sum())})
(O/'tie_diagnostics.json').write_text(json.dumps(ties,indent=2),encoding='utf-8')
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
colors={'LR':'#738195','G1':'#187c9a','G2':'#b46a36'}
fig,axs=plt.subplots(1,2,figsize=(12,4.6),layout='constrained')
for ax,name in zip(axs,['G1','G2']):
    g=curve[curve.model==name];best=r['locked_models']['models'][name]['best_iteration']
    ax.plot(g.iteration,g.development_log_loss,color=colors[name]);ax.axvline(best,ls='--',color='#666',label=f'Selected round {best}')
    ax.axhline(dev.loc['P_train','log_loss'],color='#999',ls=':',label='Training-rate constant')
    ax.set(title=name,xlabel='Boosting round',ylabel='December development log loss');ax.legend(fontsize=8);ax.grid(alpha=.15)
fig.suptitle('Fixed early stopping: log loss only, patience 50, min_delta 0.00001',fontsize=12)
fig.savefig(O/'learning_curves.png',dpi=170);plt.close(fig)
fig,axs=plt.subplots(1,2,figsize=(12,4.9),layout='constrained')
for i,name in enumerate(['LR','G1','G2']):
    ap=ev.loc[name,'AP'];axs[0].bar(i,ap,color=colors[name],width=.6);axs[0].text(i,ap+.0003,f'{ap:.5f}',ha='center')
axs[0].axhline(ev.loc['P_train','AP'],ls='--',color='#888',label='Constant-score baseline')
axs[0].set(xticks=[0,1,2],xticklabels=['LR (V2 B)','G1','G2 (+ region)'],ylabel='Average precision',ylim=(0,.019),title='G2 recovers G1 ranking loss, but does not exceed LR AP');axs[0].legend(fontsize=8,loc='lower right')
for name in ['LR','G1','G2']:
    g=bins[bins.model==name]
    axs[1].plot(g.predicted*100,g.observed*100,'o-',color=colors[name],label=name)
axs[1].plot([0,2.2],[0,2.2],'--',color='#888',label='Perfect calibration')
axs[1].set(xlabel='Predicted probability (%)',ylabel='Observed repeat rate (%)',title='Near-correct average does not imply calibrated subgroups',xlim=(0,2.2),ylim=(0,2.2));axs[1].legend(fontsize=8);axs[1].grid(alpha=.15)
fig.suptitle('Final exploratory comparison | Previously viewed evaluation: 7,893 customers / 87 repeats',fontsize=12)
fig.savefig(O/'model_comparison.png',dpi=170);plt.close(fig)

def tab(df):return df.to_html(index=False,border=0,escape=True,classes='data')
names={'LR':'LR：去月份逻辑回归B','G1':'G1：相同信息的树模型','G2':'G2：增加地区消费结构','P_train':'训练期固定均值','P_dev':'开发期固定均值'}
rows=[]
for name in names:
    q=ev.loc[name]
    rows.append({'模型':names[name],'平均预测':f'{100*q.mean_prediction:.3f}%','AP ↑':f'{q.AP:.6f}','ROC-AUC ↑':f'{q.ROC_AUC:.4f}','Log loss ↓':f'{q.log_loss:.6f}','Brier ↓':f'{q.Brier:.8f}','前790人复购数':'不排序' if name.startswith('P_') else str(int(q.top_hits))})
bootrows=[]
for pair in ['G1-LR','G2-G1']:
    new,old=pair.split('-')
    for metric in ['AP','log_loss','Brier']:
        row={'比较':pair,'指标':metric,'差值':f'{ev.loc[new,metric]-ev.loc[old,metric]:.8f}'}
        for b in r['bootstrap']:
            ci=b['intervals'][metric+'_'+pair];row['客户95%区间' if b['unit']=='customer' else '州95%区间']=f'[{ci["low"]:.8f}, {ci["high"]:.8f}]'
        bootrows.append(row)
fitrows=[]
for name,info in r['locked_models']['models'].items():
    fitrows.append({'模型':name,'选中轮数':info['best_iteration'],'共评估轮数':info['evaluated_iterations'],'选中开发集Log loss':f'{info["selected_dev_logloss"]:.8f}','原始数值最小轮数':info['numerical_min_iteration'],'选中模型树数':info['selected_tree_count']})
protocolrows=pd.DataFrame([{'参数':k,'固定值':str(v)} for k,v in r['protocol']['parameters'].items()])
body=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>Olist 最后一次LightGBM固定对照</title>
<style>body{{font-family:"Microsoft YaHei",system-ui,sans-serif;background:#eef3f7;color:#193249;line-height:1.8;margin:0}}main{{max-width:1160px;margin:28px auto;background:white;padding:34px;border-radius:12px}}h1{{font-size:28px}}h2{{font-size:21px;margin-top:34px}}.lead{{background:#e7f1f5;padding:18px;border-left:5px solid #187c9a}}.warn{{background:#fff2df;padding:18px}}table{{border-collapse:collapse;width:100%;font-size:14px;margin:15px 0}}th,td{{border-bottom:1px solid #dce4ec;padding:9px;text-align:right}}th:first-child,td:first-child{{text-align:left}}th{{background:#edf2f7}}.scroll{{overflow-x:auto}}img{{width:100%;height:auto}}a{{color:#1c6583}}code{{word-break:break-all}}small{{color:#526678}}@media(max-width:700px){{main{{margin:0;padding:16px}}}}@media print{{body{{background:white}}main{{margin:0;padding:0}}}}</style><main>
<h1>Olist：最后一次 LightGBM 固定对照</h1><p>2026-10-16 · LR、G1、G2 · 无追加调参、重训或校准</p>
<div class="lead"><b>结论：</b>G1更接近总体发生率，但排序AP低于逻辑回归。G2相对G1有排序和概率指标的点估计改善，前790人命中18位，然而增量区间包含零；其整体AP仍略低于逻辑回归。没有达到预先约定的明确优势标准，不据此替换逻辑回归或宣称地区作用已验证。</div>
<p class="warn">评估集在前两轮已经查看，本轮属于最终探索性算法对照，不是新的独立验证。固定方案可以限制继续追分，不能消除已经使用过评估信息的影响。</p>
<h2>1. 数据与比较对象</h2>
<p>训练为2017年7—8月6,459人/71次复购，开发集为12月6,862人/60次复购，修订评估为2018-04-01至05-02的7,893人/87次复购。时间边界满足90天标签已成熟。LR直接加载V2的B，未重训，预测逐项与V2核对一致。</p>
<ul><li><b>G1：</b>与LR相同首购背景和履约信息，直接复用LR训练期预处理、缺失填补、标准化及低频类别编码。</li><li><b>G2：</b>G1加首购前180天的当地平滑品类占比、当地/全国占比对数。追加特征的预处理只拟合训练数据。</li></ul>
<p>不包含月份，不增加其他变量，不做类别平衡、概率校准或开发集后重训。只比较一套参数和一个随机种子。地区指标延续前次时点计算，非真实文化或完整需求测量。</p>
<h2>2. 早停结果完整保留</h2>{tab(pd.DataFrame(fitrows))}
<p>只监控开发集binary_logloss，最多500轮，连续50轮没有超过0.00001的改善即停止。使用回调记录的best_iteration_；因存在min_delta，它不一定等于原始数值最小的轮数。G1选中第1轮后没有满足规则的进一步改善，所以保留仅1棵树，没有放宽参数。G2选中第146轮。</p>
<img src="learning_curves.png" alt="G1和G2完整开发集学习曲线与早停轮数">
<p>两个模型及轮数先保存并哈希锁定，随后才生成评估预测。G1/G2选择了不同轮数，因此G2-G1衡量的是“加入地区信息并按同一早停规则选择模型”的流程增量，不是固定树数量下的纯特征效应。</p>
<h2>3. 与逻辑回归的结果比较</h2><p>评估实际复购率为<b>1.102%</b>。AP为Average Precision，↑越高越好；Log loss、Brier为概率损失，↓越低越好。</p>
<div class="scroll">{tab(pd.DataFrame(rows))}</div>
<img src="model_comparison.png" alt="排序AP与五组概率校准的比较">
<p><b>G1对LR：</b>AP从0.015668降至0.012260，ROC-AUC也下降。平均概率更接近实际，Log loss点估计改善，但增量不确定区间包含零。它的概率几乎都挤在总体发生率附近，更接近总体均值并不等于更善于识别个人差异。</p>
<p><b>G2对G1：</b>AP提高约0.003051，Log loss和Brier点估计均改善；前790人中命中18位，命中率2.278%，覆盖全部复购客户20.69%，约为随机筛选预期命中率的2.07倍。但两种重采样的AP增量区间都包含零，概率指标增量区间也包含零，未达到事前约定的明确改善标准。</p>
<p><b>G2对LR：</b>前790人命中从11位变成18位，概率损失点估计更好，但整体AP和ROC-AUC略低。这是固定筛选规模与整体排序指标之间的取舍，不能只取18比11就宣布树模型整体胜出。G2相对训练期固定均值的概率改善也很小、区间包含零。</p>
<h2>4. G1的同分问题</h2>
<p>G1只有1棵树，其最高分组共有<b>1,548人、22位复购客户</b>，模型无法在这组内部排序。按照事先固定的客户ID顺序取790人，得到13位复购客户。若组内均匀随机选取，期望约<b>11.23位</b>。这是解释同分影响的计算，不是另换规则评估或重新训练。</p>
<p>所以G1的“13人”不能当作精细筛选能力。AP按同分组整体计算；五等人数校准图也可能切开同分组，G1相邻组的率差不代表模型区分出了那些客户。G2在第790名边界只有1人，不存在该边界歧义。</p>
<h2>5. 不确定区间与地区信息</h2>
<p>客户级和州级各1000次配对bootstrap。以下差值为前者减后者；AP正值为改善，Log loss/Brier负值为改善。均基于固定模型预测，未重新训练或重新选择早停轮数。</p>
<div class="scroll">{tab(pd.DataFrame(bootrows))}</div>
<p>地区特征确实被G2用于分裂，但训练集分裂次数或增益不是额外预测价值的证明，更不是因果证据。本轮只能说：在这套固定树模型流程下，地区信息出现正向点估计增量，证据不足以认定稳定有效。</p>
<p>开发集只有60个复购事件，早停本身也可能受噪声影响。当前区间不涵盖训练与早停选择的不稳定性，也不能修复评估集重复使用造成的偏差。不能把“区间包含零”写成“证明没有作用”。</p>
<h2>6. 概率方面仍需保留的限制</h2>
<p>G2平均预测1.127%，接近实际1.102%，但这只说明整体均值接近。五组校准中，低分组仍有低估，高分组可能高估；每组事件数有限，区间较宽。与历史均值相比，概率指标改善也未显示稳定优势，不能称作概率已全面校准。</p>
<h2>7. 最终判断与停止位置</h2>
<p>按事先标准，本轮没有充分证据证明增加树模型复杂度值得，也没有稳定验证地区指标的额外价值。保留LR作为简单参照，将G1/G2完整记录为有取舍、证据不足的探索对照。这里的“未证明值得”不等于“证明树模型一定更差”。</p>
<p>到此结束约定的算法尝试，不根据这些分数追加特征、种子、参数或校准。保留项目中成立的证据、失败结果与观察局限，不把该评估集再次称为全新测试集。</p>
<h2>8. 参数、复核与文件</h2>
<details><summary>完整固定参数（LightGBM {r['versions']['lightgbm']}）</summary>{tab(protocolrows)}</details>
<p>仅训练两次LightGBM。独立重算AP与Brier一致，保存的文本模型预测与训练对象一致，LR预测复现V2的B。此前V1/V2输出、工作文件和压缩包共<b>{len(json.loads((W/'previous_before.json').read_text(encoding='utf-8')))}</b>个文件前后SHA-256完全相同。无实施修复或结果驱动的追加调整。</p>
<p>方案注册UTC时间：{r['registration']['registered_utc']}；模型锁定UTC时间：{r['locked_models']['locked_utc']}。模型锁定记录发生在生成本轮评估预测之前，不代表原评估数据从未被看过。</p>
<p><a href="results.json">完整指标与区间</a> · <a href="metrics.csv">指标表</a> · <a href="learning_curves.csv">学习曲线数据</a> · <a href="locked_models_before_evaluation.json">评估前模型锁定记录</a> · <a href="protocol.json">固定方案</a> · <a href="tie_diagnostics.json">同分诊断</a> · <a href="preservation.json">旧文件保留核验</a> · <a href="REPRODUCE.txt">复现说明</a></p>
<p>来源：<a href="https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce">Olist数据</a>。本次仅生成本地结果，未发布GitHub。</p>
</main></html>'''
(O/'LightGBM固定对照结果与解释.html').write_text(body,encoding='utf-8')
shutil.copy2(W/'report.py',O/'report.py')
req=Path('outputs/olist-logistic-2026-10-16/requirements.txt').read_text(encoding='utf-8')
(O/'requirements.txt').write_text(req+'lightgbm==4.7.0\n',encoding='utf-8')
(O/'REPRODUCE.txt').write_text('''本轮为最后一次固定探索算法对照，评估数据已在V1/V2中查看。
依赖原工作目录中的 work/olist-logistic/model_frame.pkl、work/olist-calibration/B.joblib 和 evaluation_predictions.pkl。
V1/V2的数据准备、特征及逻辑回归源码保留在原有outputs目录中，不被本轮覆盖。
本轮固定方案与注册文件保存在 work/olist-lightgbm/，执行顺序：
work/olist-model-env/Scripts/python.exe work/olist-lightgbm/run_fixed.py
work/olist-model-env/Scripts/python.exe work/olist-lightgbm/report.py
run_fixed.py 先核对previous_before.json的旧文件哈希，训练G1/G2，只用开发集log loss早停，保存文本模型与锁定记录，然后预测评估集。
输出到 outputs/olist-lightgbm-v3-2026-10-16/。文本模型包含所选轮数的树，未包含原始客户记录。
客户级预测及区域预处理仅保存在work/olist-lightgbm/，不包含在此结果包。
完整复现需要按V1/V2说明重建同一数据、环境和目录；跨机器须另存保存记录，不能绕过哈希核验。
''' ,encoding='utf-8')
before=json.loads((W/'previous_before.json').read_text(encoding='utf-8'))
assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in before.items())
print('Report generated; all',len(before),'previous files unchanged.')
