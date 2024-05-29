import sys
sys.stdout.reconfigure(line_buffering=True,write_through=True)
sys.stderr.reconfigure(line_buffering=True,write_through=True)

import time

import tsfresh
from tsfresh.utilities.dataframe_functions import impute
from tsfresh import select_features, extract_features
import json

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
import xgboost as xgb

import pandas as pd
import pathlib
import pickle
import scipy
from scipy.stats import mode

from plot import PlotFog, OP_None, OP_Scale

__all__ = ['TSExt', 
           'run_cases_in_tsext']

# 时序数据处理
class TSExt:
    def __init__(self, out_path):
        self.case_name = None
        self.out_path = pathlib.Path(out_path)
        if not self.out_path.exists():
            self.out_path.mkdir(parents = True)
            print(f"TSExt: Created directory {self.out_path}")
        # 临时存储最新读入的数据
        self.labels = None
        self.gt_filter = None
        self.gt_data = None
        self.ts_data = None

        # 用于存储多个case的数据
        self.labels_hub = [] # 用于存储多个case的labels
        self.tsdata_hub = [] # 用于存储多个case的ts_data
        self.case_names_hub = [] # 用于存储多个case的case_name

        # 构造的最新特征提取训练集
        self.ts_slices = None
        self.ts_slices_y = None
        self.filtered_fc_parameters = None # 特征选择后的fc_parameters

    def set_case_name(self, case_name):
        self.case_name = case_name
    # [latest case] 生成有效的时序数据标签
    def generate_label(self, gt_csv_path):
        gt_path = pathlib.Path(gt_csv_path)
        assert gt_path.exists(), f"gt_csv_path error! [{gt_csv_path}]"        
        gt_data = pd.read_csv(gt_path).iloc[:, 1]
        gt_data_filter = scipy.signal.savgol_filter(gt_data, 1000, 4)*0.1
        # 根据数值距离进行离散化：0:[0,0.25] 3:[0.25,0.35] 4:[0.35,0.45] 5:[0.45,>0.55]
        discretized_labels = pd.cut(gt_data_filter, 
                                    bins=[0, 0.15, 0.25, 0.35, 0.45, float('inf')], 
                                    labels=[0, 0.2, 0.3, 0.4, 0.5])
        # 将discretized_labels转为一列的dataframe
        self.labels = pd.DataFrame(discretized_labels, columns=['label'])
        self.gt_filter = pd.DataFrame(gt_data_filter, columns=['gt_filter'])
        self.gt_data = (gt_data*0.1).to_frame()
        print(f"\nTSExt: generate labels done! {self.case_name}")
        # 输出self.labels的统计特征
        label_stats = self.labels.describe()
        print(label_stats)
    # [latest case] 打印数据标签对比示意图
    def plot_labels(self):
        plot_fog = PlotFog(self.out_path)
        plot_fog.rd_csvs([],[],[],[],[])
        plot_fog.ad_csvs(self.gt_data, 'gt', 0, [OP_None()], 'grey')
        plot_fog.ad_csvs(self.gt_filter, 'gt_filter', 0, [OP_None()], 'blue')
        plot_fog.ad_csvs(self.labels, 'event_labels', 0, [OP_None()], 'red')
        plot_fog.pt_csvs(f'labels_{self.case_name}')
        print(f"TSExt: plot_labels done! {self.case_name}\n\n")
    # [latest case] 读入原始时序数据
    def read_tsdata(self, ts_csv_path):
        ts_path = pathlib.Path(ts_csv_path)
        assert ts_path.exists(), f"ts_csv_path error! [{ts_csv_path}]"        
        ts_data = pd.read_csv(ts_path).iloc[:, 1]
        self.ts_data = ts_data.to_frame()
        print(f"\nTSExt: read ts_data done! {self.case_name}")
    # [case to hub] 当前case放入hub
    def case_to_hub(self):
        if self.labels is None:
            return
        if self.ts_data is None:
            return

        self.labels_hub.append(self.labels.iloc[:,0])
        self.tsdata_hub.append(self.ts_data.iloc[:,0])
        self.case_names_hub.append(self.case_name)
        print(f"TSExt: case_to_hub done! {self.case_name}")

        self.labels = None
        self.ts_data = None
        self.case_name = None
    
    # [hub] 打印当前hub的情况
    def hub_infos(self):
        print(f"\nTSExt: --- hub_infos ---")
        # case数量
        print(f"\tlabels_hub: {len(self.labels_hub)}")
        print(f"\ttsdata_hub: {len(self.tsdata_hub)}")
        print(f"\tcase_names_hub: {len(self.case_names_hub)}")
        # 数据条数
        for i,tsdata in enumerate(self.tsdata_hub):
            print(f"\t*case:{self.case_names_hub[i]}\tdata size: {len(tsdata)}")
        print("\n")

    # [hub] 生成时间切片，构造特征提取训练集
    def generate_ts_slices_fast(self, ws=600, ss=600, force=False):
        if self.ts_slices is not None and force == False:
            print("TSExt: generate_ts_slices done! (already exist)!")
            return
        
        self.ts_slices = None
        self.ts_slices_y = None
        self.ts_slices = pd.DataFrame(data=None, columns=['id', 'time', 'value'])
        self.ts_slices_y = pd.DataFrame(data=None, columns=['id', 'y'])

        ts_slices_y_id  = []
        ts_slices_y_y   = []
        ts_slices_id    = []
        ts_slices_time  = []
        ts_slices_value = []
        # id构造方法：[case_name]_ws_ss_index
        for case_name, ts_data, ts_label in zip(self.case_names_hub, self.tsdata_hub, self.labels_hub):
            ts_data_np = ts_data.to_numpy()
            ts_label_np = ts_label.to_numpy()
            slice_id = 0
            ts_data_len = len(ts_data)
            start_time = time.time()
            for i in range(0, ts_data_len-ws, ss):
                #y = float(ts_label.iloc[i:i+ws].value_counts().idxmax()) # 找出最多的label作为y
                #self.ts_slices_y.loc[ts_slices_y_len + slice_id] = [f"{case_name}_{ws}_{ss}_{slice_id}", int(y*10)]
                y = float(mode(ts_label_np[i:i+ws]).mode)
                ts_slices_y_id.append(f"{case_name}_{ws}_{ss}_{slice_id:05d}")
                ts_slices_y_y.append(int(y*10))
                for k in range(ws):
                    #self.ts_slices.loc[ts_slices_len+i*ws+k] = [f"{case_name}_{ws}_{ss}_{slice_id}", k, ts_data.iloc[i+k]]
                    ts_slices_id.append(f"{case_name}_{ws}_{ss}_{slice_id:05d}")
                    ts_slices_time.append(k)
                    ts_slices_value.append(ts_data_np[i+k])
                #self.ts_slices_y = pd.concat([self.ts_slices_y, pd.DataFrame({'id':ts_slices_y_id, 'y':ts_slices_y_y})], ignore_index=True)
                #self.ts_slices = pd.concat([self.ts_slices, pd.DataFrame({'id':ts_slices_id, 'time':ts_slices_time, 'value':ts_slices_value})], ignore_index=True)
                slice_id += 1
            print(f"TSExt: generate_ts_slices [case:{case_name}] [slide num:{slice_id}]")
            end_time = time.time()
            duration = end_time - start_time
            print(f"case_name: {case_name} 时间片构造时长:{duration}秒")
        
        self.ts_slices_y = pd.DataFrame(data=[ts_slices_y_id, ts_slices_y_y], columns=['id','y'])
        self.ts_slices_y.set_index('id', inplace=True)
        self.ts_slices = pd.DataFrame(data=[ts_slices_id, ts_slices_time, ts_slices_value], columns=['id','time','value'])
        print(f"TSExt: [Done] generate_ts_slices done! [slide sum:{len(self.ts_slices_y)} sample sum:{len(self.ts_slices)}]")
        del ts_slices_y_id
        del ts_slices_y_y
        del ts_slices_id
        del ts_slices_time
        del ts_slices_value

    
    # [hub] 生成时间切片，构造特征提取训练集
    def generate_ts_slices(self, ws=600, ss=600, force=False):
        if self.ts_slices is not None and force == False:
            print("TSExt: generate_ts_slices done! (already exist)!")
            return
        
        self.ts_slices = None
        self.ts_slices_y = None
        self.ts_slices = pd.DataFrame(data=None, columns=['id', 'time', 'value'])
        self.ts_slices_y = pd.DataFrame(data=None, columns=['id', 'y'])

        # id构造方法：[case_name]_ws_ss_index
        for case_name, ts_data, ts_label in zip(self.case_names_hub, self.tsdata_hub, self.labels_hub):
            ts_data_np = ts_data.to_numpy()
            ts_label_np = ts_label.to_numpy()
            ts_slices_y_id = []
            ts_slices_y_y = []
            ts_slices_id = []
            ts_slices_time = []
            ts_slices_value = []

            slice_id = 0
            ts_data_len = len(ts_data)
            start_time = time.time()
            for i in range(0, ts_data_len-ws, ss):
                #y = float(ts_label.iloc[i:i+ws].value_counts().idxmax()) # 找出最多的label作为y
                #self.ts_slices_y.loc[ts_slices_y_len + slice_id] = [f"{case_name}_{ws}_{ss}_{slice_id}", int(y*10)]
                y = float(mode(ts_label_np[i:i+ws]).mode)
                ts_slices_y_id.append(f"{case_name}_{ws}_{ss}_{slice_id}")
                ts_slices_y_y.append(int(y*10))

                for k in range(ws):
                    #self.ts_slices.loc[ts_slices_len+i*ws+k] = [f"{case_name}_{ws}_{ss}_{slice_id}", k, ts_data.iloc[i+k]]
                    ts_slices_id.append(f"{case_name}_{ws}_{ss}_{slice_id}")
                    ts_slices_time.append(k)
                    ts_slices_value.append(ts_data_np[i+k])
                slice_id += 1
            self.ts_slices_y = pd.concat([self.ts_slices_y, pd.DataFrame({'id':ts_slices_y_id, 'y':ts_slices_y_y})], ignore_index=True)
            self.ts_slices = pd.concat([self.ts_slices, pd.DataFrame({'id':ts_slices_id, 'time':ts_slices_time, 'value':ts_slices_value})], ignore_index=True)
            print(f"TSExt: generate_ts_slices [case:{case_name}] [slide num:{slice_id}]")
            end_time = time.time()
            duration = end_time - start_time
            print(f"case_name: {case_name} 时间片构造时长:{duration}秒")
        self.ts_slices_y.set_index('id', inplace=True)
        print(f"TSExt: [Done] generate_ts_slices done! [slide sum:{len(self.ts_slices_y)} sample sum:{len(self.ts_slices)}]")    
    
    # [hub] 生成时间切片，构造特征提取训练集
    def generate_ts_slices_bak(self, ws=600, ss=600, force=False):
        if self.ts_slices is not None and force == False:
            print("TSExt: generate_ts_slices done! (already exist)!")
            return
        
        self.ts_slices = None
        self.ts_slices_y = None

        # id构造方法：[case_name]_ws_ss_index
        for case_name, ts_data, ts_label in zip(self.case_names_hub, self.tsdata_hub, self.labels_hub):
            slice_id = 0
            ts_data_len = len(ts_data)
            ts_slices_len = len(self.ts_slices)
            ts_slices_y_len = len(self.ts_slices_y)
            for i in range(0, ts_data_len-ws, ss):
                y = float(ts_label.iloc[i:i+ws].value_counts().idxmax()) # 找出最多的label作为y
                self.ts_slices_y.loc[ts_slices_y_len + slice_id] = [f"{case_name}_{ws}_{ss}_{slice_id}", int(y*10)]

                for k in range(ws):
                    self.ts_slices.loc[ts_slices_len+i*ws+k] = [f"{case_name}_{ws}_{ss}_{slice_id}", k, ts_data.iloc[i+k]]
                slice_id += 1
            print(f"TSExt: generate_ts_slices [case:{case_name}] [slide num:{slice_id}]")
        self.ts_slices_y.set_index('id', inplace=True)
        print(f"TSExt: [Done] generate_ts_slices done! [slide sum:{len(self.ts_slices_y)} sample sum:{len(self.ts_slices)}]")
         

    # [hub] 自动化特征提取与选择
    def auto_extract_features(self):
        if self.ts_slices is None:
            print("TSExt: auto_extract_features failed! [ts_slices is None]")
            return
        if self.ts_slices_y is None:
            print("TSExt: auto_extract_features failed! [ts_slices_y is None]")
            return
        
        print(f"TSExt: auto_extract_features start! [slices sum:{len(self.ts_slices_y)}]")
        # 特征提取
        extracted_features = extract_features(self.ts_slices, column_id='id', column_sort='time', n_jobs=30)
        # 打印extracted_features的列名
        print(f"TSExt: extracted_features columns: [{len(extracted_features.columns)}] {extracted_features.columns}")
        # 特征选择
        impute(extracted_features)
        features_filtered = select_features(extracted_features, self.ts_slices_y['y'], n_jobs=30)
        # 打印features_filtered的列名
        print(f"TSExt: features_filtered columns: [{len(features_filtered.columns)}] {features_filtered.columns}")
        # 由features_filtered构造kind_to_fc_parameters
        self.filtered_fc_parameters = tsfresh.feature_extraction.settings.from_columns(features_filtered)
        print(f"TSExt: fc_parameters: {self.filtered_fc_parameters}")
        return features_filtered

    # [hub] 保存特征处理后的，CaseNames、数据集、标签、特征参数等
    def save_tsext_results(self, out_name="default"):
        out_dir = self.out_path / out_name
        if not out_dir.exists():
            out_dir.mkdir(parents=True)
            print(f"TSExt: save_tsext_result create out dir :{str(out_dir)}")
        # 保存数据集 Dataframe
        ts_slices_path = out_dir / "ts_slices.csv"
        self.ts_slices.to_csv(ts_slices_path)
        # 保存标签 Dataframe
        ts_slices_y_path = out_dir / "ts_slices_y.csv"
        self.ts_slices_y.to_csv(ts_slices_y_path)
        # 保存特征参数字典 fc_parameters
        fc_path = out_dir / "fc_params.plk"
        with open(fc_path, 'wb') as f:
            pickle.dump(self.filtered_fc_parameters, f)
        case_names_path = out_dir / "case_names.txt"
        with open(case_names_path, 'w') as f:
            for cname in self.case_names_hub:
                f.write(cname+"\n")
        print(f"TSExt: [Done] save: \n\t1.{ts_slices_path}\n\t2.{ts_slices_y_path}\n\t3.{fc_path}\n\t4.{case_names_path}")

class TSModel:
    def __init__(self, data_dir="out_slices/ws600_ss600"):
        self.data_dir = pathlib.Path(data_dir)
        self.ts_slices_path = self.data_dir / "ts_slices.csv"
        self.ts_slices_y_path = self.data_dir / "ts_slices_y.csv"
        self.fc_path = self.data_dir / "fc_params.plk"
        self.case_names_path = self.data_dir / "case_names.txt"
        assert self.data_dir.exists(), f"data_dir error! [{data_dir}]"
        assert self.ts_slices_path.exists(), f"x_path error! [{self.ts_slices_path}]"
        assert self.ts_slices_y_path.exists(), f"y_path error! [{self.ts_slices_y_path}]"
        assert self.fc_path.exists(), f"fc_path error! [{self.fc_path}]"
        assert self.case_names_path.exists(), f"case_names_path error! [{self.case_names_path}]"
        self.ts_slices = pd.read_csv(self.ts_slices_path, index_col=0)
        self.ts_slices_y = pd.read_csv(self.ts_slices_y_path, index_col=0)
        with open(self.fc_path, 'rb') as f:
            self.fc_params = pickle.load(f)
            self.fc_params = dict(self.fc_params)
        with open(self.case_names_path, 'r') as f:
            self.case_names = f.readlines()
            self.case_names = [x.strip() for x in self.case_names]
        print(f"TSModel: init done! [data_dir:{data_dir}]")
        
    def tsfresh_features(self):
        # 使用fc_params提取特征
        print(f"ts_slices:\n{self.ts_slices.head()}\n{self.ts_slices.describe()}")
        self.x = extract_features(self.ts_slices, column_id='id', column_sort='time', default_fc_parameters=self.fc_params['value'])
        self.x = tsfresh.utilities.dataframe_functions.impute(self.x)
        # 打印特征
        print(f"TSModel: feature_extract done! [feature num:{len(self.x.columns)}]")
        print(f"TSModel: x:{self.x.head()}")
        # self.x 与 self.ts_slices_y 进行以 id 为外键的 merge
        self.y = self.ts_slices_y.copy()
        self.y.loc[self.y['y'] ==0, 'y'] = 1
        self.y['y1'] = self.y['y'].map(lambda x: x-1)
        self.y.drop(columns=['y'], inplace=True)
        print(f"TSModel: y:{self.y.head()}")
        self.xy_data = pd.merge(self.x, self.y, how='inner', left_index=True, right_index=True)
        print(f"TSModel: xy:{self.xy_data.head()}\n{self.xy_data.tail()}\nxy的总行数:{len(self.xy_data)}")
        self.x = self.xy_data.drop(columns=['y1'])
        self.y = self.xy_data['y1'].copy()
        print(f"TSModel: self.x:{self.x.head()}\n{self.x.tail()}\n数量:{len(self.x)}")
        print(f"TSModel: self.y:{self.y.head()}\n{self.y.tail()}\n数量:{len(self.y)}")


    def generate_dataset(self, t=0.2):
        # 划分数据集
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.x, self.y, test_size=t, random_state=42)
        print(f"TSModel: generate_dataset done! [train:{len(self.x_train)} test:{len(self.x_test)}]")
    def model_XGBoost(self):
        self.model = xgb.XGBClassifier(enable_categorical=True, \
                                       n_estimators=200, learning_rate=0.1, \
                                       max_depth=8, gamma=0)
        self.model.fit(self.x_train, self.y_train)
        print(f"TSModel: model_XGBoost done!")
    def model_XGBoost_gridsearch(self):
        self.model = xgb.XGBClassifier(enable_categorical=True, n_jobs=-1)
        parameters = {
            'n_estimators': [300,400],
            'learning_rate':[0.05,0.1,0.15],
            'max_depth': [10,12],
            'gamma': [0, 0.05]
        }
        self.model = GridSearchCV(self.model, parameters, n_jobs=-1, cv=5, verbose=3)
        self.grid_result = self.model.fit(self.x_train, self.y_train, verbose=3)
        print(f"TSModel: Best: {self.grid_result.best_score_} using {self.grid_result.best_params_}")
        print(f"TSModel: model_XGBoost_gridsearch done!")
    
    def model_predict(self):
        y_pred = self.model.predict(self.x_test)
        print(classification_report(self.y_test, y_pred))
        print(f"TSModel: model_predict done!")

    def model_save(self):
        self.model.save_model(self.data_dir / "model.bin")
        print(f"TSModel: model_save done!")
    
    def get_feature_importance(self):
        fi = self.model.feature_importances_
        fi = pd.DataFrame(fi, index=self.x.columns, columns=['importance'])
        fi = fi.sort_values(by='importance', ascending=False)
        fi.to_csv(self.data_dir / "feature_importance.csv")
        print(f"TSModel: get_feature_importance done!")

def run_cases_in_tsext(case_names, ws, ss, batch=0):
    ts = TSExt(out_path='out_slices')
    for case_name in case_names:
        ts.set_case_name(case_name)
        if batch is None:
            ts.generate_label(f'frames_ground_truth/fqh/{case_name}.csv')
            ts.read_tsdata(f'out/{case_name}/dt_out_pro_ltm.csv')
        else:
            ts.generate_label(f'frames_ground_truth/fqh/{batch}/{case_name}.csv')
            ts.read_tsdata(f'out{batch}/{case_name}/dt_out_pro_ltm.csv')
        ts.case_to_hub()
    ts.hub_infos()
    ts.generate_ts_slices(ws=ws, ss=ss, force=True)
    ts.auto_extract_features()
    ts.save_tsext_results(out_name=f"ws{ws}_ss{ss}")

def run_cases_some(cases_csv, batch_id, name='some'):
    ts=TSExt(out_path='out_slices')
    ws=300
    ss=100
    case_info = pd.read_csv(cases_csv)
    case_names = []
    for i in range(len(case_info)):
        case_names.append(case_info['case_name'][i])
    for case_name in case_names:
        ts.set_case_name(case_name)
        ts.generate_label(f'/share/home/tj14034/data/frames_ground_truth/fqh/{batch_id}/{case_name}.csv')
        ts.plot_labels()
        ts.read_tsdata(f'/share/home/tj14034/data/out{batch_id}/{case_name}/dt_out_pro_ltm.csv')
        ts.case_to_hub()
    ts.hub_infos()
    ts.generate_ts_slices(ws=ws, ss=ss, force=True)
    ts.auto_extract_features()
    ts.save_tsext_results(out_name=f"ws{ws}_ss{ss}_{name}")

def run_cases_all(ws=500, ss=10, updata_fc_params=True, plot_labels=False):
    ts=TSExt(out_path='out_slices')

    batch_id = 0
    csv_path = 'case_info/case_batch0.csv'
    case_info = pd.read_csv(csv_path)
    case_names = []
    for i in range(len(case_info)):
        case_names.append(case_info['case_name'][i])
    for case_name in case_names:
        ts.set_case_name(case_name)
        ts.generate_label(f'/share/home/tj14034/data/frames_ground_truth/fqh/{batch_id}/{case_name}.csv')
        if plot_labels:
            ts.plot_labels()
        ts.read_tsdata(f'/share/home/tj14034/data/out{batch_id}/{case_name}/dt_out_pro_ltm.csv')
        ts.case_to_hub()

    batch_id = 1
    csv_path = 'case_info/case_batch1.csv'
    case_info = pd.read_csv(csv_path)
    case_names = []
    for i in range(len(case_info)):
        case_names.append(case_info['case_name'][i])
    for case_name in case_names:
        ts.set_case_name(case_name)
        ts.generate_label(f'/share/home/tj14034/data/frames_ground_truth/fqh/{batch_id}/{case_name}.csv')
        #ts.plot_labels()
        ts.read_tsdata(f'/share/home/tj14034/data/out{batch_id}/{case_name}/dt_out_pro_ltm.csv')
        ts.case_to_hub()
    ts.hub_infos()
    ts.generate_ts_slices(ws=ws, ss=ss, force=True)
    if updata_fc_params:
        ts.auto_extract_features()
    ts.save_tsext_results(out_name=f"ws{ws}_ss{ss}_all")

def run_cases_all_t():
    ts=TSExt(out_path='out_slices_t')
    ws=500
    ss=10

    batch_id = 0
    csv_path = 'case_info/case_info0_tsext.csv'
    case_info = pd.read_csv(csv_path)
    case_names = []
    for i in range(len(case_info)):
        case_names.append(case_info['case_name'][i])
    for case_name in case_names:
        ts.set_case_name(case_name)
        ts.generate_label(f'frames_ground_truth/fqh/{batch_id}/{case_name}.csv')
        ts.read_tsdata(f'out{batch_id}/{case_name}/t_out.csv')
        ts.case_to_hub()    

    batch_id = 1
    csv_path = 'case_info/case_info1_tsext.csv'
    case_info = pd.read_csv(csv_path)
    case_names = []
    for i in range(len(case_info)):
        case_names.append(case_info['case_name'][i])
    for case_name in case_names:
        ts.set_case_name(case_name)
        ts.generate_label(f'frames_ground_truth/fqh/{batch_id}/{case_name}.csv')
        ts.read_tsdata(f'out{batch_id}/{case_name}/t_out.csv')
        ts.case_to_hub()    
    ts.hub_infos()
    ts.generate_ts_slices(ws=ws, ss=ss, force=True)
    ts.auto_extract_features()
    ts.save_tsext_results(out_name=f"ws{ws}_ss{ss}_all")

def run_train_cases_td(ws=500, ss=100, updata_fc_params=True, plot_labels=False):
    ts=TSExt(out_path='out_slices_train')
    batch_id_set = {'1':0, '2':1, '3':1, '4':0, '5':0}
    csv_path = 'case_info/train/train_cases.csv'
    case_info = pd.read_csv(csv_path)
    case_names = []
    for i in range(len(case_info)):
        case_names.append(case_info['case_name'][i])
    for case_name in case_names:
        ts.set_case_name(case_name)
        batch_id = batch_id_set[case_name[0]]
        ts.generate_label(f'/share/home/tj14034/data/frames_ground_truth/final_gt/{batch_id}/{case_name}.csv')
        if plot_labels:
            ts.plot_labels()
        ts.read_tsdata(f'/share/home/tj14034/data/out{batch_id}/{case_name}/dt_out_pro_ltm.csv')
        ts.case_to_hub()
    ts.hub_infos()
    ts.generate_ts_slices(ws=ws, ss=ss, force=True)
    if updata_fc_params:
        ts.auto_extract_features()
    ts.save_tsext_results(out_name=f"ws{ws}_ss{ss}_train")

def run_train_cases_x(fname, ffile, ws=500, ss=100, updata_fc_params=True, plot_labels=False):
    ts=TSExt(out_path=f'out_slices_train_{fname}')
    batch_id_set = {'1':0, '2':1, '3':1, '4':0, '5':0}
    csv_path = 'case_info/train/train_cases.csv'
    case_info = pd.read_csv(csv_path)
    case_names = []
    for i in range(len(case_info)):
        case_names.append(case_info['case_name'][i])
    for case_name in case_names:
        ts.set_case_name(case_name)
        batch_id = batch_id_set[case_name[0]]
        ts.generate_label(f'/share/home/tj14034/data/frames_ground_truth/final_gt/{batch_id}/{case_name}.csv')
        if plot_labels:
            ts.plot_labels()
        ts.read_tsdata(f'/share/home/tj14034/data/out{batch_id}/{case_name}/{ffile}.csv')
        ts.case_to_hub()
    ts.hub_infos()
    ts.generate_ts_slices(ws=ws, ss=ss, force=True)
    if updata_fc_params:
        ts.auto_extract_features()
    ts.save_tsext_results(out_name=f"ws{ws}_ss{ss}_train")


def run_model(data_dir_path="out_slices/ws600_ss600"):
    ts_model = TSModel(data_dir=data_dir_path)
    ts_model.tsfresh_features()
    ts_model.generate_dataset()
    ts_model.model_XGBoost()
    ts_model.model_predict()
    ts_model.get_feature_importance()
    ts_model.model_save()

def run_model_gridsearch(data_dir_path="out_slices/ws600_ss600"):
    ts_model = TSModel(data_dir=data_dir_path)
    ts_model.tsfresh_features()
    ts_model.generate_dataset()
    ts_model.model_XGBoost_gridsearch()
    ts_model.model_predict()

if __name__ == '__main__':
    run_train_cases_td()
    #run_cases_some(cases_csv='/share/home/tj14034/data/code/HighwayFog/case_info/case_info5.csv',\
    #               batch_id=0, name='test_c5')
    #run_model(data_dir_path="out_slices/ws500_ss10_all")
    #run_model_gridsearch(data_dir_path="out_slices/ws300_ss100_all")
