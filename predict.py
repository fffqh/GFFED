import pathlib
import xgboost as xgb
import time
import numpy as np
import pandas as pd
import pickle

import tsfresh
from tsfresh.utilities.dataframe_functions import impute
from tsfresh import select_features, extract_features
import argparse



class FogPredictor:
    def __init__(self, model_path, fc_params_path, window=300):
        self.model_path = pathlib.Path(model_path)
        assert self.model_path.exists(), "Model file not found"
        self.fc_params_path = pathlib.Path(fc_params_path)
        assert self.fc_params_path.exists(), "fc_params file not found"
        with open(self.fc_params_path, 'rb') as f:
            self.fc_params = pickle.load(f)
            self.fc_params = dict(self.fc_params)
        self.model = xgb.XGBClassifier(enable_categorical=True)
        self.model.load_model(str(self.model_path))
        self.window = window
        print(f"FogPredictor: Model loaded from {model_path}, Time Series Window:{window}")
    
    def run_case_flow(self, case_path, input_path, step=1, name=''):
        self.case_path = pathlib.Path(case_path)
        assert self.case_path.exists(), "case file not found"
        self.input_path = pathlib.Path(input_path)
        assert self.input_path.exists(), "input file not found"
        self.data = pd.read_csv(self.input_path).iloc[:, 1].ffill()
        data_len=len(self.data)
        half_window = self.window // 2
        preds_id = []
        preds_pred = []
        for i in range(0, data_len-step+1, step):
            # 未达半窗
            if i < half_window:
                input_slice = self.data[:self.window]
            # 不足半窗
            elif i > data_len - half_window:
                input_slice = self.data[data_len - self.window:]
            # 正常情况
            else:
                input_slice = self.data[i - half_window: i + half_window]
            # 将input_slice转为pd.DataFrame
            x = pd.DataFrame(data=input_slice.values, columns=['value'])
            # 增加列
            x['id'] = [i for _ in range(self.window)]
            x['time'] = range(self.window)
            x = x[['id', 'time', 'value']]
            x = extract_features(x, column_id='id', column_sort='time', default_fc_parameters=self.fc_params['value'], n_jobs=4, disable_progressbar=True)
            x = tsfresh.utilities.dataframe_functions.impute(x)
            pred = self.model.predict(x)
            if i%5000 == 0:
                print(f"case:{case_path}\tid:{i}\tpred:{pred}")
            for k in range(step):
                preds_id.append(i+k)
                preds_pred.append(pred[0])
        preds_path = self.case_path / f"preds{name}.csv"
        self.preds = pd.DataFrame(data={'id':preds_id, 'pred':preds_pred})
        self.preds.to_csv(str(preds_path), index=False)

# 使用命令行参数
import argparse
parser = argparse.ArgumentParser(description='Fog Prediction')
parser.add_argument('--slices', type=str, default="out_slices/ws500_ss10_all/", help='slices path')
parser.add_argument('--window', type=int, default=500, help='window size')
parser.add_argument('--step', type=int, default=10, help='step size')
parser.add_argument('--name', type=str, default='', help='pred name')
parser.add_argument('--case_list', nargs='+', type=str, help='list of case_csv_path')
#parser.add_argument('--batch_list', nargs='+', type=int, help='list of batch_id')

args = parser.parse_args()

if __name__ == '__main__':
    # 处理模型相关参数
    slices_path = pathlib.Path(args.slices)
    model_path = slices_path / "model.bin"
    fc_params_path = slices_path / "fc_params.plk"
    assert slices_path.exists(), f"Slices path not found:{str(slices_path)}"
    assert model_path.exists(), f"Model file not found:{str(model_path)}"
    assert fc_params_path.exists(), f"fc_params file not found:{str(fc_params_path)}"
    fog = FogPredictor(model_path=str(model_path), fc_params_path=str(fc_params_path), window=args.window)

    # 处理case相关参数
    batch_id_set = {'1':0, '2':1, '3':1, '4':0, '5':0}
    case_list = args.case_list
    for case_csv in case_list:
        case_csv = pathlib.Path(case_csv)
        if not case_csv.exists():
            print(f"[跳过] Case csv file not found:{str(case_csv)}")
            continue
        case_df = pd.read_csv(case_csv)
        for i in range(len(case_df)):
            case_name = case_df['case_name'][i]
            batch_id = batch_id_set[case_name[0]]
            case_path = f"/share/home/tj14034/data/out{batch_id}/{case_name}/"
            input_path = f"/share/home/tj14034/data/out{batch_id}/{case_name}/dt_out_pro_ltm.csv"
            if not pathlib.Path(input_path).exists():
                print(f"[跳过] {case_name} input file not found: {input_path}")
                continue
            print(f"Predicting case:{case_name}...")
            start_time = time.time()
            fog.run_case_flow(case_path=str(case_path), input_path=str(input_path), step=args.step, name=args.name)
            end_time = time.time()
            duration = end_time - start_time
            print(f"Predicting case:{case_name} done. Duration:{duration:.2f}s")

