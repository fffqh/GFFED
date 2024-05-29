import plotly as py
import plotly.graph_objs as go
pyplt = py.offline.plot
import pandas as pd
import numpy as np
import pathlib
import scipy
from sklearn import metrics

__all__ = [
    'PlotFog',
    'OPChain',
    'OP_1_T',
    'OP_None',
    'OP_Flt',
    'OP_Scale',
    'OP_Norm',
    'OP_x_T',
    'OP_Add'
    'PlotFog',
    'GeminiCSVData'
]

class OPChain:
    def __init__(self, ops):
        self.ops = ops
    def __call__(self, data):
        for op in self.ops:
            data = op(data)
        return data

class OP_Add:
    def __init__(self, x):
        self.x = x
    def __call__(self, data):
        return self.x+data

class OP_1_T:
    def __init__(self):
        pass
    def __call__(self, data):
        return 1 - data
    
class OP_x_T:
    def __init__(self, x):
        self.x = x
    def __call__(self, data):
        return self.x-data

class OP_None:
    def __init__(self):
        pass
    def __call__(self, data):
        return data

class OP_Flt:
    def __init__(self):
        pass
    def __call__(self, data):
        print(f"\tsavgol_filter: {len(data)}")
        data = data.ffill()
        data = scipy.signal.savgol_filter(data, 1000, 4)
        return data

class OP_Scale:
    def __init__(self, s):
        self.s = s
    def __call__(self, data):
        return data * self.s

class OP_Norm:
    def __init__(self, minv, maxv):
        self.minv = minv
        self.maxv = maxv
        assert self.maxv-self.minv > 1e-6, "Error: maxv and minv should not be too close."
    def __call__(self, data):
        return (data - self.minv) / (2*(self.maxv - self.minv))

class PlotFog:
    def __init__(self, out_path):
        self.out_path = pathlib.Path(out_path)
        if not self.out_path.exists():
            self.out_path.mkdir(parents = True)
            print(f"PlotFog: Created directory {self.out_path}")
        
    def get_metrics(self, gt_id=4):
        self.csv_metrics = []
        gt_data = self.data_csvs[gt_id]
        for i, data in enumerate(self.data_csvs):
            Ndata = len(data)
            Ngt = len(gt_data)
            N = min(Ndata, Ngt)
            m_list = []
            m_list.append(metrics.mean_squared_error(gt_data[:N], data[:N]))
            m_list.append(metrics.r2_score(gt_data[:N], data[:N]))
            self.csv_metrics.append(m_list)
        # 把metrics写入文件
        with open(self.out_path / 'out_metrics.csv', 'a+') as f:
            for _, ms in enumerate(self.csv_metrics):
                for m in ms:
                    f.write(f'{m},')
            f.write('\n')

    def rd_csvs(self, csv_paths, csv_names, csv_cols, ops, colors = None):
        self.data_csvs = []

        for csv_path, csv_col in zip(csv_paths, csv_cols):
            df = pd.read_csv(csv_path).iloc[:,csv_col]
            df = df.ffill()
            self.data_csvs.append(df)

        for i, data in enumerate(self.data_csvs):
            data_ops = ops[i]
            new_data = data
            for op in data_ops:
                new_data = op(new_data)
            self.data_csvs[i] = new_data
        
        self.csv_names = csv_names
        self.csv_colors = colors
        print(f"PlotFog: Read {len(self.data_csvs)} csv files")

    def ad_csvs(self, data, name, col, data_op, c):
        new_data = data.iloc[:,col]
        for op in data_op:
            new_data = op(new_data)
        self.data_csvs.append(new_data)
        self.csv_names.append(name)
        if self.csv_colors is not None:
            self.csv_colors.append(c)
        print(f"PlotFog: Add {name} data to csv data. num:{len(self.data_csvs)}.")
        
    def pt_csvs(self, plot_name):
        print(f"PlotFog: Plotting {len(self.data_csvs)} csv files")
        # 将时间序列数据绘制在一张折线图上
        trace = []
        for i, data in enumerate(self.data_csvs):
            trace.append(go.Scatter(
                y = data,
                mode = 'lines',
                name = self.csv_names[i],
                line_color = self.csv_colors[i] if self.csv_colors else None,
                line=dict(width=2),
            ))
        title_txt = ''
        for i, ms in enumerate(self.csv_metrics):
            title_txt += f'{self.csv_names[i]}:'
            for m in ms:
                title_txt += f'{m:.4f} '
        
        layout = go.Layout(
            #title = title_txt,
            xaxis = dict(title = 'Frame Index'),
            yaxis = dict(title = 'Value'),
            legend=dict(yanchor="top",y=0.99,xanchor="left",x=0.01),
            #legend=dict(orientation="h", yanchor="bottom",y=1.02,xanchor="right",x=1),
            # 确定纵坐标范围
            yaxis_range = [0, 1],
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            width=900,
            height=400,
        )
        fig = go.Figure(data = trace, layout = layout)
        fig.update_xaxes(showline=True, linewidth=3, linecolor='black', mirror=True)
        fig.update_yaxes(showline=True, linewidth=3, linecolor='black', mirror=True)
        fig.update_layout(font=dict(size=16))
        # 设置折线粗细为1
        #fig.update_traces(line=dict(width=2))
        # fig.add_annotation(text='South Korea: Asia <br>China: Asia <br>Canada: North America', 
        #             align='left',
        #             showarrow=False,
        #             xref='paper',
        #             yref='paper',
        #             x=1.1,
        #             y=0.8,
        #             bordercolor='black',
        #             borderwidth=1)
    
        pyplt(fig, filename = str(self.out_path / f'{plot_name}.html'), show_link=False, auto_open=False)
        print(f"PlotFog: Plot saved to {self.out_path / f'{plot_name}.html'}")
    

class GeminiCSVData:
    def __init__(self, case_name, batch_id):
        gemini_csv_path = f'out{batch_id}/{case_name}/gemini_out.csv'
        self.gemini_csv_path = pathlib.Path(gemini_csv_path)
        assert self.gemini_csv_path.exists(), f"Gemini csv 文件不存在: {gemini_csv_path}."

        dt_csv_path = f'out{batch_id}/{case_name}/t_out.csv'
        self.dt_csv_path = pathlib.Path(dt_csv_path)
        assert self.dt_csv_path.exists(), f"dt csv 文件不存在: {dt_csv_path}"
        
        self.opath = pathlib.Path(f'out{batch_id}/{case_name}/gemin_out_pro.csv')

    def revise(self):
        # 补充所有时序数据
        gm_df = pd.read_csv(self.gemini_csv_path, header=None)
        gm_df.columns = ['id','rank']
        dt_df = pd.read_csv(self.dt_csv_path, header=None)
        dt_df.columns = ['id','dt_m','dt_s']
        # 根据id将gm_df与dt_df进行合并
        df = pd.merge(dt_df,gm_df, on='id', how='left')
        # 把rank列中的-1改为Nan
        df['rank'] = df['rank'].replace(-1, np.nan)
        # 对rank中的NaN进行线性插值
        df['rank'] = df['rank'].interpolate()
        # 删除dt_m,dt_s列
        df = df.drop(['dt_m','dt_s'], axis=1)
        # 保存到新的csv文件，不保存表头
        df.to_csv(self.opath, header=False, index=False)

# Plot函数
def plot_case(case_name, batch=0):
    if batch is None:
        csv_paths = [f'out/{case_name}/dt_out_pro_ltm.csv',
                        f'out/{case_name}/dt_out_pro_ltm.csv',
                        f'out/{case_name}/dt_out.csv',
                        f'out/{case_name}/t_out.csv',
                        f'frames_ground_truth/final_gt/{case_name}.csv']
        out_name='out_plot'
    else:
        csv_paths = [f'out{batch}/{case_name}/dt_out_pro_ltm.csv',
                    f'out{batch}/{case_name}/dt_out_pro_ltm.csv',
                    f'out{batch}/{case_name}/dt_out.csv',
                    f'out{batch}/{case_name}/t_out.csv',
                    f'frames_ground_truth/final_gt/{batch}/{case_name}.csv']
        out_name=f'out_plot{batch}'
    #out_name=f'out_plot_debug'
    plot_fog = PlotFog(out_name)
    csv_names = ['dtp_more','dtp', 'dt', 't', 'score']
    csv_cols = [1, 1, 1, 1, 1]
    ops = [[OP_None()], [OP_Flt()],  [OP_Flt()], [OP_1_T(), OP_Flt()], [OP_Scale(0.1), OP_Flt()]]
    #ops = [[OP_None()], [OP_Flt()],  [OP_None()], [OP_1_T()], [OP_Scale(0.1), OP_Flt()]]
    colors = [ 'grey','red', 'blue', 'yellow', 'green']
    plot_fog.rd_csvs(csv_paths, csv_names, csv_cols, ops, colors=colors)
    plot_fog.get_metrics(gt_id=4)
    plot_fog.pt_csvs(case_name)

def plot_case_gemini(case_name, batch=0):
    if batch is None:
        csv_paths = [f'out/{case_name}/dt_out_pro_ltm.csv',
                        f'out/{case_name}/t_out.csv',
                        f'out/{case_name}/dt_out_pro_ltm.csv',
                        f'out/{case_name}/t_out.csv',
                        f'out/{case_name}/gemin_out_pro.csv',
                        f'frames_ground_truth/final_gt/{case_name}.csv']
        out_name='out_plot_gemini_beautiful'
    else:
        csv_paths = [f'out{batch}/{case_name}/dt_out_pro_ltm.csv',
                    f'out{batch}/{case_name}/t_out.csv',
                    f'out{batch}/{case_name}/dt_out_pro_ltm.csv',
                    f'out{batch}/{case_name}/t_out.csv',
                    f'out{batch}/{case_name}/gemin_out_pro.csv',
                    f'frames_ground_truth/final_gt/{batch}/{case_name}.csv']
        out_name=f'out_plot{batch}_gemini_beautiful'
    #out_name=f'out_plot_debug'
    plot_fog = PlotFog(out_name)
    csv_names = ['dtp_ori', 't_ori', 'dtp', 't', 'gemini', 'score']
    csv_cols = [1, 1, 1, 1, 1, 1]
    ops = [[OP_None()], [OP_1_T()], [OP_Flt()], [OP_1_T(), OP_Flt()], [OP_Scale(0.1),OP_Flt()], [OP_Scale(0.1), OP_Flt()]]
    #ops = [[OP_None()], [OP_Flt()],  [OP_None()], [OP_1_T()], [OP_Scale(0.1), OP_Flt()]]
    colors = [ 'rgba(143,150,216,0.6)','rgba(230,140,176,0.6)', '#222E97',  '#BC1458', '#777304', '#007730']
    plot_fog.rd_csvs(csv_paths, csv_names, csv_cols, ops, colors=colors)
    plot_fog.get_metrics(gt_id=5)
    plot_fog.pt_csvs(case_name)

def plot_case_gemini_final(case_name, batch=0):
    if batch is None:
        csv_paths = [
                        f'out/{case_name}/dt_out_pro_ltm.csv',
                        f'out/{case_name}/gemin_out_pro.csv',
                        f'frames_ground_truth/final_gt/{case_name}.csv'
                    ]
        out_name='out_plot_gemini_final'
    else:
        csv_paths = [
                        f'out{batch}/{case_name}/dt_out_pro_ltm.csv',
                        f'out{batch}/{case_name}/gemin_out_pro.csv',
                        f'frames_ground_truth/final_gt/{batch}/{case_name}.csv'
                    ]
        out_name=f'out_plot{batch}_gemini_final'
    
    plot_fog = PlotFog(out_name)
    csv_names = ['TD', 'Gemini', 'Ground Truth']
    csv_cols = [1, 1, 1]
    ops = [[OP_Flt()], [OP_Scale(0.1), OP_Flt()], [OP_Scale(0.1), OP_Flt()]]
    colors = ['#222E97', '#777304', '#007730']
    plot_fog.rd_csvs(csv_paths, csv_names, csv_cols, ops, colors=colors)
    plot_fog.get_metrics(gt_id=2)
    plot_fog.pt_csvs(case_name)

def plot_case_fade(case_name, batch=0):
    print(f"PlotFog: {case_name} start!")
    if batch is None:
        csv_paths = [f'out/{case_name}/dt_out_pro_ltm.csv',
                        f'out/{case_name}/tp_out.csv',
                        f'out/{case_name}/fade_out.csv',
                        f'out/{case_name}/dt_out_pro_ltm.csv',
                        f'out/{case_name}/tp_out.csv',
                        f'out/{case_name}/fade_out.csv',
                        f'frames_ground_truth/final_gt/{case_name}.csv']
        out_name='out_plot_fade'
    else:
        csv_paths = [f'out{batch}/{case_name}/dt_out_pro_ltm.csv',
                    f'out{batch}/{case_name}/tp_out.csv',
                    f'out{batch}/{case_name}/fade_out.csv',
                    f'out{batch}/{case_name}/dt_out_pro_ltm.csv',
                    f'out{batch}/{case_name}/tp_out.csv',
                    f'out{batch}/{case_name}/fade_out.csv',
                    f'frames_ground_truth/final_gt/{batch}/{case_name}.csv']
        out_name=f'out_plot{batch}_fade'
    #out_name=f'out_plot_debug'777304
    plot_fog = PlotFog(out_name)
    csv_names = ['dtp_ori', 'tp_ori', 'fade_ori', 'dtp', 'tp', 'fade','score']
    csv_cols = [1, 1, 1, 1, 1, 1, 1]
    ops_bak = [[OP_Norm(0.0154, 0.4152)], 
           [OP_Norm(0.4579, 0.8838), OP_x_T(0.5)], 
           [OP_Norm(0.2153, 2.5405)], 
           [OP_Norm(0.1106, 0.4152), OP_Flt()], 
           [OP_Norm(0.4579, 0.8838), OP_x_T(0.5), OP_Flt()], 
           [OP_Norm(0.7467, 2.5405), OP_Flt()],
           [OP_Scale(0.1), OP_Flt()]]
    ops = [[OP_None()], 
           [OP_x_T(1)], 
           [OP_Norm(0.2153, 2.5405)], 
           [OP_Flt()], 
           [OP_x_T(1), OP_Flt()], 
           [OP_Norm(0.2153, 2.5405), OP_Flt()],
           [OP_Scale(0.1), OP_Flt()]]

    #ops = [[OP_None()], [OP_Flt()],  [OP_None()], [OP_1_T()], [OP_Scale(0.1), OP_Flt()]]
    colors = [ 'rgba(143,150,216,0.6)','rgba(230,140,176,0.6)', 'rgba(189,197,115, 0.6)', '#222E97',  '#BC1458', '#777304', '#007730']
    plot_fog.rd_csvs(csv_paths, csv_names, csv_cols, ops, colors=colors)
    plot_fog.get_metrics(gt_id=6)
    plot_fog.pt_csvs(case_name)

def plot_case_fade_final(case_name, batch=0):
    print(f"PlotFog: {case_name} start!")
    if batch is None:
        csv_paths = [f'out/{case_name}/dt_out_pro_ltm.csv',
                        f'out/{case_name}/tp_out.csv',
                        f'out/{case_name}/fade_out.csv',
                        f'out/{case_name}/dt_out_pro_ltm.csv',
                        f'out/{case_name}/tp_out.csv',
                        f'out/{case_name}/fade_out.csv',
                        f'frames_ground_truth/final_gt/{case_name}.csv']
        out_name='out_plot_fade'
    else:
        csv_paths = [f'out{batch}/{case_name}/dt_out_pro_ltm.csv',
                    f'out{batch}/{case_name}/tp_out.csv',
                    f'out{batch}/{case_name}/fade_out.csv',
                    f'out{batch}/{case_name}/dt_out_pro_ltm.csv',
                    f'out{batch}/{case_name}/tp_out.csv',
                    f'out{batch}/{case_name}/fade_out.csv',
                    f'frames_ground_truth/final_gt/{batch}/{case_name}.csv']
        out_name=f'out_plot{batch}_fade'
    plot_fog = PlotFog(out_name)
    csv_names = ['TD', 'T', 'Fade', 'TD*', 'T*', 'Fade*', 'Ground Truth']
    csv_cols = [1, 1, 1, 1, 1, 1, 1]
    ops_bak = [ [OP_Norm(0.0154, 0.4152)], 
                [OP_Norm(0.4579, 0.8838), OP_x_T(0.5)], 
                [OP_Norm(0.2153, 2.5405)], 
                [OP_Norm(0.1106, 0.4152), OP_Flt()], 
                [OP_Norm(0.4579, 0.8838), OP_x_T(0.5), OP_Flt()], 
                [OP_Norm(0.7467, 2.5405), OP_Flt()],
                [OP_Scale(0.1), OP_Flt()]]
    ops = [ [OP_None()], 
            [OP_x_T(1)], 
            [OP_Norm(0.2153, 2.5405)], 
            [OP_Flt()], 
            [OP_x_T(1), OP_Flt()], 
            [OP_Norm(0.2153, 2.5405), OP_Flt()],
            [OP_Scale(0.1), OP_Flt()]]
    colors = [ 'rgba(143,150,216,0.4)','rgba(230,140,176,0.4)', 'rgba(189,197,115, 0.4)', \
               '#222E97', '#BC1458', '#777304', '#007730']
    plot_fog.rd_csvs(csv_paths, csv_names, csv_cols, ops, colors=colors)
    plot_fog.get_metrics(gt_id=6)
    plot_fog.pt_csvs(case_name)

def plot_case_feature_t_final(case_name, batch=0):
    if batch is None:
        csv_paths = [
                        f'out/{case_name}/tp_out.csv',
                        f'out/{case_name}/dt_out_pro_ltm.csv',
                        f'out/{case_name}/tp_out.csv',
                        f'out/{case_name}/dt_out_pro_ltm.csv',
                        f'frames_ground_truth/final_gt/{case_name}.csv'
                    ]
        out_name='out_plot_feature_t_final'
    else:
        csv_paths = [
                        f'out{batch}/{case_name}/tp_out.csv',
                        f'out{batch}/{case_name}/dt_out_pro_ltm.csv',
                        f'out{batch}/{case_name}/tp_out.csv',
                        f'out{batch}/{case_name}/dt_out_pro_ltm.csv',
                        f'frames_ground_truth/final_gt/{batch}/{case_name}.csv'
                    ]
        out_name=f'out_plot{batch}_feature_t_final'
    
    plot_fog = PlotFog(out_name)
    csv_names = ['T', 'TD', 'T*', 'TD*', 'Ground Truth']
    csv_cols = [1, 1, 1, 1, 1]
    ops = [ [OP_x_T(1)],  #T
            [OP_None()],  #TD
            [OP_x_T(1), OP_Flt()], #T
            [OP_Flt()],            #TD
            [OP_Scale(0.1), OP_Flt()]] #GT
    colors = ['rgba(230,140,176,0.4)','rgba(143,150,216,0.4)',
              '#BC1458', '#222E97', '#007730']
    plot_fog.rd_csvs(csv_paths, csv_names, csv_cols, ops, colors=colors)
    plot_fog.get_metrics(gt_id=4)
    plot_fog.pt_csvs(case_name)

def plot_case_feature_fade_final(case_name, batch=0):
    if batch is None:
        csv_paths = [
                        f'out/{case_name}/fade_out.csv',
                        f'out/{case_name}/dt_out_pro_ltm.csv',
                        f'out/{case_name}/fade_out.csv',
                        f'out/{case_name}/dt_out_pro_ltm.csv',
                        f'frames_ground_truth/final_gt/{case_name}.csv'
                    ]
        out_name='out_plot_feature_fade_final'
    else:
        csv_paths = [
                        f'out{batch}/{case_name}/fade_out.csv',
                        f'out{batch}/{case_name}/dt_out_pro_ltm.csv',
                        f'out{batch}/{case_name}/fade_out.csv',
                        f'out{batch}/{case_name}/dt_out_pro_ltm.csv',
                        f'frames_ground_truth/final_gt/{batch}/{case_name}.csv'
                    ]
        out_name=f'out_plot{batch}_feature_fade_final'
    
    plot_fog = PlotFog(out_name)
    csv_names = ['FADE', 'TD', 'FADE*', 'TD*', 'Ground Truth']
    csv_cols = [1, 1, 1, 1, 1]
    ops = [ [OP_Norm(0.2153, 2.5405)],  #T
            [OP_None()],  #TD
            [OP_Norm(0.2153, 2.5405), OP_Flt()], #T
            [OP_Flt()],            #TD
            [OP_Scale(0.1), OP_Flt()]] #GT
    colors = ['rgba(230,140,176,0.4)','rgba(143,150,216,0.4)',
              '#BC1458', '#222E97', '#007730']
    plot_fog.rd_csvs(csv_paths, csv_names, csv_cols, ops, colors=colors)
    plot_fog.get_metrics(gt_id=4)
    plot_fog.pt_csvs(case_name)

def plot_case_feature_slp_final(case_name, batch=0):
    if batch is None:
        csv_paths = [
                        f'out/{case_name}/slp_out.csv',
                        f'out/{case_name}/dt_out_pro_ltm.csv',
                        f'out/{case_name}/slp_out.csv',
                        f'out/{case_name}/dt_out_pro_ltm.csv',
                        f'frames_ground_truth/final_gt/{case_name}.csv'
                    ]
        out_name='out_plot_feature_slp_final'
    else:
        csv_paths = [
                        f'out{batch}/{case_name}/slp_out.csv',
                        f'out{batch}/{case_name}/dt_out_pro_ltm.csv',
                        f'out{batch}/{case_name}/slp_out.csv',
                        f'out{batch}/{case_name}/dt_out_pro_ltm.csv',
                        f'frames_ground_truth/final_gt/{batch}/{case_name}.csv'
                    ]
        out_name=f'out_plot{batch}_feature_slp_final'
    
    plot_fog = PlotFog(out_name)
    csv_names = ['SLP', 'TD', 'SLP*', 'TD*', 'Ground Truth']
    csv_cols = [1, 1, 1, 1, 1]
    ops = [ [OP_x_T(1)],  #T
            [OP_None()],  #TD
            [OP_x_T(1), OP_Flt()], #T
            [OP_Flt()],            #TD
            [OP_Scale(0.1), OP_Flt()]] #GT
    colors = ['rgba(230,140,176,0.4)','rgba(143,150,216,0.4)',
              '#BC1458', '#222E97', '#007730']
    plot_fog.rd_csvs(csv_paths, csv_names, csv_cols, ops, colors=colors)
    plot_fog.get_metrics(gt_id=4)
    plot_fog.pt_csvs(case_name)


def plot_case_dtpmt(case_name, batch=0):
    print(f"PlotFog: {case_name} start!")
    if batch is None:
        csv_paths = [   f'out/{case_name}/dt_out_pro_ltm.csv',
                        f'out/{case_name}/dtp_mt.csv',
                        f'out/{case_name}/tp_out.csv',
                        f'out/{case_name}/fade_out.csv',
                        f'out/{case_name}/dt_out_pro_ltm.csv',
                        f'out/{case_name}/dtp_mt.csv',
                        f'out/{case_name}/tp_out.csv',
                        f'out/{case_name}/fade_out.csv',
                        f'frames_ground_truth/final_gt/{case_name}.csv']
        out_name='out_plot_fade_dtpmt'
    else:
        csv_paths = [f'out{batch}/{case_name}/dt_out_pro_ltm.csv',
                    f'out{batch}/{case_name}/dtp_mt.csv',
                    f'out{batch}/{case_name}/tp_out.csv',
                    f'out{batch}/{case_name}/fade_out.csv',
                    f'out{batch}/{case_name}/dt_out_pro_ltm.csv',
                    f'out{batch}/{case_name}/dtp_mt.csv',
                    f'out{batch}/{case_name}/tp_out.csv',
                    f'out{batch}/{case_name}/fade_out.csv',
                    f'frames_ground_truth/final_gt/{batch}/{case_name}.csv']
        out_name=f'out_plot{batch}_dtpmt'
    #out_name=f'out_plot_debug'777304
    plot_fog = PlotFog(out_name)
    csv_names = ['dtp_ori', 'dtpmt_ori', 'tp_ori', 'fade_ori', 'dtp', 'dtpmt', 'tp', 'fade','score']
    csv_cols = [1, 1, 1, 1, 1, 1, 1, 1, 1]
    ops_bak = [[OP_Norm(0.0154, 0.4152)], 
           [OP_Norm(0.4579, 0.8838), OP_x_T(0.5)], 
           [OP_Norm(0.2153, 2.5405)], 
           [OP_Norm(0.1106, 0.4152), OP_Flt()], 
           [OP_Norm(0.4579, 0.8838), OP_x_T(0.5), OP_Flt()], 
           [OP_Norm(0.7467, 2.5405), OP_Flt()],
           [OP_Scale(0.1), OP_Flt()]]
    ops = [[OP_None()],
           [OP_None()],
           [OP_x_T(1)], 
           [OP_Norm(0.2153, 2.5405)], 
           [OP_Flt()],
           [OP_Flt()],
           [OP_x_T(1), OP_Flt()], 
           [OP_Norm(0.2153, 2.5405), OP_Flt()],
           [OP_Scale(0.1), OP_Flt()]]

    #ops = [[OP_None()], [OP_Flt()],  [OP_None()], [OP_1_T()], [OP_Scale(0.1), OP_Flt()]]
    colors = [ 'rgba(143,150,216,0.6)', 'rgba(84,143,148,0.6)','rgba(230,140,176,0.6)', 'rgba(189,197,115, 0.6)', '#222E97', '#548F94', '#BC1458', '#777304', '#007730']
    plot_fog.rd_csvs(csv_paths, csv_names, csv_cols, ops, colors=colors)
    plot_fog.get_metrics(gt_id=8)
    plot_fog.pt_csvs(case_name)

def plot_case_pred(case_name, batch=0, preds_name='300_50'):
    print(f"PlotFog: {case_name} start!")
    if batch is None:
        csv_paths = [f'out/{case_name}/dt_out_pro_ltm.csv',
                        f'out/{case_name}/t_out.csv',
                        f'out/{case_name}/preds{preds_name}.csv',
                        f'out/{case_name}/dt_out_pro_ltm.csv',
                        f'out/{case_name}/t_out.csv',
                        f'out/{case_name}/preds{preds_name}.csv',
                        f'frames_ground_truth/final_gt/{case_name}.csv']
        out_name=f'out_plot_preds{preds_name}'
    else:
        csv_paths = [f'out{batch}/{case_name}/dt_out_pro_ltm.csv',
                    f'out{batch}/{case_name}/t_out.csv',
                    f'out{batch}/{case_name}/preds{preds_name}.csv',
                    f'out{batch}/{case_name}/dt_out_pro_ltm.csv',
                    f'out{batch}/{case_name}/t_out.csv',
                    f'out{batch}/{case_name}/preds{preds_name}.csv',
                    f'frames_ground_truth/final_gt/{batch}/{case_name}.csv']
        out_name=f'out_plot{batch}_preds{preds_name}'
    #out_name=f'out_plot_debug'777304
    plot_fog = PlotFog(out_name)
    csv_names = ['dtp_ori', 't_ori', 'preds_ori', 'dtp', 't', 'preds', 'score']
    csv_cols = [1, 1, 1, 1, 1, 1, 1]
    ops_bak = [[OP_Norm(0.0154, 0.4152)], 
           [OP_Norm(0.4579, 0.8838), OP_x_T(0.5)], 
           [OP_Norm(0.2153, 2.5405)], 
           [OP_Norm(0.1106, 0.4152), OP_Flt()], 
           [OP_Norm(0.4579, 0.8838), OP_x_T(0.5), OP_Flt()], 
           [OP_Norm(0.7467, 2.5405), OP_Flt()],
           [OP_Scale(0.1), OP_Flt()]]
    ops = [[OP_None()], 
           [OP_x_T(1)], 
           [OP_Scale(0.1)], 
           [OP_Flt()], 
           [OP_x_T(1), OP_Flt()], 
           [OP_Scale(0.1), OP_Flt()],
           [OP_Scale(0.1), OP_Flt()]]

    #ops = [[OP_None()], [OP_Flt()],  [OP_None()], [OP_1_T()], [OP_Scale(0.1), OP_Flt()]]
    colors = [ 'rgba(143,150,216,0.6)','rgba(230,140,176,0.6)', 'rgba(189,197,115, 0.6)', '#222E97',  '#BC1458', '#777304', '#007730']
    plot_fog.rd_csvs(csv_paths, csv_names, csv_cols, ops, colors=colors)
    plot_fog.get_metrics(gt_id=6)
    plot_fog.pt_csvs(case_name)

def plot_case_slpt(case_name, batch=0):
    print(f"PlotFog: {case_name} start!")
    if batch is None:
        csv_paths = [f'out/{case_name}/dt_out_pro_ltm.csv',
                        f'out/{case_name}/tp_out.csv',
                        f'out/{case_name}/slp_out.csv',
                        f'out/{case_name}/dt_out_pro_ltm.csv',
                        f'out/{case_name}/tp_out.csv',
                        f'out/{case_name}/slp_out.csv',
                        f'frames_ground_truth/final_gt/{case_name}.csv']
        out_name='out_plot_fade'
    else:
        csv_paths = [f'out{batch}/{case_name}/dt_out_pro_ltm.csv',
                    f'out{batch}/{case_name}/tp_out.csv',
                    f'out{batch}/{case_name}/slp_out.csv',
                    f'out{batch}/{case_name}/dt_out_pro_ltm.csv',
                    f'out{batch}/{case_name}/tp_out.csv',
                    f'out{batch}/{case_name}/slp_out.csv',
                    f'frames_ground_truth/final_gt/{batch}/{case_name}.csv']
        out_name=f'out_plot{batch}_slp'
    plot_fog = PlotFog(out_name)
    csv_names = ['TD', 'T', 'SLP_T', 'TD*', 'T*', 'SLP_T*', 'Ground Truth']
    csv_cols = [1, 1, 1, 1, 1, 1, 1]
    ops = [ [OP_None()], 
            [OP_x_T(1)], 
            [OP_x_T(1)], 
            [OP_Flt()],  
            [OP_x_T(1), OP_Flt()], 
            [OP_x_T(1), OP_Flt()],
            [OP_Scale(0.1), OP_Flt()] ]
    colors = [ 'rgba(143,150,216,0.4)','rgba(230,140,176,0.4)', 'rgba(189,197,115, 0.4)', \
               '#222E97', '#BC1458', '#777304', '#007730']
    plot_fog.rd_csvs(csv_paths, csv_names, csv_cols, ops, colors=colors)
    plot_fog.get_metrics(gt_id=6)
    plot_fog.pt_csvs(case_name)

def plot_case_pred_final(case_name, batch=0, preds_name='500_100_50_out', test=True):
    print(f"PlotFog: {case_name} start!")
    if batch is None:
        csv_paths = [f'out/{case_name}/dt_out_pro_ltm.csv',
                        f'out/{case_name}/preds{preds_name}.csv',
                        f'frames_ground_truth/final_gt/{case_name}.csv']
        out_name=f'out_plot_preds{preds_name}'
    else:
        csv_paths = [f'out{batch}/{case_name}/dt_out_pro_ltm.csv',
                    f'out{batch}/{case_name}/preds{preds_name}.csv',
                    f'frames_ground_truth/final_gt/{batch}/{case_name}.csv']
        out_name=f'out_plot_preds{preds_name}'
    if test:
        out_name += '_test'
    else:
        out_name += '_train'
    plot_fog = PlotFog(out_name)
    csv_names = ['TD', 'Fog Event', 'Ground Truth']
    csv_cols = [1, 1, 1]
    ops = [[OP_None()], 
           [OP_Add(1), OP_Scale(0.1)],
           [OP_Scale(0.1), OP_Flt()]]

    colors = [ 'rgba(143,150,216,0.6)', '#BC1458', '#007730']
    plot_fog.rd_csvs(csv_paths, csv_names, csv_cols, ops, colors=colors)
    plot_fog.get_metrics(gt_id=2)
    plot_fog.pt_csvs(case_name)



# 批处理函数
def batch_fade():
    batch0_csv = pathlib.Path('case_info/case_batch0.csv')
    batch1_csv = pathlib.Path('case_info/case_batch1.csv')
    batch0_df = pd.read_csv(batch0_csv)
    batch1_df = pd.read_csv(batch1_csv)
    case_names = []
    for i in range(len(batch0_df)):
        case_names.append(batch0_df['case_name'][i])
    for case_name in case_names:
        try:
            plot_case_fade_final(case_name=case_name, batch=0)
        except Exception as e:
            print(f"Error: {case_name}")
            print(f"\treason:{e}")
        else:
            print(f"Success: {case_name}")
    case_names = []
    for i in range(len(batch1_df)):
        case_names.append(batch1_df['case_name'][i])
    for case_name in case_names:
        try:
            plot_case_fade_final(case_name=case_name, batch=1)
        except Exception as e:
            print(f"Error: {case_name}")
            print(f"\treason:{e}")
        else:
            print(f"Success: {case_name}")

def batch_slp():
    batch0_csv = pathlib.Path('case_info/case_batch0.csv')
    batch1_csv = pathlib.Path('case_info/case_batch1.csv')
    batch0_df = pd.read_csv(batch0_csv)
    batch1_df = pd.read_csv(batch1_csv)
    case_names = []
    for i in range(len(batch0_df)):
        case_names.append(batch0_df['case_name'][i])
    for case_name in case_names:
        try:
            plot_case_slpt(case_name=case_name, batch=0)
        except Exception as e:
            print(f"Error: {case_name}")
            print(f"\treason:{e}")
        else:
            print(f"Success: {case_name}")
    case_names = []
    for i in range(len(batch1_df)):
        case_names.append(batch1_df['case_name'][i])
    for case_name in case_names:
        try:
            plot_case_slpt(case_name=case_name, batch=1)
        except Exception as e:
            print(f"Error: {case_name}")
            print(f"\treason:{e}")
        else:
            print(f"Success: {case_name}")

def batch_feature_t():
    batch0_csv = pathlib.Path('case_info/case_batch0.csv')
    batch1_csv = pathlib.Path('case_info/case_batch1.csv')
    batch0_df = pd.read_csv(batch0_csv)
    batch1_df = pd.read_csv(batch1_csv)
    case_names = []
    for i in range(len(batch0_df)):
        case_names.append(batch0_df['case_name'][i])
    for case_name in case_names:
        try:
            plot_case_feature_slp_final(case_name=case_name, batch=0)
        except Exception as e:
            print(f"Error: {case_name}")
            print(f"\treason:{e}")
        else:
            print(f"Success: {case_name}")
    case_names = []
    for i in range(len(batch1_df)):
        case_names.append(batch1_df['case_name'][i])
    for case_name in case_names:
        try:
            plot_case_feature_slp_final(case_name=case_name, batch=1)
        except Exception as e:
            print(f"Error: {case_name}")
            print(f"\treason:{e}")
        else:
            print(f"Success: {case_name}")

def batch_gemini():
    batch0_csv = pathlib.Path('case_info/case_batch0.csv')
    batch1_csv = pathlib.Path('case_info/case_batch1.csv')
    batch0_df = pd.read_csv(batch0_csv)
    batch1_df = pd.read_csv(batch1_csv)
    case_names = []
    for i in range(len(batch0_df)):
        case_names.append(batch0_df['case_name'][i])
    for case_name in case_names:
        try:
            plot_case_gemini_final(case_name=case_name, batch=0)
        except Exception as e:
            print(f"Error: {case_name}")
            print(f"\treason:{e}")
        else:
            print(f"Success: {case_name}")

    case_names = []
    for i in range(len(batch1_df)):
        case_names.append(batch1_df['case_name'][i])
    for case_name in case_names:
        try:
            plot_case_gemini_final(case_name=case_name, batch=1)
        except Exception as e:
            print(f"Error: {case_name}")
            print(f"\treason:{e}")
        else:
            print(f"Success: {case_name}")

def batch_pred(pred_name, test=True):
    if test:
        csv_path = 'case_info/train/test_cases.csv'
    else:
        csv_path = 'case_info/train/train_cases.csv'
    batch_id_set = {'1':0, '2':1, '3':1, '4':0, '5':0}
    case_info = pd.read_csv(csv_path)
    case_names = []
    for i in range(len(case_info)):
        case_names.append(case_info['case_name'][i])
    for case_name in case_names:
        try:
            batch_id = batch_id_set[case_name[0]]
            plot_case_pred_final(case_name=case_name, batch=batch_id, preds_name=pred_name, test=test)
        except Exception as e:
            print(f"Error: {case_name}")
            print(f"\treason:{e}")
        else:
            print(f"Success: {case_name}")

if __name__ == '__main__':
    batch_pred(pred_name='500_100_100_out', test=True)
    exit()
    batch_feature_t()
    exit()
    batch_slp()
    exit()
    batch_fade()
    exit()
    batch0_csv = pathlib.Path('case_info/case_batch0.csv')
    batch1_csv = pathlib.Path('case_info/case_batch1.csv')
    batch0_df = pd.read_csv(batch0_csv)
    batch1_df = pd.read_csv(batch1_csv)
    case_names = []
    # 绘制pred结果对比图
    for i in range(len(batch0_df)):
        case_names.append(batch0_df['case_name'][i])
    for case_name in case_names:
        try:
            plot_case_pred(case_name=case_name, batch=0, preds_name='500_10_out')
        except:
            print(f"Error: {case_name}")
        else:
            print(f"Success: {case_name}")
    case_names = []
    for i in range(len(batch1_df)):
        case_names.append(batch1_df['case_name'][i])
    for case_name in case_names:
        try:
            plot_case_pred(case_name=case_name, batch=1, preds_name='500_10_out')
        except:
            print(f"Error: {case_name}")
        else:
            print(f"Success: {case_name}")

    exit()
    # 绘制fade对比图
    batch_id = 0
    case_info_path = 'case_info/case_info0.csv'
    case_info = pd.read_csv(case_info_path)
    case_names = []
    for i in range(len(case_info)):
        case_names.append(case_info['case_name'][i])
    for case_name in case_names:
        plot_case_dtpmt(case_name=case_name, batch=batch_id)
        
    exit()
    # 绘制gemini对比图
    batch_id = 1
    case_info_path = 'case_info/case_batch1.csv'
    case_info = pd.read_csv(case_info_path)
    case_names = []
    for i in range(len(case_info)):
        case_names.append(case_info['case_name'][i])
    for case_name in case_names:
        GeminiCSVData(case_name=case_name, batch_id=batch_id).revise()
        plot_case_gemini(case_name=case_name, batch=batch_id)

    #csv_path = 'case_info1.csv'
    #case_info = pd.read_csv(csv_path)
    #case_names = []
    #for i in range(len(case_info)):
    #    case_names.append(case_info['case_name'][i])
    #for case_name in case_names:
    #    plot_case(case_name=case_name, batch=1)