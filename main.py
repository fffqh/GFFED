import cv2
import math
import numpy as np
import pandas as pd
import pathlib

from scipy.optimize import curve_fit
import plotly as py
import plotly.graph_objs as go
pyplt = py.offline.plot

# Dark Prior
def DarkChannel(im,sz):
    b,g,r = cv2.split(im)
    dc = cv2.min(cv2.min(r,g),b);
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(sz,sz)) # 矩形结构元素
    dark = cv2.erode(dc,kernel) # 腐蚀操作
    return dark # 返回暗通道图像

def AtmLight(im,dark):
    [h,w] = im.shape[:2]
    imsz = h*w
    numpx = int(max(math.floor(imsz/1000),1)) # 前千分之一
    darkvec = dark.reshape(imsz);
    imvec = im.reshape(imsz,3);

    indices = darkvec.argsort(); # 返回数组值从小到大的索引
    indices = indices[imsz-numpx::] # 索引数组的后千分之一项（即暗通道数值最大的千分之一）

    atmsum = np.zeros([1,3])
    for ind in range(1,numpx):
       atmsum = atmsum + imvec[indices[ind]]

    A = atmsum / numpx;
    return A

def AtmLight_Pro(im, dark, mask):
    dark_mask = dark.copy()
    dark_mask[mask] = 0
    
    [h,w] = im.shape[:2]
    imsz = h*w
    numpx = int(max(math.floor(imsz/1000),1))
    
    darkvec = dark_mask.reshape(imsz);
    imvec = im.reshape(imsz,3);
    indices = darkvec.argsort(); # 返回数组值从小到大的索引
    indices = indices[imsz-numpx::] # 索引数组的后千分之一项（即暗通道数值最大的千分之一）

    atmsum = np.zeros([1,3])
    for ind in range(1,numpx):
       atmsum = atmsum + imvec[indices[ind]]

    A = atmsum / numpx;
    return A

def TransmissionEstimate(im,A,sz):
    omega = 0.95;
    im3 = np.empty(im.shape,im.dtype);

    for ind in range(0,3):
        im3[:,:,ind] = im[:,:,ind]/A[0,ind]

    transmission = 1 - omega*DarkChannel(im3,sz)
    return transmission

def Guidedfilter(im,p,r,eps):
    mean_I = cv2.boxFilter(im,cv2.CV_64F,(r,r))
    mean_p = cv2.boxFilter(p, cv2.CV_64F,(r,r))
    mean_Ip = cv2.boxFilter(im*p,cv2.CV_64F,(r,r))
    cov_Ip = mean_Ip - mean_I*mean_p

    mean_II = cv2.boxFilter(im*im,cv2.CV_64F,(r,r))
    var_I   = mean_II - mean_I*mean_I

    a = cov_Ip/(var_I + eps)
    b = mean_p - a*mean_I

    mean_a = cv2.boxFilter(a,cv2.CV_64F,(r,r))
    mean_b = cv2.boxFilter(b,cv2.CV_64F,(r,r))

    q = mean_a*im + mean_b
    return q

def TransmissionRefine(im,et):
    gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    gray = np.float64(gray)/255
    r = 60
    eps = 0.0001
    t = Guidedfilter(gray,et,r,eps)
    return t

def Recover(im,t,A,tx = 0.1):
    res = np.empty(im.shape,im.dtype)
    t = cv2.max(t,tx)
    for ind in range(0,3):
        res[:,:,ind] = (im[:,:,ind]-A[0,ind])/t + A[0,ind]
    return res


# Remove Txt
def midpoint(x1,y1,x2,y2):
    x_mid = int((x1 + x2)/2)
    y_mid = int((y1 + y2)/2)
    return (x_mid, y_mid)
def remove_txt(img_path):
    img = cv2.imread(img_path)
    # ocr = CnOcr()
    # ocr_out = ocr.ocr(img_path)
    # out_len = len(ocr_out)
    # for i in range(out_len):
    #     box = ocr_out[i]['position']
    #     point0 = midpoint(box[0][0],box[0][1],box[3][0],box[3][1])
    #     point1 = midpoint(box[1][0],box[1][1],box[2][0],box[2][1])
    #     thickness = int(math.sqrt( (box[3][0] - box[0][0])**2 + (box[0][1] - box[3][1])**2 ))
    #     mask = np.zeros(img.shape[:2], dtype="uint8")
    #     cv2.line(mask, point0, point1, 255, thickness)
    #     img = cv2.inpaint(img, mask, 7, cv2.INPAINT_NS)
    return img


# Run Processor
class DirProcessor:
    def __init__(self, dir_path, out_path):
        self.dirpath = pathlib.Path(dir_path)
        assert self.dirpath.exists(), f"video path error: {dir_path}"
        # jpg或png
        self.file_walk_list = sorted(list(self.dirpath.rglob('*.jpg')))
        self.opath = pathlib.Path(out_path)
        if not self.opath.exists():
            self.opath.mkdir(parents=True)
            print(f"create out path: {out_path}")

    def get_mask(self, img):
        """cv2.img uint8 255"""
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        mask = img_gray > 220
        return mask

    def get_t_pro(self, img):
        l_mask = self.get_mask(img)
        ori_img = img.astype('float32')/255
        ori_dark = DarkChannel(ori_img, 15)
        ori_A = AtmLight_Pro(ori_img, ori_dark, l_mask)
        ori_t = TransmissionRefine(ori_img, TransmissionEstimate(ori_img, ori_A, 15))
        return ori_t

    def get_t(self,img):
        ori_img = img.astype('float32')/255
        ori_dark = DarkChannel(ori_img, 15)
        ori_A = AtmLight(ori_img, ori_dark)
        ori_t = TransmissionRefine(ori_img, TransmissionEstimate(ori_img, ori_A, 15))
        return ori_t
    
    def get_dt_pro(self, img):
        l_mask = self.get_mask(img)
        ori_img = img.astype('float32')/255
        ori_dark = DarkChannel(ori_img, 15)
        ori_A = AtmLight_Pro(ori_img, ori_dark, l_mask)
        ori_t = TransmissionRefine(ori_img, TransmissionEstimate(ori_img, ori_A, 15))
        dfg_img = Recover(ori_img, ori_t, ori_A, 0.1)
        dfg_dark = DarkChannel(dfg_img, 15)
        dfg_A = AtmLight_Pro(dfg_img, dfg_dark, l_mask)
        dfg_t = TransmissionRefine(dfg_img, TransmissionEstimate(dfg_img, dfg_A, 15))
        dt = dfg_t - ori_t
        return dt        

    def get_dt(self, img):
        """img:(0,255)"""     
        ori_img = img.astype('float32')/255
        ori_dark = DarkChannel(ori_img, 15)
        ori_A = AtmLight(ori_img, ori_dark)
        ori_t = TransmissionRefine(ori_img, TransmissionEstimate(ori_img, ori_A, 15))
        dfg_img = Recover(ori_img, ori_t, ori_A, 0.1)
        dfg_dark = DarkChannel(dfg_img, 15)
        dfg_A = AtmLight(dfg_img, dfg_dark)
        dfg_t = TransmissionRefine(dfg_img, TransmissionEstimate(dfg_img, dfg_A, 15))
        dt = dfg_t - ori_t
        return dt

    def get_t_mask(self, t_img):
        t_mask = t_img > 0.001
        return t_mask

    def run_remove_txt(self):
        for (idx, img_file) in enumerate(self.file_walk_list):
            img_out_path = self.opath / img_file.name
            if img_out_path.exists():
                print(f"跳过：{str(img_out_path)}")
                continue            
            img = remove_txt(str(img_file))
            cv2.imwrite(str(img_out_path), img)
            print(f"Done:{str(img_file)}")

    def run_t_pro(self, save_csv=True):
        if save_csv:
            csv_path = self.opath/'tp_out.csv'
            f = open(str(csv_path), 'w')
        for (idx, img_file) in enumerate(self.file_walk_list):
            #print(f"img_file:{img_file}")
            img = cv2.imread(str(img_file))
            t_img = self.get_t_pro(img)
            t_mean = np.mean(t_img)
            t_var = np.var(t_img)
            if save_csv:
                f.write(f"{idx},{t_mean},{t_var}\n")
            #print(f"done frame {idx}")
            if idx % 1000 == 0:
                print(f"Done tp img_file:{img_file}")
        if save_csv:
            f.close()

    def run_dtpro(self, save_csv=True):
        if save_csv:
            csv_path = self.opath/'dt_out_pro_ltm.csv'
            f = open(str(csv_path), 'w')
        
        for (idx, img_file) in enumerate(self.file_walk_list):
            img = cv2.imread(str(img_file))
            dt_img = self.get_dt_pro(img)
            l_mask = self.get_mask(img)
            dt_matric = dt_img[~l_mask]
            dt_mean = np.mean(dt_matric)
            dt_var = np.var(dt_matric)
            if save_csv:
                f.write(f"{idx},{dt_mean},{dt_var}\n")
            if idx % 1000 == 0:
                print(f"Done img_file:{img_file}")
        if save_csv:
            f.close()

    def run_slp(self, save_csv=True):
        import matlab.engine
        eng = matlab.engine.start_matlab()
        eng.addpath(r'./SLP',nargout=0)
        start_idx = 0
        if save_csv:
            csv_path = self.opath/'fade_out.csv'
            if csv_path.exists():
                try:
                    df = pd.read_csv(csv_path, header=None)
                    start_idx = int(df.iloc[-1,0]) + 1
                except:
                    start_idx = 0
                print(f"csv文件已存在，从{start_idx}开始计算")
            f = open(str(csv_path), 'a+')
        try:
            for (idx, img_file) in enumerate(self.file_walk_list):
                if idx < start_idx:
                    print(f"跳过： idx{idx} {str(img_file)}")
                    continue
                img = cv2.imread(str(img_file))
                fade_value = eng.FADE(img)
                f.write(f"{idx},{fade_value}\n")
                f.flush()
                print(f"Done: fade={fade_value} img:{str(img_file)}")
        except:
            print(f"Except Error: {str(img_file)}")
            if save_csv:
                f.close()
            eng.quit()
        else:
            if save_csv:
                f.close()
            eng.quit()

    def run_fade(self, save_csv=True):
        import matlab.engine
        eng = matlab.engine.start_matlab()
        eng.addpath(r'./FADE',nargout=0)
        start_idx = 0
        if save_csv:
            csv_path = self.opath/'fade_out.csv'
            if csv_path.exists():
                try:
                    df = pd.read_csv(csv_path, header=None)
                    start_idx = int(df.iloc[-1,0]) + 1
                except:
                    start_idx = 0
                print(f"csv文件已存在，从{start_idx}开始计算")
            f = open(str(csv_path), 'a+')
        try:
            for (idx, img_file) in enumerate(self.file_walk_list):
                if idx < start_idx:
                    print(f"跳过： idx{idx} {str(img_file)}")
                    continue
                img = cv2.imread(str(img_file))
                fade_value = eng.FADE(img)
                f.write(f"{idx},{fade_value}\n")
                f.flush()
                print(f"Done: fade={fade_value} img:{str(img_file)}")
        except:
            print(f"Except Error: {str(img_file)}")
            if save_csv:
                f.close()
            eng.quit()
        else:
            if save_csv:
                f.close()
            eng.quit()

    def run_plot_t(self):
        # 绘制t的热力图
        for (idx, img_file) in enumerate(self.file_walk_list):
            img = cv2.imread(str(img_file))
            t_img = self.get_t_pro(img)
            # plotly绘制热力图, t_img的形状为(x,y)
            x = list(range(t_img.shape[1]))
            y = list(range(t_img.shape[0]))
            X,Y = np.meshgrid(x,y)
            trace = go.Heatmap(x=X.ravel(),y=Y.ravel(),z=t_img.ravel(), colorscale='Viridis')
            data = [trace]
            layout = go.Layout(title=f"t_img_{img_file.name}")
            fig = go.Figure(data=data, layout=layout)
            plt_path = self.opath/f"t_img_{img_file.name}.html"
            pyplt(fig, filename=str(plt_path), show_link=False, auto_open=False)
            print(f"Done: plot t_img_{img_file.name}.html")

class VideoProcessor:
    def __init__(self, video_path, out_path):
        self.vpath = pathlib.Path(video_path)
        assert self.vpath.exists(), f"video path error: {video_path}"
        self.opath = pathlib.Path(out_path)
        if not self.opath.exists():
            self.opath.mkdir(parents=True)
            print(f"create out path: {out_path}")
        print(f"VideoProcessor init: video path: {video_path}\t out path: {out_path}")

    def get_t(self, img):
        ori_img = img.astype('float32')/255
        ori_dark = DarkChannel(ori_img, 15)
        ori_A = AtmLight(ori_img, ori_dark)
        ori_t = TransmissionRefine(ori_img, TransmissionEstimate(ori_img, ori_A, 15))
        return ori_t

    def get_t_pro(self, img):
        l_mask = self.get_mask(img)
        ori_img = img.astype('float32')/255
        ori_dark = DarkChannel(ori_img, 15)
        ori_A = AtmLight_Pro(ori_img, ori_dark, l_mask)
        ori_t = TransmissionRefine(ori_img, TransmissionEstimate(ori_img, ori_A, 15))
        return ori_t
    
    def get_mask(self, img):
        """cv2.img uint8 255"""
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        l_mask = img_gray > 220
        return l_mask

    def get_dt(self, img):
        """img:(0,255)"""     
        ori_img = img.astype('float32')/255
        ori_dark = DarkChannel(ori_img, 15)
        ori_A = AtmLight(ori_img, ori_dark)
        ori_t = TransmissionRefine(ori_img, TransmissionEstimate(ori_img, ori_A, 15))
        dfg_img = Recover(ori_img, ori_t, ori_A, 0.1)
        dfg_dark = DarkChannel(dfg_img, 15)
        dfg_A = AtmLight(dfg_img, dfg_dark)
        dfg_t = TransmissionRefine(dfg_img, TransmissionEstimate(dfg_img, dfg_A, 15))
        dt = dfg_t - ori_t
        return dt
    
    def get_dt_pro(self, img):
        l_mask = self.get_mask(img)
        ori_img = img.astype('float32')/255
        ori_dark = DarkChannel(ori_img, 15)
        ori_A = AtmLight_Pro(ori_img, ori_dark, l_mask)
        ori_t = TransmissionRefine(ori_img, TransmissionEstimate(ori_img, ori_A, 15))
        dfg_img = Recover(ori_img, ori_t, ori_A, 0.1)
        dfg_dark = DarkChannel(dfg_img, 15)
        dfg_A = AtmLight_Pro(dfg_img, dfg_dark, l_mask)
        dfg_t = TransmissionRefine(dfg_img, TransmissionEstimate(dfg_img, dfg_A, 15))
        dt = dfg_t - ori_t
        return dt        

    # 大气光强增强，传输矩阵增强
    def run_apro_dtpro(self, save_csv=True, save_dtimg=False, save_img=False, max_idx=50000):
        cap = cv2.VideoCapture(str(self.vpath))
        assert cap.isOpened(), f"video open failed: {str(self.vpath)}"
        fps = cap.get(cv2.CAP_PROP_FPS)
        fcnt = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        print(f"**video info: {fps} fps\t {fcnt} frames.")
        idx = 0
        if save_csv:
            csv_path = self.opath/'dt_out_pro_ltm.csv'
            f = open(str(csv_path), 'w')
        
        while True:
            ret, img = cap.read()
            if (idx > max_idx) or (not ret):
                break
            idx = idx + 1
            dt_img = self.get_dt_pro(img)
            l_mask = self.get_mask(img)
            dt_matric = dt_img[~l_mask]
            dt_mean = np.mean(dt_matric)
            dt_var = np.var(dt_matric)

            if save_dtimg:
                dir_name = f"dt_frame{idx//5000:04}"
                img_dir = self.opath/dir_name
                if not img_dir.exists():
                    img_dir.mkdir(parents=True)
                img_path = img_dir/f"dt_frame{idx:04}.jpg"
                cv2.imwrite(str(img_path), dt_img)
            if save_img:
                dir_name = f"frame{idx//5000:04}"
                img_dir = self.opath/dir_name
                if not img_dir.exists():
                    img_dir.mkdir(parents=True)
                img_path = img_dir/f"frame{idx:04}.jpg"
                cv2.imwrite(str(img_path), img)
            if save_csv:
                f.write(f"{idx},{dt_mean},{dt_var}\n")
            if idx % 1000 == 0:
                print(f"vp done frame {idx}")
        if save_csv:
            f.close()

    # 大气光强增强，传输矩阵未增强
    def run_t_pro(self, save_csv=True, save_dtimg=False, save_img=False, max=5000):
        cap = cv2.VideoCapture(str(self.vpath))
        assert cap.isOpened(), f"video open failed: {str(self.vpath)}"
        fps = cap.get(cv2.CAP_PROP_FPS)
        fcnt = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        print(f"**video info: {fps} fps\t {fcnt} frames.")
        idx = 0
        if save_csv:
            csv_path = self.opath/'tp_out.csv'
            f = open(str(csv_path), 'w')
        
        while True:
            ret, img = cap.read()
            if not ret:
                break
            idx = idx + 1
            t_img = self.get_t_pro(img)
            t_mean = np.mean(t_img)
            t_var = np.var(t_img)

            if save_dtimg:
                dir_name = f"dt_frame{idx//5000:04}"
                img_dir = self.opath/dir_name
                if not img_dir.exists():
                    img_dir.mkdir(parents=True)
                img_path = img_dir/f"dt_frame{idx:04}.jpg"
                cv2.imwrite(str(img_path), t_img)
            if save_img:
                dir_name = f"frame{idx//5000:04}"
                img_dir = self.opath/dir_name
                if not img_dir.exists():
                    img_dir.mkdir(parents=True)
                img_path = img_dir/f"frame{idx:04}.jpg"
                cv2.imwrite(str(img_path), img)
            if save_csv:
                f.write(f"{idx},{t_mean},{t_var}\n")
            if idx % 1000 == 0:
                print(f"Done tp frame {idx}")
        if save_csv:
            f.close()
    
    def run_fade(self, save_csv=True, save_img=False, max_idx=50000):
        # 视频读取
        cap = cv2.VideoCapture(str(self.vpath))
        assert cap.isOpened(), f"video open failed: {str(self.vpath)}"
        fps = cap.get(cv2.CAP_PROP_FPS)
        fcnt = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        print(f"**video info: {fps} fps\t {fcnt} frames.")
        idx = 0        
        
        # matlab引擎启动
        import matlab.engine
        eng = matlab.engine.start_matlab()
        eng.addpath(r'./FADE',nargout=0)
        start_idx = 0
        
        # csv文件打开
        if save_csv:
            csv_path = self.opath/'fade_out.csv'
            if csv_path.exists():
                try:
                    df = pd.read_csv(csv_path, header=None)
                    start_idx = int(df.iloc[-1,0]) + 1
                except:
                    start_idx = 0
                print(f"csv文件已存在，从{start_idx}开始计算")
            f = open(str(csv_path), 'a+')
        
        while True:
            ret, img = cap.read()
            if (idx > max_idx) or (not ret):
                break
            # 判断是否要保存
            if save_img:
                dir_name = f"frame{idx//5000:04}"
                img_dir = self.opath/dir_name
                if not img_dir.exists():
                    img_dir.mkdir(parents=True)
                img_path = img_dir/f"frame{idx:04}.jpg"
                cv2.imwrite(str(img_path), img)
            # 判断是否需要计算FADE
            if idx < start_idx:
                idx+=1
                print(f"跳过： idx{idx}",flush=True)
                continue
            # 开始计算FADE
            try:
                fade_value = eng.FADE(img)
            except:
                print(f"Matlab Except Error!",flush=True)
                break
            f.write(f"{idx},{fade_value}\n")
            f.flush()
            if idx % 500 == 0:
                print(f"Done: fade={fade_value} idx{idx}",flush=True)
            idx+=1

        eng.quit()
        if save_csv:
            f.close()

    def run_slp(self, save_csv=True, save_img=False, max_idx=50000):
            # 视频读取
            cap = cv2.VideoCapture(str(self.vpath))
            assert cap.isOpened(), f"video open failed: {str(self.vpath)}"
            fps = cap.get(cv2.CAP_PROP_FPS)
            fcnt = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            print(f"**video info: {fps} fps\t {fcnt} frames.")
            idx = 0        
            
            # matlab引擎启动
            import matlab.engine
            eng = matlab.engine.start_matlab()
            eng.addpath(r'./SLP',nargout=0)
            start_idx = 0
            
            # csv文件打开
            if save_csv:
                csv_path = self.opath/'slp_out.csv'
                if csv_path.exists():
                    try:
                        df = pd.read_csv(csv_path, header=None)
                        start_idx = int(df.iloc[-1,0]) + 1
                    except:
                        start_idx = 0
                    print(f"csv文件已存在，从{start_idx}开始计算")
                f = open(str(csv_path), 'a+')
            
            while True:
                ret, img = cap.read()
                if (idx > max_idx) or (not ret):
                    break
                # 判断是否要保存
                if save_img:
                    dir_name = f"frame{idx//5000:04}"
                    img_dir = self.opath/dir_name
                    if not img_dir.exists():
                        img_dir.mkdir(parents=True)
                    img_path = img_dir/f"frame{idx:04}.jpg"
                    cv2.imwrite(str(img_path), img)
                # 判断是否需要计算FADE
                if idx < start_idx:
                    idx+=1
                    print(f"跳过： idx{idx}",flush=True)
                    continue
                # 开始计算FADE
                try:
                    slp_t = eng.get_transmission(img)
                except:
                    print(f"Matlab Except Error!",flush=True)
                    break
                f.write(f"{idx},{slp_t}\n")
                f.flush()
                if idx % 500 == 0:
                    print(f"Done: slp={slp_t} idx{idx}",flush=True)
                idx+=1

            eng.quit()
            if save_csv:
                f.close()

# 得到一个视频case的 dt_pro, dt, t 以及对应的帧数据
def run_case(case_video_path, case_name):
    try:
        vp = VideoProcessor(case_video_path,f"out/{case_name}")
        vp.run_apro_dtpro(save_csv=True, save_dtimg=False, save_img=True)
        dp = DirProcessor(f"out/{case_name}", f"out/{case_name}")
        dp.run_dt()
        dp.run_t()
    except:
        print(f"{case_name} failed!")
    else:
        print(f"{case_name} success!")
def run_case_t_pro(case_name, batch_id):
    try:
        dp = DirProcessor(f"out{batch_id}/{case_name}", f"out{batch_id}/{case_name}")
        dp.run_t_pro()
    except:
        print(f"{case_name} failed!")
    else:
        print(f"{case_name} success!")
def run_case_dt_pro_mt(case_name, batch_id):
    try:
        dp = DirProcessor(f"out{batch_id}/{case_name}", f"out{batch_id}/{case_name}")
        dp.run_dtpro_mt()
    except:
        print(f"{case_name} failed!")
    else:
        print(f"{case_name} success!")
def run_baseline_fade(case_name, batch_id):
    try:
        dp = DirProcessor(f"out{batch_id}/{case_name}", f"out{batch_id}/{case_name}")
        dp.run_fade()
    except:
        print(f"{case_name} failed!")
    else:
        print(f"{case_name} success!")
def run_baseline_slp(case_video_path, case_name, batch_id):
    try:
        vp = VideoProcessor(case_video_path,f"out{batch_id}/{case_name}")
        vp.run_slp()
    except Exception as e:
        print(f"{case_name} failed!")
        print(f"\terror:{e}")
    else:
        print(f"{case_name} success!")

if __name__=='__main__':
    print('-- 开始 --')
    batch_id = 0
    case_df = pd.read_csv("case_info/case_5.csv")
    case_names = []
    for i in range(len(case_df)):
        case_names.append(case_df['case_name'][i])
    for case_name in case_names:
        run_case_t_pro(case_name, batch_id)
    print('-- 结束 --')
