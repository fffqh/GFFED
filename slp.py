import cv2
import pathlib
import numpy as np
import matlab.engine

from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr


class SLPRunner:
    def __init__(self, image_path, mask_path, output_path):
        self.img_dir = pathlib.Path(image_path)
        self.msk_dir = pathlib.Path(mask_path)        
        assert self.img_dir.is_dir(), "Image directory does not exist"
        assert self.msk_dir.is_dir(), "Mask directory does not exist"
        self.out_dir = pathlib.Path(output_path)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        print(f"Output directory: {self.out_dir}")
    
    def run_dir(self):
        eng = matlab.engine.start_matlab()
        eng.addpath('./SLP', nargout=0)
        for img_file in self.img_dir.iterdir():
            img = cv2.imread(str(img_file))
            re_img = eng.dehaze_slp(img)
            re_img = np.array(re_img)*255
            re_img = re_img.astype(np.uint8)
            # 计算ssim和psnr
            out_file = self.out_dir / f"out_{img_file.stem}.png"
            cv2.imwrite(str(out_file), re_img)
            print(f"Saved: {out_file}")
        eng.quit()

    def run_O_HAZE(self):
        eng = matlab.engine.start_matlab()
        eng.addpath('./SLP', nargout=0)
        out_csv = self.out_dir / "slp_results.csv"
        f = open(out_csv, "w")
        for img_file in self.img_dir.iterdir():
            img = cv2.imread(str(img_file))
            re_img = eng.dehaze_slp(img)
            msk_path = self.msk_dir / f"{img_file.stem[:-5]}_GT.jpg"
            print(f"msk_img:{msk_path}")
            msk_img = cv2.imread(str(msk_path))
            re_img = np.array(re_img)*255
            re_img = re_img.astype(np.uint8)
            print(f"re_img:{re_img.shape} msk_img:{msk_img.shape}")
            # 计算ssim和psnr
            out_ssim = ssim(msk_img, re_img, data_range=255, channel_axis=2)
            out_psnr = psnr(msk_img, re_img, data_range=255)
            print(f"SSIM: {out_ssim}")
            print(f"PSNR: {out_psnr}")
            out_file = self.out_dir / f"out_{img_file.stem}.png"
            cv2.imwrite(str(out_file), re_img)
            print(f"Saved: {out_file}")
            f.write(f"{img_file.stem},{out_ssim},{out_psnr}\n")
        f.close()
        eng.quit()

    def run_D_HAZY(self):
        eng = matlab.engine.start_matlab()
        eng.addpath('./SLP', nargout=0)
        out_csv = self.out_dir / "slp_results.csv"
        f = open(out_csv, "w")
        for img_file in self.img_dir.iterdir():
            img = cv2.imread(str(img_file))
            re_img = eng.dehaze_slp(img)
            msk_path = self.msk_dir / f"{img_file.stem[:-5]}_im0.png"
            print(f"msk_img:{msk_path}")
            msk_img = cv2.imread(str(msk_path))
            re_img = np.array(re_img)*255
            re_img = re_img.astype(np.uint8)
            print(f"re_img:{re_img.shape} msk_img:{msk_img.shape}")
            # 计算ssim和psnr
            out_ssim = ssim(msk_img, re_img, data_range=255, channel_axis=2)
            out_psnr = psnr(msk_img, re_img, data_range=255)
            print(f"SSIM: {out_ssim}")
            print(f"PSNR: {out_psnr}")
            out_file = self.out_dir / f"out_{img_file.stem}.png"
            cv2.imwrite(str(out_file), re_img)
            print(f"Saved: {out_file}")
            f.write(f"{img_file.stem},{out_ssim},{out_psnr}\n")
        f.close()
        eng.quit()

    def run_SOTS(self):
        eng = matlab.engine.start_matlab()
        eng.addpath('./SLP', nargout=0)
        out_csv = self.out_dir / "slp_results.csv"
        f = open(out_csv, "w")
        for img_file in self.img_dir.iterdir():
            img = cv2.imread(str(img_file))
            re_img = eng.dehaze_slp(img)
            msk_path = self.msk_dir / f"{img_file.stem[:4]}.png"
            msk_img = cv2.imread(str(msk_path))[10:-10,10:-10,:]
            re_img = np.array(re_img)*255
            re_img = re_img.astype(np.uint8)
            print(f"re_img:{re_img.shape} msk_img:{msk_img.shape}")
            # 计算ssim和psnr
            out_ssim = ssim(msk_img, re_img, data_range=255, channel_axis=2)
            out_psnr = psnr(msk_img, re_img, data_range=255)
            print(f"SSIM: {out_ssim}")
            print(f"PSNR: {out_psnr}")
            out_file = self.out_dir / f"out_{img_file.stem}.png"
            cv2.imwrite(str(out_file), re_img)
            print(f"Saved: {out_file}")
            f.write(f"{img_file.stem},{out_ssim},{out_psnr}\n")
        f.close()
        eng.quit()

if __name__ == '__main__':
    #r = SLPRunner(image_path='E:/slp_data/SOTS/nyuhaze500/hazy', 
    #              mask_path='E:/slp_data/SOTS/nyuhaze500/gt', 
    #              output_path='E:/slp_data/SOTS/nyuhaze500/out')
    #r.run_SOTS()
    #r = SLPRunner(image_path='E:/slp_data/DHazy/Middlebury_Hazy',
    #              mask_path='E:/slp_data/DHazy/Middlebury_GT',
    #              output_path='E:/slp_data/DHazy/Middlebury_out')
    #r.run_D_HAZY()
    #r = SLPRunner(image_path='E:/slp_data/O-HAZE/hazy',
    #              mask_path='E:/slp_data/O-HAZE/GT',
    #              output_path='E:/slp_data/O-HAZE/out')
    #r.run_O_HAZE()
    r = SLPRunner(image_path='out0/4_1208_070325/frame0005',
                  mask_path='out0/4_1208_070325/frame0005',
                  output_path='E:/slp_data/fog_out')
    r.run_dir()

    print('Done!')


