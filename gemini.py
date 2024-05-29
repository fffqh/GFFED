import os
import json
import pandas as pd
import requests
import pathlib
import google.generativeai as genai
from IPython.display import Image
import time

class GeminiFogDirectory:
    def __init__(self, dir_path, out_path):
        self.dir_path = pathlib.Path(dir_path)
        assert self.dir_path.exists(), f"video path error: {dir_path}"
        self.file_walk_list = sorted(list(self.dir_path.rglob('*.jpg')))
        self.opath = pathlib.Path(out_path)
        if not self.opath.exists():
            self.opath.mkdir(parents=True)
            print(f"create out path: {out_path}")
                
        res1=requests.get('https://myip.ipip.net/')
        print(res1.text)
        GOOGLE_API_KEY=os.environ.get('GOOGLE_API_KEY')
        genai.configure(api_key=GOOGLE_API_KEY, transport='rest')
        for model in genai.list_models():
            if 'generateContent' in model.supported_generation_methods:
                print(model.name)

        self.model = genai.GenerativeModel('gemini-1.0-pro-vision-latest')
        print(f"已加载Gemini模型: gemini-1.0-pro-vision-latest !")
    
    def set_refer(self):
        self.img1 = Image('./llm/reference/1.jpg')
        self.img2 = Image('./llm/reference/2.jpg')
        self.img3 = Image('./llm/reference/3.jpg')
        self.img4 = Image('./llm/reference/4.jpg')
        self.img5 = Image('./llm/reference/5.jpg')

    def get_prompt(self, img_path):
        img = Image(str(img_path))
        prompt = [  
            'This is an image with a fog density level of 1', self.img1,\
            'This is an image with a fog density level of 2', self.img2,\
            'This is an image with a fog density level of 3', self.img3,\
            'This is an image with a fog density level of 4', self.img4,\
            'This is an image with a fog density level of 5', self.img5,\
            'What do you think is the fog density level of this image?',img,\
            'Please select a fog density level from 1, 2, 3, 4, 5, and respond in JSON format, for example: {"rank": 2}'
            ]
        return prompt
    
    def check_and_process_response(self, res, img_path=''):
        try:
            res_str = str.strip(str(res.candidates[0].content.parts[0].text))
            res_json = json.loads(res_str)
            rank = int(res_json['rank'])
            if rank in [1,2,3,4,5]:
                return rank
            else:
                print(f"res outofrank: {img_path} \n \t res: {res_str}")
                return None        
        except:
            print(f"res error: {img_path} \n \t res: {res_str}")
            return None

    def run(self, step=100, save_csv=True, append=True):
        max_idx = -1
        if save_csv:
            csv_path = self.opath/'speed_gemini.csv'
            if append and csv_path.exists():
                try:
                    df = pd.read_csv(str(csv_path))
                    # 遍历df，找出已存在的最大idx
                    for _, row in df.iterrows():
                        eidx = int(row[0])
                        max_idx = eidx if eidx > max_idx else max_idx
                except:
                    print(f"读取csv文件失败：{csv_path}")
                print(f"继续未完成的工作，当前已完成的最大id：{max_idx}")
                f = open(str(csv_path), 'a+')
            else:
                f = open(str(csv_path), 'w')
        t = 0
        for (idx, img_file) in enumerate(self.file_walk_list):
            if idx < max_idx:
                print(f"跳过:{idx} 文件:{img_file}")
                continue
            if t % step != 0:
                t += 1
                continue
            st_time = time.time()
            p = self.get_prompt(img_file)
            r = self.model.generate_content(p)
            rank = self.check_and_process_response(r, img_path=img_file)
            ed_time = time.time()
            duration = ed_time - st_time
            rank = -1 if rank is None else rank
            if save_csv:
                f.write(f"{idx},{rank},{duration}\n")
            print(f"Done idx:{idx} rank:{rank} duration:{duration} s.")
            t += 1
        if save_csv:
            f.close()

if __name__ == '__main__':
    gfd = GeminiFogDirectory(dir_path='out1/2_1126_070641/', out_path='out1/2_1126_070641/')
    gfd.set_refer()
    gfd.run()
