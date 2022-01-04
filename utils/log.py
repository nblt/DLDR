import os
import yaml 
import logging
import numpy as np 

def checkdir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path, exist_ok=True)
 
def dump_args(args, output_dir):
    checkdir(output_dir)
    args_text = yaml.safe_dump(args.__dict__, default_flow_style=False)
    with open(os.path.join(output_dir, 'args.yaml'), 'w') as f:
        f.write(args_text)

def console_out(logFilename):
    ''' Output log to file and console '''
    # Define a Handler and set a format which output to file
    logging.basicConfig(
                    level    = logging.DEBUG,              # 定义输出到文件的log级别，                                                            
                    format   = '%(asctime)s  %(filename)s : %(levelname)s  %(message)s',    # 定义输出log的格式
                    datefmt  = '%Y-%m-%d %A %H:%M:%S',                                     # 时间
                    filename = logFilename,                # log文件名
                    filemode = 'w')                        # 写入模式“w”或“a”
    # Define a Handler and set a format which output to console
    console = logging.StreamHandler()                  # 定义console handler
    console.setLevel(logging.INFO)                     # 定义该handler级别
    formatter = logging.Formatter('%(asctime)s  %(filename)s : %(levelname)s  %(message)s')  #定义该handler格式
    console.setFormatter(formatter)
    # Create an instance
    logging.getLogger().addHandler(console)           # 实例化添加handler

def log_dump_metrics(output_dir, **metrics):
    for name, nums in metrics.items():
        logging.info(f'{name}: {nums}')
        np.save(os.path.join(output_dir, f'{name}.npy'), nums)