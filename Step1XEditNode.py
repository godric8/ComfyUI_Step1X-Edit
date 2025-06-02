import torch
import numpy as np
from PIL import Image
import os
import json
import tempfile
import folder_paths # comfyUI 路径管理模块
import subprocess
import sys
import glob
import shutil
import threading
from concurrent.futures import ThreadPoolExecutor

class Step1XEdit:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "instruction": ("STRING", {"multiline": True, "default": "Make the sky more dramatic"}),
                "model_type": (["Step1X-Edit", "Step1X-Edit-FP8"], {"default": "Step1X-Edit"}),
                "use_offload": ("BOOLEAN", {"default": False}),
                "use_quantized": ("BOOLEAN", {"default": False}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "steps": ("INT", {"default": 28, "min": 1, "max": 100}),
                "cfg_scale": ("FLOAT", {"default": 7.5, "min": 0.1, "max": 20.0, "step": 0.1}),
                "size_level": ("INT", {"default": 512, "min": 512, "max": 1024, "step": 256}),
            },
            "optional": {
                "lora_path": ("STRING", {"default": ""}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("edited_image",)
    FUNCTION = "edit_image"
    CATEGORY = "image/editing"
    
    def __init__(self):
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.script_dir = self.find_script_dir()
        print(f"Step1X-Edit script directory: {self.script_dir}")
        
    def find_script_dir(self):
        """查找 inference.py 脚本所在的目录"""
        script_dir=os.path.join(os.path.dirname(folder_paths.__file__), "custom_nodes", "ComfyUI_Step1X-Edit")
        if os.path.exists(script_dir):
            return script_dir
        raise FileNotFoundError("Could not find Step1X-Edit repository. Please set STEP1X_EDIT_PATH environment variable.")
    
    def get_model_path(self, model_type):
        """获取模型路径"""
        # 1. 检查默认模型目录
        model_dir = os.path.join(folder_paths.models_dir, "step1x", model_type)
        if os.path.exists(model_dir):
            return model_dir
        
        raise FileNotFoundError(
            f"Step1X-Edit model not found. Please download weights to one of these locations:\n"
            f"1. {os.path.join(folder_paths.models_dir, 'step1x', model_type)}\n"
        )
    
    def edit_image(self, image, instruction, model_type, use_offload, use_quantized, seed, steps, cfg_scale, size_level, lora_path=""):
        # 转换为 PIL 图像
        pil_image = tensor2pil(image)
        
        # 创建临时工作目录
        with tempfile.TemporaryDirectory() as tmp_dir:
            # 1. 保存输入图像
            input_dir = os.path.join(tmp_dir, "input")
            os.makedirs(input_dir, exist_ok=True)
            input_path = os.path.join(input_dir, "input.png")
            pil_image.save(input_path)
            
            # 2. 创建 JSON 指令文件
            json_path = os.path.join(tmp_dir, "instructions.json")
            with open(json_path, "w") as f:
                json.dump({
                    "input.png":instruction
                }, f)
            
            # 3. 设置输出目录
            output_dir = os.path.join(tmp_dir, "output")
            os.makedirs(output_dir, exist_ok=True)
            
            # 4. 获取模型路径
            model_path = self.get_model_path(model_type)
            
            # 5. 执行图像编辑
            future = self.executor.submit(
                self.run_inference,
                model_path,
                input_dir,
                json_path,
                output_dir,
                seed,
                steps,
                cfg_scale,
                use_offload,
                use_quantized,
                lora_path,
                size_level
            )
            
            # 等待任务完成
            future.result()
            
            # 6. 加载结果图像
            output_path = os.path.join(output_dir, "input.png")
            if os.path.exists(output_path):
                edited_image = Image.open(output_path)
                return (pil2tensor(edited_image),)
        
        # 失败时返回原始图像
        return (image,)
    
    def run_inference(self, model_path, input_dir, json_path, output_dir, seed, steps, cfg_scale, use_offload, use_quantized, lora_path, size_level):
        """执行推理命令"""
        try:
            # 构建命令
            cmd = [
                sys.executable,  # 使用当前 Python 解释器
                os.path.join(self.script_dir, "inference.py"),
                "--model_path", model_path,
                "--input_dir", input_dir,
                "--json_path", json_path,
                "--output_dir", output_dir,
                "--seed", str(seed),
                "--num_steps", str(steps),
                "--cfg_guidance", str(cfg_scale),
                "--size_level", str(size_level)
            ]
            
            # 添加可选参数
            if use_offload:
                cmd.append("--offload")
            if use_quantized:
                cmd.append("--quantized")
            if lora_path and os.path.exists(lora_path):
                cmd.extend(["--lora", lora_path])
            
            print(f"Executing command: {' '.join(cmd)}")
            
            # 设置环境变量
            env = os.environ.copy()
            env["PYTHONPATH"] = self.script_dir + os.pathsep + env.get("PYTHONPATH", "")
            
            # 执行命令
            result = subprocess.run(
                cmd,
                env=env,
                cwd=self.script_dir,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            # 打印输出
            if result.stdout:
                print("Step1X-Edit output:")
                print(result.stdout)
            if result.stderr:
                print("Step1X-Edit errors:")
                print(result.stderr)
            
            # 检查结果
            if result.returncode != 0:
                raise RuntimeError(f"Inference failed with code {result.returncode}")
            
            return True
        except Exception as e:
            print(f"Error running Step1X-Edit inference: {str(e)}")
            return False

# 实用函数
def tensor2pil(tensor):
    """将 ComfyUI 张量转换为 PIL 图像"""
    # 处理批处理维度
    if len(tensor.shape) == 4:
        tensor = tensor[0]  # 取第一张图像
    
    # 转换为 [0, 255] 范围的 uint8 数组
    image_np = tensor.cpu().numpy()
    image_np = np.clip(255. * image_np, 0, 255).astype(np.uint8)
    
    # 处理单通道图像
    if image_np.shape[-1] == 1:
        image_np = image_np[..., 0]
    
    # 创建 PIL 图像
    return Image.fromarray(image_np)

def pil2tensor(image):
    """将 PIL 图像转换为 ComfyUI 张量"""
    # 转换为 numpy 数组
    image_np = np.array(image).astype(np.float32) / 255.0
    
    # 处理灰度图像
    if len(image_np.shape) == 2:
        image_np = np.expand_dims(image_np, axis=-1)
    
    # 添加批处理维度
    return torch.from_numpy(image_np).unsqueeze(0)
