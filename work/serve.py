"""Tiny web demo for the Ultraman LoRA (stdlib only, no gradio/flask).

    CUDA_VISIBLE_DEVICES=1 python3 serve.py --lora ultraman-lora-r32.safetensors --port 8200

Open http://<host>:8200 , type a Chinese name (迪迦 / 赛罗 / 哥莫拉 ...), get an image.
"""
import compat_shim  # noqa: F401  (must run before diffusers imports)
import os, io, json, argparse, threading, urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import names

ap = argparse.ArgumentParser()
ap.add_argument("--model", default=os.path.expanduser("~/ultraman/models/sdxl-base-1.0"))
ap.add_argument("--lora", required=True)
ap.add_argument("--rank", type=int, default=32)
ap.add_argument("--lora-scale", type=float, default=1.0,
                help="LoRA strength; lower it if the base model's general ability drifts")
ap.add_argument("--port", type=int, default=8200)
ap.add_argument("--host", default="0.0.0.0")
ap.add_argument("--steps", type=int, default=28)
ap.add_argument("--cfg", type=float, default=6.5)
ap.add_argument("--size", type=int, default=1024)
args = ap.parse_args()

import torch
from diffusers import StableDiffusionXLPipeline, UNet2DConditionModel
from lora_util import load_into, upcast_vae

print("loading pipeline...", flush=True)
pipe = StableDiffusionXLPipeline.from_pretrained(
    args.model, torch_dtype=torch.float16, variant="fp16", use_safetensors=True).to("cuda")
upcast_vae(pipe)
unet = UNet2DConditionModel.from_pretrained(
    args.model, subfolder="unet", torch_dtype=torch.float16, variant="fp16",
    use_safetensors=True)
unet, missing, unexpected = load_into(unet, args.lora, args.rank, args.rank, args.lora_scale)
pipe.unet = unet.to("cuda")
print("lora loaded (missing %d / unexpected %d)" % (len(missing), len(unexpected)), flush=True)

LOCK = threading.Lock()

PAGE = """<!doctype html><meta charset=utf-8><title>奥特曼生成器</title>
<style>
 body{font-family:system-ui,sans-serif;max-width:760px;margin:40px auto;padding:0 16px;background:#111;color:#eee}
 input[type=text]{width:70%;padding:10px;font-size:16px;border-radius:6px;border:1px solid #444;background:#1c1c1c;color:#eee}
 button{padding:10px 18px;font-size:16px;border-radius:6px;border:0;background:#e33;color:#fff;cursor:pointer}
 img{max-width:100%;margin-top:18px;border-radius:8px}
 code{color:#8cf}
 .hint{color:#888;font-size:14px;margin-top:10px;line-height:1.6}
</style>
<h2>奥特曼 / 怪兽生成器</h2>
<form method=post action=/gen>
  <input type=text name=prompt placeholder="生迪迦 / 赛罗奥特曼在战斗 / 哥莫拉" autofocus>
  <button>生成</button>
</form>
<div class=hint>已训练身份：迪迦 赛罗 初代 赛文 杰克 艾斯 泰罗 雷欧 爱迪 戴拿 盖亚 阿古茹 高斯
奈克瑟斯 梦比优斯 希卡利 银河 欧布 捷德 泽塔 特利迦 德凯 布莱泽 亚刻 ｜ 哥莫拉 杰顿 巴尔坦 雷德王
艾雷王 美菲拉斯 泰兰特 加坦杰厄 哥尔赞 安培拉<br>
中文名会先归一成训练时用的英文 tag，所以 <code>生迪迦</code> 和 <code>diga</code> 等价。</div>
"""


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_GET(self):
        if self.path == "/health":
            body = json.dumps({"ok": True, "identities": len(names.META)}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(body)
            return
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.end_headers()
        self.wfile.write(PAGE.encode())

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        q = urllib.parse.parse_qs(self.rfile.read(n).decode())
        text = (q.get("prompt") or [""])[0].strip()
        if not text:
            self.send_response(400)
            self.end_headers()
            return
        prompt = names.normalize(text)
        print("prompt: %s -> %s" % (text, prompt), flush=True)
        with LOCK:
            g = torch.Generator(device="cuda").manual_seed(1234)
            im = pipe(prompt=prompt, negative_prompt=names.NEGATIVE,
                      num_inference_steps=args.steps, guidance_scale=args.cfg,
                      width=args.size, height=args.size, generator=g).images[0]
        buf = io.BytesIO()
        im.save(buf, format="PNG")
        self.send_response(200)
        self.send_header("Content-Type", "image/png")
        self.send_header("X-Prompt", urllib.parse.quote(prompt[:180]))
        self.end_headers()
        self.wfile.write(buf.getvalue())


print("serving on %s:%d" % (args.host, args.port), flush=True)
ThreadingHTTPServer((args.host, args.port), H).serve_forever()
